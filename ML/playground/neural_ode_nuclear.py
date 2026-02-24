# Neural ODE for U238 trajectory prediction with power as external forcing
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import ML.datamodule.dataset_helper as data_help
import ML.datamodule.data_scalers as data_scalers

from sklearn.preprocessing import MinMaxScaler, StandardScaler

RESULTS_DIR = '/mnt/iusers01/fse-ugpgt01/compsci01/t97807fs/scratch/Nuclear_Transport_With_ML/ML/results/NODE'


# ─── Model ───────────────────────────────────────────────────────────────────

class ODEFuncForced(nn.Module):
    """
    Neural ODE that learns dU238/dt = f(power(t), U238(t)).
    
    Power is NOT part of the ODE state — it's an external forcing function
    that is interpolated at whatever time the solver requests.
    The ODE state is just [U238], a 1D variable.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.nfe = 0
        self.t_points = None       # time grid for power interpolation
        self.power_profile = None  # known power values at those times

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),     # inputs: [power_at_t, U238]
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),     # output: dU238/dt
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_forcing(self, t_points, power_profile):
        """Set the known power schedule before calling odeint."""
        self.t_points = t_points
        self.power_profile = power_profile

    def _interpolate_power(self, t):
        """Linearly interpolate power at an arbitrary time t."""
        t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
        idx = torch.searchsorted(self.t_points, t_clamped) - 1
        idx = idx.clamp(0, len(self.t_points) - 2)

        t0 = self.t_points[idx]
        t1 = self.t_points[idx + 1]
        p0 = self.power_profile[idx]
        p1 = self.power_profile[idx + 1]

        frac = (t_clamped - t0) / (t1 - t0 + 1e-8)
        return p0 + frac * (p1 - p0)

    def forward(self, t, y):
        """
        t: scalar time (called by the ODE solver)
        y: (batch, 1) — just U238
        """
        self.nfe += 1
        power = self._interpolate_power(t)       # scalar
        power = power.expand(y.shape[0], 1)       # (batch, 1)
        combined = torch.cat([power, y], dim=-1)  # (batch, 2)
        return self.net(combined)                  # (batch, 1) = dU238/dt


# ─── Training & Evaluation ──────────────────────────────────────────────────

def train_one_epoch(func, trajectories, t_span, optimizer, device):
    """
    trajectories: (num_runs, steps_per_run, 2) where [:, :, 0]=power, [:, :, 1]=U238
    """
    func.train()
    func.nfe = 0
    epoch_loss = 0.0
    num_runs = trajectories.shape[0]

    # Shuffle run order each epoch
    perm = torch.randperm(num_runs)

    for i in perm:
        traj = trajectories[i].to(device)          # (steps, 2)
        power_profile = traj[:, 0]                  # (steps,)
        u238_true = traj[:, 1:2]                    # (steps, 1)
        y0 = u238_true[0:1]                         # (1, 1) — initial U238

        func.set_forcing(t_span, power_profile)

        optimizer.zero_grad()
        # odeint with y0 shape (1, 1) → output shape (steps, 1, 1)
        u238_pred = odeint(func, y0, t_span, method='dopri5', rtol=1e-5, atol=1e-7)
        u238_pred = u238_pred.squeeze(-1)           # (steps, 1)

        loss = ((u238_pred - u238_true) ** 2).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / num_runs


@torch.no_grad()
def evaluate_trajectories(func, trajectories, t_span, device):
    func.eval()
    total_loss = 0.0
    num_runs = trajectories.shape[0]

    for i in range(num_runs):
        traj = trajectories[i].to(device)
        power_profile = traj[:, 0]
        u238_true = traj[:, 1:2]
        y0 = u238_true[0:1]

        func.set_forcing(t_span, power_profile)
        u238_pred = odeint(func, y0, t_span, method='dopri5', rtol=1e-5, atol=1e-7)
        u238_pred = u238_pred.squeeze(-1)

        loss = ((u238_pred - u238_true) ** 2).mean()
        total_loss += loss.item()

    return total_loss / num_runs


@torch.no_grad()
def predict_trajectory(func, trajectory, t_span, device):
    """Predict a single trajectory. Returns (u238_pred, u238_true, power) as numpy."""
    func.eval()
    traj = trajectory.to(device)
    power_profile = traj[:, 0]
    u238_true = traj[:, 1:2]
    y0 = u238_true[0:1]

    func.set_forcing(t_span, power_profile)
    u238_pred = odeint(func, y0, t_span, method='dopri5', rtol=1e-5, atol=1e-7)
    u238_pred = u238_pred.squeeze(-1)

    return (
        u238_pred.cpu().numpy(),
        u238_true.cpu().numpy(),
        power_profile.cpu().numpy(),
    )


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Validation')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_trajectory_prediction(t, u238_pred, u238_true, power, title, save_path):
    """Three-panel plot: power input, U238 prediction vs truth, residual."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 8), height_ratios=[1, 2, 1],
        sharex=True, gridspec_kw={'hspace': 0.1},
    )

    # Power forcing
    ax1.plot(t, power, color='tab:orange', linewidth=1.0)
    ax1.set_ylabel('Power (scaled)')
    ax1.set_title(title)

    # U238 predictions vs truth
    ax2.plot(t, u238_true, label='Truth', linewidth=1.0)
    ax2.plot(t, u238_pred, label='Prediction', linewidth=1.0, linestyle='--')
    ax2.set_ylabel('U238 (scaled)')
    ax2.legend()

    # Residuals
    residuals = u238_pred - u238_true
    ax3.plot(t, residuals, color='tab:red', linewidth=0.8)
    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax3.set_ylabel('Residual')
    ax3.set_xlabel('Time')

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_multi_trajectory_summary(t, all_preds, all_trues, title, save_path):
    """Overlay multiple trajectory predictions."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[3, 1],
        sharex=True, gridspec_kw={'hspace': 0.05},
    )

    all_residuals = []
    for i, (pred, true) in enumerate(zip(all_preds, all_trues)):
        ax1.plot(t, true, color='tab:blue', alpha=0.3, linewidth=0.5,
                 label='Truth' if i == 0 else None)
        ax1.plot(t, pred, color='tab:red', alpha=0.3, linewidth=0.5,
                 label='Prediction' if i == 0 else None)
        all_residuals.append(pred - true)

    ax1.set_ylabel('U238 (scaled)')
    ax1.set_title(title)
    ax1.legend()

    # Mean absolute residual across all trajectories
    mean_abs_res = np.mean(np.abs(np.array(all_residuals)), axis=0)
    ax2.plot(t, mean_abs_res, color='tab:red', linewidth=1.0)
    ax2.set_ylabel('Mean |Residual|')
    ax2.set_xlabel('Time')

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ─── Data Reshaping ─────────────────────────────────────────────────────────

def reshape_to_trajectories(X, steps_per_run):
    """
    Reshape flat array (num_runs * steps_per_run, features) 
    → (num_runs, steps_per_run, features)
    """
    num_samples, num_features = X.shape
    num_runs = num_samples // steps_per_run
    assert num_runs * steps_per_run == num_samples, \
        f"Data length {num_samples} not divisible by steps_per_run {steps_per_run}"
    return torch.tensor(
        X.reshape(num_runs, steps_per_run, num_features),
        dtype=torch.float32,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_frac = 3 / 3305
    steps_per_run = 101

    data_df, run_length, time_array = data_help.read_data(
        '/mnt/iusers01/fse-ugpgt01/compsci01/t97807fs/scratch/Nuclear_Transport_With_ML/ML/data/casl_3305_runs_inter.h5',
        data_frac, drop_run_label=True,
    )
    data_help.print_dataset_stats(data_df)

    all_columns = ['power_W_g', 'U238']
    data_df = data_help.filter_columns(data_df, all_columns)

    input_data_arr, col_index_map = data_help.split_df(data_df, all_columns)

    # ── Scale the data ──
    # For trajectory mode we scale inputs only (no separate target scaler needed 
    # since U238 appears in the input and we compare absolute values)
    input_scaler = data_scalers.create_column_transformer(
        {'power_W_g': MinMaxScaler(), 'U238': StandardScaler()}, col_index_map,
    )
    scaled_data = input_scaler.fit_transform(input_data_arr)

    # ── Split into runs, THEN into train/val/test by run ──
    # Shape: (num_runs, steps_per_run, 2)
    all_trajectories = reshape_to_trajectories(scaled_data, steps_per_run)

    # Drop last timestep: power at t is used to predict U238 at t+1,
    # so the final row has NaN power
    all_trajectories = all_trajectories[:, :-1, :]
    actual_steps = steps_per_run - 1
 

    print(all_trajectories)
    # Verify no NaNs remain
    assert not torch.isnan(all_trajectories).any(), \
        f"NaNs found in trajectories! Count: {torch.isnan(all_trajectories).sum()}"

    num_runs = all_trajectories.shape[0]
    print(f"Total trajectories: {num_runs}, each {actual_steps} steps (after dropping last)")

    # Split by run (not by timestep!) so no trajectory is split across sets
    perm = torch.randperm(num_runs)
    all_trajectories = all_trajectories[perm]

    if num_runs <= 3:
        # Small dataset: 1 train, 1 val, rest test
        train_trajs = all_trajectories[:1]
        val_trajs   = all_trajectories[1:2]
        test_trajs  = all_trajectories[2:]
    else:
        n_train = int(num_runs * 0.6)
        n_val   = int(num_runs * 0.2)
        train_trajs = all_trajectories[:n_train]
        val_trajs   = all_trajectories[n_train:n_train + n_val]
        test_trajs  = all_trajectories[n_train + n_val:]

    print(f"Train: {len(train_trajs)} runs, Val: {len(val_trajs)} runs, Test: {len(test_trajs)} runs")

    # ── Time span ──
    t_span = torch.tensor(time_array[:actual_steps], dtype=torch.float32).to(device)

    # ── Model, optimiser, scheduler ──
    hidden_dim = 64
    func = ODEFuncForced(hidden_dim).to(device)
    optimizer = torch.optim.AdamW(func.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5,
    )

    # ── Training loop ──
    num_epochs = 500
    best_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        avg_train = train_one_epoch(func, train_trajs, t_span, optimizer, device)
        avg_val   = evaluate_trajectories(func, val_trajs, t_span, device)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            best_model_state = {k: v.clone() for k, v in func.state_dict().items()}

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {lr:.2e} | NFE: {func.nfe}")

    print(f"\nBest val loss: {best_loss:.6f} — restoring best weights")
    func.load_state_dict(best_model_state)

    # ── Plot loss curves ──
    plot_loss_curves(train_losses, val_losses, f'{RESULTS_DIR}/loss_curves.png')

    # ── Plot individual test trajectory predictions ──
    t_np = t_span.cpu().numpy()
    num_to_plot = min(5, len(test_trajs))
    all_preds, all_trues = [], []

    for i in range(num_to_plot):
        u238_pred, u238_true, power = predict_trajectory(func, test_trajs[i], t_span, device)

        plot_trajectory_prediction(
            t_np, u238_pred.squeeze(), u238_true.squeeze(), power,
            title=f'Test Trajectory {i+1}',
            save_path=f'{RESULTS_DIR}/test_traj_{i+1}.png',
        )
        all_preds.append(u238_pred.squeeze())
        all_trues.append(u238_true.squeeze())

    # ── Summary plot: all test trajectories overlaid ──
    # Predict ALL test trajectories for the summary
    all_preds_full, all_trues_full = [], []
    for i in range(len(test_trajs)):
        u238_pred, u238_true, _ = predict_trajectory(func, test_trajs[i], t_span, device)
        all_preds_full.append(u238_pred.squeeze())
        all_trues_full.append(u238_true.squeeze())

    plot_multi_trajectory_summary(
        t_np, all_preds_full, all_trues_full,
        title=f'All Test Trajectories ({len(test_trajs)} runs)',
        save_path=f'{RESULTS_DIR}/test_all_trajectories.png',
    )

    # ── Print test metrics ──
    all_preds_arr = np.array(all_preds_full)
    all_trues_arr = np.array(all_trues_full)
    residuals = all_preds_arr - all_trues_arr

    print(f"\nTEST SET ({len(test_trajs)} trajectories):")
    print(f"  MSE:  {np.mean(residuals**2):.6f}")
    print(f"  MAE:  {np.mean(np.abs(residuals)):.6f}")
    print(f"  Mean residual: {np.mean(residuals):.6f}")
    print(f"  Std residual:  {np.std(residuals):.6f}")
    print(f"\nAll plots saved to {RESULTS_DIR}/")

if __name__ == '__main__':
    main() # python playground/neural_ode_nuclear.py