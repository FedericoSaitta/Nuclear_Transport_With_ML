# Reshape and mold the csv/h5 dataset to be more approachable for ML
import re
import h5py
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt

import ML.datamodule.dataset_helper as data_help
import ML.datamodule.data_scalers as data_scalers

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)

RESULTS_DIR = '/mnt/iusers01/fse-ugpgt01/compsci01/t97807fs/scratch/Nuclear_Transport_With_ML/ML/results/NODE'


## Now defining the NODE module
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.nfe = 0  # number of function evaluations
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),   # U238 + power as input
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),   # only output dU238/dt
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.nfe += 1
        # y contains [power, U238] but we only evolve U238
        return torch.cat([torch.zeros_like(y[..., :1]), self.net(y)], dim=-1)


def train_one_epoch(func, train_loader, optimizer, t_span, device):
    func.train()
    func.nfe = 0
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        # y_pred shape: (len(t_span), batch, 2)
        y_pred = odeint(func, X_batch, t_span, method='dopri5')
        y_end = y_pred[-1]  # state at t=dt

        # Target is delta concentration, so predicted change = end - start
        delta_pred = y_end[:, 1:] - X_batch[:, 1:]  # only U238 change
        loss = ((delta_pred - y_batch) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


@torch.no_grad()
def evaluate(func, loader, t_span, device):
    func.eval()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = odeint(func, X_batch, t_span, method='dopri5')
        y_end = y_pred[-1]
        delta_pred = y_end[:, 1:] - X_batch[:, 1:]
        loss = ((delta_pred - y_batch) ** 2).mean()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict_all(func, loader, t_span, device):
    """Collect all predictions and targets from a loader."""
    func.eval()
    all_X, all_y_true, all_delta_pred = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = odeint(func, X_batch, t_span, method='dopri5')
        y_end = y_pred[-1]
        delta_pred = y_end[:, 1:] - X_batch[:, 1:]

        all_X.append(X_batch.cpu().numpy())
        all_y_true.append(y_batch.cpu().numpy())
        all_delta_pred.append(delta_pred.cpu().numpy())

    return (
        np.concatenate(all_X),
        np.concatenate(all_y_true),
        np.concatenate(all_delta_pred),
    )


# ─── Plotting helpers ────────────────────────────────────────────────────────

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


def plot_predictions_vs_truth(y_true, y_pred, title, ylabel, save_path):
    """Scatter + residual subplot for a single variable."""
    residuals = y_pred - y_true

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[3, 1],
        sharex=True, gridspec_kw={'hspace': 0.05},
    )

    sample_idx = np.arange(len(y_true))

    ax1.plot(sample_idx, y_true, label='Truth', linewidth=0.8)
    ax1.plot(sample_idx, y_pred, label='Prediction', linewidth=0.8, alpha=0.8)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend()

    ax2.plot(sample_idx, residuals, color='tab:red', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Residual')
    ax2.set_xlabel('Sample Index')

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_scatter(y_true, y_pred, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=5, alpha=0.5)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, 'k--', linewidth=0.8, label='y = x')
    ax.set_xlabel('True ΔU238')
    ax.set_ylabel('Predicted ΔU238')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_residual_histogram(y_true, y_pred, title, save_path):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Residual (Pred − True)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_frac = 300 / 3305

    data_df, run_length, time_array = data_help.read_data(
        '/mnt/iusers01/fse-ugpgt01/compsci01/t97807fs/scratch/Nuclear_Transport_With_ML/ML/data/casl_3305_runs_inter.h5',
        data_frac, drop_run_label=True,
    )
    data_help.print_dataset_stats(data_df)

    all_columns = ['power_W_g', 'U238']
    target = ['U238']

    data_df = data_help.filter_columns(data_df, all_columns)

    input_data_arr, col_index_map = data_help.split_df(data_df, all_columns)
    target_data_arr, target_index_map = data_help.split_df(data_df, target)

    input_scaler = data_scalers.create_column_transformer(
        {'power_W_g': MinMaxScaler(), 'U238': StandardScaler()}, col_index_map,
    )
    target_scaler = data_scalers.create_column_transformer(
        {'U238': StandardScaler()}, target_index_map,
    )

    X, Y = data_help.create_timeseries_targets(
        input_data_arr, target_data_arr, time_array,
        col_index_map, target_index_map, delta_conc=True,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = data_help.timeseries_train_val_test_split(
        X, Y, train_frac=1/3.0, val_frac=1/3.0, test_frac=1/3.0,
        steps_per_run=100, shuffle_within_train=False,
    )

    # ── Pre-scaling diagnostic plots ──
    plt.plot(X_train[:, 0]); plt.plot(X_val[:, 0]); plt.plot(X_test[:, 0])
    plt.title('Power'); plt.savefig(f'{RESULTS_DIR}/Power.png'); plt.close()

    plt.plot(X_train[:, 1]); plt.plot(X_val[:, 1]); plt.plot(X_test[:, 1])
    plt.title('U238'); plt.savefig(f'{RESULTS_DIR}/U238_data.png'); plt.close()

    X_train, X_val, X_test, y_train, y_val, y_test = data_help.scale_datasets(
        X_train, X_val, X_test, y_train, y_val, y_test, input_scaler, target_scaler,
    )

    # ── Post-scaling diagnostic plots ──
    plt.plot(X_train[:, 0]); plt.plot(X_val[:, 0]); plt.plot(X_test[:, 0])
    plt.title('Scaled Power'); plt.savefig(f'{RESULTS_DIR}/Scaled_Power.png'); plt.close()

    plt.plot(X_train[:, 1]); plt.plot(X_val[:, 1]); plt.plot(X_test[:, 1])
    plt.title('Scaled U238'); plt.savefig(f'{RESULTS_DIR}/Scaled_U238_data.png'); plt.close()

    plt.plot(y_train[:, 0]); plt.plot(y_val[:, 0]); plt.plot(y_test[:, 0])
    plt.title('Scaled U238 Target'); plt.savefig(f'{RESULTS_DIR}/Scaled_U238_Target_data.png'); plt.close()

    # ── Create datasets and loaders ──
    train_dataset, val_dataset, test_dataset = data_help.create_tensor_datasets(
      X_train, X_val, X_test, y_train, y_val, y_test,
    )

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ── Model, optimiser, scheduler — all on device ──
    dt = float(time_array[1] - time_array[0])
    t_span = torch.tensor([0.0, dt]).to(device)

    hidden_dim = 64
    func = ODEFunc(hidden_dim).to(device)
    optimizer = torch.optim.AdamW(func.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5,
    )

    # ── Training loop ──
    num_epochs = 5_000
    best_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        avg_train = train_one_epoch(func, train_loader, optimizer, t_span, device)
        avg_val   = evaluate(func, val_loader, t_span, device)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            best_model_state = {k: v.clone() for k, v in func.state_dict().items()}

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {lr:.2e} | NFE: {func.nfe}")

    print(f"\nBest validation loss: {best_loss:.6f} — restoring best model weights")
    func.load_state_dict(best_model_state)

    # ── Plots: learning curves ──
    plot_loss_curves(train_losses, val_losses, f'{RESULTS_DIR}/loss_curves.png')

    # ── Plots: predictions on each split ──
    for name, loader in [('test', test_loader)]:
        X_all, y_true_all, delta_pred_all = predict_all(func, loader, t_span, device)

        plot_scatter(
            y_true_all[:, 0], delta_pred_all[:, 0],
            title=f'ΔU238 Parity Plot ({name})',
            save_path=f'{RESULTS_DIR}/{name}_scatter.png',
        )

        plot_residual_histogram(
            y_true_all[:, 0], delta_pred_all[:, 0],
            title=f'Residual Distribution ({name})',
            save_path=f'{RESULTS_DIR}/{name}_residual_hist.png',
        )

        # Print summary statistics
        residuals = delta_pred_all[:, 0] - y_true_all[:, 0]
        print(f"\n{name.upper()} set:")
        print(f"  MSE:  {np.mean(residuals**2):.6f}")
        print(f"  MAE:  {np.mean(np.abs(residuals)):.6f}")
        print(f"  Mean residual: {np.mean(residuals):.6f}")
        print(f"  Std residual:  {np.std(residuals):.6f}")

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
    # To run: python playground/neural_ode_nuclear.py