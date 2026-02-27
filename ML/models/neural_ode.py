# ML/models/node_model.py
import os
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

from loguru import logger
import lightning as L
from torchdiffeq import odeint

from ML.utils import plot
from ML.models.model_architectures import ODEFuncForced
from ML.models.model_helper import get_loss_fn
import ML.datamodule.data_scalers as data_scaler

# ─── Lightning Module ────────────────────────────────────────────────────────

class NODE_Model(L.LightningModule):
  def __init__(self, config_object):
    super().__init__()
    self.save_hyperparameters(OmegaConf.to_container(config_object, resolve=True))
    self.cfg = config_object

    # Model
    self.func = ODEFuncForced(config_object)

    # Training Config
    self.loss_fn = get_loss_fn(config_object.train.loss)

    # Optimizer Config
    self.rtol = getattr(config_object.train, 'rtol', 1e-5)
    self.atol = getattr(config_object.train, 'atol', 1e-7)

    # Results directory
    self.result_dir = f'results/{config_object.model.name}/'
    os.makedirs(self.result_dir, exist_ok=True)

    # Collect losses for plotting
    self._train_losses = []
    self._val_losses = []

    # Collect data for test-level metrics
    self._test_preds = []
    self._test_trues = []
    self._test_input_trajs = []  # Scaled input features per trajectory

  def setup(self, stage=None):
    """Called after datamodule.setup — grab t_span and feature layout."""
    dm = self.trainer.datamodule
    self.t_span = dm.t_span.to(self.device)
    self.n_input_features = dm.n_input_features
    self.n_target_features = dm.n_target_features

  def _forward_batch(self, batch):
    """
    Shared forward pass for train/val/test.

    Trajectory layout: [input_features..., target_features...]
    batch[0] is (batch_size, steps, n_input_features + n_target_features)
    
    Returns: target_pred (batch_size, steps, n_target), target_true (batch_size, steps, n_target)
    """
    trajectories = batch[0]  # TensorDataset wraps in a tuple

    n_in = self.n_input_features
    
    power_profiles = trajectories[:, :, 0]            # (batch_size, steps)
    target_true = trajectories[:, :, n_in:]            # (batch_size, steps, n_target)
    y0 = target_true[:, 0, :]                          # (batch_size, n_target)

    t_span = self.t_span.to(trajectories.device)
    self.func.set_forcing(t_span, power_profiles)

    # odeint: y0 (batch, n_target) → (steps, batch, n_target)
    target_pred = odeint(
        self.func, y0, t_span,
        method='dopri5', rtol=self.rtol, atol=self.atol,
    )
    # Rearrange to (batch, steps, n_target)
    target_pred = target_pred.permute(1, 0, 2)

    return target_pred, target_true

  # ─── Unscaling helpers ─────────────────────────────────────────────

  def _unscale_targets(self, scaled_2d):
    """Unscale target values using the target_scaler. Input shape: (N, n_target)."""
    target_scaler = self.trainer.datamodule.target_scaler
    return data_scaler.inverse_transformer(target_scaler, scaled_2d)

  def _unscale_inputs(self, scaled_2d):
    """Unscale input values using the input_scaler. Input shape: (N, n_input)."""
    input_scaler = self.trainer.datamodule.input_scaler
    return data_scaler.inverse_transformer(input_scaler, scaled_2d)

  # ── Training ─────────────────────────────────────────────────────────

  def training_step(self, batch, batch_idx):
    self.func.nfe = 0
    target_pred, target_true = self._forward_batch(batch)

    loss = self.loss_fn(target_pred, target_true)

    self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log('nfe', float(self.func.nfe), on_step=False, on_epoch=True)
    return loss

  def on_train_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["train_loss"].item()
    self._train_losses.append(epoch_loss)

    lr = self.optimizers().param_groups[0]['lr']
    self.log('lr', lr, on_epoch=True, prog_bar=True)

  def on_train_end(self):
    plot.plot_losses(self._train_losses, self._val_losses, self.result_dir)
    logger.info("Training complete.")

  # ── Validation ───────────────────────────────────────────────────────

  def validation_step(self, batch, batch_idx):
    target_pred, target_true = self._forward_batch(batch)
    loss = self.loss_fn(target_pred, target_true)
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss

  def on_validation_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["val_loss"].item()
    self._val_losses.append(epoch_loss)

  # ── Test ─────────────────────────────────────────────────────────────

  def test_step(self, batch, batch_idx):
    target_pred, target_true = self._forward_batch(batch)
    trajectories = batch[0]
    n_in = self.n_input_features

    loss = self.loss_fn(target_pred, target_true)
    self.log('test_loss', loss, on_step=False, on_epoch=True)

    # Store scaled data
    self._test_preds.append(target_pred.cpu().numpy())                    # (batch, steps, n_target)
    self._test_trues.append(target_true.cpu().numpy())                    # (batch, steps, n_target)
    self._test_input_trajs.append(trajectories[:, :, :n_in].cpu().numpy()) # (batch, steps, n_input)

    return loss

  def on_test_epoch_end(self):
    all_preds_scaled = np.concatenate(self._test_preds, axis=0)        # (num_runs, steps, n_target)
    all_trues_scaled = np.concatenate(self._test_trues, axis=0)        # (num_runs, steps, n_target)
    all_inputs_scaled = np.concatenate(self._test_input_trajs, axis=0) # (num_runs, steps, n_input)

    num_runs, steps, n_target = all_preds_scaled.shape
    n_input = all_inputs_scaled.shape[2]

    # ── Unscale using separate scalers (same pattern as DNN) ──
    preds_unscaled = self._unscale_targets(all_preds_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)
    trues_unscaled = self._unscale_targets(all_trues_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)
    inputs_unscaled = self._unscale_inputs(all_inputs_scaled.reshape(-1, n_input)).reshape(num_runs, steps, n_input)

    # Flatten for overall metrics
    flat_preds = preds_unscaled.reshape(-1, n_target)
    flat_trues = trues_unscaled.reshape(-1, n_target)

    # ── Compute metrics on UNSCALED data ──
    mae = mean_absolute_error(flat_trues, flat_preds)
    rmse = np.sqrt(np.mean((flat_trues - flat_preds) ** 2))
    r2 = r2_score(flat_trues, flat_preds)

    logger.info(f"TEST SET ({num_runs} trajectories) — UNSCALED metrics:")
    logger.info(f"  R²:   {r2:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")

    self.log('Mean Absolute Error (avg)', float(mae))
    self.log('Root Mean Squared Error (avg)', float(rmse))
    self.log('R-squared coefficient (avg)', float(r2))

    # ── Standard ML plots (same as DNN) — using UNSCALED data ──
    flat_preds_1d = flat_preds.flatten()
    flat_trues_1d = flat_trues.flatten()
    plot.plot_predictions_vs_actuals(flat_trues_1d, flat_preds_1d, mae, rmse, r2, self.result_dir)
    plot.plot_residuals_combined(flat_trues_1d, flat_preds_1d, self.result_dir)

    # ── Trajectory-specific plots — using UNSCALED data ──
    t_np = self.t_span.cpu().numpy()

    # Extract first target and first input for trajectory plots
    all_preds_traj = preds_unscaled[:, :, 0]     # (num_runs, steps)
    all_trues_traj = trues_unscaled[:, :, 0]
    all_powers_traj = inputs_unscaled[:, :, 0]    # (num_runs, steps)

    num_to_plot = min(5, num_runs)
    for i in range(num_to_plot):
        plot.plot_node_trajectory(
            t_np, all_preds_traj[i], all_trues_traj[i], all_powers_traj[i],
            title=f'Test Trajectory {i+1}',
            save_path=f'{self.result_dir}/test_traj_{i+1}.png',
        )

    plot.plot_node_trajectory_summary(
        t_np, all_preds_traj, all_trues_traj,
        title=f'All Test Trajectories ({num_runs} runs)',
        save_path=f'{self.result_dir}/test_all_trajectories.png',
    )

    # ── Cleanup ──
    self._test_preds.clear()
    self._test_trues.clear()
    self._test_input_trajs.clear()

  # ── Predict ──────────────────────────────────────────────────────────

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    target_pred, target_true = self._forward_batch(batch)
    return {
        'pred': target_pred.squeeze(-1).cpu(),
        'true': target_true.squeeze(-1).cpu(),
    }

  # ── Optimizer ────────────────────────────────────────────────────────
  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self.parameters(),
      lr=self.cfg.train.learning_rate,
      weight_decay=self.cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5,
      patience=self.cfg.train.lr_scheduler_patience,
    )

    logger.info(f"Optimizer: AdamW | LR: {self.cfg.train.learning_rate} | Weight Decay: {self.cfg.train.weight_decay}")
    logger.info(f"Scheduler: ReduceLROnPlateau | Patience: {self.cfg.train.lr_scheduler_patience} | Factor: 0.5")

    return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]