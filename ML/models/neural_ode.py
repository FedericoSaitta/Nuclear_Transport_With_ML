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

    # Collect predictions for test-level metrics
    self._test_preds = []
    self._test_trues = []
    self._test_powers = []

  def setup(self, stage=None):
    """Called after datamodule.setup — grab t_span from the datamodule."""
    self.t_span = self.trainer.datamodule.t_span.to(self.device)

  def _forward_batch(self, batch):
    """
    Shared forward pass for train/val/test.

    batch: tuple from TensorDataset, batch[0] is (batch_size, steps, features)
    Returns: u238_pred (batch_size, steps, 1), u238_true (batch_size, steps, 1)
    """
    trajectories = batch[0]  # TensorDataset wraps in a tuple

    power_profiles = trajectories[:, :, 0]      # (batch_size, steps)
    u238_true = trajectories[:, :, 1:2]          # (batch_size, steps, 1)
    y0 = u238_true[:, 0, :]                      # (batch_size, 1)

    t_span = self.t_span.to(trajectories.device)
    self.func.set_forcing(t_span, power_profiles)

    # odeint: y0 (batch, 1) → (steps, batch, 1)
    u238_pred = odeint(
        self.func, y0, t_span,
        method='dopri5', rtol=self.rtol, atol=self.atol,
    )
    # Rearrange to (batch, steps, 1)
    u238_pred = u238_pred.permute(1, 0, 2)

    return u238_pred, u238_true

  # ── Training ─────────────────────────────────────────────────────────

  def training_step(self, batch, batch_idx):
    self.func.nfe = 0
    u238_pred, u238_true = self._forward_batch(batch)

    loss = self.loss_fn(u238_pred, u238_true)

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
    u238_pred, u238_true = self._forward_batch(batch)
    loss = self.loss_fn(u238_pred, u238_true)
    self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss

  def on_validation_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["val_loss"].item()
    self._val_losses.append(epoch_loss)

  # ── Test ─────────────────────────────────────────────────────────────

  def test_step(self, batch, batch_idx):
    u238_pred, u238_true = self._forward_batch(batch)
    trajectories = batch[0]

    loss = self.loss_fn(u238_pred, u238_true)
    self.log('test_loss', loss, on_step=False, on_epoch=True)

    # Collect for plotting
    self._test_preds.append(u238_pred.squeeze(-1).cpu().numpy())
    self._test_trues.append(u238_true.squeeze(-1).cpu().numpy())
    self._test_powers.append(trajectories[:, :, 0].cpu().numpy())

    return loss

  def on_test_epoch_end(self):
    all_preds = np.concatenate(self._test_preds, axis=0)  # (num_runs, steps)
    all_trues = np.concatenate(self._test_trues, axis=0)
    all_powers = np.concatenate(self._test_powers, axis=0)

    # ── Compute metrics on flattened predictions ──
    flat_preds = all_preds.flatten()
    flat_trues = all_trues.flatten()

    mae = mean_absolute_error(flat_trues, flat_preds)
    rmse = np.sqrt(np.mean((flat_trues - flat_preds) ** 2))
    r2 = r2_score(flat_trues, flat_preds)

    logger.info(f"TEST SET ({len(all_preds)} trajectories):")
    logger.info(f"  R²:   {r2:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")

    # ── Standard ML plots (from plot.py) ──
    plot.plot_predictions_vs_actuals(flat_trues, flat_preds, mae, rmse, r2, self.result_dir)
    plot.plot_residuals_combined(flat_trues, flat_preds, self.result_dir)

    # ── Trajectory-specific plots (from plot.py) ──
    t_np = self.t_span.cpu().numpy()

    num_to_plot = min(5, len(all_preds))
    for i in range(num_to_plot):
        plot.plot_node_trajectory(
            t_np, all_preds[i], all_trues[i], all_powers[i],
            title=f'Test Trajectory {i+1}',
            save_path=f'{self.result_dir}/test_traj_{i+1}.png',
        )

    plot.plot_node_trajectory_summary(
        t_np, all_preds, all_trues,
        title=f'All Test Trajectories ({len(all_preds)} runs)',
        save_path=f'{self.result_dir}/test_all_trajectories.png',
    )

    self._test_preds.clear()
    self._test_trues.clear()
    self._test_powers.clear()

  # ── Predict ──────────────────────────────────────────────────────────

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    u238_pred, u238_true = self._forward_batch(batch)
    return {
        'pred': u238_pred.squeeze(-1).cpu(),
        'true': u238_true.squeeze(-1).cpu(),
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