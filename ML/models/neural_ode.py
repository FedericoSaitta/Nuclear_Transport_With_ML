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
from ML.utils import metrics
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
    self._test_input_trajs = []

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
    trajectories = batch[0]

    n_in = self.n_input_features
    
    power_profiles = trajectories[:, :, 0]
    target_true = trajectories[:, :, n_in:]
    y0 = target_true[:, 0, :]

    t_span = self.t_span.to(trajectories.device)
    self.func.set_forcing(t_span, power_profiles)

    target_pred = odeint(
        self.func, y0, t_span,
        method='dopri5', rtol=self.rtol, atol=self.atol,
    )
    target_pred = target_pred.permute(1, 0, 2)

    return target_pred, target_true

  # ─── Unscaling helpers ─────────────────────────────────────────────

  def _unscale_targets(self, scaled_2d):
    """Unscale target values. Input shape: (N, n_target)."""
    target_scaler = self.trainer.datamodule.target_scaler
    return data_scaler.inverse_transformer(target_scaler, scaled_2d)

  def _unscale_inputs(self, scaled_2d):
    """Unscale input values. Input shape: (N, n_input)."""
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

    self._test_preds.append(target_pred.cpu().numpy())
    self._test_trues.append(target_true.cpu().numpy())
    self._test_input_trajs.append(trajectories[:, :, :n_in].cpu().numpy())

    return loss

  def on_test_epoch_end(self):
    datamodule = self.trainer.datamodule
    
    all_preds_scaled = np.concatenate(self._test_preds, axis=0)        # (num_runs, steps, n_target)
    all_trues_scaled = np.concatenate(self._test_trues, axis=0)        # (num_runs, steps, n_target)
    all_inputs_scaled = np.concatenate(self._test_input_trajs, axis=0) # (num_runs, steps, n_input)

    num_runs, steps, n_target = all_preds_scaled.shape
    n_input = all_inputs_scaled.shape[2]

    # ── Unscale using separate scalers ──
    ar_preds_unscaled = self._unscale_targets(all_preds_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)
    trues_unscaled = self._unscale_targets(all_trues_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)
    inputs_unscaled = self._unscale_inputs(all_inputs_scaled.reshape(-1, n_input)).reshape(num_runs, steps, n_input)

    # ── Teacher-forced predictions (single-step NODE) ──
    logger.info("Computing teacher-forced (single-step) predictions...")
    tf_preds_scaled = self._teacher_forced_predictions(all_inputs_scaled, all_trues_scaled)
    tf_preds_unscaled = self._unscale_targets(tf_preds_scaled.reshape(-1, n_target)).reshape(num_runs, steps, n_target)

    target_names = list(datamodule.target.keys())
    logger.info(f"Test set: {num_runs} trajectories, {steps} steps, {n_target} targets")
    
    # 1. Overall metrics + per-target plots
    flat_ar_preds = ar_preds_unscaled.reshape(-1, n_target)
    flat_trues = trues_unscaled.reshape(-1, n_target)
    
    mae_per_output = metrics.mae(flat_trues, flat_ar_preds)
    rmse_per_output = metrics.rmse(flat_trues, flat_ar_preds)
    r2_per_output = metrics.r2(flat_trues, flat_ar_preds)
    
    self.log('Mean Absolute Error (avg)', float(mae_per_output.mean()))
    self.log('Root Mean Squared Error (avg)', float(rmse_per_output.mean()))
    self.log('R-squared coefficient (avg)', float(r2_per_output.mean()))
    
    logger.info(f"TEST SET — UNSCALED metrics (Autoregressive):")
    logger.info(f"  R² (avg):   {r2_per_output.mean():.6f}")
    logger.info(f"  RMSE (avg): {rmse_per_output.mean():.6f}")
    logger.info(f"  MAE (avg):  {mae_per_output.mean():.6f}")

    # 2. Per-target: predictions vs actuals + residuals
    per_target_metrics = []
    for idx, target_name in enumerate(target_names):
      output_dir = os.path.join(self.result_dir, target_name)
      os.makedirs(output_dir, exist_ok=True)
      
      y_true_flat = flat_trues[:, idx]
      y_pred_flat = flat_ar_preds[:, idx]
      
      plot.plot_predictions_vs_actuals(
        y_true_flat, y_pred_flat,
        mae_per_output[idx], rmse_per_output[idx], r2_per_output[idx],
        output_dir
      )
      plot.plot_residuals_combined(y_true_flat, y_pred_flat, output_dir)
      
      per_target_metrics.append({
        'name': target_name,
        'mae': float(mae_per_output[idx]),
        'rmse': float(rmse_per_output[idx]),
        'r2': float(r2_per_output[idx])
      })

    # 3. MARE comparison (Teacher-Forcing vs Autoregressive)
    self._compute_mare_comparison(
      trues_unscaled, ar_preds_unscaled, tf_preds_unscaled,
      target_names, per_target_metrics
    )

    # 4. Feature importance
    self._compute_feature_importance(
      all_inputs_scaled, all_trues_scaled, target_names, datamodule
    )
    
    # 5. Prediction comparison plots (TF vs AR)
    self._plot_prediction_comparisons(
      trues_unscaled, ar_preds_unscaled, tf_preds_unscaled, target_names
    )

    # 6. Error growth (MAE, MALE)
    self._plot_error_growth(
      trues_unscaled, ar_preds_unscaled, tf_preds_unscaled, target_names
    )

    # 7. Trajectory-specific plots
    t_np = self.t_span.cpu().numpy()
    num_to_plot = min(5, num_runs)
    for i in range(num_to_plot):
      plot.plot_node_trajectory(
        t_np, ar_preds_unscaled[i, :, 0], trues_unscaled[i, :, 0], inputs_unscaled[i, :, 0],
        title=f'Test Trajectory {i+1}',
        save_path=f'{self.result_dir}/test_traj_{i+1}.png',
      )

    plot.plot_node_trajectory_summary(
      t_np, ar_preds_unscaled[:, :, 0], trues_unscaled[:, :, 0],
      title=f'All Test Trajectories ({num_runs} runs)',
      save_path=f'{self.result_dir}/test_all_trajectories.png',
    )

    # 8. Log to database
    if hasattr(self.trainer, 'logger') and hasattr(self.trainer.logger, 'update_final_results'):
      test_metrics = {
        'mae_avg': float(mae_per_output.mean()),
        'rmse_avg': float(rmse_per_output.mean()),
        'r2_avg': float(r2_per_output.mean()),
        'per_target': per_target_metrics
      }
      self.trainer.logger.update_final_results(
        train_losses=self._train_losses,
        val_losses=self._val_losses,
        val_r2_scores=[],
        val_mae_scores=[],
        test_metrics=test_metrics
      )

    # Cleanup
    self._test_preds.clear()
    self._test_trues.clear()
    self._test_input_trajs.clear()

  # ─── Teacher-Forced Predictions ──────────────────────────────────────

  def _teacher_forced_predictions(self, all_inputs_scaled, all_trues_scaled):
    """
    Single-step NODE predictions: at each timestep t, use ground truth y(t) 
    as initial condition and integrate one dt forward to predict y(t+1).
    This is analogous to the DNN's teacher-forcing mode.
    
    Returns: (num_runs, steps, n_target) array of scaled predictions.
    """
    num_runs, steps, n_target = all_trues_scaled.shape
    tf_preds = np.zeros_like(all_trues_scaled)
    
    # First timestep: prediction = ground truth (no prior step to predict from)
    tf_preds[:, 0, :] = all_trues_scaled[:, 0, :]
    
    device = self.device
    t_span = self.t_span.to(device)
    
    # Process in batches to avoid OOM
    batch_size = 64
    
    for batch_start in range(0, num_runs, batch_size):
      batch_end = min(batch_start + batch_size, num_runs)
      
      inputs_batch = torch.tensor(
        all_inputs_scaled[batch_start:batch_end], dtype=torch.float32, device=device
      )
      trues_batch = torch.tensor(
        all_trues_scaled[batch_start:batch_end], dtype=torch.float32, device=device
      )
      
      power_profiles = inputs_batch[:, :, 0]
      self.func.set_forcing(t_span, power_profiles)
      
      # Single-step integration for each timestep
      for t in range(steps - 1):
        y_t = trues_batch[:, t, :]  # Ground truth at t
        t_short = t_span[t:t+2]     # Integrate from t to t+1
        
        with torch.no_grad():
          pred_t1 = odeint(
            self.func, y_t, t_short,
            method='dopri5', rtol=self.rtol, atol=self.atol,
          )[-1]  # Take last timepoint = t+1
        
        tf_preds[batch_start:batch_end, t+1, :] = pred_t1.cpu().numpy()
    
    return tf_preds

  # ─── MARE Comparison ─────────────────────────────────────────────────

  def _compute_mare_comparison(self, trues_unscaled, ar_preds_unscaled, 
                               tf_preds_unscaled, target_names, per_target_metrics=None):
    """Compare MARE between teacher-forcing and autoregressive (same as DNN)."""
    logger.info(f"\n{'='*20}")
    logger.info("MARE: Teacher-Forcing vs Autoregressive")
    logger.info(f"{'='*20}")
    
    for idx, target_name in enumerate(target_names):
      # Flatten across runs and timesteps
      ar_gt = trues_unscaled[:, :, idx].flatten()
      ar_pred = ar_preds_unscaled[:, :, idx].flatten()
      tf_gt = trues_unscaled[:, :, idx].flatten()
      tf_pred = tf_preds_unscaled[:, :, idx].flatten()
      
      mare_ar = metrics.mare(ar_gt, ar_pred)
      mare_tf = metrics.mare(tf_gt, tf_pred)
      
      self.log(f'{target_name}/MARE_TeacherForcing', float(mare_tf))
      self.log(f'{target_name}/MARE_Autoregressive', float(mare_ar))
      
      logger.info(f"  {target_name}: MARE(TF)={mare_tf:.6f}, MARE(AR)={mare_ar:.6f}")
      
      if per_target_metrics is not None:
        per_target_metrics[idx]['mare_tf'] = float(mare_tf)
        per_target_metrics[idx]['mare_ar'] = float(mare_ar)

  # ─── Feature Importance ──────────────────────────────────────────────

  def _compute_feature_importance(self, all_inputs_scaled, all_trues_scaled, 
                                  target_names, datamodule):
    """
    Permutation feature importance for the NODE.
    Shuffles each input feature across runs and measures the impact on R².
    """
    logger.info(f"\n{'='*20}")
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info(f"{'='*20}")
    
    feature_names = [
      key for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])
    ]
    
    num_runs, steps, n_input = all_inputs_scaled.shape
    n_target = all_trues_scaled.shape[2]
    n_repeats = 5
    device = self.device
    t_span = self.t_span.to(device)
    
    # Compute baseline predictions (already have them, but recompute for consistency)
    baseline_preds = self._batch_forward_numpy(all_inputs_scaled, all_trues_scaled)
    
    for idx, target_name in enumerate(target_names):
      output_dir = os.path.join(self.result_dir, target_name)
      os.makedirs(output_dir, exist_ok=True)
      
      logger.info(f"\nComputing feature importance for: {target_name}")
      
      # Baseline R²
      baseline_r2 = r2_score(
        all_trues_scaled[:, :, idx].flatten(),
        baseline_preds[:, :, idx].flatten()
      )
      
      importance_means = np.zeros(n_input)
      importance_stds = np.zeros(n_input)
      
      for feat_idx in range(n_input):
        feat_scores = []
        
        for repeat in range(n_repeats):
          # Shuffle this feature across runs
          inputs_permuted = all_inputs_scaled.copy()
          perm = np.random.permutation(num_runs)
          inputs_permuted[:, :, feat_idx] = all_inputs_scaled[perm, :, feat_idx]
          
          # Get predictions with permuted feature
          permuted_preds = self._batch_forward_numpy(inputs_permuted, all_trues_scaled)
          
          permuted_r2 = r2_score(
            all_trues_scaled[:, :, idx].flatten(),
            permuted_preds[:, :, idx].flatten()
          )
          
          # Importance = drop in R² (higher = more important)
          feat_scores.append(baseline_r2 - permuted_r2)
        
        importance_means[feat_idx] = np.mean(feat_scores)
        importance_stds[feat_idx] = np.std(feat_scores)
      
      # Plot using same function as DNN
      plot.plot_feature_importance(
        importance_means, importance_stds, feature_names,
        baseline_r2, output_dir, 'r2_score', n_top=20
      )
      
      logger.info(f"  ✓ Feature importance plots saved to: {output_dir}")

  def _batch_forward_numpy(self, inputs_scaled, trues_scaled):
    """
    Run NODE forward pass on numpy arrays in batches.
    Returns: (num_runs, steps, n_target) numpy array of scaled predictions.
    """
    num_runs = inputs_scaled.shape[0]
    n_target = trues_scaled.shape[2]
    steps = inputs_scaled.shape[1]
    all_preds = np.zeros((num_runs, steps, n_target))
    
    device = self.device
    t_span = self.t_span.to(device)
    batch_size = 64
    
    with torch.no_grad():
      for batch_start in range(0, num_runs, batch_size):
        batch_end = min(batch_start + batch_size, num_runs)
        
        inputs_batch = torch.tensor(
          inputs_scaled[batch_start:batch_end], dtype=torch.float32, device=device
        )
        trues_batch = torch.tensor(
          trues_scaled[batch_start:batch_end], dtype=torch.float32, device=device
        )
        
        power_profiles = inputs_batch[:, :, 0]
        y0 = trues_batch[:, 0, :]
        
        self.func.set_forcing(t_span, power_profiles)
        
        pred = odeint(
          self.func, y0, t_span,
          method='dopri5', rtol=self.rtol, atol=self.atol,
        ).permute(1, 0, 2)
        
        all_preds[batch_start:batch_end] = pred.cpu().numpy()
    
    return all_preds

  # ─── Prediction Comparisons ──────────────────────────────────────────

  def _plot_prediction_comparisons(self, trues_unscaled, ar_preds_unscaled, 
                                   tf_preds_unscaled, target_names):
    """Plot teacher-forcing vs autoregressive predictions (same as DNN)."""
    logger.info(f"\n{'='*20}")
    logger.info("PREDICTION COMPARISON (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")
    
    for idx, target_name in enumerate(target_names):
      logger.info(f"\nPlotting prediction comparison for: {target_name}")
      
      # Use first run for comparison plot
      ground_truth = trues_unscaled[0, :, idx]
      teacher_forced_preds = tf_preds_unscaled[0, :, idx]
      autoregressive_preds = ar_preds_unscaled[0, :, idx]
      
      output_dir = os.path.join(self.result_dir, target_name)
      plot.plot_prediction_comparison(
        ground_truth, teacher_forced_preds,
        autoregressive_preds, target_name, output_dir
      )
      
      logger.info(f"  ✓ Comparison plot saved to: {output_dir}")

  # ─── Error Growth ───────────────────────────────────────────────────

  def _plot_error_growth(self, trues_unscaled, ar_preds_unscaled, 
                         tf_preds_unscaled, target_names):
    """Calculate and plot how prediction error grows over time (same as DNN)."""
    logger.info(f"\n{'='*20}")
    logger.info("ERROR GROWTH ANALYSIS (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")
    
    num_runs = trues_unscaled.shape[0]
    logger.info(f"Analyzing error growth across {num_runs} runs")
    
    for idx, target_name in enumerate(target_names):
      logger.info(f"\nAnalyzing error growth for: {target_name}")
      
      # Extract per-target data: (num_runs, steps)
      gt = trues_unscaled[:, :, idx]
      ar = ar_preds_unscaled[:, :, idx]
      tf = tf_preds_unscaled[:, :, idx]
      
      # MAE per run per timestep
      tf_mae_errors = np.abs(tf - gt)       # (num_runs, steps)
      ar_mae_errors = np.abs(ar - gt)
      
      # MALE per run per timestep
      epsilon = 1e-20
      tf_male_errors = np.abs(
        np.log10(np.abs(tf) + epsilon) - np.log10(np.abs(gt) + epsilon)
      )
      ar_male_errors = np.abs(
        np.log10(np.abs(ar) + epsilon) - np.log10(np.abs(gt) + epsilon)
      )
      
      # Statistics across runs for each timestep
      avg_tf_mae = np.mean(tf_mae_errors, axis=0)
      avg_ar_mae = np.mean(ar_mae_errors, axis=0)
      std_tf_mae = np.std(tf_mae_errors, axis=0)
      std_ar_mae = np.std(ar_mae_errors, axis=0)
      
      avg_tf_male = np.mean(tf_male_errors, axis=0)
      avg_ar_male = np.mean(ar_male_errors, axis=0)
      std_tf_male = np.std(tf_male_errors, axis=0)
      std_ar_male = np.std(ar_male_errors, axis=0)
      
      # Log final MALE values
      self.log(f'{target_name}/Final MALE (AR)', float(avg_ar_male[-1]))
      self.log(f'{target_name}/Final MALE (TF)', float(avg_tf_male[-1]))
      
      output_dir = os.path.join(self.result_dir, target_name)
      
      # Plot MAE over time
      plot.plot_error_growth_metric(
        avg_tf_mae, avg_ar_mae,
        std_tf_mae, std_ar_mae,
        target_name, output_dir, num_runs,
        metric_name='MAE',
        ylabel='Mean Absolute Error',
        skip_first_n=0,
        tf_errors_all=tf_mae_errors,
        ar_errors_all=ar_mae_errors
      )
      
      # Plot MALE over time
      plot.plot_error_growth_metric(
        avg_tf_male, avg_ar_male,
        std_tf_male, std_ar_male,
        target_name, output_dir, num_runs,
        metric_name='MALE',
        ylabel='Mean Absolute Log Error',
        skip_first_n=0,
        tf_errors_all=tf_male_errors,
        ar_errors_all=ar_male_errors
      )
      
      logger.info(f"  ✓ Error growth plots saved to: {output_dir}")

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