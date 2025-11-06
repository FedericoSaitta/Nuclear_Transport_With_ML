# === Standard imports ===
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

# === External libraries ===
from omegaconf import DictConfig
from loguru import logger
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
from ML.utils import plot
from ML.utils import metrics
import ML.datamodule.data_scalers as data_scaler

# This modules will contain the Machine Learning Models used
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleDNN(nn.Module):
  def __init__(self, n_inputs, n_outputs, hidden_layers, dropout_prob, activation, output_activation, residual):
    super(SimpleDNN, self).__init__()
    self.residual = residual

    layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
    self.layers = nn.ModuleList([
      nn.Linear(in_size, out_size)
      for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ])
    self.dropout = nn.Dropout(p=dropout_prob)

    self.activation_fn = self._get_activation(activation)
    self.output_activation_fn = self._get_activation(output_activation)

  def _get_activation(self, activation):
    activations = {
      'relu': nn.ReLU(),
      'tanh': nn.Tanh(),
      'sigmoid': nn.Sigmoid(),
      'leaky_relu': nn.LeakyReLU(0.1),
      'elu': nn.ELU(),
      'gelu': nn.GELU(),
      'selu': nn.SELU(),
      'softplus': nn.Softplus(),
      'none': nn.Identity()
    }
    return activations.get(activation.lower(), nn.ReLU())

  def forward(self, x):
    for i, layer in enumerate(self.layers[:-1]):
      residual = x  # store input for skip connection
      x = layer(x)
      x = self.activation_fn(x)
      x = self.dropout(x)

      # Only add residual if dimensions match and flag is True
      if self.residual and x.shape == residual.shape:
        x = x + residual

    y = self.layers[-1](x)
    y = self.output_activation_fn(y)
    return y

class DNN_Model(L.LightningModule):
  def __init__(self, config_object : DictConfig):
    super().__init__()

    self.n_outputs = len(config_object.dataset.targets)
    self.dropout_prob = config_object.train.dropout_probability
    self.NN_layers = config_object.model.layers
    self.activation = config_object.model.activation
    self.output_activation = config_object.model.output_activation
    self.residual_connections = config_object.model.residual_connections

    ## If residual connections are turned on but layer inputs and outputs dont match they wont be used
    self.residual_map = [(in_size == out_size) for in_size, out_size in zip(self.NN_layers[:-1], self.NN_layers[1:])]
    if self.residual_connections and not any(self.residual_map):
      logger.error("Residual connections requested, but no adjacent layers have matching input/output dimensions.")
    
    self.learning_rate = config_object.train.learning_rate
    self.weight_decay = config_object.train.weight_decay
    
    # Make loss function configurable too
    self.loss_fn = self._get_loss_fn(config_object.train.loss)

    self.lr_scheduler_patience = config_object.train.lr_scheduler_patience
    
    # Keep track of losses
    self.train_losses = []
    self.val_losses = []

    # Results
    self.result_dir = 'results/' + config_object.model.name + '/'
    self._has_setup = False

  # Gets called once datamodule has setup
  def setup(self, stage=None, datamodule=None):
    # Avoiding setting up again if we have already done so
    if self._has_setup: return
    self._has_setup = True

    if datamodule is None: dm = self.trainer.datamodule
    else: dm = datamodule
    
    # Create the model based 
    input_dim = dm.train_dataset.tensors[0].shape[1]
    output_dim = self.n_outputs
    self.model = SimpleDNN(input_dim, output_dim, self.NN_layers, self.dropout_prob, self.activation, self.output_activation, self.residual_connections)

  # Gets called for each train batch
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
 
    loss = self.loss_fn(y_hat, y)
    self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    return loss

  # Gets called for each train epoch
  def on_train_epoch_end(self):
    # Lightning automatically aggregates epoch metrics, so we just grab it
    epoch_loss = self.trainer.callback_metrics["train_loss"].item()
    self.train_losses.append(epoch_loss)
  
  # Gets called when training ends
  def on_train_end(self):
    plot.plot_losses(self.train_losses, self.val_losses, self.result_dir)
    
    self.train_losses.clear()
    self.val_losses.clear()
  
  # Gets called called for each validation batch
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    
    # No shape manipulation needed - everything is already 2D from scale_datasets
    loss = self.loss_fn(y_hat, y)
    self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    return loss

  # Gets called for each validation epoch
  def on_validation_epoch_end(self):
    epoch_loss = self.trainer.callback_metrics["val_loss"].item()
    self.val_losses.append(epoch_loss)

  # Gets for each predict batch
  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    x, y = batch
    preds = self.model(x)

    target_scaler = self.trainer.datamodule.target_scaler

    # Move to CPU
    y_cpu = y.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    
    # No shape manipulation needed - everything is already 2D from scale_datasets
    labels = data_scaler.inverse_transformer(target_scaler, y_cpu)
    predictions = data_scaler.inverse_transformer(target_scaler, preds_cpu)

    return {
      "labels": torch.from_numpy(labels),
      "predictions": torch.from_numpy(predictions)
    }

  # Gets called for each test batch
  def test_step(self, batch, batch_idx):
    pass
  
  # Gets called for each test epoch
  def on_test_epoch_end(self):

    # Manually run prediction over test set AND collect X, Y in one pass
    datamodule = self.trainer.datamodule
    loader = datamodule.test_dataloader()
    
    y_true_list, y_pred_list = [], []
    X_test_list, Y_test_list = [], []
    
    # Get the teacher forced predictions from the model
    for batch in loader:
      batch = self.transfer_batch_to_device(batch, self.device, 0)
      
      # Collect X and Y for later use
      X_test_list.append(batch[0].cpu().numpy())
      Y_test_list.append(batch[1].cpu().numpy())
      
      # Get predictions
      res = self.predict_step(batch, batch_idx=0)
      y_true_list.append(res["labels"].cpu())
      y_pred_list.append(res["predictions"].cpu())

    y_true_test = torch.cat(y_true_list).cpu().detach().numpy()
    y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy()
    X_test = np.concatenate(X_test_list, axis=0)
    Y_test = np.concatenate(Y_test_list, axis=0)

    # Get target names
    target_names = list(datamodule.target.keys())
    
    # Ensure data is 2D for multi-output handling
    if y_true_test.ndim == 1: y_true_test = y_true_test.reshape(-1, 1)
    if y_pred_test.ndim == 1: y_pred_test = y_pred_test.reshape(-1, 1)
    
    logger.info(f"Test set shape - True: {y_true_test.shape}, Pred: {y_pred_test.shape}")
    
    # Compute per-output metrics manually using numpy
    mae_per_output = metrics.mae(y_true_test, y_pred_test)
    mse_per_output = metrics.mse(y_true_test, y_pred_test)
    rmse_per_output = metrics.rmse(y_true_test, y_pred_test)
    r2_per_output = metrics.r2(y_true_test, y_pred_test)
    
    # Log overall averaged metrics
    logger.info(f"\n{'='*20}")
    logger.info(f"OVERALL METRICS (averaged across {len(target_names)} outputs)")
    logger.info(f"{'='*20}")
    logger.info(f"Overall MAE: {mae_per_output.mean():.4f}")
    logger.info(f"Overall RMSE: {rmse_per_output.mean():.4f}")
    logger.info(f"Overall R²: {r2_per_output.mean():.4f}")
    
    self.log('Mean Absolute Error (avg)', float(mae_per_output.mean()))
    self.log('Root Mean Squared Error (avg)', float(rmse_per_output.mean()))
    self.log('R-squared coefficient (avg)', float(r2_per_output.mean()))
    
    # === Process each output separately ===
    for idx, target_name in enumerate(target_names):
      # Create output-specific directory
      output_dir = os.path.join(self.result_dir, target_name)
      os.makedirs(output_dir, exist_ok=True)

      logger.info(f"\n{'='*20}")
      logger.info(f"Output {idx+1}/{len(target_names)}: {target_name}")
      logger.info(f"{'='*20}")
      
      # Extract data and metrics for this output
      y_true_output = y_true_test[:, idx]
      y_pred_output = y_pred_test[:, idx]
      mae_output = mae_per_output[idx]
      mse_output = mse_per_output[idx]
      rmse_output = rmse_per_output[idx]
      r2_output = r2_per_output[idx]
      
      logger.info(f"MAE:  {mae_output:.6f}")
      logger.info(f"MSE:  {mse_output:.6f}")
      logger.info(f"RMSE: {rmse_output:.6f}")
      logger.info(f"R²:   {r2_output:.6f}")
      
      plot.plot_predictions_vs_actuals(y_true_output, y_pred_output, mae_output, rmse_output, r2_output, output_dir)
      plot.plot_residuals_combined(y_true_output, y_pred_output, output_dir)
      
      logger.info(f"Plots saved to: {output_dir}")
    
    # === MARE Comparison: Teacher-Forcing vs Autoregressive ===
    logger.info(f"\n{'='*20}")
    logger.info("MARE: Teacher-Forcing vs Autoregressive")
    logger.info(f"{'='*20}")
    
    col_index_map = datamodule.col_index_map
    input_scaler = datamodule.input_scaler
    y_scaler = datamodule.target_scaler
    
    # Create target_col_indices dict
    target_col_indices = datamodule.target_index_map
    
    # Get ground truth from Y_test
    if Y_test.ndim == 1: Y_test_2d = Y_test.reshape(-1, 1)
    else: Y_test_2d = Y_test
    
    # Inverse transform Y_test to get ground truth
    Y_test_original = data_scaler.inverse_transformer(y_scaler, Y_test_2d)
    
    # Teacher-Forcing: Use y_true_test (already computed from predictions above)
    tf_ground_truth = Y_test_original
    tf_predictions = y_pred_test  # Already in original scale from predict_step
    
    # Autoregressive: Model outputs predictions of t+2 from its own t+1 predictions
    delta_conc = datamodule.delta_conc
    steps_per_run = 100
    ar_predictions_dict, ar_ground_truth_dict = metrics.model_autoregress(self.model, X_test, Y_test, input_scaler, y_scaler, steps_per_run, col_index_map, target_col_indices)
    
    # Calculate MARE for each target
    for idx, target_name in enumerate(target_names):
      # Teacher-forcing MARE (from already-computed predictions)
      tf_gt = tf_ground_truth[:, idx] if tf_ground_truth.ndim > 1 else tf_ground_truth
      tf_pred = tf_predictions[:, idx] if tf_predictions.ndim > 1 else tf_predictions

      mare_tf = metrics.mare(tf_gt, tf_pred)
      
      # Autoregressive MARE
      ar_predictions = ar_predictions_dict[target_name]
      ar_ground_truth = ar_ground_truth_dict[target_name]
      
      mare_ar = metrics.mare(ar_ground_truth, ar_predictions)
      
      # Log results
      logger.info(f"\n{target_name}:")
      logger.info(f"  MARE (Teacher-Forcing): {mare_tf:.6f}")
      logger.info(f"  MARE (Autoregressive):  {mare_ar:.6f}")
      
      # Log to tensorboard/wandb
      self.log(f'{target_name}/MARE_TeacherForcing', float(mare_tf))
      self.log(f'{target_name}/MARE_Autoregressive', float(mare_ar))
    
    # === Feature Importance (computed per output) ===
    logger.info(f"\n{'='*20}")
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info(f"{'='*20}")
    
    feature_names = [key for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])]
    
    for idx, target_name in enumerate(target_names):
      logger.info(f"\nComputing feature importance for: {target_name}")
      
      output_dir = os.path.join(self.result_dir, target_name)
      
      # R2-based feature importance for this output
      logger.info(f"  Computing R²-based feature importance...")
      importance_means, importance_stds, baseline_r2 = metrics.calculate_feature_importance(
        self.model, loader, self.device, n_repeats=5, 
        metric={'name': 'r2', 'direction': 'increasing'},
        output_idx=idx
      )
      plot.plot_feature_importance(
        importance_means, importance_stds, feature_names, baseline_r2, output_dir, 'r2_score', n_top=20
      )
      
      # MSE-based feature importance for this output
      logger.info(f"  Computing MSE-based feature importance...")
      importance_means, importance_stds, baseline_mse = metrics.calculate_feature_importance(
        self.model, loader, self.device, n_repeats=5, 
        metric={'name': 'mse', 'direction': 'decreasing'},
        output_idx=idx
      )
      plot.plot_feature_importance(
        importance_means, importance_stds, feature_names, baseline_mse, output_dir, 'mse_score', n_top=20
      )
      
      logger.info(f"  ✓ Feature importance plots saved to: {output_dir}")
    
    # === PREDICTION COMPARISON (for each output) ===
    logger.info(f"\n{'='*20}")
    logger.info("PREDICTION COMPARISON (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*20}")

    # Use the FIRST RUN from the TEST SET, not from the original data
    first_run_start = 0
    first_run_end = steps_per_run  # This should match the steps_per_run from autoregress

    # Plot for each target
    for idx, target_name in enumerate(target_names):
        logger.info(f"\nPlotting prediction comparison for: {target_name}")
        
        # Get ground truth from the TEST SET (already unscaled in ar_ground_truth_dict)
        ground_truth = ar_ground_truth_dict[target_name][first_run_start:first_run_end]
        
        logger.info(f'The ground truth has length {len(ground_truth)}')
        
        # Get teacher-forced predictions for first run from TEST SET
        teacher_forced_preds = tf_predictions[first_run_start:first_run_end, idx] if tf_predictions.ndim > 1 else tf_predictions[first_run_start:first_run_end]
        
        # Apply delta_conc correction if needed
        if delta_conc:
            # Need to get the initial concentration and add cumulative deltas
            # This is tricky - you might need to cumsum the predictions
            logger.warning("delta_conc correction for teacher forcing may need review")
            teacher_forced_preds = teacher_forced_preds + ground_truth[0]  # Simplified - may need cumsum
        
        # Get autoregressive predictions for first run
        autoregressive_preds = ar_predictions_dict[target_name][first_run_start:first_run_end]
        
        # Verify all arrays have the same length
        logger.info(f"  Ground truth length: {len(ground_truth)}")
        logger.info(f"  Teacher-forced length: {len(teacher_forced_preds)}")
        logger.info(f"  Autoregressive length: {len(autoregressive_preds)}")
        
        # Save to output-specific directory
        output_dir = os.path.join(self.result_dir, target_name)
        plot.plot_prediction_comparison(ground_truth, teacher_forced_preds, autoregressive_preds, target_name, output_dir)
        
        logger.info(f"  ✓ Comparison plot saved to: {output_dir}")

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.lr_scheduler_patience)
    return {
      'optimizer': optimizer,
      'lr_scheduler': {
          'scheduler': scheduler,
          'interval': 'epoch',   # call scheduler.step() every epoch
          'monitor': 'val_loss', # metric to monitor for ReduceLROnPlateau
      }
    }
  
  def _get_loss_fn(self, loss_name):
    losses = {
      'mse': nn.MSELoss(),
      'mae': nn.L1Loss(),
      'huber': nn.HuberLoss(),
      'smooth_l1': nn.SmoothL1Loss()
    }
    return losses[loss_name.lower()]