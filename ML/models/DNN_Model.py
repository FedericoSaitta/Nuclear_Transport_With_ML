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

import torchmetrics

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
    labels = data_scaler.inverse_transform_column_transformer(target_scaler, y_cpu)
    predictions = data_scaler.inverse_transform_column_transformer(target_scaler, preds_cpu)

    return {
      "labels": torch.from_numpy(labels),
      "predictions": torch.from_numpy(predictions)
    }

  # Gets called for each test batch
  def test_step(self, batch, batch_idx):
    pass
  
  # Gets called for each test epoch
  def on_test_epoch_end(self):

    # Manually run prediction over test set
    datamodule = self.trainer.datamodule
    loader = datamodule.test_dataloader()
    y_true_list, y_pred_list = [], []
    for batch in loader:
      batch = self.transfer_batch_to_device(batch, self.device, 0)
      res = self.predict_step(batch, batch_idx=0)
      y_true_list.append(res["labels"].cpu())
      y_pred_list.append(res["predictions"].cpu())

    y_true_test = torch.cat(y_true_list).cpu().detach().numpy()
    y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy()

    # Get target names
    target_names = list(datamodule.target.keys())
    
    # Ensure data is 2D for multi-output handling
    if y_true_test.ndim == 1:
      y_true_test = y_true_test.reshape(-1, 1)
    if y_pred_test.ndim == 1:
      y_pred_test = y_pred_test.reshape(-1, 1)
    
    logger.info(f"Test set shape - True: {y_true_test.shape}, Pred: {y_pred_test.shape}")
    
    # Compute per-output metrics manually using numpy
    mae_per_output = np.mean(np.abs(y_true_test - y_pred_test), axis=0)
    mse_per_output = np.mean((y_true_test - y_pred_test) ** 2, axis=0)
    rmse_per_output = np.sqrt(mse_per_output)
    
    # Compute R² per output manually
    r2_per_output = np.zeros(y_true_test.shape[1])
    for i in range(y_true_test.shape[1]):
      ss_res = np.sum((y_true_test[:, i] - y_pred_test[:, i]) ** 2)
      ss_tot = np.sum((y_true_test[:, i] - np.mean(y_true_test[:, i])) ** 2)
      r2_per_output[i] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Ensure all metrics are arrays
    if mae_per_output.ndim == 0:
      mae_per_output = np.array([mae_per_output])
    if mse_per_output.ndim == 0:
      mse_per_output = np.array([mse_per_output])
    if rmse_per_output.ndim == 0:
      rmse_per_output = np.array([rmse_per_output])
    
    # Log overall averaged metrics
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL METRICS (averaged across {len(target_names)} outputs)")
    logger.info(f"{'='*60}")
    logger.info(f"Overall MAE: {mae_per_output.mean():.4f}")
    logger.info(f"Overall RMSE: {rmse_per_output.mean():.4f}")
    logger.info(f"Overall R²: {r2_per_output.mean():.4f}")
    
    self.log('Mean Absolute Error (avg)', float(mae_per_output.mean()))
    self.log('Root Mean Squared Error (avg)', float(rmse_per_output.mean()))
    self.log('R-squared coefficient (avg)', float(r2_per_output.mean()))
    
    # === Process each output separately ===
    for idx, target_name in enumerate(target_names):
      logger.info(f"\n{'='*60}")
      logger.info(f"Output {idx+1}/{len(target_names)}: {target_name}")
      logger.info(f"{'='*60}")
      
      # Extract data for this output
      y_true_output = y_true_test[:, idx]
      y_pred_output = y_pred_test[:, idx]
      
      # Get metrics for this output
      mae_output = mae_per_output[idx]
      mse_output = mse_per_output[idx]
      rmse_output = rmse_per_output[idx]
      r2_output = r2_per_output[idx]
      
      logger.info(f"MAE:  {mae_output:.6f}")
      logger.info(f"MSE:  {mse_output:.6f}")
      logger.info(f"RMSE: {rmse_output:.6f}")
      logger.info(f"R²:   {r2_output:.6f}")
      
      # Compute MARE for this output
      max_val = np.max(np.abs(y_true_output))
      if max_val > 0:
        ARE = np.abs(y_true_output - y_pred_output) / max_val
        mare_output = np.mean(ARE)
        logger.info(f'MARE: {mare_output:.6f}')
        self.log(f'MARE_{target_name}', float(mare_output))
      else:
        logger.warning(f'Cannot compute MARE for {target_name}: max value is 0')
      
      # Compute additional statistics
      mean_error = np.mean(y_pred_output - y_true_output)
      std_error = np.std(y_pred_output - y_true_output)
      logger.info(f"Mean Error: {mean_error:.6f} ± {std_error:.6f}")
      
      # Create output-specific directory
      output_dir = os.path.join(self.result_dir, target_name)
      os.makedirs(output_dir, exist_ok=True)
      
      # Plot predictions vs actuals for this output
      plot.plot_predictions_vs_actuals(
        y_true_output, 
        y_pred_output, 
        mae_output, 
        rmse_output, 
        r2_output, 
        output_dir
      )
      
      # Plot residuals for this output
      plot.plot_residuals_combined(y_true_output, y_pred_output, output_dir)
      
      logger.info(f"Plots saved to: {output_dir}")
    
    # === Feature Importance (computed per output) ===
    logger.info(f"\n{'='*60}")
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info(f"{'='*60}")
    
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
        importance_means, 
        importance_stds, 
        feature_names, 
        baseline_r2, 
        output_dir, 
        'r2_score', 
        n_top=20
      )
      
      # MSE-based feature importance for this output
      logger.info(f"  Computing MSE-based feature importance...")
      importance_means, importance_stds, baseline_mse = metrics.calculate_feature_importance(
        self.model, loader, self.device, n_repeats=5, 
        metric={'name': 'mse', 'direction': 'decreasing'},
        output_idx=idx
      )
      plot.plot_feature_importance(
        importance_means, 
        importance_stds, 
        feature_names, 
        baseline_mse, 
        output_dir, 
        'mse_score', 
        n_top=20
      )
      
      logger.info(f"  ✓ Feature importance plots saved to: {output_dir}")
    
    # === PREDICTION COMPARISON (for each output) ===
    logger.info(f"\n{'='*60}")
    logger.info("PREDICTION COMPARISON (Teacher-Forcing vs Autoregressive)")
    logger.info(f"{'='*60}")
    
    first_run = datamodule.first_run
    col_index_map = datamodule.col_index_map
    input_scaler = datamodule.input_scaler
    y_scaler = datamodule.target_scaler
    
    first_run_original = data_scaler.inverse_transform_column_transformer(input_scaler, first_run)
    
    for idx, target_name in enumerate(target_names):
      logger.info(f"\nPrediction comparison for: {target_name}")
      
      # Get ground truth for this target
      ground_truth = first_run_original[:, col_index_map[target_name]]
      
      # Get teacher-forced predictions for this output
      teacher_forced_preds = metrics.get_teacher_forced_predictions(
        self.model, 
        first_run, 
        y_scaler, 
        self.device,
        output_idx=idx
      )
      
      # Get autoregressive predictions for this output
      autoregressive_preds = metrics.get_autoregressive_predictions(
        self.model, 
        first_run, 
        col_index_map[target_name], 
        input_scaler, 
        y_scaler, 
        self.device,
        output_idx=idx
      )
      
      # Save to output-specific directory
      output_dir = os.path.join(self.result_dir, target_name)
      plot.plot_prediction_comparison(
        ground_truth, 
        teacher_forced_preds, 
        autoregressive_preds, 
        target_name, 
        output_dir
      )
      
      logger.info(f"  ✓ Comparison plot saved to: {output_dir}")
    
    logger.info(f"\n{'='*60}")
    logger.info("TESTING COMPLETE!")
    logger.info(f"{'='*60}\n")

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