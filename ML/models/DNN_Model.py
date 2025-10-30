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
  def __init__(self, n_inputs, hidden_layers, dropout_prob=0.1):
    super(SimpleDNN, self).__init__()

    # Complete layer sizes: [n_inputs, hidden1, hidden2, ..., 1]
    layer_sizes = [n_inputs] + hidden_layers + [1]

    # Create a list of Linear layers
    self.layers = nn.ModuleList([
      nn.Linear(in_size, out_size)
      for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ])

    self.dropout = nn.Dropout(p=dropout_prob)

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = self.dropout(F.relu(layer(x)))
    y = self.layers[-1](x)
    return y

class DNN_Model(L.LightningModule):
  def __init__(self, config_object : DictConfig):
    super().__init__()

    self.dropout_prob = config_object.train.dropout_probability
    self.NN_layers = config_object.model.layers

    self.learning_rate = config_object.train.learning_rate
    self.weight_decay = config_object.train.weight_decay

    # Change this in the future
    self.loss_fn = nn.MSELoss()

    self.lr_scheduler_patience = config_object.train.lr_scheduler_patience
    
    # Keep track of losses
    self.train_losses = []
    self.val_losses = []

    # Metrics
    self.mae = torchmetrics.MeanAbsoluteError()
    self.mse = torchmetrics.MeanSquaredError()
    self.r2 = torchmetrics.R2Score()

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
    self.model = SimpleDNN(input_dim, self.NN_layers, self.dropout_prob)

  # Gets called for each train batch
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x).squeeze()
    loss = self.loss_fn(y_hat, y)

    self.log('train_loss', loss, on_epoch=True, prog_bar=True)
    self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True, prog_bar=True)
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
    y_hat = self.model(x).squeeze()
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

    # Move to CPU and reshape for sklearn
    y_cpu = y.cpu()
    preds_cpu = preds.cpu()
    
    # Ensure 2D shape for sklearn scaler
    if y_cpu.ndim == 1:
      y_cpu = y_cpu.reshape(-1, 1)
    if preds_cpu.ndim == 1:
      preds_cpu = preds_cpu.reshape(-1, 1)

    return {"labels": torch.from_numpy(target_scaler.inverse_transform(y_cpu.numpy())),
            "predictions": torch.from_numpy(target_scaler.inverse_transform(preds_cpu.numpy()))}

  # Gets called for each test batch
  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x).squeeze()
    for metric in [self.mae, self.mse, self.r2]:
      metric.update(y_hat, y)

  # Gets called for each test epoch
  def on_test_epoch_end(self):
    # Manually run prediction over test set
    datamodule = self.trainer.datamodule
    loader = datamodule.test_dataloader()
    y_true_list, y_pred_list = [], []
    for batch in loader:
      batch = self.transfer_batch_to_device(batch, self.device, 0)
      res = self.predict_step(batch, batch_idx=0)   # use predict_step to get {"labels","predictions"}
      y_true_list.append(res["labels"].cpu())
      y_pred_list.append(res["predictions"].cpu())
        
    y_true_test = torch.cat(y_true_list).cpu().detach().numpy().squeeze()
    y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy().squeeze()


    ## FIX The metric inputs to this function
    plot.plot_predictions_vs_actuals(y_true_test, y_pred_test, self.mae.compute().item(), np.sqrt(self.mse.compute().item()), self.r2.compute().item(), self.result_dir)
    plot.plot_residuals_combined(y_true_test, y_pred_test, self.result_dir)

    self.log('Mean Absolute Error', self.mae.compute())
    self.log('Mean Squared Error', self.mse.compute())
    self.log('R-squared coefficient', self.r2.compute())

    feature_names = [key for key, _ in sorted(datamodule.col_index_map.items(), key=lambda x: x[1])]

    importance_means, importance_stds, baseline_r2 = metrics.calculate_feature_importance(
      self.model, loader, self.device, n_repeats=10,
      metric={'name': 'r2', 'direction': 'increasing'}
    )

    plot.plot_feature_importance(importance_means, importance_stds, feature_names, baseline_r2, self.result_dir, 'r2_score', n_top=20)

    # ===== PREDICTION COMPARISON ===== #
    # Get ground truth in original scale
    first_run = datamodule.first_run
    col_index_map = datamodule.col_index_map
    target_name = datamodule.target
    input_scaler = datamodule.input_scaler
    y_scaler = datamodule.target_scaler

    target_name = datamodule.target
    input_scaler = datamodule.input_scaler
    y_scaler = datamodule.target_scaler

    first_run_original = data_scaler.inverse_transform_column_transformer(input_scaler, first_run)
    ground_truth = first_run_original[:, col_index_map[target_name]]

    # Get teacher-forced predictions
    teacher_forced_preds = metrics.get_teacher_forced_predictions(self.model, first_run, y_scaler, self.device)
    autoregressive_preds = metrics.get_autoregressive_predictions(self.model, first_run, col_index_map[target_name], input_scaler, y_scaler, self.device)
    plot.plot_prediction_comparison(ground_truth, teacher_forced_preds, autoregressive_preds, target_name, self.result_dir)

    for metric in [self.mae, self.mse, self.r2]: metric.reset()

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