# === Standard imports ===
import os
import torch
import torch.nn as nn
import numpy as np

# === External libraries ===
from loguru import logger
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error

from ML.utils import plot
from ML.utils import metrics
import ML.datamodule.data_scalers as data_scaler
from ML.models.model_architectures import Deep_Neural_Network
from ML.models.model_helper import get_loss_fn


class NODE_Model(L.LightningModule):
  def __init__(self, config_object):
    super().__init__()

  # Gets called after datamodule has setup
  def setup(self, stage=None):
    pass
   
  # Gets called for each train batch
  def training_step(self, batch, batch_idx):
    pass
  
  # Gets called for each train epoch
  def on_train_epoch_end(self):
    pass
  
  # Gets called when training ends
  def on_train_end(self):
    pass

  # Gets called for each validation batch
  def validation_step(self, batch, batch_idx):
    pass

  def on_validation_epoch_end(self):
    pass
    
  # Gets called for each predict batch
  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    pass
  
  def test_step(self, batch, batch_idx):
    pass

  def on_test_epoch_end(self):
    pass
        
  def configure_optimizers(self):
    pass