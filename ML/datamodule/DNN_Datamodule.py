# === Core Libraries === #
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from loguru import logger
import lightning as L
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import re
from sklearn.base import clone

import ML.datamodule.dataset_helper as data_help
import ML.utils.plot as plot
import ML.datamodule.data_scalers as data_scalers


class DNN_Datamodule(L.LightningDataModule):
  def __init__(self, cfg_object: DictConfig):
    super().__init__()

    # === Dataset & model configuration === #
    dataset_cfg, model_cfg, train_cfg = cfg_object.dataset, cfg_object.model, cfg_object.train

    self.path_to_data = dataset_cfg.path_to_data
    self.fraction_of_data = dataset_cfg.fraction_of_data
    self.train_batch_size = dataset_cfg.train.batch_size
    self.val_batch_size = dataset_cfg.val.batch_size
    self.num_workers = cfg_object.runtime.num_workers
    
    # Get the inputs and target dictionaries that include their respective scaling
    self.inputs = data_scalers.create_scaler_dict(dataset_cfg['inputs'])
    self.target = data_scalers.create_scaler_dict(dataset_cfg['targets'])

    # Check that target is present in inputs
    target_keys = set(self.target.keys())
    input_keys = set(self.inputs.keys())

    if not target_keys.issubset(input_keys):
      missing = target_keys - input_keys
      raise ValueError(f"Target features not found in inputs: {missing}")

    # === Result output directory === #
    self.result_dir = f'results/{model_cfg.name}/'
    os.makedirs(self.result_dir, exist_ok=True)

    # Private variable to ensure set up is not done twice when calling training and test scripts back to back
    self._has_setup = False
  
  def setup(self, stage: str):
    if self._has_setup: 
      return
    self._has_setup = True

    logger.info("Setting up the data module...")

    data_df = data_help.read_data(self.path_to_data, self.fraction_of_data, drop_run_label=True)
    data_help.print_dataset_stats(data_df)

    data_df = data_help.filter_columns(data_df, list(self.inputs.keys()))
    data_arr, self.col_index_map = data_help.split_df(data_df)
    
    target_index_map = {str(list(self.target.keys())[0]) : 0}
    

    logger.info(f"Inputs chosen and their respective indices: {self.col_index_map}")

    # Create Column Transformers for inputs and get the scaler for the targets
    self.input_scaler = data_scalers.create_column_transformer(self.inputs, self.col_index_map)
    self.target_scaler = list(self.target.values())[0]
    logger.info(f"Target scaler type: {type(self.target_scaler).__name__}")

    X, Y = data_help.create_timeseries_targets(data_arr, self.col_index_map['time_days'], self.col_index_map, list(self.target.keys()))

    self.first_run = X[0:100, :]

    # Split data 80/10/10 -- Train/validation/Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=0)
    
    plot.plot_data_distributions(X_train, self.col_index_map, save_dir=self.result_dir, name='Raw_Inputs')
    plot.plot_data_distributions(y_train, target_index_map, save_dir=self.result_dir, name='Raw_Targets')
    
    # Scale all datasets
    X_train, X_val, X_test, y_train, y_val, y_test, self.first_run = data_help.scale_datasets(
      X_train, X_val, X_test, y_train, y_val, y_test, self.first_run, self.input_scaler, self.target_scaler
    )

    plot.plot_data_distributions(X_train, self.col_index_map, save_dir=self.result_dir, name='Scaled_Inputs')
    plot.plot_data_distributions(y_train, target_index_map, save_dir=self.result_dir, name='Scaled_Targets')
    
    # Create tensor datasets and log their sizes
    self.train_dataset, self.val_dataset, self.test_dataset = data_help.create_tensor_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info(f"Training dataset size: {len(y_train)}")
    logger.info(f"Validation dataset size: {len(y_val)}")
    logger.info(f"Test dataset size: {len(y_test)}")

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, drop_last=False, pin_memory=False)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, drop_last=False)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

  def predict_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
