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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "datamodule")))
import dataset_helper as data_help
import plot

class DNN_Datamodule(L.LightningDataModule):
    def __init__(self, cfg_object: DictConfig):
        super().__init__()

        # === Dataset & model configuration === #
        dataset_cfg, model_cfg, train_cfg = cfg_object.dataset, cfg_object.model, cfg_object.train

        self.path_to_data = dataset_cfg.path_to_data
        self.train_batch_size = dataset_cfg.train.batch_size
        self.val_batch_size = dataset_cfg.val.batch_size
        self.train_num_workers = dataset_cfg.train.num_workers
        self.val_num_workers = dataset_cfg.val.num_workers

        self.target = dataset_cfg.target
        self.state_features = dataset_cfg.state_features
        self.input_elements = dataset_cfg.input_elements

        ## Check that target is in the input elements as well: 
        if self.target not in self.input_elements:
          logger.error(f"Target '{self.target}' not found in input elements.")
          raise ValueError(f"Target '{self.target}' not found in input elements.")
  
        # Notably these two have not been yet set!!!!!!
        self.input_scaling = dataset_cfg.input_scaling
        self.target_scaling = dataset_cfg.target_scaling

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

        data_df = data_help.read_data(self.path_to_data, drop_run_label=True)
        data_help.print_dataset_stats(data_df)

        data_df = data_help.filter_columns(data_df, self.input_elements, self.state_features)


        data_arr, col_index_map = data_help.split_df(data_df)

        self.col_index_map = col_index_map

        X, Y = data_help.create_timeseries_targets(data_arr, col_index_map['time_days'], col_index_map, [self.target])
        plot.plot_data_distributions(X, Y, col_index_map, target_name='Target', save_dir=self.result_dir, name='Raw_Data')

        self.first_run = X[0:100, :]

        # First split: Train and Temp (80/20)
        X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

        # Second split: Val and Test from Temp (50/50 of Temp => 10% each of total)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=0)
        

        # Scaling the data: 
        self.input_scaler =  MinMaxScaler()

        X_train = self.input_scaler.fit_transform(X_train)
        X_val   = self.input_scaler.transform(X_val)
        X_test  = self.input_scaler.transform(X_test)

        self.first_run = self.input_scaler.transform(self.first_run)
        
        self.target_scaler = StandardScaler()
        y_train = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        plot.plot_data_distributions(X_train, y_train, col_index_map, target_name='Target', save_dir=self.result_dir, name='Scaled_Data')
        
        X_train_tensor = torch.nan_to_num(torch.tensor(X_train, dtype=torch.float32), nan=-1)
        X_val_tensor   = torch.nan_to_num(torch.tensor(X_val, dtype=torch.float32), nan=-1)
        X_test_tensor  = torch.nan_to_num(torch.tensor(X_test, dtype=torch.float32), nan=-1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)

        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
        self.test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
    
        # === Log sizes === #
        logger.info(f"Training dataset size: {len(y_train)}")
        logger.info(f"Validation dataset size: {len(y_val)}")
        logger.info(f"Test dataset size: {len(y_test)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.train_num_workers, persistent_workers=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, persistent_workers=True, drop_last=True)

    def test_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def predict_dataloader(self):
      return DataLoader(self.test_dataset, batch_size=self.train_batch_size, shuffle=False, num_workers=4, persistent_workers=True)
