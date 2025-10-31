# This file is used to reshape and mold the csv dataset to be more approachable for ML
import pandas as pd
import numpy as np
import re
import glob
import os
from loguru import logger
import torch
from torch.utils.data import DataLoader, TensorDataset

def check_duplicates(pandas_df):
  # Get all columns except 'run_label'
  cols_to_check = [col for col in pandas_df.columns if col != 'run_label']
  
  if pandas_df.duplicated(subset=cols_to_check).any():
    logger.warning("The DataFrame contains duplicate rows (ignoring run_label).")
    return True
  else:
    logger.info("The DataFrame has no duplicate rows (ignoring run_label).")
    return False


def remove_empty_columns(pandas_df):
  removed_columns = []
  
  for col in pandas_df.columns:
    # Check if all values are 0 or NaN
    if pandas_df[col].replace(0, np.nan).isna().all():
      removed_columns.append(col)
      pandas_df.drop(columns=col, inplace=True)
  
  if removed_columns: logger.warning(f"Removed columns: {removed_columns}")
  else: logger.info("No empty columns to remove.")
  
  return pandas_df

def add_burnup_to_df(pandas_df):
  power_MW_per_kg = pandas_df['power_W_g'] * 1e-3  # Convert power from W/g to MW/kg
  dt_days = 10  # time step (constant)

  energy_step = power_MW_per_kg * dt_days

  # Compute cumulative sum in chunks of 100 rows
  chunk_size = 100
  burnup_chunks = [
      np.cumsum(energy_step[i:i + chunk_size])
      for i in range(0, len(energy_step), chunk_size)
  ]
  burnup_values = np.concatenate(burnup_chunks)

  pandas_df['burnup_MWd_per_kg'] = burnup_values
  return pandas_df


def read_data(folder_path, drop_run_label=True):
  # Get all CSV files in the folder
  csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
  
  # Read each CSV into a list of DataFrames
  dfs = []
  for file in csv_files:
    logger.info(f'Reading data from: {file}')
    df = pd.read_csv(file)
    df = remove_empty_columns(df)
    df = add_burnup_to_df(df)
    
    # Drop the 'run_label' column if it exists
    if drop_run_label and 'run_label' in df.columns:
      df = df.drop(columns=['run_label'])
    
    dfs.append(df)
  
  combined_df = pd.concat(dfs, ignore_index=True)
  check_duplicates(combined_df) # Checking if entire dataset has duplicates

  return combined_df


def print_dataset_stats(df):
  logger.info("=== Dataset Statistics ===")
  
  # Number of rows and columns
  logger.info(f"Number of rows (time steps): {df.shape[0]}")
  logger.info(f"Number of columns: {df.shape[1]}")

  numeric_cols = df.select_dtypes(include=[np.number]).columns

  # Select only columns that look like isotopes
  isotope_cols = [col for col in numeric_cols if re.match(r'^[A-Za-z]+[0-9]+(_.*)?$', col)]

  first_row = df.iloc[0]
  nonzero_isotopes = [col for col in isotope_cols if first_row[col] != 0]

  logger.info(f"Number of element columns: {len(isotope_cols)}")
  logger.info(f"Number of state columns: {len(df.columns) - len(isotope_cols)}")
  logger.info("Isotopes with non-zero concentration at t=0:", nonzero_isotopes)
  

def filter_columns(df, elements_mask, features_mask):
  # Keep all numeric columns
  numeric_cols = df.select_dtypes(include=[np.number]).columns
  element_cols = [col for col in numeric_cols if re.match(r'^[A-Za-z]+[0-9]+(_.*)?$', col)]
  elements_to_keep = [col for col in elements_mask if col in element_cols]

  # Keep features_mask regardless of type
  non_element_to_keep = [col for col in features_mask if col in df.columns and col not in element_cols]

  # Error checking
  if len(elements_to_keep) != len(elements_mask):
    logger.warning("Warning: Some specified elements were not found in the DataFrame.")
  if len(non_element_to_keep) != len(features_mask):
    logger.warning("Warning: Some specified features were not found in the DataFrame.")

  final_cols = non_element_to_keep + elements_to_keep
  new_df = df[final_cols]

  logger.info(f"Kept {len(elements_to_keep)} elements and {len(non_element_to_keep)} state columns. Total columns: {new_df.shape[1]}")
  return new_df


def split_df(df):
  data_array = df.to_numpy()
  col_index_map = {col: idx for idx, col in enumerate(df.columns)}
  return data_array, col_index_map


def create_timeseries_targets(data, time_col_idx, element_dict, target_elements):
  
  if time_col_idx is None: raise ValueError("No time column found")
  for element in target_elements:
    if element not in element_dict: raise ValueError(f"Target element '{element}' not found in element_dict")
  
  target_indices = [element_dict[element] for element in target_elements]
  
  # Find run boundaries (where time resets)
  time_values = data[:, time_col_idx]
  run_end_indices = []
  
  for i in range(1, len(time_values)):
      if time_values[i] <= time_values[i-1]:
          run_end_indices.append(i - 1)
  run_end_indices.append(len(data) - 1)  # Last row always ends a run

  # Create mask excluding run-ending rows
  valid_mask = np.ones(len(data), dtype=bool)
  valid_mask[run_end_indices] = False
  valid_indices = np.where(valid_mask)[0]
  
  # Create X (all columns) and y (target columns at t+1)
  X = data[valid_indices]
  y = data[valid_indices + 1][:, target_indices]
  
  logger.info(f"Detected {len(run_end_indices)} runs")
  logger.info(f"Created {len(X)} training samples")
  logger.info(f"Input shape: {X.shape} | Target shape: {y.shape}")
  logger.info(f"Targets: {target_elements}")
  
  return X, y


def scale_datasets(X_train, X_val, X_test, y_train, y_val, y_test, X_first_run, input_scaler, target_scaler):
  # Scale inputs
  X_train = input_scaler.fit_transform(X_train)
  X_val = input_scaler.transform(X_val)
  X_test = input_scaler.transform(X_test)
  X_first_run = input_scaler.transform(X_first_run)
  
  # Scale targets
  y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
  y_val = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
  y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
  
  return X_train, X_val, X_test, y_train, y_val, y_test, X_first_run


def create_tensor_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
  # Convert inputs to tensors (replace NaN with -1)
  X_train_tensor = torch.nan_to_num(torch.tensor(X_train, dtype=torch.float32), nan=-1)
  X_val_tensor = torch.nan_to_num(torch.tensor(X_val, dtype=torch.float32), nan=-1)
  X_test_tensor = torch.nan_to_num(torch.tensor(X_test, dtype=torch.float32), nan=-1)
  
  # Convert targets to tensors
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
  
  # Create datasets
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
  
  return train_dataset, val_dataset, test_dataset
