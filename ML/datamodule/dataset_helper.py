# This file is used to reshape and mold the csv dataset to be more approachable for ML
import pandas as pd
import numpy as np
import re
import glob
import os
from loguru import logger
import torch
from torch.utils.data import TensorDataset

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
  burnup_chunks = [np.cumsum(energy_step[i:i + chunk_size]) for i in range(0, len(energy_step), chunk_size)]
  burnup_values = np.concatenate(burnup_chunks)

  pandas_df['burnup_MWd_per_kg'] = burnup_values
  return pandas_df


def detect_run_length(pandas_df, time_col='time_days'):
  time_values = pandas_df[time_col].values
  
  # Find where time resets (time[i] <= time[i-1])
  for i in range(1, len(time_values)):
    if time_values[i] <= time_values[i-1]:
      # Found first reset, this is the run length
      return i

  # If no reset found, entire dataframe is one run
  return len(pandas_df)


def read_data(folder_path, fraction_of_data, drop_run_label=True):
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

  run_length = detect_run_length(combined_df, time_col='time_days')
  logger.info(f'Detected Run Lenght: {run_length}')

  if (fraction_of_data < 1.0):
    total_runs = combined_df.shape[0] / run_length 
    runs_kept = int(fraction_of_data * total_runs) # Find the closest integer run number
    combined_df = combined_df.iloc[0:runs_kept*run_length]
    logger.info(f"Cutting Down Dataset to {fraction_of_data*100}%, runs present: {runs_kept}")

  return combined_df, run_length


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
  

def filter_columns(df, input_features):
  return df[input_features].copy() 


def split_df(df):
  data_array = df.to_numpy()
  col_index_map = {col: idx for idx, col in enumerate(df.columns)}
  return data_array, col_index_map


def create_timeseries_targets(data, time_col_idx, element_dict, target_elements, delta_conc):
  
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
  
  if delta_conc:
    # Compute differences: y[t+1] - y[t] for the target columns
    y = data[valid_indices + 1][:, target_indices] - data[valid_indices][:, target_indices]
  else:
    y = data[valid_indices + 1][:, target_indices]

  logger.info(y)
  logger.info(f"Detected {len(run_end_indices)} runs")
  logger.info(f"Created {len(X)} training samples")
  logger.info(f"Input shape: {X.shape} | Target shape: {y.shape}")
  
  return X, y


def timeseries_train_val_test_split(X, Y, train_frac=0.8, val_frac=0.1, test_frac=0.1, 
                                     steps_per_run=100, shuffle_within_train=True):
    # Validate fractions
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}")
    
    # Calculate total number of runs
    total_samples = len(X)
    total_runs = total_samples // steps_per_run
    
    if total_samples % steps_per_run != 0:
        logger.warning(f"Total samples ({total_samples}) not evenly divisible by steps_per_run ({steps_per_run}). "
                      f"Last {total_samples % steps_per_run} samples will be discarded.")
        # Trim to exact multiple of steps_per_run
        total_samples = total_runs * steps_per_run
        X = X[:total_samples]
        Y = Y[:total_samples]
    
    logger.info(f"Total runs detected: {total_runs}")
    logger.info(f"Steps per run: {steps_per_run}")
    
    # Calculate number of runs for each split
    n_train_runs = int(total_runs * train_frac)
    n_val_runs = int(total_runs * val_frac)
    n_test_runs = total_runs - n_train_runs - n_val_runs  # Remainder goes to test
    
    logger.info(f"Split: {n_train_runs} train runs, {n_val_runs} val runs, {n_test_runs} test runs")
    
    # Calculate indices
    train_end_idx = n_train_runs * steps_per_run
    val_end_idx = train_end_idx + (n_val_runs * steps_per_run)
    
    # Split data sequentially (first runs for train, middle for val, last for test)
    X_train = X[:train_end_idx]
    y_train = Y[:train_end_idx]
    
    X_val = X[train_end_idx:val_end_idx]
    y_val = Y[train_end_idx:val_end_idx]
    
    X_test = X[val_end_idx:]
    y_test = Y[val_end_idx:]
    
    logger.info(f"Before shuffle - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Optionally shuffle within training set (but keep time steps from same run together)
    if shuffle_within_train and n_train_runs > 1:
        # Create array of run indices
        train_run_indices = np.arange(n_train_runs)
        np.random.shuffle(train_run_indices)
        
        # Reorder entire runs
        X_train_shuffled = []
        y_train_shuffled = []
        
        for run_idx in train_run_indices:
            start_idx = run_idx * steps_per_run
            end_idx = start_idx + steps_per_run
            X_train_shuffled.append(X_train[start_idx:end_idx])
            y_train_shuffled.append(y_train[start_idx:end_idx])
        
        X_train = np.concatenate(X_train_shuffled, axis=0)
        y_train = np.concatenate(y_train_shuffled, axis=0)
        
        logger.info("Training runs shuffled (keeping time steps within each run intact)")
    
    logger.info(f"Final split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_datasets(X_train, X_val, X_test, y_train, y_val, y_test, input_scaler, target_scaler):
  # Scale inputs
  X_train = input_scaler.fit_transform(X_train)
  X_val = input_scaler.transform(X_val)
  X_test = input_scaler.transform(X_test)
  
  # Scale targets - handle both single and multiple outputs
  # Ensure targets are 2D for sklearn scalers
  if y_train.ndim == 1:
    y_train = y_train.reshape(-1, 1)
  if y_val.ndim == 1:
    y_val = y_val.reshape(-1, 1)
  if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)
  
  # Fit and transform targets
  y_train = target_scaler.fit_transform(y_train)
  y_val = target_scaler.transform(y_val)
  y_test = target_scaler.transform(y_test)
  
  # REMOVED: The flattening logic that was causing inconsistency
  # Keep everything 2D for consistency
  return X_train, X_val, X_test, y_train, y_val, y_test

def create_tensor_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
  # Convert inputs to tensors (replace NaN with -1)
  X_train_tensor = torch.nan_to_num(torch.tensor(X_train, dtype=torch.float32), nan=-1)
  X_val_tensor = torch.nan_to_num(torch.tensor(X_val, dtype=torch.float32), nan=-1)
  X_test_tensor = torch.nan_to_num(torch.tensor(X_test, dtype=torch.float32), nan=-1)
  
  # Convert targets to tensors - should already be 2D from scale_datasets
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
  
  # Create datasets
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
  
  return train_dataset, val_dataset, test_dataset