# This file is used to reshape and mold the csv dataset to be more approachable for ML
import pandas as pd
import numpy as np
import re
import glob
import os

def check_duplicates(pandas_df):
  # Get all columns except 'run_label'
  cols_to_check = [col for col in pandas_df.columns if col != 'run_label']
  
  if pandas_df.duplicated(subset=cols_to_check).any():
    print("The DataFrame contains duplicate rows (ignoring run_label).")
  else:
    print("The DataFrame has no duplicate rows (ignoring run_label).")


def remove_empty_columns(pandas_df):
  removed_columns = []
  
  for col in pandas_df.columns:
    # Check if all values are 0 or NaN
    if pandas_df[col].replace(0, np.nan).isna().all():
      removed_columns.append(col)
      pandas_df.drop(columns=col, inplace=True)
  
  if removed_columns: print("Removed columns:", removed_columns)
  else: print("No empty columns to remove.")
  
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
    df = pd.read_csv(file)
    check_duplicates(df)
    df = remove_empty_columns(df)
    df = add_burnup_to_df(df)
    
    # Drop the 'run_label' column if it exists
    if drop_run_label and 'run_label' in df.columns:
      df = df.drop(columns=['run_label'])
    
    dfs.append(df)
  
  combined_df = pd.concat(dfs, ignore_index=True)
  
  return combined_df


def print_dataset_stats(df):
  print("=== Dataset Statistics ===")
  
  # Number of rows and columns
  print(f"Number of rows (time steps): {df.shape[0]}")
  print(f"Number of columns: {df.shape[1]}")
  
  # Column names
  print("\nColumns:")
  print(df.columns.tolist())

  numeric_cols = df.select_dtypes(include=[np.number]).columns

  # Select only columns that look like isotopes
  isotope_cols = [col for col in numeric_cols if re.match(r'^[A-Za-z]+[0-9]+(_.*)?$', col)]

  first_row = df.iloc[0]
  nonzero_isotopes = [col for col in isotope_cols if first_row[col] != 0]

  print("Isotopes with non-zero concentration at t=0:", nonzero_isotopes)

  print("==========================")
  

def filter_columns(df, elements_to_keep, features_to_keep):
  numeric_cols = df.select_dtypes(include=[np.number]).columns

  # Identify columns that look like elements (letters followed by digits)
  element_cols = [col for col in numeric_cols if re.match(r'^[A-Za-z]+[0-9]+(_.*)?$', col)]
  elements_existing = [col for col in elements_to_keep if col in element_cols]

  if len(elements_existing) != len(elements_to_keep):
    print("Warning: Some specified elements were not found in the DataFrame.")

  non_element_to_keep = [col for col in df.columns if col in features_to_keep]

  final_cols = non_element_to_keep + elements_existing # Combine non-element columns + selected elements
  new_df = df[final_cols]

  print(f"Kept {len(elements_existing)} elements and {len(non_element_to_keep)} non-element columns. Total columns: {new_df.shape[1]}")
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
  
  print(f"Detected {len(run_end_indices)} runs")
  print(f"Created {len(X)} training samples")
  print(f"Input shape: {X.shape} | Target shape: {y.shape}")
  print(f"Targets: {target_elements}")
  
  return X, y
