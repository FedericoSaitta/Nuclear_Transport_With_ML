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

  power_MW_per_kg = pandas_df['power_W_g'] * 1e-3 # Convert power from W/g to MW/kg

  dt_days = pandas_df['time_days'].diff().fillna(0) # Computes time difference between steps

  energy_step = power_MW_per_kg * dt_days
  pandas_df['burnup_MWd_per_kg'] = np.cumsum(energy_step)

  return pandas_df


def read_data(folder_path):
  # Get all CSV files in the folder
  csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
  
  # Read each CSV into a list of DataFrames
  dfs = []
  for file in csv_files:
    df = pd.read_csv(file)
    check_duplicates(df)
    df = remove_empty_columns(df)
    df = add_burnup_to_df(df)
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


def remove_columns_from_data(df, keys):
  cols_to_drop = [col for col in keys if col in df.columns]
  if len(cols_to_drop) != len(keys): print("Warning: Some specified columns were not found in the DataFrame.")
  new_df = df.drop(columns=cols_to_drop)
  
  print(f"New Number of columns: {new_df.shape[1]}")
  return new_df

  
def keep_only_columns(df, keys):
  existing_cols = [col for col in keys if col in df.columns]
  if len(existing_cols) != len(keys): print("Warning: Some specified columns were not found in the DataFrame.")
  new_df = df[existing_cols]
  
  print(f"New Number of columns: {new_df.shape[1]}")
  return new_df


def keep_only_elements(df, elements_to_keep):
  numeric_cols = df.select_dtypes(include=[np.number]).columns

  # Identify columns that look like elements (letters followed by digits)
  element_cols = [col for col in numeric_cols if re.match(r'^[A-Za-z]+[0-9]+(_.*)?$', col)]

  # Only keep the specified elements that exist in the DataFrame
  elements_existing = [col for col in elements_to_keep if col in element_cols]
  if len(elements_existing) != len(elements_to_keep):
    print("Warning: Some specified elements were not found in the DataFrame.")

  non_element_cols = [col for col in df.columns if col not in element_cols]

  final_cols = non_element_cols + elements_existing # Combine non-element columns + selected elements
  new_df = df[final_cols]

  print(f"Kept {len(elements_existing)} elements and {len(non_element_cols)} non-element columns. Total columns: {new_df.shape[1]}")
  return new_df
