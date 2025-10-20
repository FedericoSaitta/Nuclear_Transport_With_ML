# This file is used to reshape and mold the csv dataset to be more approachable for ML
import pandas as pd
import numpy as np

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

