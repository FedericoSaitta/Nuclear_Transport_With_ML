# This is the main file which will be called to begin ML training
# General Imports
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Custom python files
import dataset_helper as data_help
import plot


def main(path_to_data, plots_folder):
  data_df = data_help.read_data(path_to_data, drop_run_label=True)

  data_help.print_dataset_stats(data_df)

  print(data_df.columns)
  data_df = data_help.keep_only_elements(data_df, ['U235'])

  print(data_df.columns)

  data_arr, col_index_map = data_help.split_df(data_df)

  print(f"Data array shape: {data_arr.shape}")
  print(f"Data array: {data_arr}")
  print(col_index_map)

  # X are the inputs, Y are the targets, notably each col in the input is still defined by col_index_map
  target_name = 'U235'
  X, Y = data_help.create_timeseries_targets(data_arr, col_index_map['time_days'], col_index_map, [target_name])

  # Plot variable correlations and distributions
  plot.plot_feature_target_correlations(X, Y, col_index_map, target_name=target_name, save_dir=plots_folder)
  plot.plot_data_distributions(X, Y, col_index_map, target_name='Target', save_dir=plots_folder)

  # Choose the device to do ML on: 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # and choosing device for training
  print(f"Using {device} device")

  indices = np.arange(len(X)) # Get original indices

  # First split: Train and Temp (80/20)
  X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
      X, Y, indices, test_size=0.2, random_state=0, shuffle=True
  )

  # Second split: Val and Test from Temp (50/50 of Temp => 10% each of total)
  X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
      X_temp, y_temp, idx_temp, test_size=0.5, shuffle=True, random_state=0
  )

if (__name__ == '__main__'):
  plots_folder = './plots'
  # Get the folder of the current script
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Build the path to the data file in a portable way
  data_file_path = os.path.join(script_dir, "data")
  main(data_file_path, plots_folder)