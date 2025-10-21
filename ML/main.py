# This is the main file which will be called to begin ML training
# General Imports
import os
import pandas as pd
import numpy as np


# Custom python files
import dataset_helper as data_help
import plot

def main(path_to_data):
  data_df = data_help.read_data(path_to_data)

  data_help.print_dataset_stats(data_df)

  print(data_df.columns)
  data_df = data_help.keep_only_elements(data_df, ['Zr96'])

  print(data_df.columns)



if (__name__ == '__main__'):

  # Get the folder of the current script
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Build the path to the data file in a portable way
  data_file_path = os.path.join(script_dir, "data", "data.csv")
  main(data_file_path)