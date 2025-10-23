# This is the main file which will be called to begin ML training
# General Imports
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === PyTorch === #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary


# Custom python files
import dataset_helper as data_help
import plot
import model as MLmodel
import metrics


def main(path_to_data, plots_folder):
  data_df = data_help.read_data(path_to_data, drop_run_label=True)

  data_help.print_dataset_stats(data_df)

  target_name = 'Xe135'
  extra_name = 'U238'
  data_df = data_help.filter_columns(data_df, [target_name,'U238', 'U235'], ['time_days','power_W_g','int_p_W'])

  print(data_df.columns)

  data_arr, col_index_map = data_help.split_df(data_df)

  print(f"Data array shape: {data_arr.shape}")
  print(f"Data array: {data_arr}")
  print(f"Column Index Dictionary: {col_index_map}")

  # X are the inputs, Y are the targets, notably each col in the input is still defined by col_index_map
  X, Y = data_help.create_timeseries_targets(data_arr, col_index_map['time_days'], col_index_map, [target_name])

  # Plot variable correlations and distributions
  plot.plot_correlation_matrix(X, Y, col_index_map, target_name=target_name, save_dir=plots_folder)
  plot.plot_data_distributions(X, Y, col_index_map, target_name='Target', save_dir=plots_folder, name='Raw_Data')

  # Choose the device to do ML on: 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # and choosing device for training
  print(f"Using {device} device")

  indices = np.arange(len(X)) # Get original indices

  first_run = X[0:100, :]

  # First split: Train and Temp (80/20)
  X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

  # Second split: Val and Test from Temp (50/50 of Temp => 10% each of total)
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=0)
  
  # Scaling the data: 
  scaler =  MinMaxScaler()

  X_train = scaler.fit_transform(X_train)
  X_val   = scaler.transform(X_val)
  X_test  = scaler.transform(X_test)

  first_run = scaler.transform(first_run)
  
  y_scaler = StandardScaler()
  y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
  y_val = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
  y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

  plot.plot_data_distributions(X_train, y_train, col_index_map, target_name='Target', save_dir=plots_folder, name='Scaled_Data')

  # Convert to torch tensors
  X_train_tensor = torch.nan_to_num(torch.tensor(X_train, dtype=torch.float32), nan=-1)
  X_val_tensor   = torch.nan_to_num(torch.tensor(X_val, dtype=torch.float32), nan=-1)
  X_test_tensor  = torch.nan_to_num(torch.tensor(X_test, dtype=torch.float32), nan=-1)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
  y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
  
  Hidden_layers = [64, 64]
  Drop_out = 0.1
  LR = 0.00003
  Weight_decay = 0.0
  LR_SCHEDULER_PATIENCE = 10

  model = MLmodel.SimpleDNN(len(X_train[0]), Hidden_layers, Drop_out).to(device)
  summary(model, input_size=(1, len(X_train[0]))) # len(X_train[0]) Is the number of inputs

  optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=Weight_decay)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE)
  criterion = nn.MSELoss()

  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
  test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

  Num_workers = 10
  Train_batch = 2048
  Eval_batch = 2048

  drop_last = False

  # Create DataLoader objects for training, validation, and testing in batches
  train_loader = DataLoader(train_dataset, batch_size=Train_batch,  shuffle=True, num_workers=Num_workers, persistent_workers= True, drop_last=drop_last, pin_memory=True)
  val_loader   = DataLoader(val_dataset, batch_size=Eval_batch, shuffle=False, num_workers=Num_workers, persistent_workers= True, drop_last=drop_last, pin_memory=True)
  test_loader  = DataLoader(test_dataset, batch_size=Eval_batch, shuffle=False, num_workers=Num_workers)

  num_epochs = 1_000
  patience = 5
  train_losses = []
  val_losses = []

  best_val_loss = float('inf')
  counter = 0
  best_model_state = None

  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for inputs, targets in train_loader:
          optimizer.zero_grad()
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = model(inputs)

          if outputs.shape != targets.shape:
            targets = targets.view_as(outputs)

          loss = criterion(outputs, targets)
      
          loss.backward()
          optimizer.step()
          running_loss += loss.item()

      train_loss = running_loss / len(train_loader)
      train_losses.append(train_loss)

      # Validation
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for inputs, targets in val_loader:
              inputs = inputs.to(device)
              targets = targets.to(device)
              outputs = model(inputs)

              if outputs.shape != targets.shape:
                targets = targets.view_as(outputs)

              loss = criterion(outputs, targets)

              val_loss += loss.item()

      val_loss /= len(val_loader)
      val_losses.append(val_loss)

      # Step the scheduler (assuming ReduceLROnPlateau)
      scheduler.step(val_loss)

      current_lr = optimizer.param_groups[0]['lr']
      print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}", end='\r')

      # Early stopping check
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model_state = model.state_dict()  # Save the best model weights
          counter = 0
      else:
          counter += 1

      if counter >= patience:
          print(f"\nEarly stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
          break

  # Load best model weights before testing or saving
  model.load_state_dict(best_model_state)

  # ============== EVALUATION AND PLOTTING ============== #

  # 1. Plot Training and Validation Loss
  plot.plot_losses(train_losses, val_losses, plots_folder)

  # 2. Get predictions on test set
  model.eval()
  predictions, actuals = MLmodel.get_predictions(model, test_loader, device)

  # Inverse transform predictions and actuals back to original scale
  predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
  actuals = y_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

  # 3. Calculate metrics
  mae = mean_absolute_error(actuals, predictions)
  rmse = np.sqrt(mean_squared_error(actuals, predictions))
  r2 = r2_score(actuals, predictions)

  # 4. Predictions vs Actuals Scatter Plot and the residual plot and distribution
  plot.plot_predictions_vs_actuals(actuals, predictions, mae, rmse, r2, plots_folder)
  plot.plot_residuals_combined(actuals, predictions, plots_folder)

  # 5. Calculate and plot feature importances, get the sorted feature names
  feature_names = [key for key, _ in sorted(col_index_map.items(), key=lambda x: x[1])]

  importance_means, importance_stds, baseline_r2 = metrics.calculate_feature_importance(model, test_loader, device, n_repeats=10)

  plot.plot_feature_importance(importance_means, importance_stds, feature_names, baseline_r2, plots_folder, n_top=20)

  # ===== PREDICTION COMPARISON ===== #
  # Get ground truth in original scale
  first_run_original = scaler.inverse_transform(first_run)
  ground_truth = first_run_original[:, col_index_map[target_name]]

  # Get teacher-forced predictions
  teacher_forced_preds = metrics.get_teacher_forced_predictions(model, first_run, y_scaler, device)
  autoregressive_preds = metrics.get_autoregressive_predictions(model, first_run, col_index_map[target_name], scaler, y_scaler, device)
  plot.plot_prediction_comparison(ground_truth, teacher_forced_preds, autoregressive_preds, target_name, plots_folder)


if (__name__ == '__main__'):
  

  plots_folder = './plots'
  # Get the folder of the current script
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Build the path to the data file in a portable way
  data_file_path = os.path.join(script_dir, "data")
  main(data_file_path, plots_folder)