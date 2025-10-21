# This is the main file which will be called to begin ML training
# General Imports
import os
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


# === PyTorch === #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary


# Custom python files
import dataset_helper as data_help
import plot
import model as MLmodel




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
  plot.plot_data_distributions(X, Y, col_index_map, target_name='Target', save_dir=plots_folder, name='Raw_Data')

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

  # Scaling the data: 
  scaler =  MinMaxScaler()

  X_train = scaler.fit_transform(X_train)
  X_val   = scaler.transform(X_val)
  X_test  = scaler.transform(X_test)

  plot.plot_data_distributions(X_train, y_train, col_index_map, target_name='Target', save_dir=plots_folder, name='Scaled_Data')

  # Convert to torch tensors
  X_train_tensor = torch.nan_to_num(torch.tensor(X_train, dtype=torch.float32), nan=-1)
  X_val_tensor   = torch.nan_to_num(torch.tensor(X_val, dtype=torch.float32), nan=-1)
  X_test_tensor  = torch.nan_to_num(torch.tensor(X_test, dtype=torch.float32), nan=-1)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  y_val_tensor   = torch.tensor(y_val, dtype=torch.float32)
  y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
  
  Hidden_layers = [128, 128]
  Drop_out = 0.1
  LR = 0.00003
  Weight_decay = 0.0
  LR_SCHEDULER_PATIENCE = 10

  model = MLmodel.SimpleDNN(len(X_train[0]), Hidden_layers, Drop_out).to(device)
  summary(model, input_size=(1, len(X_train[0]))) # len(X_train[0]) Is the number of inputs

  optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=Weight_decay)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE)
  criterion = nn.L1Loss()

  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
  test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

  Num_workers = 10
  Train_batch = 2048
  Eval_batch = 2048

  # Create DataLoader objects for training, validation, and testing in batches
  train_loader = DataLoader(train_dataset, batch_size=Train_batch,  shuffle=True, num_workers=Num_workers, persistent_workers= True, drop_last=True, pin_memory=True)
  val_loader   = DataLoader(val_dataset, batch_size=Eval_batch, shuffle=False, num_workers=Num_workers, persistent_workers= True, drop_last=True, pin_memory=True)
  test_loader  = DataLoader(test_dataset, batch_size=Eval_batch, shuffle=False, num_workers=Num_workers)

  num_epochs = 2_000
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
  import matplotlib.pyplot as plt

  plt.figure(figsize=(10, 6))
  plt.plot(train_losses, label='Training Loss', linewidth=2)
  plt.plot(val_losses, label='Validation Loss', linewidth=2)
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('Loss (MSE)', fontsize=12)
  plt.title('Training and Validation Loss Over Time', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'training_loss.png'), dpi=300)
  plt.show()

  # 2. Get predictions on test set
  model.eval()
  all_predictions = []
  all_targets = []

  with torch.no_grad():
      for inputs, targets in test_loader:
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = model(inputs)
          
          if outputs.shape != targets.shape:
              targets = targets.view_as(outputs)
          
          all_predictions.append(outputs.cpu().numpy())
          all_targets.append(targets.cpu().numpy())

  # Concatenate all batches
  predictions = np.concatenate(all_predictions, axis=0).flatten()
  actuals = np.concatenate(all_targets, axis=0).flatten()

  # 3. Calculate metrics
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

  mae = mean_absolute_error(actuals, predictions)
  rmse = np.sqrt(mean_squared_error(actuals, predictions))
  r2 = r2_score(actuals, predictions)

  print(f"\n{'='*50}")
  print(f"TEST SET PERFORMANCE")
  print(f"{'='*50}")
  print(f"Mean Absolute Error (MAE):  {mae:.4f}")
  print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
  print(f"R² Score: {r2:.4f}")
  print(f"{'='*50}\n")

  # 4. Predictions vs Actuals Scatter Plot
  plt.figure(figsize=(10, 8))
  plt.scatter(actuals, predictions, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)

  # Perfect prediction line
  min_val = min(actuals.min(), predictions.min())
  max_val = max(actuals.max(), predictions.max())
  plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

  plt.xlabel('Actual Values', fontsize=12)
  plt.ylabel('Predicted Values', fontsize=12)
  plt.title(f'Predictions vs Actual Values\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'predictions_vs_actual.png'), dpi=300)
  plt.show()

  # 5. Residual Plot
  residuals = actuals - predictions

  plt.figure(figsize=(10, 6))
  plt.scatter(predictions, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
  plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
  plt.xlabel('Predicted Values', fontsize=12)
  plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
  plt.title('Residual Plot', fontsize=14)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'residuals.png'), dpi=300)
  plt.show()

  # 6. Residual Distribution
  plt.figure(figsize=(10, 6))
  plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
  plt.xlabel('Residuals', fontsize=12)
  plt.ylabel('Frequency', fontsize=12)
  plt.title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}', fontsize=14)
  plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'residual_distribution.png'), dpi=300)
  plt.show()

  # 7. Sample predictions table
  print("\nSample Predictions (first 10):")
  print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12}")
  print("-" * 36)
  for i in range(min(10, len(actuals))):
      print(f"{actuals[i]:<12.4f} {predictions[i]:<12.4f} {residuals[i]:<12.4f}")


if (__name__ == '__main__'):
  plots_folder = './plots'
  # Get the folder of the current script
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Build the path to the data file in a portable way
  data_file_path = os.path.join(script_dir, "data")
  main(data_file_path, plots_folder)