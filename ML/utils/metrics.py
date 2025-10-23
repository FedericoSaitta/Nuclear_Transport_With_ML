# This file contains functions to calculate metrics to probe the performance of the ML models
import numpy as np
import torch 
from sklearn.metrics import r2_score

def calculate_feature_importance(model, test_loader, device, n_repeats=10):
  print("\nCalculating feature importance...")
  
  # Prepare test data
  X_test = []
  y_test = []
  
  for inputs, targets in test_loader:
      X_test.append(inputs.numpy())
      y_test.append(targets.numpy())
  
  X_test = np.concatenate(X_test, axis=0)
  y_test = np.concatenate(y_test, axis=0).flatten()
  
  # Get baseline performance (R² score)
  model.eval()
  with torch.no_grad():
      X_tensor = torch.FloatTensor(X_test).to(device)
      baseline_predictions = model(X_tensor).cpu().numpy().flatten()
  
  baseline_r2 = r2_score(y_test, baseline_predictions)
  
  # Calculate permutation importance for each feature
  n_features = X_test.shape[1]
  feature_importances = np.zeros((n_features, n_repeats))
  
  for feature_idx in range(n_features):
      print(f"Processing feature {feature_idx + 1}/{n_features}...", end='\r')
      
      for repeat in range(n_repeats):
          # Copy the test set
          X_permuted = X_test.copy()
          
          # Shuffle the feature column
          np.random.shuffle(X_permuted[:, feature_idx])
          
          # Get predictions with permuted feature
          with torch.no_grad():
              X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
              permuted_predictions = model(X_permuted_tensor).cpu().numpy().flatten()
          
          # Calculate the drop in R² score
          permuted_r2 = r2_score(y_test, permuted_predictions)
          feature_importances[feature_idx, repeat] = baseline_r2 - permuted_r2
  
  print("\nFeature importance calculation complete!      ")
  
  # Calculate mean and std
  importance_means = feature_importances.mean(axis=1)
  importance_stds = feature_importances.std(axis=1)
  
  return importance_means, importance_stds, baseline_r2


def get_teacher_forced_predictions(model, data, y_scaler, device):
  # Convert to tensor
  data_tensor = torch.nan_to_num(torch.tensor(data, dtype=torch.float32), nan=-1)
  data_tensor = data_tensor.to(device)
  
  # Get predictions
  model.eval()
  with torch.no_grad():
      predictions_scaled = model(data_tensor).cpu().numpy().flatten()
  
  # Inverse transform to original scale
  predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
  
  # Trim last prediction to align with autoregressive
  return predictions[:-1]


def get_autoregressive_predictions(model, data, target_col_idx, x_scaler, y_scaler, device):
  predictions = []
  
  # Extract scaling parameters for target column
  target_min = x_scaler.data_min_[target_col_idx]
  target_max = x_scaler.data_max_[target_col_idx]
  
  # Start with first time step
  current_input = data[0:1].copy()
  
  model.eval()
  with torch.no_grad():
      for step in range(len(data) - 1):
          # Convert to tensor and predict
          input_tensor = torch.nan_to_num(torch.tensor(current_input, dtype=torch.float32), nan=-1)
          input_tensor = input_tensor.to(device)
          
          pred_scaled = model(input_tensor).cpu().numpy().flatten()[0]
          predictions.append(pred_scaled)
          
          # Prepare next input
          if step < len(data) - 1:
              # Get features from next time step
              current_input = data[step + 1:step + 2].copy()
              
              # Convert prediction: StandardScaler → Original → MinMaxScaler
              pred_original = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
              pred_minmax = (pred_original - target_min) / (target_max - target_min)
              
              # Replace target column with prediction
              current_input[0, target_col_idx] = pred_minmax
  
  # Convert to original scale
  predictions = np.array(predictions)
  return y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()