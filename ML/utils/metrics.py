# This file contains functions to calculate metrics to probe the performance of the ML models
from loguru import logger
import numpy as np
import torch 
from sklearn.metrics import r2_score
def calculate_feature_importance(model, test_loader, device, n_repeats=10, 
                                 metric={'name': 'r2', 'direction': 'increasing'}):
  from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
  
  # Prepare test data
  X_test = []
  y_test = []
  
  for inputs, targets in test_loader:
    X_test.append(inputs.numpy())
    y_test.append(targets.numpy())
  
  X_test = np.concatenate(X_test, axis=0)
  y_test = np.concatenate(y_test, axis=0).flatten()
  
  # Define metric function
  metric_functions = {
    'r2': r2_score,
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
  }
  
  metric_name = metric['name'].lower()
  metric_direction = metric['direction'].lower()
  
  if metric_name not in metric_functions:
      raise ValueError(f"Unsupported metric: {metric_name}. Choose from {list(metric_functions.keys())}")
  
  if metric_direction not in ['increasing', 'decreasing']:
      raise ValueError("Direction must be 'increasing' or 'decreasing'")
  
  metric_func = metric_functions[metric_name]
  
  # Get baseline performance
  model.eval()
  with torch.no_grad():
      X_tensor = torch.FloatTensor(X_test).to(device)
      baseline_predictions = model(X_tensor).cpu().numpy().flatten()
  
  baseline_score = metric_func(y_test, baseline_predictions)
  
  # Calculate permutation importance for each feature
  n_features = X_test.shape[1]
  feature_importances = np.zeros((n_features, n_repeats))
  logger.info(f"Calculating Feature Importances using {metric_name} ({metric_direction} is better)")
  
  for feature_idx in range(n_features):
      for repeat in range(n_repeats):
          # Copy the test set
          X_permuted = X_test.copy()
          
          # Shuffle the feature column
          np.random.shuffle(X_permuted[:, feature_idx])
          
          # Get predictions with permuted feature
          with torch.no_grad():
              X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
              permuted_predictions = model(X_permuted_tensor).cpu().numpy().flatten()
          
          # Calculate the change in score
          permuted_score = metric_func(y_test, permuted_predictions)
          
          # Calculate importance based on direction
          if metric_direction == 'increasing':
              # For metrics where higher is better (e.g., R²)
              # Positive importance = performance dropped when feature was shuffled
              feature_importances[feature_idx, repeat] = baseline_score - permuted_score
          else:  # decreasing
              # For metrics where lower is better (e.g., MSE, MAE)
              # Positive importance = performance got worse (score increased) when feature was shuffled
              feature_importances[feature_idx, repeat] = permuted_score - baseline_score
  
  # Calculate mean and std
  importance_means = feature_importances.mean(axis=1)
  importance_stds = feature_importances.std(axis=1)
  
  return importance_means, importance_stds, baseline_score


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
        
        # Convert prediction: y_scaler space → Original → x_scaler space
        pred_original = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
        
        # Create a dummy row with the prediction in the target column
        dummy_row = np.zeros((1, data.shape[1]))
        dummy_row[0, target_col_idx] = pred_original
        
        # Transform to x_scaler space
        pred_x_scaled = x_scaler.transform(dummy_row)[0, target_col_idx]
        
        # Replace target column with prediction
        current_input[0, target_col_idx] = pred_x_scaled
  
  # Convert to original scale
  predictions = np.array(predictions)
  return y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()