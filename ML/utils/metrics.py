# This file contains functions to calculate metrics to probe the performance of the ML models
from loguru import logger
import numpy as np
import torch 


def calculate_feature_importance(model, test_loader, device, n_repeats=10, 
                                 metric={'name': 'r2', 'direction': 'increasing'},
                                 output_idx=None):
  from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
  
  # Prepare test data
  X_test = []
  y_test = []
  
  for inputs, targets in test_loader:
    X_test.append(inputs.numpy())
    y_test.append(targets.numpy())
  
  X_test = np.concatenate(X_test, axis=0)
  y_test = np.concatenate(y_test, axis=0)
  
  # Ensure y_test is 2D
  if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)
  
  # If output_idx is specified, extract only that output column
  if output_idx is not None:
    y_test = y_test[:, output_idx:output_idx+1]  # Keep 2D shape
  
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
    baseline_predictions = model(X_tensor).cpu().numpy()
  
  # Ensure predictions are 2D
  if baseline_predictions.ndim == 1:
    baseline_predictions = baseline_predictions.reshape(-1, 1)
  
  # If output_idx is specified, extract only that output column
  if output_idx is not None:
    baseline_predictions = baseline_predictions[:, output_idx:output_idx+1]  # Keep 2D shape

  baseline_score = metric_func(y_test, baseline_predictions)
  
  # Calculate permutation importance for each feature
  n_features = X_test.shape[1]
  feature_importances = np.zeros((n_features, n_repeats))
  
  if output_idx is not None:
    logger.info(f"Calculating Feature Importances for output {output_idx} using {metric_name} ({metric_direction} is better)")
  else:
    logger.info(f"Calculating Feature Importances (all outputs) using {metric_name} ({metric_direction} is better)")
  
  for feature_idx in range(n_features):
    for repeat in range(n_repeats):
      # Copy the test set
      X_permuted = X_test.copy()
      
      # Shuffle the feature column
      np.random.shuffle(X_permuted[:, feature_idx])
      
      # Get predictions with permuted feature
      with torch.no_grad():
        X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
        permuted_predictions = model(X_permuted_tensor).cpu().numpy()
      
      # Ensure predictions are 2D
      if permuted_predictions.ndim == 1:
        permuted_predictions = permuted_predictions.reshape(-1, 1)
      
      # If output_idx is specified, extract only that output column
      if output_idx is not None:
        permuted_predictions = permuted_predictions[:, output_idx:output_idx+1]  # Keep 2D shape
      
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


def get_teacher_forced_predictions(model, data, y_scaler, device, output_idx=0):
  # Convert to tensor
  data_tensor = torch.nan_to_num(torch.tensor(data, dtype=torch.float32), nan=-1)
  data_tensor = data_tensor.to(device)
  
  # Get predictions
  model.eval()
  with torch.no_grad():
    predictions_scaled = model(data_tensor).cpu().numpy()
  
  # Handle multi-output: extract specific output
  if predictions_scaled.ndim == 2 and predictions_scaled.shape[1] > 1:
    # Multi-output model
    predictions_scaled = predictions_scaled[:, output_idx]
  else:
    # Single output or already 1D
    predictions_scaled = predictions_scaled.flatten()
  
  # Inverse transform to original scale
  # Need to handle ColumnTransformer vs single scaler
  from sklearn.compose import ColumnTransformer
  import ML.datamodule.data_scalers as data_scaler
  
  if isinstance(y_scaler, ColumnTransformer):
    # For ColumnTransformer, create a dummy array with all outputs
    dummy = np.zeros((len(predictions_scaled), y_scaler.n_features_in_))
    dummy[:, output_idx] = predictions_scaled
    predictions_original = data_scaler.inverse_transform_column_transformer(y_scaler, dummy)
    predictions = predictions_original[:, output_idx]
  else:
    # Single scaler
    predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
  
  # Trim last prediction to align with autoregressive
  return predictions[:-1]


def get_autoregressive_predictions(model, data, target_col_idx, x_scaler, y_scaler, device, output_idx=0):
  from sklearn.compose import ColumnTransformer
  import ML.datamodule.data_scalers as data_scaler
  
  predictions = []
  
  # Start with first time step
  current_input = data[0:1].copy()
  
  model.eval()
  with torch.no_grad():
    for step in range(len(data) - 1):
      # Convert to tensor and predict
      input_tensor = torch.nan_to_num(torch.tensor(current_input, dtype=torch.float32), nan=-1)
      input_tensor = input_tensor.to(device)
      
      pred_output = model(input_tensor).cpu().numpy()
      
      # Handle multi-output: extract specific output
      if pred_output.ndim == 2 and pred_output.shape[1] > 1:
        pred_scaled = pred_output[0, output_idx]
      else:
        pred_scaled = pred_output.flatten()[0]
      
      predictions.append(pred_scaled)
      
      # Prepare next input
      if step < len(data) - 1:
        # Get features from next time step (everything except target)
        current_input = data[step + 1:step + 2].copy()
        
        # Step 1: Convert prediction from y_scaler space to original space
        if isinstance(y_scaler, ColumnTransformer):
          # For ColumnTransformer - need all target dimensions
          dummy_y = np.zeros((1, y_scaler.n_features_in_))
          dummy_y[0, output_idx] = pred_scaled
          pred_original_array = data_scaler.inverse_transform_column_transformer(y_scaler, dummy_y)
          pred_original = pred_original_array[0, output_idx]
        else:
          # Single scaler
          pred_original = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
        
        # Step 2: Convert from original space to x_scaler space
        # Need to create a full row with the prediction in the correct column
        if isinstance(x_scaler, ColumnTransformer):
          # For ColumnTransformer, we need to transform properly
          # Create a dummy row in ORIGINAL space with just the target value
          dummy_x_original = np.zeros((1, data.shape[1]))
          dummy_x_original[0, target_col_idx] = pred_original
          
          # Transform to x_scaler space
          dummy_x_scaled = x_scaler.transform(dummy_x_original)
          pred_x_scaled = dummy_x_scaled[0, target_col_idx]
        else:
          # Single scaler - transform the whole row
          dummy_x_original = np.zeros((1, data.shape[1]))
          dummy_x_original[0, target_col_idx] = pred_original
          dummy_x_scaled = x_scaler.transform(dummy_x_original)
          pred_x_scaled = dummy_x_scaled[0, target_col_idx]
        
        # Replace target column with prediction
        current_input[0, target_col_idx] = pred_x_scaled
  
  # Convert predictions to original scale for plotting
  predictions = np.array(predictions)
  
  if isinstance(y_scaler, ColumnTransformer):
    # For ColumnTransformer
    dummy = np.zeros((len(predictions), y_scaler.n_features_in_))
    dummy[:, output_idx] = predictions
    predictions_original = data_scaler.inverse_transform_column_transformer(y_scaler, dummy)
    return predictions_original[:, output_idx]
  else:
    # Single scaler
    return y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()