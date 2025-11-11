# This file contains functions to calculate metrics to probe the performance of the ML models
from loguru import logger
import numpy as np
import torch 
from sklearn.compose import ColumnTransformer
import ML.datamodule.data_scalers as data_scaler
from tqdm import tqdm
  


# These always return arrays even if with only one output (eg. [single_output])
def mae(y_true, y_pred):
    """Mean Absolute Error per output"""
    result = np.mean(np.abs(y_true - y_pred), axis=0)
    return np.atleast_1d(result)

def mse(y_true, y_pred):
    """Mean Squared Error per output"""
    result = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.atleast_1d(result)

def rmse(y_true, y_pred):
    """Root Mean Squared Error per output"""
    result = np.sqrt(mse(y_true, y_pred))
    return np.atleast_1d(result)

def r2(y_true, y_pred):
    """R² (Coefficient of Determination) per output"""
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    r2_scores = np.zeros(n_outputs)
    if n_outputs == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    for i in range(n_outputs):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2_scores[i] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return np.atleast_1d(r2_scores)

# Returns Float
def mare(tf_gt, tf_pred):
    tf_gt = np.asarray(tf_gt)
    tf_pred = np.asarray(tf_pred)

    max_val_tf = np.max(np.abs(tf_gt))
    if max_val_tf > 0:
        are_tf = np.abs(tf_gt - tf_pred) / max_val_tf
        mare_tf = np.mean(are_tf)
    else:
        mare_tf = float('nan')

    return mare_tf




def calculate_feature_importance(model, test_loader, device, n_repeats=10, 
                                 metric={'name': 'r2', 'direction': 'increasing'},output_idx=None):
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

def get_model_prediction(model, input, y_scaler):
    # Ensure input is 2D: (1, n_features)
    if input.ndim == 1:
      input = input.reshape(1, -1)
    
    input_tensor = torch.tensor(input, dtype=torch.float32)
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(input_tensor)  # Shape: (1, n_targets)
    
    # Convert back to numpy - ensure 2D
    prediction_scaled = prediction_scaled.cpu().numpy()
    if prediction_scaled.ndim == 1:
      prediction_scaled = prediction_scaled.reshape(1, -1)
    
    # Inverse transform to get original scale - pass 2D array, not list
    prediction_original = data_scaler.inverse_transformer(y_scaler, prediction_scaled)
    return prediction_original

def get_model_prediction(model, input, y_scaler):
    # Ensure input is 2D: (1, n_features)
    if input.ndim == 1:
      input = input.reshape(1, -1)
    
    input_tensor = torch.tensor(input, dtype=torch.float32)
    
    # Move input to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
      prediction_scaled = model(input_tensor)  # Shape: (1, n_targets)
    
    # Convert back to numpy - ensure 2D
    prediction_scaled = prediction_scaled.cpu().numpy()
    if prediction_scaled.ndim == 1:
      prediction_scaled = prediction_scaled.reshape(1, -1)
    
    # Inverse transform to get original scale - pass 2D array, not list
    prediction_original = data_scaler.inverse_transformer(y_scaler, prediction_scaled)
    return prediction_original

def model_autoregress(model, X_data, Y_data, x_scaler, y_scaler, steps_per_run, inputs_indices, target_col_indices, delta_conc):
    total_samples = len(X_data)
    n_runs = total_samples // steps_per_run
    
    if total_samples % steps_per_run != 0:
      logger.warning(f"Dataset has {total_samples} samples, not divisible by {steps_per_run}")
      total_samples = n_runs * steps_per_run
      X_data = X_data[:total_samples]
      Y_data = Y_data[:total_samples]

    unscaled_x_data = data_scaler.inverse_transformer(x_scaler, X_data)

    logger.info(f"Running autoregressive predictions for {n_runs} runs ({steps_per_run} steps each)")
    logger.info(steps_per_run)

    # Initialize dictionaries with empty lists for each target
    predictions_dict = {target_name: [] for target_name in target_col_indices.keys()}
    ground_truth_dict = {target_name: [] for target_name in target_col_indices.keys()}
    
    current_concentrations = {}
    
    for depletion_run in tqdm(range(n_runs), desc="Autoregressive MARE", unit="run"):
        run_start = depletion_run * steps_per_run
        run_end = run_start + steps_per_run
        
        # Initialize concentrations at start of each run
        if delta_conc:
            # Get initial concentrations from the first timestep
            initial_x = data_scaler.inverse_transformer(x_scaler, X_data[run_start].reshape(1, -1))[0]
            for target_name in target_col_indices.keys():
                if target_name in inputs_indices:
                    current_concentrations[target_name] = initial_x[inputs_indices[target_name]]

        for run_idx in range(run_start, run_end):
            # Get model prediction (delta if delta_conc=True)
            model_output_unscaled = get_model_prediction(model, X_data[run_idx], y_scaler)
            
            # Get ground truth (delta if delta_conc=True)
            ground_truth_unscaled = data_scaler.inverse_transformer(
                y_scaler, Y_data[run_idx].reshape(1, -1)
            )

            # Save results for each target
            for target_name, target_index in target_col_indices.items():
                predictions_dict[target_name].append(model_output_unscaled[0, target_index])
                ground_truth_dict[target_name].append(ground_truth_unscaled[0, target_index])

            # Update next timestep's input
            if run_idx < run_end - 1:
                for target_name, target_index in target_col_indices.items():
                    if target_name in inputs_indices:
                        if delta_conc:
                            # Convert delta to absolute concentration
                            delta = model_output_unscaled[0, target_index]
                            current_concentrations[target_name] += delta
                            # Put absolute concentration in input
                            unscaled_x_data[run_idx + 1, inputs_indices[target_name]] = \
                                current_concentrations[target_name]
                        else:
                            # Already absolute, just use directly
                            unscaled_x_data[run_idx + 1, inputs_indices[target_name]] = \
                                model_output_unscaled[0, target_index]

                # Transform back to scaled space
                X_data[run_idx + 1] = x_scaler.transform(
                    unscaled_x_data[run_idx + 1].reshape(1, -1)
                )[0]
    
    # Convert lists to numpy arrays
    for target_name in target_col_indices.keys():
      predictions_dict[target_name] = np.array(predictions_dict[target_name])
      ground_truth_dict[target_name] = np.array(ground_truth_dict[target_name])
    
    logger.info(f"Collected predictions for targets: {list(predictions_dict.keys())}")
    
    return predictions_dict, ground_truth_dict