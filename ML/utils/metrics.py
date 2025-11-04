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


def calculate_mare_autoregressive(model, X_data, Y_data, target_col_indices, x_scaler, y_scaler, device, steps_per_run=100):
  """
  Calculate MARE using autoregressive prediction with ALL outputs fed back.
  
  Args:
    model: Trained model
    X_data: Input data array (scaled)
    Y_data: Target data array (scaled) - these are the true next-step values
    target_col_indices: Dict mapping target names to their column indices in X_data
    x_scaler: Input scaler
    y_scaler: Target scaler
    device: Device to run model on
    steps_per_run: Number of time steps per run
  
  Returns:
    predictions_dict: Dict of predictions for each target
    ground_truth_dict: Dict of ground truth for each target
  """
  # Calculate number of complete runs
  total_samples = len(X_data)
  n_runs = total_samples // steps_per_run
  
  # Raise error if steps are not exactly divisible
  if total_samples % steps_per_run != 0:
    raise ValueError(f"Dataset has {total_samples} samples, trimming to {n_runs * steps_per_run} for complete runs")

  # Get number of outputs and target names
  n_outputs = Y_data.shape[1] if Y_data.ndim > 1 else 1
  target_names = list(target_col_indices.keys())
  
  logger.info(f"Running autoregressive predictions for {n_runs} runs ({steps_per_run} steps each)")
  
  # Initialize storage for all targets
  all_predictions = {name: [] for name in target_names}
  all_ground_truth = {name: [] for name in target_names}
  
  model.eval()
  with torch.no_grad():
    for run_idx in tqdm(range(n_runs), desc="Autoregressive MARE", unit="run"):
      # Get this run's data
      run_start = run_idx * steps_per_run
      run_end = run_start + steps_per_run
      run_x_data = X_data[run_start:run_end]
      run_y_data = Y_data[run_start:run_end]
      
      # Start with first time step
      current_input = run_x_data[0:1].copy()
      
      for step in range(len(run_x_data) - 1):
        # Predict ALL outputs
        input_tensor = torch.nan_to_num(torch.tensor(current_input, dtype=torch.float32), nan=-1)
        input_tensor = input_tensor.to(device)
        pred_output = model(input_tensor).cpu().numpy()
        
        # pred_output shape: [1, n_outputs]
        if pred_output.ndim == 1:
          pred_output = pred_output.reshape(1, -1)
        
        # Convert ALL predictions from y_scaler space to original space
        if isinstance(y_scaler, ColumnTransformer):
          pred_original = data_scaler.inverse_transform_column_transformer(y_scaler, pred_output)
          pred_original = pred_original[0]  # Get the single row
        else:
          pred_original = y_scaler.inverse_transform(pred_output)[0]
        
        # Store predictions for this step (for all targets)
        for output_idx, target_name in enumerate(target_names):
          all_predictions[target_name].append(pred_original[output_idx])
        
        # Prepare next input
        if step < len(run_x_data) - 1:
          current_input = run_x_data[step + 1:step + 2].copy()
          
          # Update ALL target columns with their predictions
          for output_idx, target_name in enumerate(target_names):
            target_col_idx = target_col_indices[target_name]
            pred_val_original = pred_original[output_idx]
            
            # Convert from original space to x_scaler space
            if isinstance(x_scaler, ColumnTransformer):
              dummy_x_original = np.zeros((1, run_x_data.shape[1]))
              dummy_x_original[0, target_col_idx] = pred_val_original
              dummy_x_scaled = x_scaler.transform(dummy_x_original)
              pred_x_scaled = dummy_x_scaled[0, target_col_idx]
            else:
              dummy_x_original = np.zeros((1, run_x_data.shape[1]))
              dummy_x_original[0, target_col_idx] = pred_val_original
              dummy_x_scaled = x_scaler.transform(dummy_x_original)
              pred_x_scaled = dummy_x_scaled[0, target_col_idx]
            
            # Replace target column with prediction
            current_input[0, target_col_idx] = pred_x_scaled
      
      # Get ground truth from Y_data for all targets
      run_y_data_subset = run_y_data[:-1]  # Exclude last since we don't predict beyond run
      
      if isinstance(y_scaler, ColumnTransformer):
        run_ground_truth_original = data_scaler.inverse_transform_column_transformer(y_scaler, run_y_data_subset)
      else:
        run_ground_truth_original = y_scaler.inverse_transform(run_y_data_subset)
      
      # Store ground truth for all targets
      for output_idx, target_name in enumerate(target_names):
        if run_ground_truth_original.ndim == 1:
          all_ground_truth[target_name].extend(run_ground_truth_original)
        else:
          all_ground_truth[target_name].extend(run_ground_truth_original[:, output_idx])
  
  # Convert lists to arrays
  predictions_dict = {name: np.array(preds) for name, preds in all_predictions.items()}
  ground_truth_dict = {name: np.array(gt) for name, gt in all_ground_truth.items()}
  
  logger.info(f"Autoregressive predictions complete!")
  
  return predictions_dict, ground_truth_dict