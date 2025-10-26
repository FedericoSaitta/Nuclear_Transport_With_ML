# This file is used to write functions to help plot the training variables for the Machine Learning Models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(X, Y, col_index_map, target_name='Target', save_dir=None, name='correlation_matrix'):
  # Ensure Y is 1D
  if Y.ndim > 1 and Y.shape[1] == 1: Y = Y.flatten()
  
  # Get sorted feature names
  feature_list = sorted(col_index_map.items(), key=lambda x: x[1])
  feature_names = [name for name, _ in feature_list]
  
  # Combine X and Y into single array
  data_combined = np.column_stack([X, Y])
  all_names = feature_names + [f'{target_name}(t+1)']
  
  # Calculate correlation matrix
  corr_matrix = np.corrcoef(data_combined.T)
  
  # Create figure
  fig, ax = plt.subplots(figsize=(12, 10))
  
  # Create heatmap
  im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
  
  # Set ticks and labels
  ax.set_xticks(range(len(all_names)))
  ax.set_yticks(range(len(all_names)))
  ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
  ax.set_yticklabels(all_names, fontsize=9)
  
  # Add colorbar
  cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=11)
  
  # Add correlation values to cells
  for i in range(len(all_names)):
    for j in range(len(all_names)):
      corr_val = corr_matrix[i, j]
      # Choose text color based on background
      text_color = 'white' if abs(corr_val) > 0.5 else 'black'
      # Only show values for off-diagonal or if significant
      if i != j or abs(corr_val) > 0.01:
        ax.text(j, i, f'{corr_val:.2f}', 
                ha='center', va='center', 
                color=text_color, fontsize=7)
  
  # Highlight the target row and column
  target_idx = len(all_names) - 1

  ax.set_title(f'Correlation Matrix: All Features and {target_name}(t+1)', 
                fontsize=14, fontweight='bold', pad=20)
  
  plt.tight_layout()
  
  # Save plot
  os.makedirs(save_dir, exist_ok=True)
  filepath = os.path.join(save_dir, f'{name}.png')
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  print(f"Correlation matrix saved to: {filepath}")
  plt.close()
  
  # Print top correlations with target
  print(f"\n{'='*60}")
  print(f"TOP CORRELATIONS WITH {target_name}(t+1)")
  print(f"{'='*60}")
  target_corrs = corr_matrix[target_idx, :-1]  # Exclude self-correlation
  sorted_indices = np.argsort(np.abs(target_corrs))[::-1]
  
  for rank, idx in enumerate(sorted_indices[:10], 1):
    print(f"{rank:2d}. {feature_names[idx]:25s} | {target_corrs[idx]:+.4f}")
  print(f"{'='*60}\n")
  

def plot_data_distributions(X, Y, col_index_map, target_name='Target', save_dir=None, name='Raw_Data'):
  # Ensure Y is 1D
  if Y.ndim > 1 and Y.shape[1] == 1: Y = Y.flatten()
  
  # Get all feature names sorted by index
  feature_list = sorted(col_index_map.items(), key=lambda x: x[1])
  feature_names = [name for name, _ in feature_list]
  
  # Calculate grid dimensions
  n_features = len(feature_names)
  n_cols = 4
  n_rows = int(np.ceil(n_features / n_cols))
  
  # Create figure
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
  axes = axes.flatten() if n_features > 1 else [axes]
  
  # Plot histogram for each feature
  for idx, (feature_name, col_idx) in enumerate(feature_list):
    ax = axes[idx]
    data = X[:, col_idx]
    
    # Create histogram
    n, bins, patches = ax.hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color code if this is the target feature
    if feature_name == target_name:
      for patch in patches:
        patch.set_facecolor('red')
        patch.set_alpha(0.7)
      ax.set_title(f'{feature_name} (TARGET at t)', fontweight='bold', color='red')
    else:
      for patch in patches:
        patch.set_facecolor('steelblue')
      ax.set_title(feature_name)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3e}')
    ax.legend(fontsize=8)
  
  # Hide unused subplots
  for idx in range(n_features, len(axes)):
    axes[idx].axis('off')
  
  plt.suptitle('Feature Distributions (Input Features at time t)', 
                fontsize=16, fontweight='bold', y=1.0)
  plt.tight_layout()
  
  os.makedirs(save_dir, exist_ok=True)
  filepath = os.path.join(save_dir, name + '_distribution_features.png')
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  print(f"Feature distributions saved to: {filepath}")
  plt.close()


def plot_losses(train_losses, val_losses, save_dir):
  plt.figure(figsize=(10, 6))
  plt.plot(train_losses, label='Training Loss', linewidth=2)
  plt.plot(val_losses, label='Validation Loss', linewidth=2)
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('Loss (MSE)', fontsize=12)
  plt.title('Training and Validation Loss Over Time', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300)


def plot_predictions_vs_actuals(actuals, predictions, mae, rmse, r2, plots_folder):
  plt.figure(figsize=(10, 8))
  plt.scatter(actuals, predictions, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
  
  # Perfect prediction line
  min_val = min(actuals.min(), predictions.min())
  max_val = max(actuals.max(), predictions.max())
  plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
  
  plt.xlabel('Actual Values', fontsize=12)
  plt.ylabel('Predicted Values', fontsize=12)
  plt.title(f'Predictions vs Actual Values\nR² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}', 
            fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'predictions_vs_actual.png'), dpi=300)
  plt.close()


def plot_residuals_combined(actuals, predictions, plots_folder):
  residuals = actuals - predictions
  
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))
  
  # Left plot: Residual scatter
  axes[0].scatter(predictions, residuals, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
  axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
  axes[0].set_xlabel('Predicted Values', fontsize=12)
  axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
  axes[0].set_title('Residual Plot', fontsize=14)
  axes[0].grid(True, alpha=0.3)
  
  # Right plot: Residual distribution
  axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
  axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
  axes[1].set_xlabel('Residuals', fontsize=12)
  axes[1].set_ylabel('Frequency', fontsize=12)
  axes[1].set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}', 
                    fontsize=14)
  axes[1].grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'residuals_combined.png'), dpi=300)
  plt.close()

def plot_feature_importance(importance_means, importance_stds, feature_names, 
                          baseline_r2, plots_folder, n_top=20):
  n_features = len(importance_means)
  
  # Sort features by importance
  indices = np.argsort(importance_means)[::-1]
  
  # Plot top N features (or all if less than N)
  n_top = min(n_top, n_features)
  top_indices = indices[:n_top]
  
  plt.figure(figsize=(10, 8))
  colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_top))
  plt.barh(range(n_top), importance_means[top_indices], 
            xerr=importance_stds[top_indices], 
            align='center', alpha=0.8, edgecolor='black',
            color=colors)
  plt.yticks(range(n_top), [feature_names[i] for i in top_indices])
  plt.xlabel('Permutation Importance (Decrease in R²)', fontsize=12)
  plt.ylabel('Features', fontsize=12)
  plt.title(f'Top {n_top} Most Important Features\nBaseline R² = {baseline_r2:.4f}', 
            fontsize=14)
  plt.gca().invert_yaxis()
  plt.grid(True, alpha=0.3, axis='x')
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'feature_importance.png'), dpi=300)
  plt.close()


def plot_prediction_comparison(ground_truth, teacher_forced, autoregressive, 
                               target_name, plots_folder):
  time_indices = np.arange(len(ground_truth))
  prediction_indices = time_indices[1:len(autoregressive) + 1]
  
  plt.figure(figsize=(14, 7))
  plt.plot(time_indices, ground_truth, 'b-o', label='Ground Truth', 
            linewidth=2, markersize=4)
  plt.plot(prediction_indices, teacher_forced[:len(autoregressive)], 'r--s', 
            label='Teacher Forced', linewidth=2, markersize=4, alpha=0.7)
  plt.plot(prediction_indices, autoregressive, 'g--^', 
            label='Autoregressive', linewidth=2, markersize=4, alpha=0.7)
  
  plt.xlabel('Time Step', fontsize=12)
  plt.ylabel(target_name, fontsize=12)
  plt.title('Prediction Comparison: Teacher Forced vs Autoregressive', fontsize=14)
  plt.legend(fontsize=10)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(os.path.join(plots_folder, 'prediction_comparison.png'), dpi=300)
  plt.close()