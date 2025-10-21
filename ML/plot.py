# This file is used to write functions to help plot the training variables for the Machine Learning Models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_distributions(X, Y, col_index_map, target_name='Target', save_dir=None):
  # Ensure Y is 1D
  if Y.ndim > 1 and Y.shape[1] == 1:
      Y = Y.flatten()
  
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
  filepath = os.path.join(save_dir, 'distribution_features.png')
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  print(f"Feature distributions saved to: {filepath}")
  plt.close()


def plot_feature_target_correlations(X, Y, col_index_map, target_name='Target', save_dir=None):
  # Ensure Y is 1D
  if Y.ndim > 1 and Y.shape[1] == 1:  Y = Y.flatten()
  
  # Calculate correlations for each feature
  feature_names = []
  corr_values = []
  
  for feature_name, col_idx in col_index_map.items():
    corr = np.corrcoef(X[:, col_idx], Y)[0, 1]
    feature_names.append(feature_name)
    corr_values.append(corr)
  
  # Sort by absolute correlation (most important first)
  sorted_indices = np.argsort(np.abs(corr_values))[::-1]
  sorted_features = [feature_names[i] for i in sorted_indices]
  sorted_corrs = [corr_values[i] for i in sorted_indices]
  
  # Create heatmap
  fig, ax = plt.subplots(figsize=(5, 8))
  
  corr_matrix = np.array(sorted_corrs).reshape(-1, 1)
  im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
  
  ax.set_yticks(range(len(sorted_features)))
  ax.set_yticklabels(sorted_features)
  ax.set_xticks([0])
  ax.set_xticklabels([f'{target_name}(t+1)'])
  ax.set_title(f'Feature Correlations with {target_name}(t+1)', fontsize=14, fontweight='bold')
  
  # Add colorbar
  cbar = plt.colorbar(im, ax=ax)
  cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
  
  # Add correlation values on heatmap
  for i, corr in enumerate(sorted_corrs):
    color = 'white' if abs(corr) > 0.5 else 'black'
    ax.text(0, i, f'{corr:.3f}', ha='center', va='center', color=color, fontsize=9, fontweight='bold')
  
  plt.tight_layout()
  
  os.makedirs(save_dir, exist_ok=True)
  filename = f'correlation_{target_name}.png'
  filepath = os.path.join(save_dir, filename)
  plt.savefig(filepath, dpi=300, bbox_inches='tight')
  print(f"Plot saved to: {filepath}")
  plt.close()
