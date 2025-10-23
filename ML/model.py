# This modules will contain the Machine Learning Models used
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
  def __init__(self, n_inputs, hidden_layers, dropout_prob=0.1):
    super(SimpleDNN, self).__init__()

    # Complete layer sizes: [n_inputs, hidden1, hidden2, ..., 1]
    layer_sizes = [n_inputs] + hidden_layers + [1]

    # Create a list of Linear layers
    self.layers = nn.ModuleList([
      nn.Linear(in_size, out_size)
      for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ])

    self.dropout = nn.Dropout(p=dropout_prob)

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = self.dropout(F.relu(layer(x)))
    y = self.layers[-1](x)
    return y
  
def get_predictions(model, data_loader, device):
  model.eval()
  all_predictions = []
  all_targets = []
  
  with torch.no_grad():
      for inputs, targets in data_loader:
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = model(inputs)
          
          # Ensure shapes match
          if outputs.shape != targets.shape:
              targets = targets.view_as(outputs)
          
          all_predictions.append(outputs.cpu().numpy())
          all_targets.append(targets.cpu().numpy())
  
  # Concatenate all batches and flatten
  predictions = np.concatenate(all_predictions, axis=0).flatten()
  actuals = np.concatenate(all_targets, axis=0).flatten()
  
  return predictions, actuals