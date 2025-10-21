# This modules will contain the Machine Learning Models used
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
    y = F.softplus(self.layers[-1](x))
    return y