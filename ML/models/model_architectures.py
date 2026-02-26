import torch.nn as nn
from ML.models.model_helper import get_activation
import torch

class Deep_Neural_Network(nn.Module):
  def __init__(self, n_inputs, n_outputs, hidden_layers, dropout_prob, activation, output_activation, residual):
    super(Deep_Neural_Network, self).__init__()
    self.residual = residual

    layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
    self.layers = nn.ModuleList([
      nn.Linear(in_size, out_size)
      for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ])
    self.dropout = nn.Dropout(p=dropout_prob)

    self.activation_fn = get_activation(activation)
    self.output_activation_fn = get_activation(output_activation)

  def forward(self, x):
    for layer in self.layers[:-1]:
      residual = x  # store input for skip connection
      x = layer(x)
      x = self.activation_fn(x)
      x = self.dropout(x)

      # Only add residual if dimensions match and flag is True
      if self.residual and x.shape == residual.shape:
        x = x + residual

    y = self.layers[-1](x)
    y = self.output_activation_fn(y)
    return y


# ─── ODE Function ────────────────────────────────────────────────────────────

class ODEFuncForced(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.nfe = 0
    self.t_points = None
    self.power_profiles = None

    self.net = Deep_Neural_Network(
      n_inputs=2,              # power + U238
      n_outputs=1,             # dU238/dt
      hidden_layers=cfg.model.layers,
      dropout_prob=cfg.model.dropout_probability,
      activation=cfg.model.activation,
      output_activation=cfg.model.output_activation,
      residual=cfg.model.residual_connections,
    )

  def set_forcing(self, t_points, power_profiles):
    self.t_points = t_points
    self.power_profiles = power_profiles

  def _interpolate_power(self, t):
    t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
    idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
    idx = idx.clamp(0, len(self.t_points) - 2)

    t0 = self.t_points[idx]
    t1 = self.t_points[idx + 1]
    p0 = self.power_profiles[:, idx]
    p1 = self.power_profiles[:, idx + 1]

    frac = (t_clamped - t0) / (t1 - t0 + 1e-8)
    return (p0 + frac * (p1 - p0)).unsqueeze(-1)


  def forward(self, t, y):
    self.nfe += 1
    power = self._interpolate_power(t)
    combined = torch.cat([power, y], dim=-1)
    return self.net(combined)