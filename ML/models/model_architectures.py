import torch.nn as nn
import torch.nn.functional as F
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
    self.forcing_profiles = None

    n_input = len(cfg.dataset.inputs)
    n_target = len(cfg.dataset.targets)

    self.net = Deep_Neural_Network(
      n_inputs=n_input + n_target,
      n_outputs=n_target,
      hidden_layers=cfg.model.layers,
      dropout_prob=cfg.model.dropout_probability,
      activation=cfg.model.activation,
      output_activation=cfg.model.output_activation,
      residual=cfg.model.residual_connections,
    )

  def set_forcing(self, t_points, forcing_profiles):
    self.t_points = t_points
    self.forcing_profiles = forcing_profiles  # (batch, steps, n_input)

  def _interpolate_forcing(self, t):
    """Piecewise-constant (zero-order hold) forcing interpolation.
    
    Returns the forcing value at the left endpoint of whichever interval
    t falls into, i.e. the value is held constant until the next grid point.
    This is correct for power profiles that are sampled independently at
    each timestep (step-function behaviour).
    """
    t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
    idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
    idx = idx.clamp(0, len(self.t_points) - 2)

    return self.forcing_profiles[:, idx, :]   # (batch, n_input)

  def forward(self, t, y):
    self.nfe += 1
    forcing = self._interpolate_forcing(t)      # (batch, n_input)
    combined = torch.cat([forcing, y], dim=-1)   # (batch, n_input + n_target)
    return self.net(combined)
  
# ─── Matrix ODE Function ─────────────────────────────────────────────────────
 
class ODEFuncMatrix(nn.Module):
  """ODE function that predicts a depletion matrix A(t) from forcing inputs.

  dy/dt = A(forcing(t)) @ y(t)

  Physical constraints:
    - Diagonal entries are negative (decay/absorption losses)
    - Off-diagonal entries are positive (production from transmutation)

  Sparsity modes (cfg.model.matrix_sparsity):
    - "full"  : all n²  entries are learnable (default)
    - "chain" : bidiagonal with mass conservation — the network outputs n
                rates r₁…rₙ; diagonal = -rᵢ, sub-diagonal = +r_{i-1}.
                Atoms lost from isotope i-1 exactly appear in isotope i.
                Only n parameters instead of n².
  """

  def __init__(self, cfg):
    super().__init__()
    self.nfe = 0
    self.t_points = None
    self.forcing_profiles = None

    n_input = len(cfg.dataset.inputs)
    n_target = len(cfg.dataset.targets)
    self.n_target = n_target

    # Sparsity configuration
    sparsity = getattr(cfg.model, 'matrix_sparsity', 'full')
    self.sparsity = sparsity

    if sparsity == 'chain':
      # The network outputs n rates r₁…rₙ.  The matrix is built as:
      #   diagonal(i,i)     = -rᵢ          (loss from isotope i)
      #   sub-diag(i,i-1)   = +r_{i-1}     (production: what leaves i-1 enters i)
      # This enforces mass conservation along the chain automatically.
      n_outputs = n_target   # one rate per isotope
    else:
      n_outputs = n_target * n_target

    # Network takes forcing only -> outputs learnable matrix entries
    self.net = Deep_Neural_Network(
      n_inputs=n_input,
      n_outputs=n_outputs,
      hidden_layers=cfg.model.layers,
      dropout_prob=cfg.model.dropout_probability,
      activation=cfg.model.activation,
      output_activation='none',  # We apply our own constraints
      residual=cfg.model.residual_connections,
    )

  def set_forcing(self, t_points, forcing_profiles):
    self.t_points = t_points
    self.forcing_profiles = forcing_profiles  # (batch, steps, n_input)

  def _interpolate_forcing(self, t):
    """Piecewise-constant (zero-order hold) forcing interpolation."""
    t_clamped = t.clamp(self.t_points[0], self.t_points[-1])
    idx = torch.searchsorted(self.t_points, t_clamped.unsqueeze(0)).squeeze() - 1
    idx = idx.clamp(0, len(self.t_points) - 2)

    return self.forcing_profiles[:, idx, :]   # (batch, n_input)

  def _build_matrix(self, forcing):
    """Build constrained depletion matrix from forcing input.

    Returns: (batch, n_target, n_target) matrix with negative diagonal
             and positive off-diagonal entries.  Zero where the sparsity
             mask forbids a connection.
    """
    raw = self.net(forcing)   # (batch, n_outputs)
    batch = raw.shape[0]
    n = self.n_target

    if self.sparsity == 'chain':
      # Network outputs n raw values -> n positive rates via softplus
      rates = F.softplus(raw)                               # (batch, n)
      A = raw.new_zeros(batch, n, n)
      # Diagonal: -rᵢ (loss from each isotope)
      idx = torch.arange(n, device=raw.device)
      A[:, idx, idx] = -rates
      # Sub-diagonal: +r_{i-1} (what leaves isotope i-1 enters isotope i)
      if n > 1:
        A[:, idx[1:], idx[:-1]] = rates[:, :-1]
    else:
      raw = raw.view(batch, n, n)
      # Off-diagonal: positive (production terms)
      A = F.softplus(raw)
      # Diagonal: negative (loss terms)
      diag_mask = torch.eye(n, device=raw.device, dtype=torch.bool)
      A = torch.where(diag_mask, -F.softplus(raw), A)

    return A

  def forward(self, t, y):
    self.nfe += 1
    forcing = self._interpolate_forcing(t)       # (batch, n_input)
    A = self._build_matrix(forcing)              # (batch, n_target, n_target)

    # dy/dt = A @ y
    dydt = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)   # (batch, n_target)
    return dydt