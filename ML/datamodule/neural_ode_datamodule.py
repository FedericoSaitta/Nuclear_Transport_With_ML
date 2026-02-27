# ML/datamodule/node_datamodule.py
from loguru import logger
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os

import ML.datamodule.dataset_helper as data_help
import ML.datamodule.data_scalers as data_scalers
import ML.utils.plot as plot


class NODE_Datamodule(L.LightningDataModule):
  """
  DataModule for Neural ODE trajectory training.

  Key difference from the standard datamodule: instead of serving
  (input, target) pairs, this serves FULL TRAJECTORIES.
  Each item in the dataset is one complete run of shape (steps, features).
  A batch is (batch_size, steps, features).

  Uses separate input_scaler and target_scaler, matching the DNN datamodule.
  """

  def __init__(self, cfg_object):
    super().__init__()

    self.path_to_data = cfg_object.dataset.path_to_data
    self.fraction_of_data = cfg_object.dataset.fraction_of_data

    self.train_batch_size = cfg_object.dataset.train.batch_size
    self.val_batch_size = cfg_object.dataset.val.batch_size

    self.num_workers = cfg_object.runtime.num_workers

    # Separate scaler configs for inputs and targets (same as DNN)
    self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset['inputs'])
    self.target = data_scalers.create_scaler_dict(cfg_object.dataset['targets'])

    self.result_dir = f'results/{cfg_object.model.name}/'
    os.makedirs(self.result_dir, exist_ok=True)

    self._has_setup = False

  def setup(self, stage=None):
    if self._has_setup:
        return
    self._has_setup = True

    logger.info("Setting up NODE trajectory data module...")

    data_df, self.steps_per_run, self.time_array = data_help.read_data(
        self.path_to_data, self.fraction_of_data, drop_run_label=True,
    )
    data_help.print_dataset_stats(data_df)

    # Preserve order: inputs first, then targets (no duplicates) — same as DNN
    all_columns = list(self.inputs.keys()) + [k for k in self.target.keys() if k not in self.inputs.keys()]
    data_df = data_help.filter_columns(data_df, all_columns)

    # Get separate arrays and index maps for inputs and targets
    input_data_arr, self.col_index_map = data_help.split_df(data_df, self.inputs.keys())
    target_data_arr, self.target_index_map = data_help.split_df(data_df, self.target.keys())

    # ── Drop the last row of each run BEFORE scaling ──
    num_total = input_data_arr.shape[0]
    num_runs = num_total // self.steps_per_run
    assert num_runs * self.steps_per_run == num_total, \
        f"Data length {num_total} not divisible by steps_per_run {self.steps_per_run}"

    keep_mask = np.ones(num_total, dtype=bool)
    for r in range(num_runs):
        keep_mask[r * self.steps_per_run + (self.steps_per_run - 1)] = False
    input_data_arr = input_data_arr[keep_mask]
    target_data_arr = target_data_arr[keep_mask]

    self.actual_steps = self.steps_per_run - 1
    logger.info(f"Dropped {num_runs} NaN rows, {input_data_arr.shape[0]} samples remain")
    assert not np.isnan(input_data_arr).any(), "NaNs in input data after dropping last rows!"
    assert not np.isnan(target_data_arr).any(), "NaNs in target data after dropping last rows!"

    # ── Reshape to trajectories BEFORE scaling ──
    n_input_features = input_data_arr.shape[1]
    n_target_features = target_data_arr.shape[1]
    
    input_trajs = input_data_arr.reshape(num_runs, self.actual_steps, n_input_features)
    target_trajs = target_data_arr.reshape(num_runs, self.actual_steps, n_target_features)

    logger.info(f"Total trajectories: {num_runs}, each {self.actual_steps} steps")
    logger.info(f"Input features: {n_input_features}, Target features: {n_target_features}")

    # ── Split by run BEFORE scaling (prevents data leakage) ──
    perm = np.random.permutation(num_runs)
    input_trajs = input_trajs[perm]
    target_trajs = target_trajs[perm]

    n_train = int(num_runs * 0.6)
    n_val = int(num_runs * 0.2)

    train_input_raw = input_trajs[:n_train]
    val_input_raw = input_trajs[n_train:n_train + n_val]
    test_input_raw = input_trajs[n_train + n_val:]

    train_target_raw = target_trajs[:n_train]
    val_target_raw = target_trajs[n_train:n_train + n_val]
    test_target_raw = target_trajs[n_train + n_val:]

    logger.info(f"Train: {len(train_input_raw)}, Val: {len(val_input_raw)}, Test: {len(test_input_raw)} runs")

    # ── Plot raw (unscaled) distributions from training data ──
    train_input_flat = train_input_raw.reshape(-1, n_input_features)
    train_target_flat = train_target_raw.reshape(-1, n_target_features)
    plot.plot_data_distributions(train_input_flat, self.col_index_map, save_dir=self.result_dir, name='Raw_Inputs')
    plot.plot_data_distributions(train_target_flat, self.target_index_map, save_dir=self.result_dir, name='Raw_Targets')

    # ── Create and fit scalers on TRAINING data only (no leakage) ──
    self.input_scaler = data_scalers.create_column_transformer(self.inputs, self.col_index_map)
    self.target_scaler = data_scalers.create_column_transformer(self.target, self.target_index_map)

    self.input_scaler.fit(train_input_flat)
    self.target_scaler.fit(train_target_flat)

    # ── Scale all splits ──
    def scale_split(input_raw, target_raw):
      n_runs = len(input_raw)
      input_scaled = self.input_scaler.transform(input_raw.reshape(-1, n_input_features)).reshape(n_runs, self.actual_steps, n_input_features)
      target_scaled = self.target_scaler.transform(target_raw.reshape(-1, n_target_features)).reshape(n_runs, self.actual_steps, n_target_features)
      return input_scaled, target_scaled

    train_input_scaled, train_target_scaled = scale_split(train_input_raw, train_target_raw)
    val_input_scaled, val_target_scaled = scale_split(val_input_raw, val_target_raw)
    test_input_scaled, test_target_scaled = scale_split(test_input_raw, test_target_raw)

    # ── Plot scaled distributions from training data ──
    plot.plot_data_distributions(train_input_scaled.reshape(-1, n_input_features), self.col_index_map, save_dir=self.result_dir, name='Scaled_Inputs')
    plot.plot_data_distributions(train_target_scaled.reshape(-1, n_target_features), self.target_index_map, save_dir=self.result_dir, name='Scaled_Targets')

    # ── Combine input and target into trajectory tensors: (batch, steps, input_features + target_features) ──
    def combine_to_tensor(input_scaled, target_scaled):
      combined = np.concatenate([input_scaled, target_scaled], axis=-1)
      return torch.tensor(combined, dtype=torch.float32)

    train_trajs = combine_to_tensor(train_input_scaled, train_target_scaled)
    val_trajs = combine_to_tensor(val_input_scaled, val_target_scaled)
    test_trajs = combine_to_tensor(test_input_scaled, test_target_scaled)

    # ── Store feature layout info ──
    self.n_input_features = n_input_features
    self.n_target_features = n_target_features
    # In combined trajectory: columns [0:n_input_features] are inputs, [n_input_features:] are targets

    # ── Verify no NaNs ──
    for split_name, trajs in [('train', train_trajs), ('val', val_trajs), ('test', test_trajs)]:
        assert not torch.isnan(trajs).any(), \
            f"NaNs in {split_name} trajectories: {torch.isnan(trajs).sum()}"

    # ── Store time span (normalized to [0, 1]) ──
    raw_t = self.time_array[:self.actual_steps]
    self.t_span = torch.tensor(
        (raw_t - raw_t[0]) / (raw_t[-1] - raw_t[0]),
        dtype=torch.float32,
    )

    # ── Wrap in TensorDatasets ──
    self.train_dataset = TensorDataset(train_trajs)
    self.val_dataset = TensorDataset(val_trajs)
    self.test_dataset = TensorDataset(test_trajs)

    self.test_trajs = test_trajs

    logger.info(f"Training dataset size: {len(train_trajs)} trajectories")
    logger.info(f"Validation dataset size: {len(val_trajs)} trajectories")
    logger.info(f"Test dataset size: {len(test_trajs)} trajectories")

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset, batch_size=self.train_batch_size, shuffle=True,
        num_workers=self.num_workers, persistent_workers=self.num_workers > 0, pin_memory=True,
    )

  def val_dataloader(self):
    return DataLoader(
        self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
        num_workers=self.num_workers, persistent_workers=self.num_workers > 0, pin_memory=True,
    )

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset, batch_size=self.val_batch_size, shuffle=False,
        num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
    )

  def predict_dataloader(self):
    return self.test_dataloader()