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
  """

  def __init__(self, cfg_object):
    super().__init__()

    self.path_to_data = cfg_object.dataset.path_to_data
    self.fraction_of_data = cfg_object.dataset.fraction_of_data

    self.train_batch_size = cfg_object.dataset.train.batch_size
    self.val_batch_size = cfg_object.dataset.val.batch_size

    self.num_workers = cfg_object.runtime.num_workers

    # Scaler configs from yaml
    self.inputs = data_scalers.create_scaler_dict(cfg_object.dataset['inputs'])

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

    all_columns = list(self.inputs.keys())
    data_df = data_help.filter_columns(data_df, all_columns)

    input_data_arr, self.col_index_map = data_help.split_df(data_df, all_columns)

    # ── Drop the last row of each run BEFORE scaling ──
    # Power at t predicts U238 at t+1, so final row has NaN power
    num_total = input_data_arr.shape[0]
    num_runs = num_total // self.steps_per_run
    assert num_runs * self.steps_per_run == num_total, \
        f"Data length {num_total} not divisible by steps_per_run {self.steps_per_run}"

    keep_mask = np.ones(num_total, dtype=bool)
    for r in range(num_runs):
        keep_mask[r * self.steps_per_run + (self.steps_per_run - 1)] = False
    input_data_arr = input_data_arr[keep_mask]

    self.actual_steps = self.steps_per_run - 1
    logger.info(f"Dropped {num_runs} NaN rows, {input_data_arr.shape[0]} samples remain")
    assert not np.isnan(input_data_arr).any(), "NaNs still present after dropping last rows!"

    # ── Reshape to trajectories BEFORE scaling ──
    num_features = input_data_arr.shape[1]
    all_trajectories = input_data_arr.reshape(num_runs, self.actual_steps, num_features)

    logger.info(f"Total trajectories: {num_runs}, each {self.actual_steps} steps, {num_features} features")

    # ── Split by run BEFORE scaling (prevents data leakage) ──
    perm = np.random.permutation(num_runs)
    all_trajectories = all_trajectories[perm]

    n_train = int(num_runs * 0.6)
    n_val = int(num_runs * 0.2)

    train_trajs_raw = all_trajectories[:n_train]
    val_trajs_raw = all_trajectories[n_train:n_train + n_val]
    test_trajs_raw = all_trajectories[n_train + n_val:] # train split is 1 - train - val

    logger.info(f"Train: {len(train_trajs_raw)}, Val: {len(val_trajs_raw)}, Test: {len(test_trajs_raw)} runs")

    # ── Plot raw (unscaled) distributions from training data ──
    train_flat_raw = train_trajs_raw.reshape(-1, num_features)
    plot.plot_data_distributions(train_flat_raw, self.col_index_map, save_dir=self.result_dir, name='Raw_Inputs')

    # ── Fit scaler on TRAINING data only (no leakage) ──
    self.input_scaler = data_scalers.create_column_transformer(
        self.inputs, self.col_index_map,
    )
    self.input_scaler.fit(train_flat_raw)

    # ── Transform all splits using the train-fitted scaler ──
    train_scaled = self.input_scaler.transform(train_flat_raw).reshape(
        len(train_trajs_raw), self.actual_steps, num_features)
    val_scaled = self.input_scaler.transform(val_trajs_raw.reshape(-1, num_features)).reshape(
        len(val_trajs_raw), self.actual_steps, num_features)
    test_scaled = self.input_scaler.transform(test_trajs_raw.reshape(-1, num_features)).reshape(
        len(test_trajs_raw), self.actual_steps, num_features)

    # ── Plot scaled distributions from training data ──
    plot.plot_data_distributions(train_scaled.reshape(-1, num_features), self.col_index_map, save_dir=self.result_dir, name='Scaled_Inputs')

    # ── Convert to tensors ──
    train_trajs = torch.tensor(train_scaled, dtype=torch.float32)
    val_trajs = torch.tensor(val_scaled, dtype=torch.float32)
    test_trajs = torch.tensor(test_scaled, dtype=torch.float32)

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