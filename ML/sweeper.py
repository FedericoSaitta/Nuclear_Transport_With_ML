# sweeper.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf, DictConfig


class ConfigSweeper:
    """
    Helper for modifying a YAML/OmegaConf config to sweep over:
      - model layers
      - input/target features and their scalings
      - train/val batch sizes

    It keeps:
      - base_cfg: original config loaded from disk
      - cfg:      mutable working copy

    You can treat an instance like your "model config object" and call:
      model.add_input(...)
      model.set_scaling(...)
      model.set_layers(...)
    """

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.base_cfg: DictConfig = OmegaConf.load(self.config_path)
        self.cfg: DictConfig = self._clone_cfg(self.base_cfg)

    # ---------- internals ----------

    def _clone_cfg(self, cfg: DictConfig) -> DictConfig:
        """
        Deep copy an OmegaConf DictConfig.
        """
        # Convert to a plain container and recreate
        as_container: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=False)  # type: ignore[assignment]
        return OmegaConf.create(as_container)

    def reset(self) -> None:
        """
        Reset cfg to the original loaded config.
        """
        self.cfg = self._clone_cfg(self.base_cfg)

    def get_cfg(self) -> DictConfig:
        """
        Return the current working config. This is what you pass to DNN_Datamodule/DNN_Model.
        """
        return self.cfg

    # ---------- model helpers ----------

    def set_layers(self, layers: List[int]) -> None:
        """
        Set the list of hidden layer sizes under model.layers.
        Example: [64, 64] or [128, 64, 32]
        """
        if "model" not in self.cfg:
            self.cfg.model = {}
        self.cfg.model.layers = list(layers)

    # ---------- dataset feature / scaling helpers ----------

    def _ensure_dataset_section(self) -> None:
        if "dataset" not in self.cfg or self.cfg.dataset is None:
            self.cfg.dataset = {}

    def _get_feature_dict(self, section: str) -> Dict[str, str]:
        """
        Get cfg.dataset.inputs or cfg.dataset.targets as a plain dict-like.

        section: "inputs" or "targets"
        """
        if section not in ("inputs", "targets"):
            raise ValueError("section must be 'inputs' or 'targets'.")

        self._ensure_dataset_section()

        if section not in self.cfg.dataset or self.cfg.dataset[section] is None:
            self.cfg.dataset[section] = {}

        feature_dict = self.cfg.dataset[section]

        if not isinstance(feature_dict, dict):
            # OmegaConf sometimes uses DictConfig, but dict-like is fine
            feature_dict = dict(feature_dict)
            self.cfg.dataset[section] = feature_dict

        return feature_dict

    def set_scaling(self, section: str, scaling: str) -> None:
        """
        Set scaling for ALL features in dataset.inputs or dataset.targets.

        Example:
            model.set_scaling("inputs", "MinMax")
            model.set_scaling("targets", "robust")
        """
        feature_dict = self._get_feature_dict(section)
        for key in list(feature_dict.keys()):
            feature_dict[key] = scaling

    def add_input(self, input_name: str, scaling: str) -> None:
        """
        Add or update a single input feature in dataset.inputs.

        If it exists, scaling is UPDATED.
        Example:
            model.add_input("time_days", "MinMax")
        """
        inputs = self._get_feature_dict("inputs")
        inputs[input_name] = scaling

    def add_target(self, target_name: str, scaling: str) -> None:
        """
        Add or update a single target feature in dataset.targets.
        """
        targets = self._get_feature_dict("targets")
        targets[target_name] = scaling

    def set_feature_scaling(
        self,
        section: str,
        feature_name: str,
        scaling: str,
        create_if_missing: bool = True,
    ) -> None:
        """
        Explicitly set scaling for a single feature in inputs/targets.

        section: 'inputs' or 'targets'
        feature_name: name of the feature
        scaling: e.g. 'MinMax', 'robust'
        """
        feature_dict = self._get_feature_dict(section)
        if feature_name not in feature_dict and not create_if_missing:
            raise KeyError(
                f"{feature_name!r} not found in dataset.{section} "
                f"and create_if_missing=False."
            )
        feature_dict[feature_name] = scaling

    # ---------- batch size helpers ----------

    def set_batch_size(self, split: str, batch_size: int) -> None:
        """
        Set batch_size under dataset.train or dataset.val.

        split: 'train' or 'val'
        """
        self._ensure_dataset_section()

        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'.")

        if split not in self.cfg.dataset or self.cfg.dataset[split] is None:
            self.cfg.dataset[split] = {}

        if not isinstance(self.cfg.dataset[split], dict):
            self.cfg.dataset[split] = dict(self.cfg.dataset[split])

        self.cfg.dataset[split]["batch_size"] = int(batch_size)

    def set_batch_sizes(self, train_batch: int, val_batch: int) -> None:
        """
        Convenience: set both train and val batch sizes.
        """
        self.set_batch_size("train", train_batch)
        self.set_batch_size("val", val_batch)
