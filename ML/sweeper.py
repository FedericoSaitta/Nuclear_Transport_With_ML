# sweeper.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union, Iterable, Tuple

from omegaconf import OmegaConf, DictConfig

from itertools import product

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

# sweeper helpers for the dict generation

ConfigPathForSweeper = Tuple[str, ...]  # e.g. ("dataset", "inputs", "U235")

def generate_permutations(
    sweep_space: Dict[Path, List[Any]]
) -> Iterable[Dict[Path, Any]]:
    """
    Given a mapping:
        (path_tuple) -> [list, of, options]
    yield dictionaries mapping each path_tuple to one chosen value.

    Example:
      {
        ("model", "activation"): ["relu", "gelu"],
        ("train", "loss"): ["mse", "mae"],
      }

      => 4 combinations total.
    """
    paths = list(sweep_space.keys())
    value_lists = [sweep_space[p] for p in paths]

    for combo in product(*value_lists):
        yield {path: value for path, value in zip(paths, combo)}


def set_cfg_value(cfg: DictConfig, path: Path, value: Any) -> None:
    """
    Set cfg[path[0]][path[1]]...[path[-1]] = value,
    creating intermediate dicts if needed.
    """
    node = cfg
    *parents, last = path
    for key in parents:
        if key not in node or node[key] is None:
            node[key] = {}
        node = node[key]
    node[last] = value
