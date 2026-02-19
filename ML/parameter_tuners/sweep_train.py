# sweep_train.py

from omegaconf import OmegaConf
import torch

# Personal imports (same as your main)
import ML.datamodule.dnn_datamodule as dnn_datamodule
import ML.models.dnn_model as dnn_model
import ML.models.modes as modes

from ML.parameter_tuners.sweeper import ConfigSweeper
from ML.parameter_tuners.sweeper import ConfigPathForSweeper
from ML.parameter_tuners.sweeper import generate_permutations, set_cfg_value

sweep_space = {
    ("dataset", "inputs", "U235"): ["MinMax", "robust", "standard", "quantile","log"],
    ("dataset", "targets", "U235"): ["MinMax", "robust", "standard", "quantile","log"],

    ("model", "layers"): [[64, 64], [128, 64], [64, 32, 64]],
    ("model", "activation"): ["relu", "gelu", "tanh"],
    
    ("train", "loss"): ["mse", "mae", "huber", "smooth_l1"],
}


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # This object holds the base config and the helpers:
    model = ConfigSweeper("base_simple_U235.yaml")  

    run_id = 0

    for combo in generate_permutations(sweep_space):
        run_id += 1

        # start from clean base YAML each time
        model.reset()
        cfg = model.get_cfg()

        # apply all chosen values for this permutation
        for path, value in combo.items():
            set_cfg_value(cfg, path, value)

        # make sure we are in train mode
        cfg.runtime.mode = "train"

        # optional: encode hyperparams in model name for logging
        cfg.model.name = f"run_{run_id}"

        # build datamodule + train
        datamodule = dnn_datamodule.DNN_Datamodule(cfg)
        modes.train_and_test(datamodule, dnn_model.DNN_Model, cfg)
        
