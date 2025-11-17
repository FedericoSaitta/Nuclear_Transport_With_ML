# sweep_train.py

from omegaconf import OmegaConf
import torch

# Personal imports (same as your main)
import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model
import ML.models.modes as modes

from sweeper import ConfigSweeper
from sweeper import ConfigPathForSweeper
from sweeper import generate_permutations

sweep_space = {

    ("dataset", "inputs", "U235"): ["MinMax", "robust", "standard", "quantile","log"],
    ("dataset", "targets", "U235"): ["MinMax", "robust", "standard", "quantile","log"],


    ("model", "layers"): [
        [64, 64],
        [128, 64],
        [64, 32, 64],
    ],
    ("model", "activation"): ["relu", "gelu", "tanh"],
    
    
    ("train", "loss"): ["mse", "mae"],
}
#model = ConfigSweeper("ML/base_simple_U235.yaml") 
#for combo in generate_permutations(sweep_space):
#    print(combo)

 

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # This object holds the base config and the helpers:
    model = ConfigSweeper("base_simple_U235.yaml")  

    run_id = 0

    for combo in generate_permutations(sweep_space):
        run_id += 1

        # start from clean base YAML each time
        model.reset()
        cfg: DictConfig = model.get_cfg()

        # apply all chosen values for this permutation
        for path, value in combo.items():
            set_cfg_value(cfg, path, value)

        # make sure we are in train mode
        cfg.runtime.mode = "train"

        # optional: encode hyperparams in model name for logging
        cfg.model.name = f"run_{run_id}"

        # build datamodule + train
        datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
        modes.train_and_test(datamodule, DNN_Model.DNN_Model, cfg)
        
