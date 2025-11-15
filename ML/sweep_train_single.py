# sweep_train_single.py

#read this to understand the sweeper and also read the sweeper.py

from omegaconf import OmegaConf
import torch

import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model
import ML.models.modes as modes

from sweeper import ConfigSweeper


if __name__ == "__main__":
    # allow tensor core acceleration
    torch.set_float32_matmul_precision("high")

    # Load base config through sweeper
    model = ConfigSweeper("test_config.yaml")

    # ---------------------------------------------------------
    # MODIFY THE CONFIG ONE TIME (example)
    # ---------------------------------------------------------

    # Change model architecture
    model.set_layers([256, 128, 64])

    # Change the scaling of ALL inputs
    model.set_scaling("inputs", "MinMax")

    # Change the scaling of ALL targets
    model.set_scaling("targets", "robust")

    # Add or override an input
    model.add_input("new_feature_x", "robust")

    # Change train and val batch sizes
    model.set_batch_sizes(train_batch=2048, val_batch=8192)

    # ---------------------------------------------------------
    # GET THE FINAL CONFIG AND TRAIN ONE MODEL
    # ---------------------------------------------------------
    cfg = model.get_cfg()
    cfg.runtime.mode = "train"        # force train mode
    cfg.model.name = "single_run_test"  # name the run

    # Build datamodule and train
    datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
    modes.train_and_test(datamodule, DNN_Model.DNN_Model, cfg)
