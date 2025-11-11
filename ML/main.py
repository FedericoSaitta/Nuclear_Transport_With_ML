from omegaconf import OmegaConf
import os
import lightning as L

# Personal Imports
import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model
import ML.models.modes as modes

if __name__ == "__main__":

  cfg = OmegaConf.load("test_config.yaml")
  datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
  lightning_mode = cfg.runtime.mode

  if (lightning_mode == 'train'):
    modes.train_and_test(datamodule, DNN_Model.DNN_Model, cfg)
  elif (lightning_mode == 'train_from_ckp'):
    modes.train_from_checkpoint_and_test(datamodule, DNN_Model.DNN_Model, cfg)
  else: 
    raise ValueError(f"The mode {lightning_mode} is not one of the available ones")
