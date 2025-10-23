from omegaconf import OmegaConf
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig
from loguru import logger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "datamodule")))
import DNN_Datamodule
import DNN_Model

import torch.multiprocessing as mp

def train(datamodule, model, cfg: DictConfig):

    callbacks = []
    device = "cuda" if (cfg.runtime.device == "cuda" and torch.cuda.is_available()) else "cpu"

    
    # Create early stopping callback with patience from config
    if hasattr(cfg.train, 'early_stopping_patience'):
            callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=getattr(cfg.train, 'early_stopping_patience', 10),
            mode="min",
            verbose=False
        ))

    # Reduced the number of open files that each worker creates, needed to avoid exceeding file open limit when using
    # many workers or opening very large files (about 65k files for Linux systems by default)
    mp.set_sharing_strategy('file_system')
    
    model = model(config_object=cfg)
    trainer = L.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator=device,            # Use GPU
        devices="auto",               # Automatically use all available GPUs
        callbacks=callbacks,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
  cfg = OmegaConf.load("test_config.yaml")
  datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
  train(datamodule, DNN_Model.DNN_Model, cfg)
