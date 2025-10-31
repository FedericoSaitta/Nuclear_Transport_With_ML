from omegaconf import OmegaConf
import os
import lightning as L
import torch
from loguru import logger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch.multiprocessing as mp

# Personal Imports
import ML.datamodule.DNN_Datamodule as DNN_Datamodule
import ML.models.DNN_Model as DNN_Model


def train_and_test(datamodule, model, cfg):
  model_name = cfg.model.name
  result_dir = f'results/{model_name}/'
  os.makedirs(result_dir, exist_ok=True)

  callbacks = []
  device = "cuda" if (cfg.runtime.device == "cuda" and torch.cuda.is_available()) else "cpu"

  checkpoint_callback = ModelCheckpoint(
    dirpath=result_dir,
    filename=f'best-{model_name}-'+'{epoch:02d}',
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_last=False,
    verbose=False
  )
  callbacks.append(checkpoint_callback)
  
  # Create early stopping callback with patience from config
  if hasattr(cfg.train, 'early_stopping_patience'):
    callbacks.append(EarlyStopping(
      monitor="val_loss",
      patience=getattr(cfg.train, 'early_stopping_patience', 10),
      mode="min",
      verbose=False
    ))

  mp.set_sharing_strategy('file_system') # Reduced the number of open files that each worker creates
  
  # Create CSV logger that saves to results folder instead of lightnig logs
  csv_logger = CSVLogger(save_dir=result_dir, name='', version='')
  
  model = model(config_object=cfg)
  trainer = L.Trainer(
    max_epochs=cfg.train.num_epochs,
    accelerator=device,
    devices="auto",
    callbacks=callbacks,
    logger=csv_logger, 
  )

  # Train the model
  trainer.fit(model=model, datamodule=datamodule)
  
  # Load best checkpoint for testing
  best_model_path = checkpoint_callback.best_model_path
  logger.info(f"Best model saved at: {best_model_path}")
  
  # Test with best model
  trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)

if __name__ == "__main__":

  cfg = OmegaConf.load("test_config.yaml")
  datamodule = DNN_Datamodule.DNN_Datamodule(cfg)
  train_and_test(datamodule, DNN_Model.DNN_Model, cfg)