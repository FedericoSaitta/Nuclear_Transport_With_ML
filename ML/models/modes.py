import os
import lightning as L
import torch
from loguru import logger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch.multiprocessing as mp

# Personal imports
from ML.utils.sql_lite_logger import SQLiteLogger

def train_and_test(datamodule, model, cfg):
  model_name = cfg.model.name
  database_name = cfg.runtime.model_database
  result_dir = f'results/{model_name}/'
  os.makedirs(result_dir, exist_ok=True)

  callbacks = []

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
  sqlite_logger = SQLiteLogger(
    db_path=database_name,
    name=cfg.model.name,  # or 'my_experiment'
    config=cfg  # Make sure you're passing the config!
  )
  
  model = model(config_object=cfg)
  trainer = L.Trainer(
    max_epochs=cfg.train.num_epochs,
    accelerator=cfg.runtime.device,
    devices="auto",
    callbacks=callbacks,
    logger=sqlite_logger, 
  )

  # Train the model
  trainer.fit(model=model, datamodule=datamodule)
  
  # Load best checkpoint for testing
  best_model_path = checkpoint_callback.best_model_path
  logger.info(f"Best model saved at: {best_model_path}")
  
  trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)


def load_checkpoint_into_model(model, ckpt_path, save_fixed=True):
  """
  Load checkpoint into model while fixing a leading 'model.' prefix.
  """
  checkpoint = torch.load(ckpt_path, map_location="cpu")
  state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
  model_keys = set(model.state_dict().keys())
  ckpt_keys = set(state_dict.keys())
  
  # Fix 'model.' prefix if needed
  if model_keys == ckpt_keys:
      new_state_dict = state_dict
  elif any(k.startswith("model.") for k in model_keys) and not any(k.startswith("model.") for k in ckpt_keys):
      new_state_dict = {"model." + k: v for k, v in state_dict.items()}
  elif any(k.startswith("model.") for k in ckpt_keys) and not any(k.startswith("model.") for k in model_keys):
      new_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
  else:
      new_state_dict = state_dict
      # We let the code run which will likely crash as lightning throws an error explaining which keys don't match
  
  # Optionally save fixed checkpoint
  if save_fixed:
      ckpt_copy = checkpoint.copy() if isinstance(checkpoint, dict) else {"state_dict": new_state_dict}
      ckpt_copy["state_dict"] = new_state_dict
      torch.save(ckpt_copy, ckpt_path + ".fixed")
  
  # Load strictly
  model.load_state_dict(new_state_dict, strict=True)
  return model

def train_from_checkpoint_and_test(datamodule, model_class, cfg):
  model_name = cfg.model.name
  result_dir = f'results/{model_name}/'
  os.makedirs(result_dir, exist_ok=True)

  callbacks = []

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
  
  if hasattr(cfg.train, 'early_stopping_patience'):
    callbacks.append(EarlyStopping(
      monitor="val_loss",
      patience=getattr(cfg.train, 'early_stopping_patience', 10),
      mode="min",
      verbose=False
    ))

  mp.set_sharing_strategy('file_system')
  
  csv_logger = CSVLogger(save_dir=result_dir, name='', version='')
  
  logger.info(f"Loading model from checkpoint: {cfg.runtime.ckp_path}")
  
  # 1. Create model instance
  model = model_class(cfg)
  
  # 2. Setup datamodule and model (IMPORTANT: do this before loading weights)
  datamodule.setup(stage="fit")  # Use "fit" for training
  model.setup(stage="fit", datamodule=datamodule)
  
  # 3. NOW load the checkpoint weights
  model = load_checkpoint_into_model(model, cfg.runtime.ckp_path, save_fixed=True)
  logger.info(f"Successfully loaded checkpoint into model")
  
  trainer = L.Trainer(
    max_epochs=cfg.train.num_epochs,
    accelerator=cfg.runtime.device,
    devices="auto",
    callbacks=callbacks,
    logger=csv_logger,
    enable_progress_bar=True,
  )

  logger.info("Starting training...")
  trainer.fit(model=model, datamodule=datamodule)
  
  best_model_path = checkpoint_callback.best_model_path
  logger.info(f"Best model saved at: {best_model_path}")
  trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
