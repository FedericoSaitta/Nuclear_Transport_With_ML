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
  
  if hasattr(cfg.train, 'early_stopping_patience'):
      callbacks.append(EarlyStopping(
          monitor="val_loss",
          patience=getattr(cfg.train, 'early_stopping_patience', 10),
          mode="min",
          verbose=False
      ))

  mp.set_sharing_strategy('file_system')
  
  sqlite_logger = SQLiteLogger(
      db_path=database_name,
      name=cfg.model.name,
      config=cfg
  )
  
  model = model(config_object=cfg)
  trainer = L.Trainer(
      max_epochs=cfg.train.num_epochs,
      accelerator=cfg.runtime.device,
      devices="auto",
      callbacks=callbacks,
      logger=sqlite_logger, 
      gradient_clip_val=1.0, 
      gradient_clip_algorithm="norm",
  )

  # Train the model
  trainer.fit(model=model, datamodule=datamodule)
  
  # ========== SAVE VALIDATION METRICS BEFORE TESTING ==========
  val_r2 = trainer.callback_metrics.get("val_r2", -float("inf"))
  val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
  val_mae = trainer.callback_metrics.get("val_mae", float("inf"))
  # ============================================================
  
  # Load best checkpoint for testing
  best_model_path = checkpoint_callback.best_model_path
  logger.info(f"Best model saved at: {best_model_path}")
  
  trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
  
  # Return the saved validation metrics (not overwritten by test)
  return {
      "val_r2": float(val_r2),
      "val_loss": float(val_loss),
      "val_mae": float(val_mae),
  }

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