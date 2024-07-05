"""Training"""
import os
from typing import Optional

import hydra
import torch
from lightning.pytorch import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf



torch.set_default_dtype(torch.float32)


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.

    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning loggers
    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
    # - this makes it easy to push the model to huggingface
    use_wandb_logger = False
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            use_wandb_logger = True
            wandb_logger = logger
            break

    if use_wandb_logger:
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                # Need to call the .experiment property to initialise the logger
                wandb_logger.experiment
                callback.dirpath = "/".join(
                    callback.dirpath.split("/")[:-1] + [wandb_logger.version]
                )
                # Also save model config here - this makes for easy model push to huggingface
                os.makedirs(callback.dirpath, exist_ok=True)
                OmegaConf.save(config.model, f"{callback.dirpath}/model_config.yaml")

                # Similarly save the data config
                OmegaConf.save(config.datamodule, f"{callback.dirpath}/data_config.yaml")

                break


    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    # Train the model completely
    trainer.fit(model=model, datamodule=datamodule)
    
    
if __name__ == "__main__":
    train()