"""Train the model using parameters in the supplied config files."""

if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

import os

import hydra
import rich.syntax
import rich.tree
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
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from sat_pred.load_model_from_checkpoint import get_model_from_checkpoints
from sat_pred.loss import LossFunction

# TODO: is this line needed?
torch.set_default_dtype(torch.float32)


def resolve_loss_name(loss):
    """Return the desired metric to monitor based on the loss being used.

    The adds the option to use something like:
        monitor: "${resolve_loss_name:${model.output_quantiles}}"
    """

    if isinstance(loss, str):
        return loss
    loss = hydra.utils.instantiate(loss, _convert_="all")
    if isinstance(loss, LossFunction):
        return loss.name
    msg = f"Unknown loss type: {type(loss)}"
    raise ValueError(msg)


OmegaConf.register_new_resolver("resolve_loss_name", resolve_loss_name)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: list[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


@rank_zero_only
@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def train(config: DictConfig):
    """Train the model using parameters in the supplied config files.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    print_config(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if config.model.model.get("from_pretrained", False):
        # Load the model from the checkpoint
        torch_model, model_config, data_config = get_model_from_checkpoints(
            config.model.model.checkpoint_dir, val_best=config.model.model.val_best
        )

        # Overwtie the model config with the loaded model config
        config.model.model = OmegaConf.create(model_config).model

        # Create a new lightning wrapped model
        model: LightningModule = hydra.utils.instantiate(config.model)

        # Replace the untrained model with the loaded model
        model.model = torch_model

    else:
        # Instantiate the model
        model: LightningModule = hydra.utils.instantiate(config.model)

    model.multi_gpu = len(config.trainer.devices) > 1

    # Instantiate the loggers
    loggers: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))

    # Instantiate callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Align the wandb id with the checkpoint path
    # - only works if wandb logger and model checkpoint used
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

                #  skip for non-rank-0 processes:
                # see https://github.com/Lightning-AI/pytorch-lightning/issues/13166#issuecomment-1139765549
                if wandb_logger.version is None:
                    break

                callback.dirpath = "/".join(
                    callback.dirpath.split("/")[:-1] + [wandb_logger.version]
                )
                # Also save model config to this path
                os.makedirs(callback.dirpath, exist_ok=True)
                OmegaConf.save(config.model, f"{callback.dirpath}/model_config.yaml")

                # Similarly save the data config
                OmegaConf.save(config.datamodule, f"{callback.dirpath}/data_config.yaml")

                break

    # Instantiate the datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _convert_="all")

    datamodule.zarr_path = list(datamodule.zarr_path)

    # Instantiate the trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        _convert_="partial",
        callbacks=callbacks,
    )

    # Train the model
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
