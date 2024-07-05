"""Base model for all PVNet submodels"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import yaml
import numpy as np
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)


class DictListAccumulator:
    """Abstract class for accumulating dictionaries of lists"""

    @staticmethod
    def _dict_list_append(d1, d2):
        for k, v in d2.items():
            d1[k].append(v)

    @staticmethod
    def _dict_init_list(d):
        return {k: [v] for k, v in d.items()}
    
    
class MetricAccumulator(DictListAccumulator):
    """Dictionary of metrics accumulator.

    A class for accumulating, and finding the mean of logging metrics when using grad
    accumulation and the batch size is small.

    Attributes:
        _metrics (Dict[str, list[float]]): Dictionary containing lists of metrics.
    """

    def __init__(self):
        """Dictionary of metrics accumulator."""
        self._metrics = {}

    def __bool__(self):
        return self._metrics != {}

    def append(self, loss_dict: dict[str, float]):
        """Append lictionary of metrics to self"""
        if not self:
            self._metrics = self._dict_init_list(loss_dict)
        else:
            self._dict_list_append(self._metrics, loss_dict)

    def flush(self) -> dict[str, float]:
        """Calculate mean of all accumulated metrics and clear"""
        mean_metrics = {k: np.mean(v) for k, v in self._metrics.items()}
        self._metrics = {}
        return mean_metrics
    

class AdamW:
    """AdamW optimizer"""

    def __init__(self, lr=0.0005, **kwargs):
        """AdamW optimizer"""
        self.lr = lr
        self.kwargs = kwargs

    def __call__(self, model):
        """Return optimizer"""
        return torch.optim.AdamW(model.parameters(), lr=self.lr, **self.kwargs)


def plot_sat_images(y, y_hat, channel_inds=[8, 1], n_frames=6):
    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()
        
    y[y<=0] = np.nan
    y_hat[y_hat<=0] = np.nan
    
    seq_len = y.shape[1]
    
    n_frames = min(n_frames, seq_len)
    plot_frames = np.linspace(0, seq_len-1, n_frames).astype(int)
    
    fig, axes = plt.subplots(len(channel_inds), 1, sharex=True, sharey=True)
    
    for p, channel_ind in enumerate(channel_inds):

        y_frames = [y[channel_ind, f_num][:, ::-1] for f_num in plot_frames]
        y_hat_frames = [y_hat[channel_ind, f_num][:, ::-1] for f_num in plot_frames]

        ims = np.concatenate([np.concatenate(y_frames, axis=1), np.concatenate(y_hat_frames, axis=1)], axis=0) 

        vmax = min(ims.max(), 1)
        axes[p].imshow(ims, cmap='gist_grey', interpolation='nearest', origin="lower", vmin=0, vmax=1)
        axes[p].set_title(f"Channel number : {channel_ind}")
    
    xlabels = []
    ylabels = ["y", "y_hat"]
    y_ticks = np.array([0.5, 1.5])*y.shape[2]
    
    x_ticks = (np.arange(n_frames)+0.5)*y.shape[3]
    
    plt.yticks(ticks=y_ticks, labels=ylabels)
    plt.xticks(ticks=x_ticks, labels=plot_frames)
    plt.xlabel("Frame number")
    plt.tight_layout()
    return fig
    
    
class TrainingModule(pl.LightningModule):
    """Abstract base class for PVNet submodels"""

    def __init__(
        self,
        model,
        num_channels: int,
        history_mins: int,
        forecast_mins: int, 
        sample_freq_mins: int,
        optimizer = AdamW(),
    ):
        """tbc

        """
        super().__init__()

        self._optimizer = optimizer

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None

        self.history_minutes = history_mins
        self.forecast_minutes = forecast_mins
        self.interval_minutes = sample_freq_mins

        self.history_len = history_mins // sample_freq_mins + 1
        self.forecast_len = (forecast_mins) // sample_freq_mins

        self._accumulated_metrics = MetricAccumulator()
        
        self.model = model(
            num_channels=num_channels,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
        )


    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, test, and val"""
        
        losses = {}

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        losses.update(
            {
                "MSE": mse_loss,
                "MAE": mae_loss,
            }
        )

        return losses

    def _step_mae_and_mse(self, y, y_hat, dict_key_root):
        """Calculate the MSE and MAE at each forecast step"""
        losses = {}

        mse_each_step = torch.mean((y_hat - y) ** 2, dim=0)
        mae_each_step = torch.mean(torch.abs(y_hat - y), dim=0)

        losses.update({f"MSE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mse_each_step)})
        losses.update({f"MAE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mae_each_step)})

        return losses

    def _calculate_val_losses(self, y, y_hat):
        """Calculate additional validation losses"""

        losses = {}

        # Log the loss at each time horizon
        #losses.update(self._step_mae_and_mse(y, y_hat, dict_key_root="horizon"))

        return losses

    def _calculate_test_losses(self, y, y_hat):
        """Calculate additional test losses"""
        # No additional test losses
        losses = {}
        return losses

    def _training_accumulate_log(self, batch, batch_idx, losses, y_hat):
        """Internal function to accumulate training batches and log results.

        This is used when accummulating grad batches. Should make the variability in logged training
        step metrics indpendent on whether we accumulate N batches of size B or just use a larger
        batch size of N*B with no accumulaion.
        """

        losses = {k: v.detach().cpu() for k, v in losses.items()}
        y_hat = y_hat.detach().cpu()

        self._accumulated_metrics.append(losses)

        if not self.trainer.fit_loop._should_accumulate():
            losses = self._accumulated_metrics.flush()

            self.log_dict(
                losses,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        """Run training step"""
        X, y = batch
        
        y_hat = self.model(X)

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self._training_accumulate_log(batch, batch_idx, losses, y_hat)

        return losses["MSE/train"]

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""
        X, y = batch
        
        y_hat = self.model(X)
    
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        logged_losses = {f"{k}/val": v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=False,
            on_epoch=True,
        )

        accum_batch_num = batch_idx // self.trainer.accumulate_grad_batches
        
        if batch_idx in [0, 1]:
            fig = plot_sat_images(y[0], y_hat[0], channel_inds=[8, 1, 2], n_frames=6)
            
            plot_name = f"val_samples/batch_idx_{batch_idx}"

            self.logger.experiment.log({plot_name: wandb.Image(fig)})

            plt.close(fig)

        return logged_losses

    def on_validation_epoch_end(self):
        """Run on epoch end"""

        return


    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self)