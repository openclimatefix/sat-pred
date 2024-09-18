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
import wandb
from torch.utils.data import default_collate


logger = logging.getLogger(__name__)

activities = [torch.profiler.ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(torch.profiler.ProfilerActivity.CUDA)


def check_nan_and_finite(X, y, y_hat):
    if X is not None:
        assert not np.isnan(X.cpu().numpy()).any(), "NaNs in X"
        assert np.isfinite(X.cpu().numpy()).all(), "infs in X"
    
    if y is not None:
        assert not np.isnan(y.cpu().numpy()).any(), "NaNs in y"
        assert np.isfinite(y.cpu().numpy()).all(), "infs in y"

    if y_hat is not None:
        assert not np.isnan(y_hat.detach().cpu().numpy()).any(), "NaNs in y_hat"
        assert np.isfinite(y_hat.detach().cpu().numpy()).all(), "infs in y_hat"


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

    
class AdamWReduceLROnPlateau:
    """AdamW optimizer and reduce on plateau scheduler"""

    def __init__(
        self, lr=0.0005, patience=10, factor=0.2, threshold=2e-4, step_freq=None, **opt_kwargs
    ):
        """AdamW optimizer and reduce on plateau scheduler"""
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.step_freq = step_freq
        self.opt_kwargs = opt_kwargs

    def __call__(self, model):

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, **self.opt_kwargs
        )
        sch = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
            ),
            "monitor": "MAE/val",
        }

        return [opt], [sch]
    

def plot_sat_images(y, y_hat, channel_inds=[8, 1], n_frames=6):
    y = y.cpu().numpy().copy()
    y_hat = y_hat.cpu().numpy().copy()
        
    mask = y<0
    y[mask] = np.nan
    y_hat[mask] = np.nan
    
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
    
    ylabels = ["y", "y_hat"]
    y_ticks = np.array([0.5, 1.5])*y.shape[2]
    
    x_ticks = (np.arange(n_frames)+0.5)*y.shape[3]
    
    plt.yticks(ticks=y_ticks, labels=ylabels)
    plt.xticks(ticks=x_ticks, labels=plot_frames)
    plt.xlabel("Frame number")
    plt.tight_layout()
    return fig


def upload_video(y, y_hat, video_name, channel_nums=[8, 1], fps=1):
    
    check_nan_and_finite(None, y, y_hat)

    y = y.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    channel_frames = []
    
    for channel_num in channel_nums:
        y_frames = y.transpose(1,0,2,3)[:, channel_num:channel_num+1, ::-1, ::-1]
        y_hat_frames = y_hat.transpose(1,0,2,3)[:, channel_num:channel_num+1, ::-1, ::-1]
        channel_frames.append(np.concatenate([y_hat_frames, y_frames], axis=3))
        
    channel_frames = np.concatenate(channel_frames, axis=2)
    channel_frames = channel_frames.clip(0, 1)
    channel_frames = np.repeat(channel_frames, 3, axis=1)*255
    channel_frames = channel_frames.astype(np.uint8)
    wandb.log({video_name: wandb.Video(channel_frames, fps=fps)})
    
    
class TrainingModule(pl.LightningModule):

    def __init__(
        self,
        model,
        num_channels: int,
        history_mins: int,
        forecast_mins: int, 
        sample_freq_mins: int,
        optimizer = AdamWReduceLROnPlateau(),
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

    @staticmethod
    def _filter_missing_targets(y, y_hat):
        mask = y==-1
        return y[~mask], y_hat[~mask]
        
    @staticmethod
    def _calculate_common_losses(y, y_hat):
        """Calculate losses common to train, test, and val"""
        
        y, y_hat = TrainingModule._filter_missing_targets(y, y_hat)
        losses = {}

        # calculate mse, mae
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)

        assert not np.isnan(float(mae_loss)), "MAE loss is NaN"
        assert not np.isnan(float(mse_loss)), "MSE loss is NaN"

        losses.update(
            {
                "MSE": mse_loss,
                "MAE": mae_loss,
            }
        )

        return losses

    def _calculate_val_losses(self, y, y_hat):
        """Calculate additional validation losses"""

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

        check_nan_and_finite(X, y, y_hat)

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self._training_accumulate_log(batch, batch_idx, losses, y_hat)

        return losses["MAE/train"]

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""
        X, y = batch
        
        y_hat = self.model(X)

        check_nan_and_finite(X, y, y_hat)
    
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        logged_losses = {f"{k}/val": v for k, v in losses.items()}

        self.log_dict(
            logged_losses,
            on_step=False,
            on_epoch=True,
        )

        return logged_losses
        
    def on_validation_epoch_start(self):
        
        val_dataset = self.trainer.val_dataloaders.dataset
        
        dates = [val_dataset.t0_times[i] for i in [0,1,2]]
        
        X, y = default_collate([val_dataset[date]for date in dates])
        X = X.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(X)

        assert val_dataset.nan_to_num, val_dataset.nan_to_num
        check_nan_and_finite(X, y, y_hat)
                               
        for i in range(len(dates)):
                               
            plot_name = f"val_sample_plots/{dates[i]}"
            fig = plot_sat_images(y[i], y_hat[i], channel_inds=[8, 1, 2], n_frames=6)
            self.logger.experiment.log({plot_name: wandb.Image(fig)})
            plt.close(fig)

            for channel_num in [1, 8]:
                channel_name = val_dataset.ds.variable.values[channel_num]
                video_name = f"val_sample_videos/{dates[i]}_{channel_name}"

                upload_video(y[i], y_hat[i], video_name, channel_nums=[channel_num])
                
        
    def on_validation_epoch_end(self):
        """Run on epoch end"""
        return

    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self)