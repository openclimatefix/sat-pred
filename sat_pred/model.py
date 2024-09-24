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
from sat_pred.ssim import SSIM3D


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
            "monitor": f"{model.target_loss}/val",
        }

        return [opt], [sch]


def upload_video(y, y_hat, video_name, channel_nums=[8, 1], fps=1):
    
    #check_nan_and_finite(None, y, y_hat)

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
        target_loss: float = "MAE",
        optimizer = AdamWReduceLROnPlateau(),
    ):
        """tbc

        """
        super().__init__()
        
        assert target_loss in ["MAE", "MSE", "SSIM"]

        self._optimizer = optimizer

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None

        self.history_minutes = history_mins
        self.forecast_minutes = forecast_mins
        self.interval_minutes = sample_freq_mins

        self.history_len = history_mins // sample_freq_mins + 1
        self.forecast_len = (forecast_mins) // sample_freq_mins
        self.target_loss = target_loss
        
        self.model = model(
            num_channels=num_channels,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
        )
        
        self.ssim_func = SSIM3D()

    @staticmethod
    def _minus_one_to_nan(y):
        y[y==-1] = torch.nan
        
    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, test, and val"""
        
        losses = {}
        
        # calculate mse, mae
        
        mse_loss = torch.nanmean(F.mse_loss(y_hat, y, reduction="none"))
        mae_loss = torch.nanmean(F.l1_loss(y_hat, y, reduction="none"))
        ssim_loss = torch.nanmean(1-self.ssim_func(y_hat, y)) # need to maximise SSIM

        losses = {
                "MSE": mse_loss,
                "MAE": mae_loss,
                "SSIM": ssim_loss,
        }

        return losses

    def _calculate_val_losses(self, y, y_hat):
        """Calculate additional validation losses"""

        losses = {}

        return losses

    def training_step(self, batch, batch_idx):
        """Run training step"""
        
        X, y = batch
        
        y_hat = self.model(X)
        del X

        #check_nan_and_finite(X, y, y_hat)
        
        # Replace the -1 (filled) values in y with NaNs 
        # This operation is in-place
        self._minus_one_to_nan(y)

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self.log_dict(
            {k: v.detach().cpu().item() for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
        )
        
        train_loss = losses[f"{self.target_loss}/train"]
        
        # Occasionally y will be entirely NaN and we have no training targets. So the train loss
        # will also be NaN. In this case we return None so lightning skips this train step
        if torch.isnan(train_loss).item():
            return None
        else:
            return train_loss

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""
        X, y = batch
        y_hat = self.model(X)
        del X

        #check_nan_and_finite(X, y, y_hat)
        
        # Replace the -1 (filled) values in y with NaNs 
        # This operation is in-place        
        self._minus_one_to_nan(y)
    
        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        # Rename and convert metrics to float
        losses = {f"{k}/val": v.item() for k, v in losses.items()}
        
        # Occasionally y will be entirely NaN and we have no training targets. So the val loss
        # will also be NaN. We filter these out
        non_nan_losses = {k: v for k, v in losses.items() if not np.isnan(v)}

        self.log_dict(
            non_nan_losses,
            on_step=False,
            on_epoch=True,
        )

        return
        
    def on_validation_epoch_start(self):
        
        val_dataset = self.trainer.val_dataloaders.dataset
        
        dates = [val_dataset.t0_times[i] for i in [0,1,2]]
        
        X, y = default_collate([val_dataset[date]for date in dates])
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            y_hat = self.model(X)

        assert val_dataset.nan_to_num, val_dataset.nan_to_num
        #check_nan_and_finite(X, y, y_hat)
                               
        for i in range(len(dates)):

            for channel_num in [1, 8]:
                channel_name = val_dataset.ds.variable.values[channel_num]
                video_name = f"val_sample_videos/{dates[i]}_{channel_name}"

                upload_video(y[i], y_hat[i], video_name, channel_nums=[channel_num])
                
    def on_validation_epoch_end(self):
        # Clear cache at the end of validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self)