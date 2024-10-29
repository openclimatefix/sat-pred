"""Training class to wrap model and optimizer"""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import wandb
from torch.utils.data import default_collate
from sat_pred.ssim import SSIM3D
from sat_pred.optimizers import AdamWReduceLROnPlateau

    
class MetricAccumulator:
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
        """Append dictionary of metrics to self"""
        if not self:
            self._metrics = {k: [v] for k, v in loss_dict.items()}
        else:
            for k, v in loss_dict.items():
                self._metrics[k].append(v)

    def flush(self) -> dict[str, float]:
        """Calculate mean of all accumulated metrics and clear"""
        mean_metrics = {k: np.nanmean(v) for k, v in self._metrics.items()}
        self._metrics = {}
        return mean_metrics


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


def upload_video(y, y_hat, video_name, channel_nums=[8, 1], fps=1):
    
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
        model: torch.nn.Module,
        target_loss: float = "MAE",
        optimizer = AdamWReduceLROnPlateau(),
    ):
        """tbc

        """
        super().__init__()
        
        assert target_loss in ["MAE", "MSE", "SSIM"]

        self.model = model
        self._optimizer = optimizer
    
        self.ssim_func = SSIM3D()
        self.target_loss = target_loss
        
        self._accumulated_metrics = MetricAccumulator()
    
    @staticmethod
    def _minus_one_to_nan(y):
        y[y==-1] = torch.nan
        
    def _calculate_common_losses(self, y, y_hat):
        """Calculate losses common to train, test, and val"""
        
        losses = {}
        
        # calculate mse, mae
        # y shape: [N, C, T, H, W]
        
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
    
    def _training_accumulate_log(self, losses):
        """Internal function to accumulate training batches and log results.

        This is used when accummulating grad batches. Should make the variability in logged training
        step metrics indpendent on whether we accumulate N batches of size B or just use a larger
        batch size of N*B with no accumulaion.
        """

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
        del X

        # Replace the -1 (filled) values in y with NaNs 
        # This operation is in-place
        self._minus_one_to_nan(y)

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self._training_accumulate_log({k: v.detach().cpu().item() for k, v in losses.items()})
        
        train_loss = losses[f"{self.target_loss}/train"]
                
        # Occasionally y will be entirely NaN and we have no training targets. So the train loss
        # will also be NaN. In this case we return None so lightning skips this train step
        if torch.isnan(train_loss).item():
            print("\n\nTraining loss is nan\n\n")
            return None
        else:
            return train_loss
        

    def validation_step(self, batch: dict, batch_idx):
        """Run validation step"""
        X, y = batch
        y_hat = self.model(X)
        del X
        
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