try:
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from cloudcasting.dataset import load_satellite_zarrs, find_valid_t0_times

import hydra
import torch
from pyaml_env import parse_config
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from glob import glob
from numcodecs import Blosc


checkpoint = "/home/jamesfulton/repos/sat_pred/checkpoints/u12kiwmy"
save_dir = "/mnt/disks/sat_preds/simvp_preds"
compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_from_checkpoints(
    checkpoint_dir_path: str,
    val_best: bool = True,
):
    """Load a model from its checkpoint directory
    
    Args:
        checkpoint_dir_path: Path to the checkpoint directory
        val_best: Whether to use the best performing checkpoint found during training, else uses
            the last checkpoint saved during training
    """

    # Load the model
    model_config = parse_config(f"{checkpoint_dir_path}/model_config.yaml")

    lightning_wrapped_model = hydra.utils.instantiate(model_config)

    if val_best:
        # Only one epoch (best) saved per model
        files = glob(f"{checkpoint_dir_path}/epoch*.ckpt")
        if len(files) != 1:
            raise ValueError(
                f"Found {len(files)} checkpoints @ {checkpoint_dir_path}/epoch*.ckpt. Expected one."
            )
        checkpoint = torch.load(files[0], map_location="cpu")
    else:
        checkpoint = torch.load(f"{checkpoint_dir_path}/last.ckpt", map_location="cpu")

    lightning_wrapped_model.load_state_dict(state_dict=checkpoint["state_dict"])
    
    # discard the lightning wrapper on the model
    model  = lightning_wrapped_model.model

    # Check for data config
    data_config = parse_config(f"{checkpoint_dir_path}/data_config.yaml")
    

    return model, model_config, data_config



# We define a new class that inherits from AbstractModel
class MLModel:
    """A persistence model which predicts a blury version of the most recent frame"""

    def __init__(self, checkpoint_dir_path: str) -> None:

        
        model, model_config, data_config = get_model_from_checkpoints(checkpoint)

        self.model = model.to(DEVICE)
        self.history_mins = model_config['history_mins']
        self.model_config = model_config
        self.data_config = data_config
        self.checkpoint_dir_path = checkpoint_dir_path


    def __call__(self, X):
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)
                
        X = torch.Tensor(X).to(DEVICE)
        
        with torch.no_grad():
            y_hat = self.model(X).cpu().numpy()
        
        # Clip the values to be between 0 and 1
        y_hat = y_hat.clip(0, 1)

        return y_hat
    


def backtest_collate_fn(
    samples: list,
):
    # Create empty stores for the compiled batch
    X_all = np.empty((len(samples), *samples[0][0].shape), dtype=np.float32)

    # Fill the stores with the samples
    ts = []
    for i, (X, t) in enumerate(samples):
        X_all[i] = X
        ts.append(t)
    return X_all, pd.to_datetime(ts)
    

DataIndex = str | datetime | pd.Timestamp | int

class BacktestSatelliteDataset(Dataset):
    def __init__(
        self,
        zarr_path: list[str] | str,
        start_time: str | None,
        end_time: str | None,
        history_mins: int,
        sample_freq_mins: int,
        nan_to_num: bool = False,
    ):
        """A torch Dataset for loading past and future satellite data

        Args:
            zarr_path: Path to the satellite data. Can be a string or list
            start_time: The satellite data is filtered to exclude timestamps before this
            end_time: The satellite data is filtered to exclude timestamps after this
            history_mins: How many minutes of history will be used as input features
            sample_freq_mins: The sample frequency to use for the satellite data
            nan_to_num: Whether to convert NaNs to -1.
        """

        # Load the sat zarr file or list of files and slice the data to the given period
        self.ds = load_satellite_zarrs(zarr_path).sel(time=slice(start_time, end_time))

        # Convert the satellite data to the given time frequency by selection
        mask = np.mod(self.ds.time.dt.minute, 15) == 0
        self.ds = self.ds.sel(time=mask)

        # Find the valid t0 times for the available data. This avoids trying to take samples where
        # there would be a missing timestamp in the sat data required for the sample
        self.t0_times = self._find_t0_times(
            pd.DatetimeIndex(self.ds.time), history_mins, sample_freq_mins
        )

        # Only do 30 minute intervals
        self.t0_times = self.t0_times[self.t0_times.minute%30==0]

        self.history_mins = history_mins
        self.sample_freq_mins = sample_freq_mins
        self.nan_to_num = nan_to_num

    @staticmethod
    def _find_t0_times(
        date_range: pd.DatetimeIndex, 
        history_mins: int, 
        sample_freq_mins: int
    ) -> pd.DatetimeIndex:
        return find_valid_t0_times(date_range, history_mins, 0, sample_freq_mins)

    def __len__(self):
        return len(self.t0_times)

    def _get_datetime(self, t0: datetime):
        ds_input = self.ds.sel(time=slice(t0 - timedelta(minutes=self.history_mins), t0))

        # Load the data eagerly so that the same chunks aren't loaded multiple times after we split
        # further
        ds_input = ds_input.compute(scheduler="single-threaded")

        # Reshape to (channel, time, height, width)
        ds_input = ds_input.transpose("variable", "time", "y_geostationary", "x_geostationary")

        # Convert to arrays
        X = ds_input.data.values

        if self.nan_to_num:
            X = np.nan_to_num(X, nan=-1)

        return X.astype(np.float32), t0

    def __getitem__(self, key: DataIndex):
        if isinstance(key, int):
            t0 = self.t0_times[key]

        else:
            assert isinstance(key, str | datetime | pd.Timestamp)
            t0 = pd.Timestamp(key)
            assert t0 in self.t0_times

        return self._get_datetime(t0)



def run_backtest(
    model: MLModel,
    dataset: BacktestSatelliteDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    batch_limit: int | None = None,
    agg_batches: int = 1
) -> None:
    """Calculate the scoreboard metrics for the given model on the validation dataset.

    Args:
        model (AbstractModel): The model to score.
        valid_dataset (ValidationSatelliteDataset): The validation dataset to score the model on.
        batch_size (int, optional): Defaults to 1.
        num_workers (int, optional): Defaults to 0.
        batch_limit (int | None, optional): Defaults to None. Stop after this many batches.
            For testing purposes only.

    """

    backtest_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=backtest_collate_fn,
        drop_last=False,
    )

    # we probably want to accumulate metrics here instead of taking the mean of means!
    loop_steps = len(backtest_dataloader) if batch_limit is None else batch_limit

    da_y_hats = []
    save_batch_num = 0

    attrs_dict = {k:v for k,v in dataset.ds.attrs.items()}
    attrs_dict["model_checkpoint"] = model.checkpoint_dir_path

    for i, (X, t) in tqdm(enumerate(backtest_dataloader), total=loop_steps):
        
        y_hat = model(X)
        init_times = pd.DatetimeIndex(t)
        steps = pd.timedelta_range("15min", periods=y_hat.shape[2], freq="15min")

        da_y_hat = xr.DataArray(
            y_hat, 
            dims=["init_time", "variable", "step", "y_geostationary", "x_geostationary"], 
            coords={
                "init_time": init_times,
                "variable": dataset.ds.variable,
                "step": steps,
                "y_geostationary": dataset.ds.y_geostationary,
                "x_geostationary": dataset.ds.x_geostationary,
            }
        ).chunk(
            {
                "init_time": 1, 
                "variable":-1,
                "step":-1, 
                "y_geostationary": -1, 
                "x_geostationary": -1
            }
        )

        da_y_hats.append(da_y_hat)

        del da_y_hat
        
        if len(da_y_hats)==agg_batches or i==loop_steps-1:

            da_y_hats = xr.concat(da_y_hats, dim="init_time")

            da_y_hats.attrs = attrs_dict

            da_y_hats = da_y_hats.to_dataset(name="sat_pred")
            
            da_y_hats.to_zarr(
                f"{save_dir}/part_{save_batch_num}.zarr", 
                mode="w",
                encoding={var: {'compressor': compressor} for var in da_y_hats.data_vars},
            )
            
            save_batch_num += 1
            da_y_hats = []
            

        if batch_limit is not None and i == batch_limit:
            break


if __name__=="__main__":


    os.makedirs(save_dir, exist_ok=False)

    model = MLModel(checkpoint)

    dataset = BacktestSatelliteDataset(
        zarr_path=[
            "/mnt/disks/all_inputs/sat/2019_nonhrv.zarr",
            "/mnt/disks/all_inputs/sat/2020_nonhrv.zarr",
            "/mnt/disks/all_inputs/sat/2021_nonhrv.zarr",
            "/mnt/disks/all_inputs/sat/2022_nonhrv.zarr",
            "/mnt/disks/all_inputs/sat/2023_nonhrv.zarr",
        ],
        start_time=None, 
        end_time=None,
        history_mins=model.history_mins,
        sample_freq_mins=15,
        nan_to_num=model.data_config['nan_to_num'],
    )   

    run_backtest(
        model=model,
        dataset=dataset,
        batch_size=4,
        num_workers=12,
        batch_limit=None,
        agg_batches=8,
    )
