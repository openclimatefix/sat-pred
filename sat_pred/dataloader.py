"""Dataset and DataModule for past and future satellite data"""

from typing import Union
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

import torch

from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from ocf_datapipes.load.satellite import _get_single_sat_data
from ocf_datapipes.select.find_contiguous_t0_time_periods import (
    find_contiguous_time_periods, find_contiguous_t0_time_periods
)


def minutes(m):
    """Timedelta of a number of minutes"""
    return timedelta(minutes=m)


def load_satellite_zarrs(zarr_path):
    """Load the satellite data"""
    
    zarr_path = list(zarr_path)
    
    if isinstance(zarr_path, (list, tuple)):
        ds = xr.combine_nested(
            [_get_single_sat_data(path) for path in zarr_path],
            concat_dim="time",
            combine_attrs="override",
            join="override",
        )
    else:
        ds = _get_single_sat_data(zarr_path)
        
    return ds


def find_valid_t0_times(ds, history_mins, forecast_mins, sample_freq_mins):
    """Constuct an array of all t0 times which are valid considering the gaps in the sat data"""

    # Find periods where we have contiguous time steps
    contiguous_time_periods = find_contiguous_time_periods(
        datetimes=pd.DatetimeIndex(ds.time),
        min_seq_length=int((history_mins +  forecast_mins) / sample_freq_mins) + 1,
        max_gap_duration=minutes(sample_freq_mins),
    )

    # Find periods of valid init-times
    contiguous_t0_periods = find_contiguous_t0_time_periods(
        contiguous_time_periods=contiguous_time_periods,
        history_duration=minutes(history_mins),
        forecast_duration=minutes(forecast_mins),
    )

    valid_t0_times = []
    for _, row in contiguous_t0_periods.iterrows():
        valid_t0_times.append(
            pd.date_range(
                row["start_dt"], 
                row["end_dt"],  
                freq=f"{sample_freq_mins}min"
            )
        )

    valid_t0_times = pd.to_datetime(np.concatenate(valid_t0_times))
    
    return valid_t0_times


class SatelliteDataset(Dataset):
    def __init__(
        self, 
        zarr_path: Union[list, str], 
        start_time: str,
        end_time: str, 
        history_mins: int, 
        forecast_mins: int, 
        sample_freq_mins: int,
        preshuffle: bool = False
    ):
        """A torch Dataset for loading past and future satellite data
        
        Args:
            zarr_path: Path the satellite data. Can be a string or list
            start_time: The satellite data is filtered to exclude timestamps before this
            end_time: The satellite data is filtered to exclude timestamps after this
            history_mins: How many minutes of history will be used as input features
            forecast_mins: How many minutes of future will be used as target features
            sample_freq_mins: The sample frequency to use for the satellite data
            preshuffle: Whether to shuffle the data - useful for validation
        """
        
        # Load the sat zarr file or list of files and slice the data to the given period
        self.ds = load_satellite_zarrs(zarr_path).sel(time=slice(start_time, end_time))
        
        # Convert the satellite data to the given time frequency by selection
        mask = np.mod(self.ds.time.dt.minute, sample_freq_mins)==0
        self.ds = self.ds.sel(time=mask)
             
        # Find the valid t0 times for the available data. This avoids trying to take samples where 
        # there would be a missing timestamp in the sat data required for the sample
        self.valid_t0_times = find_valid_t0_times(self.ds, history_mins, forecast_mins, sample_freq_mins)
        
        if preshuffle:
            self.valid_t0_times = pd.to_datetime(np.random.permutation(self.valid_t0_times))
        
        self.history_mins = history_mins
        self.forecast_mins = forecast_mins
        self.sample_freq_mins = sample_freq_mins
        
    
    def __len__(self):
        return len(self.valid_t0_times)

    
    def __getitem__(self, idx):
        if isinstance(idx, (str)):
            t0 = pd.Timestamp(idx)
            assert t0 in self.valid_t0_times
        elif isinstance(idx, int):
            t0 = self.valid_t0_times[idx]
        else:
            raise ValueError(f"Unrecognised type {type(idx)}")
        
        ds_sel = self.ds.sel(
            time=slice(
                t0-minutes(self.history_mins), 
                t0+minutes(self.forecast_mins)
            )
        )
        
        # Load the data eagerly so that the same chunks aren't loaded multiple times after we split 
        # further
        ds_sel = ds_sel.compute(scheduler="single-threaded")
        
        # Reshape to (channel, time, height, width)
        ds_sel = ds_sel.transpose("variable", "time", "y_geostationary", "x_geostationary")
                
        ds_input = ds_sel.sel(time=slice(None, t0))
        ds_target = ds_sel.sel(time=slice(t0+minutes(self.sample_freq_mins), None))
        
        # Convert to arrays
        X = ds_input.data.values
        y = ds_target.data.values
                
        X = np.nan_to_num(X, nan=-1)
        y = np.nan_to_num(y, nan=-1)
        
        return X.astype(np.float32), y.astype(np.float32)


class SatelliteDataModule(LightningDataModule):
    """A lightning DataModule for loading past and future satellite data"""

    def __init__(
        self,
        zarr_path: Union[list, str],
        history_mins: int, 
        forecast_mins: int, 
        sample_freq_mins: int,
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
        test_period=[None, None],
    ):
        """A lightning DataModule for loading past and future satellite data

        Args:
            zarr_path: Path the satellite data. Can be a string or list
            history_mins: How many minutes of history will be used as input features
            forecast_mins: How many minutes of future will be used as target features
            sample_freq_mins: The sample frequency to use for the satellite data
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
            train_period: Date range filter for train dataloader.
            val_period: Date range filter for val dataloader.
            test_period: Date range filter for test dataloader.
        """
        super().__init__()
        
        self.zarr_path = zarr_path
        self.history_mins = history_mins
        self.forecast_mins = forecast_mins
        self.sample_freq_mins = sample_freq_mins
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period

        self._common_dataloader_kwargs = dict(
            batch_size=batch_size, 
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

    def _make_dataset(self, start_date, end_date, preshuffle=False):
        dataset = SatelliteDataset(
            self.zarr_path,
            start_date,
            end_date,
            self.history_mins, 
            self.forecast_mins, 
            self.sample_freq_mins,
            preshuffle=preshuffle,
        )
        return dataset
        
    def train_dataloader(self):
        """Construct train dataloader"""
        dataset = self._make_dataset(*self.train_period)
        return DataLoader(dataset, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self):
        """Construct val dataloader"""

        dataset = self._make_dataset(*self.val_period, preshuffle=True)
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(self):
        """Construct test dataloader"""
        dataset = self._make_dataset(*self.test_period)
        return DataLoader(dataset, shuffle=False, **self._common_dataloader_kwargs)