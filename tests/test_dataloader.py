from sat_pred.dataloader import (
    load_satellite_zarrs,
    find_valid_t0_times,
    SatelliteDataset,
    SatelliteDataModule,
)


def test_load_satellite_zarrs(sat_zarr_path):
    
    # Check can load with string and list of string(s)
    ds = load_satellite_zarrs(sat_zarr_path)
    ds = load_satellite_zarrs([sat_zarr_path])
    
    # Dataset is a full 24 hours of 5 minutely data -> 24hours * (60/5) = 288
    assert len(ds.time)==288


def test_find_valid_t0_times(sat_zarr_path):
    ds = load_satellite_zarrs(sat_zarr_path)
    
    t0_times = find_valid_t0_times(
        ds, 
        history_mins=60, 
        forecast_mins=120, 
        sample_freq_mins=5,
    )
    
    # original timesteps                  288
    # forecast length buffer      - (120 / 5) 
    # history length buffer       -  (60 / 5) 
    #                            ------------
    # Total                               252
    
    assert len(t0_times)==252
    
    t0_times = find_valid_t0_times(
        ds, 
        history_mins=60, 
        forecast_mins=120, 
        sample_freq_mins=15,
    )
    
    # original 15 minute timesteps     288 / 3
    # forecast length buffer      - (120 / 15) 
    # history length buffer       -  (60 / 15) 
    #                            ------------
    # Total                               84
    
    assert len(t0_times)==84    


def test_satellite_dataset(sat_zarr_path):
    dataset = SatelliteDataset(
        zarr_path=sat_zarr_path,
        start_time=None,
        end_time=None,
        history_mins=60, 
        forecast_mins=120, 
        sample_freq_mins=5,
    )
    
    assert len(dataset)==252
    
    X, y = dataset[0]
    
    # 11 channels
    # 372 y-dim steps
    # 614 x-dim steps
    # (60 / 5) + 1 = 13 history steps
    # (120 / 5) = 24 forecast steps
    assert X.shape==(11, 13, 372, 614)
    assert y.shape==(11, 24, 372, 614)


def test_satellite_datamodule(sat_zarr_path):
    datamodule = SatelliteDataModule(
        zarr_path=sat_zarr_path,
        history_mins=60, 
        forecast_mins=120, 
        sample_freq_mins=5,
        batch_size=2,
        num_workers=2,
        prefetch_factor=None,
    )
    
    dl = datamodule.train_dataloader()
    
    X, y = next(iter(dl))
    
    assert X.shape==(2, 11, 13, 372, 614)
    assert y.shape==(2, 11, 24, 372, 614)