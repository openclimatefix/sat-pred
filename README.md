![Actions Status][actions-badge]][actions-link]

# Satellite Image Forecasting with Neural Networks

## Installation

After cloning and entering this repo, create a fresh Python environment (e.g. via `uv`, `venv`, `conda`), then install this package and its dependencies:
```
pip install .
```

### Developer installation

As above, but install in editable mode with the `dev` dependencies:
```
pip install -e ".[dev]"
```


## Training

If you want to train the earthformer model you should clone and install the earthformer repo as well

```
cd ..
git clone https://github.com/amazon-science/earth-forecasting-transformer.git
cd earth-forecasting-transformer
pip install -e .
```

You can train a model by running

```
python train.py
```

from the root of the library.

The model and training options used are defined in the config files. The most important parts of the config files you may wish to train are:

- `configs/datamodule/default.yaml`
  - `zarr_paths` which point to your training data
  - `train/val_period` which control the train / val split used
  - `num_workers` and `batch_size` to suit your machine

- `configs/logger/wandb.yaml`
  - Set `project` to the project name you want to save the runs to on wandb

- `configs/trainer/default.yaml`
  - This control the parameters for the lightning Trainer. See https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
  - Note you might want to set `fast_dev_run` to `true` to aid with testing and getting set up

- `configs/config.yaml`
  - Set `model_name` to the name the run will be logged under on wandb
  - Set `defaults:model` to one of the model config filenames within `configs/model`

Note that since we use hydra to build up the configs, you can change the configs from the command line when running the training job. For example

```
python sat_pred/train.py model=earthformer model_name="earthformer-v1" model.optimizer.lr=0.0002
```

will train the model defined in `configs/model/earthformer.yaml` log ther training results to wandb under the name `earthformer-v1`. It will also overwrite the learning rate of the optimiser to 0.0002.


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/openclimatefix/sat_pred/workflows/CI/badge.svg
[actions-link]:             https://github.com/openclimatefix/sat_pred/actions
[pypi-link]:                https://pypi.org/project/sat_pred/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/sat_pred
[pypi-version]:             https://img.shields.io/pypi/v/sat_pred
<!-- prettier-ignore-end -->
