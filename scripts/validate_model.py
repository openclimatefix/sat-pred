from cloudcasting.validation import validate
from cloudcasting.models import AbstractModel

import glob

import hydra
import torch
from pyaml_env import parse_config


checkpoint = "/home/jamesfulton/repos/sat_pred/checkpoints/zk5vvbhk"
WANDB_PROJECT = "cloudcasting"
WANDB_RUN_NAME = "earthformer-v1"


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
        files = glob.glob(f"{checkpoint_dir_path}/epoch*.ckpt")
        if len(files) != 1:
            raise ValueError(
                f"Found {len(files)} checkpoints @ {checkpoint_dir_path}/epoch*.ckpt. Expected one."
            )
        checkpoint = torch.load(files[0], map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(f"{checkpoint_dir_path}/last.ckpt", map_location="cpu", weights_only=True)

    state_dict = checkpoint["state_dict"]
    if 'ssim_func.kernel' in state_dict:
        del state_dict['ssim_func.kernel']

    lightning_wrapped_model.load_state_dict(state_dict=state_dict)
    
    # discard the lightning wrapper on the model
    model  = lightning_wrapped_model.model

    # Check for data config
    data_config = parse_config(f"{checkpoint_dir_path}/data_config.yaml")
    

    return model, model_config, data_config



# We define a new class that inherits from AbstractModel
class MLModel(AbstractModel):
    """A persistence model which predicts a blury version of the most recent frame"""

    def __init__(self, checkpoint_dir_path: str) -> None:

        
        model, model_config, data_config = get_model_from_checkpoints(checkpoint)
        
        super().__init__(history_steps=12)


        self.model = model.to(DEVICE)
        self.model_config = model_config
        self.data_config = data_config
        self.checkpoint_dir_path = checkpoint_dir_path


    def forward(self, X):
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)
                
        X = torch.Tensor(X).to(DEVICE)
        
        with torch.no_grad():
            y_hat = self.model(X).cpu().numpy()
        
        # Clip the values to be between 0 and 1
        y_hat = y_hat.clip(0, 1)

        return y_hat

    def hyperparameters_dict(self):

        wandb_id = self.checkpoint_dir_path.split("/")[-1]
        params_dict = {
            "training_run_link": f"https://wandb.ai/openclimatefix/sat_pred/runs/{wandb_id}",
        }
        params_dict.update(self.model_config)

        return params_dict

if __name__=="__main__":

    model = MLModel(checkpoint)

    validate(
        model=model,
        data_path="/mnt/disks/sat_data/2022_test_nonhrv.zarr",
        wandb_project_name=WANDB_PROJECT,
        wandb_run_name=WANDB_RUN_NAME,
        batch_size = 2,
        num_workers = 10,
        batch_limit = None,
        nan_to_num = model.data_config['nan_to_num']
    )
