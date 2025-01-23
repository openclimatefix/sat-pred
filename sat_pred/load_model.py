from pyaml_env import parse_config
import hydra
from glob import glob
import torch



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
    model_config = model_config["model"]

    # Check for data config
    data_config = parse_config(f"{checkpoint_dir_path}/data_config.yaml")
    

    return model, model_config, data_config