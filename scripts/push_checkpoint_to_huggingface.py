"""Command line tool to push locally save model checkpoints to huggingface

use:
python push_checkpoint_to_huggingface.py.py "path/to/model/checkpoints" \
    --huggingface-repo="openclimatefix/cloudcasting_uk" \
    --wandb-repo="openclimatefix/sat_pred" \
    --local-path="~/tmp/this_model" \
    --no-push-to-hub
"""

import tempfile
import yaml
import os
import typer
import wandb

from torch.nn import Module

from sat_pred.load_model import get_model_from_checkpoints

from pathlib import Path

from safetensors.torch import save_model as save_model_as_safetensor

from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.hf_api import HfApi


def save_model_to_huggingface(
    model: Module,
    save_directory: str,
    model_config: dict,
    push_to_hub: bool = False,
    repo_id: str | None = None,
    wandb_repo: str | None = None,
    wandb_id: str | None = None,
    **kwargs,
):
    """
    Save weights in local directory.

    Args:
        save_directory:
            Path to directory in which the model weights and configuration will be saved.
        model_config:
            Model configuration specified as a key/value dictionary.
         push_to_hub:
            Whether or not to push your model to the HuggingFace Hub after saving it.
        repo_id:
            ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to
            the folder name if not provided.
        wandb_repo: Identifier of the repo on wandb.
        wandb_id: Identifier of the model on wandb.
        card_template_path: Path to the HuggingFace model card template. Defaults to card in
            PVNet library if set to None.
        kwargs:
            Additional key word arguments passed along to the
    """
    
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # Save model weights
    save_model_as_safetensor(model, f"{save_directory}/model.safetensors")

    # Save model config
    with open(save_directory / "model_config.yaml", 'w') as outfile:
        yaml.dump(model_config, outfile, default_flow_style=False)
   
    # Create and save model card.
    card_data = ModelCardData(language="en", license="mit", library_name="pytorch")
    card_template_path = (
        f"{os.path.dirname(os.path.abspath(__file__))}/model_cards/default_model_card.md"
    )

    wandb_link = f"https://wandb.ai/{wandb_repo}/runs/{wandb_id}"

    card = ModelCard.from_template(
        card_data,
        template_path=card_template_path,
        wandb_link=wandb_link,
    )

    (save_directory / "README.md").write_text(str(card))
    
    # Optionally push to huggingface
    if push_to_hub:
        api = HfApi()

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=save_directory,
        )


def push_to_huggingface(
    checkpoint_dir_path: str,
    huggingface_repo: str = "openclimatefix/cloudcasting_uk",
    wandb_repo: str = "openclimatefix/sat_pred",
    val_best: bool = True,
    wandb_id: str | None = None,
    local_path: str = None,
    push_to_hub: bool = True,
):
    
    # If the wandb run name is not supplied infer it from the checkpoint path
    if wandb_id is None:
        all_wandb_ids = [run.id for run in wandb.Api().runs(path=wandb_repo)]
        dirname = checkpoint_dir_path.split("/")[-1]
        assert dirname in all_wandb_ids
        wandb_id = dirname

    # Load the model
    model, model_config, _ = get_model_from_checkpoints(
        checkpoint_dir_path, 
        val_best=val_best
    )
    
    assert push_to_hub or local_path is not None
    
    # Push to hub
    if local_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        model_output_dir = temp_dir.name
    else:
        model_output_dir = local_path
        

    save_model_to_huggingface(
        model=model,
        save_directory=model_output_dir,
        model_config=model_config,
        wandb_repo=wandb_repo,
        wandb_id=wandb_id,
        push_to_hub=push_to_hub,
        repo_id=huggingface_repo if push_to_hub else None,
    )

    if local_path is None:
        temp_dir.cleanup()


if __name__ == "__main__":
    #typer.run(push_to_huggingface)
    push_to_huggingface(
        "/home/jamesfulton/repos/sat_pred/checkpoints/ob9v9128",
        local_path="tmp/simVP_model",
        push_to_hub=True,
    )