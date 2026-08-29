# SPDX-License-Identifier: MIT
# src/utils/checkpoint.py

"""Utilities for the saving and loading checkpoints."""

import inspect
from typing import Any
import torch
from torch import nn

from models.rbm_binary import RBM_binary
from models.rbm_exponential import RBM_exponential
from models.rbm_gaussian import RBM_gaussian
from models.rbm_vonmises import RBM_vonmises
from models.rbm_multinomial import RBM_multinomial
from utils.config import migrate_deprecated_keys

MODEL_REGISTRY = {
    "binary": RBM_binary,
    "exponential": RBM_exponential,
    "gaussian": RBM_gaussian,
    "multinomial": RBM_multinomial,
    "vonmises": RBM_vonmises,
}


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    config: dict,
    history: dict,
    path: str,
) -> None:
    """Save a checkpoint dictionary with torch to the given path.

    A checkpoint file will be saved as a dictionary with the following keys:
        - "epoch"
        - "model_state"
        - "optimizer_state"
        - "config"
        - "history"

    Args:
        model: RBM class that was trained.
        optimizer: Optimizer used in training. Currently unused.
        epoch: Number of epochs trained.
        config: Configuration used to train the model.
        history: Dictionary of training metrics per epoch.
        path: Path to which the checkpoint is saved.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
            if optimizer is not None
            else None,
            "config": config,
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: torch.device,
    strict: bool = True,
    load_model: bool = True,
) -> tuple[Any, dict]:
    """Load the checkpoint with torch from the given path and reconstruct the model.

    Args:
        path: Path from which the checkpoint is loaded.
        device: Device outputted by get_device().
            Can be either 'cpu' or 'cuda'
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by model's state_dict.
        load_model: Whether to reconstruct the model from the checkpoint.

    Returns:
        ckpt if ``load_model`` is set to False.

        Otherwise, a tuple containing:
            - **model**: Model reconstructed from the checkpoint.
            - **ckpt**: Checkpoint dictionary with the following keys:

                - "epoch"
                - "model_state"
                - "optimizer_state"
                - "config"
                - "history"
    """
    ckpt = torch.load(path, map_location=device)
    config = migrate_deprecated_keys(ckpt["config"])
    ckpt["config"] = config

    if load_model:
        model_type = config["model"]["model_type"]
        cls = MODEL_REGISTRY[model_type]

        # Pass all model config fields except "type" directly to constructor
        model_kwargs = {
            k: v for k, v in config["model"].items() if k != "model_type"
        }
        valid_params = inspect.signature(cls.__init__).parameters
        model_kwargs = {
            k: v for k, v in model_kwargs.items() if k in valid_params
        }

        model_state = ckpt["model_state"]
        model = cls(**model_kwargs)
        try:
            model.persistent_v = torch.empty_like(model_state["persistent_v"])
        except:
            model.persistent_v = None

        try:
            model.persistent_v_pt = torch.empty_like(
                model_state["persistent_v_pt"]
            )
        except:
            model.persistent_v_pt = None

        model.load_state_dict(model_state, strict=strict)
        model.to(device)
        model.eval()
        return model, ckpt
    return ckpt
