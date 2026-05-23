# src/utils/checkpoint.py
import inspect
import torch

from models.rbm_binary import RBM_binary
from models.rbm_exponential import RBM_exponential
from models.rbm_gaussian import RBM_gaussian
from models.rbm_vonmises import RBM_vonmises
from models.rbm_multinomial import RBM_multinomial

MODEL_REGISTRY = {
    "binary":      RBM_binary,
    "exponential": RBM_exponential,
    "gaussian":    RBM_gaussian,
    "multinomial": RBM_multinomial,
    "vonmises":    RBM_vonmises,
}

def save_checkpoint(model, optimizer, epoch, config, history, path):
    torch.save({
        "epoch":  epoch,
        "model_state":  model.state_dict(),
        "optimizer_state":  optimizer.state_dict() if optimizer is not None else None,
        "config":  config,
        "history":  history,
    }, path)

def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    
    model_type = config["model"]["model_type"]
    cls = MODEL_REGISTRY[model_type]
    
    # Pass all model config fields except "type" directly to constructor

    model_kwargs = {k: v for k, v in config["model"].items() if k != "model_type"}
    valid_params = inspect.signature(cls.__init__).parameters
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}

    model = cls(**model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt