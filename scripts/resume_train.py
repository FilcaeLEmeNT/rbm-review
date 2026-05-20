import argparse
import os
from os import path
import numpy as np
import math

from utils.device import get_device
from utils.config import load_config

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train RBM model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=path.join("outputs", "checkpoints", "default_run", "checkpoint.pt"),
        help="Path to checkpoint file"
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs to train, resuming from a checkpoint."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load device: Either CPU or CUDA
    device = get_device()

    # Get loaded model and checkpoint
    ckpt_path = args.checkpoint
    rbm, ckpt = load_checkpoint(ckpt_path, device)

    if rbm is None:
        raise ValueError(f"The model was not loaded from the checkpoint file properly.")
    if ckpt is None:
        raise ValueError(f"The checkpoint file was not loaded from the checkpoint file properly.")

    # Get config file from checkpoint
    config = ckpt["config"]

    data_type = config["data"]["type"]
    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    split = config["data"]["split"]
    binarize = config["data"]["binarize"]
    q = config["data"]["q"]
    T = config["data"]["T"]
    L = config["data"]["L"]

    model_type = config["model"]["type"]
    n_class = config["model"]["n_class"]
    n_visible = config["model"]["n_visible"]
    n_hidden = config["model"]["n_hidden"]
    mf = config["model"]["mf"]

    n_epochs = config["training"]["n_epochs"]
    lr = config["training"]["lr"]
    k = config["training"]["k"]
    pcd = config["training"]["pcd"]
    sm = config["training"]["sm"]
    mc = config["training"]["mc"]
    epsilon = config["training"]["epsilon"]

    out_dir = config["output"]["base_dir"]
    run_name = config["output"]["run_name"]

    # Print config summary
    print("Config summary:")
    print("Data parameters:")
    print(f"\ttype={data_type}", f"data_dir={data_dir}", f"batch_size={batch_size}", f"split={split}", f"binarize={binarize}", f"q={q}", f"T={T}", f"L={L}", sep="\n\t")
    print("Model parameters:")
    print(f"\ttype={model_type}", f"n_class={n_class}", f"n_visible={n_visible}", f"n_hidden={n_hidden}", f"mf={mf}", sep="\n\t")
    print("Training parameters:")
    print(f"\tn_epochs={n_epochs}", f"lr={lr}", f"k={k}", f"pcd={pcd}", f"sm={sm}", f"mc={mc}", f"epsilon={epsilon}", sep="\n\t")
    print(f"Output directory: {out_dir}")
    print(f"Run Name: {run_name}")
    print("")
    
    # Load data.
    train_loader, test_loader = load_data(data_type, data_dir, split, q, T, L, batch_size, binarize, model_type)

    # Print model information    
    print(f"Using model type: {model_type}")
    
    if model_type == "binary":
        print(f"Using mean-field: {mf}")
        print(f"Using binarize: {binarize}")
    elif model_type == "multinomial":
        print(f"Number of categories: {n_class}")
    
    # Get the history and n_epochs from the checkpoint.
    n_epochs = args.n_epochs  # Overwrite n_epochs with how many epochs to resume training.
    n_epochs_prev = ckpt["epoch"] # Number of epochs already trained.
    history = ckpt["history"]  # Old history file.

    print(f"\nResuming training, starting from {n_epochs_prev} epochs.")

    # Train the model
    if sm == True and model_type == "gaussian":
        history_temp = train_sm(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)
    else:
        history_temp = train_cd(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)

    # Extend the history from previous session with the temporary history.
    for key in history:
        history[key].extend(history_temp[key])
    
    # Get total epochs
    n_epochs = n_epochs_prev + n_epochs
    
    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)

    # Create new config with all parameters for saving
    new_config = {
        "data": {
            "type": data_type,
            "data_dir": data_dir,
            "batch_size": batch_size,
            "split": split,
            "binarize": binarize,
            "q": q,
            "T": T,
            "L": L,
        },
        "model": {
            "type": model_type,
            "n_class": n_class,
            "n_visible": n_visible,
            "n_hidden": n_hidden,
            "mf": mf,
        },
        "training": {
            "n_epochs": n_epochs,
            "lr": lr,
            "k": k,
            "pcd": pcd,
            "sm": sm,
            "mc": mc,
            "epsilon": epsilon,
        },
        "output": {
            "base_dir": out_dir,
            "run_name": run_name,
        }
    }

    save_checkpoint(model=rbm, optimizer=None, epoch=n_epochs, config=new_config, history=history, path=path.join(checkpoints_dir, "checkpoint.pt"))
    print(f"\nCheckpoint file, 'checkpoint.pt', saved to directory: {checkpoints_dir}")
    print("Total Epochs:", n_epochs)

    return

if __name__ == "__main__":
    main()