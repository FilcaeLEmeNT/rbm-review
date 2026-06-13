import argparse
import os
from os import path
import numpy as np
import math

from utils.device import get_device
import utils.config as cfg
import utils.sweep as swp

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Continue Training RBM model")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--checkpoint", type=str, default=None,
        help="Path to checkpoint file"
    )
    group.add_argument("--sweep", type=str, default=None,
        help="Path to sweep configuration file"
    )

    parser.add_argument("--config", type=str, default=None,
        help="Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. \
            Use as an alternative to --checkpoint, or combine with --sweep to resume all runs generated from that config/sweep combination."
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

    # Check args
    if args.sweep and not args.config:
        argparse.error("--config is required when using --sweep")

    if args.checkpoint and args.config:
        print("Warning: --config is ignored when --checkpoint is provided. Configuration file embedded in the checkpoint is used.")

    # Load device: Either CPU or CUDA
    device = get_device()

    if args.sweep:
        config = cfg.load_config(args.config)
        sweep = cfg.load_config(args.sweep)
        ckpt_paths = swp.get_checkpoints_from_sweep(config, sweep)

        for ckpt_path in ckpt_paths:
            resume_train(device, ckpt_path, args.n_epochs)

    elif args.checkpoint:
        # Get checkpoint path
        ckpt_path = args.checkpoint
        resume_train(device, ckpt_path, args.n_epochs)

    elif args.config:
        # Get checkpoint path
        config = cfg.load_config(args.config)
        ckpt_path = cfg.get_checkpoint_from_config(config)
        resume_train(device, ckpt_path, args.n_epochs)
    
    else:
        argparse.error("Atleast one of --checkpoint, --config, and --sweep must be used.")

    return

def resume_train(device, ckpt_path, n_epochs_arg):
    # Get loaded model and checkpoint
    rbm, ckpt = load_checkpoint(ckpt_path, device)

    if rbm is None:
        raise ValueError(f"The model was not loaded from the checkpoint file properly.")
    if ckpt is None:
        raise ValueError(f"The checkpoint file was not loaded from the checkpoint file properly.")

    # Get config file from checkpoint
    config = ckpt["config"]

    # Get values from configuration.
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    data_type = data_cfg.get("data_type")
    data_dir = data_cfg.get("data_dir")
    batch_size = data_cfg.get("batch_size")
    split = data_cfg.get("split")
    binarize = data_cfg.get("binarize")
    q = data_cfg.get("q")
    T = data_cfg.get("T")
    L = data_cfg.get("L")

    model_type = model_cfg.get("model_type")
    n_class = model_cfg.get("n_class")
    n_visible = model_cfg.get("n_visible")
    n_hidden = model_cfg.get("n_hidden")
    mf = model_cfg.get("mf")

    # n_epochs = training_cfg.get("n_epochs")
    lr = training_cfg.get("lr")
    k = training_cfg.get("k")
    pcd = training_cfg.get("pcd")
    sm = training_cfg.get("sm")
    mc = training_cfg.get("mc")
    epsilon = training_cfg.get("epsilon")

    out_dir = output_cfg.get("base_dir")
    run_name = output_cfg.get("run_name")

    # Print configuration summary.
    cfg.print_cfg_summary(config)
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
    n_epochs = n_epochs_arg  # Overwrite n_epochs with how many epochs to resume training.
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
            "data_type": data_type,
            "data_dir": data_dir,
            "batch_size": batch_size,
            "split": split,
            "binarize": binarize,
            "q": q,
            "T": T,
            "L": L,
        },
        "model": {
            "model_type": model_type,
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