import argparse
import os
from os import path
import numpy as np
import math

from utils.device import get_device
from utils.config import load_config

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train RBM model")

    parser.add_argument(
        "--config",
        type=str,
        default=path.join("configs", "default.yaml"),
        help="Path to config file"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load device: Either CPU or CUDA
    device = get_device()

    # Load config
    config = load_config(args.config)
    print(f"Using config file: {args.config}")

    if "data" not in config:
        config["data"] = {}
    data_type = config["data"]["type"] if "type" in config["data"] else None
    data_dir = config["data"]["data_dir"] if "data_dir" in config["data"] else None
    batch_size = config["data"]["batch_size"] if "batch_size" in config["data"] else None
    split = config["data"]["split"] if "split" in config["data"] else None
    binarize = config["data"]["binarize"] if "binarize" in config["data"] else False
    q = config["data"]["q"] if "q" in config["data"] else None
    T = config["data"]["T"] if "T" in config["data"] else None
    L = config["data"]["L"] if "L" in config["data"] else None

    if "model" not in config:
        config["model"] = {}
    model_type = config["model"]["type"] if "type" in config["model"] else None
    n_visible = config["model"]["n_visible"] if "n_visible" in config["model"] else None
    n_hidden = config["model"]["n_hidden"] if "n_hidden" in config["model"] else None
    mf = config["model"]["mf"] if "mf" in config["model"] else None

    if "training" not in config:
        config["training"] = {}
    n_epochs = config["training"]["n_epochs"] if "n_epochs" in config["training"] else None
    lr = config["training"]["lr"] if "lr" in config["training"] else None
    k = config["training"]["k"] if "k" in config["training"] else None
    pcd = config["training"]["pcd"] if "pcd" in config["training"] else None
    sm = config["training"]["sm"] if "sm" in config["training"] else None
    mc = config["training"]["mc"] if "mc" in config["training"] else None
    epsilon = config["training"]["epsilon"] if "epsilon" in config["training"] else None

    if "output" not in config:
        config["output"] = {}
    out_dir = config["output"]["base_dir"] if "base_dir" in config["output"] else None
    run_name = config["output"]["run_name"] if "run_name" in config["output"] else None

    # Print config summary
    print("Config summary:")
    print("Data parameters:")
    print(f"\ttype={data_type}", f"data_dir={data_dir}", f"batch_size={batch_size}", f"split={split}", f"binarize={binarize}", f"q={q}", f"T={T}", f"L={L}", sep="\n\t")
    print("Model parameters:")
    print(f"\ttype={model_type}", f"n_visible={n_visible}", f"n_hidden={n_hidden}", f"mf={mf}", sep="\n\t")
    print("Training parameters:")
    print(f"\tn_epochs={n_epochs}", f"lr={lr}", f"k={k}", f"pcd={pcd}", f"sm={sm}", f"mc={mc}", f"epsilon={epsilon}", sep="\n\t")
    print(f"Output directory: {out_dir}")
    print(f"Run Name: {run_name}")
    print("")
    
    # Load data: None variables are handled within the function.
    train_loader, test_loader = load_data(data_type, data_dir, split, q, T, L, batch_size, binarize=binarize)
    
    # Check if n_visible is set in config, if not infer from data. If set, check if it matches the data.

    # Get a batch in data
    batch_data = next(iter(test_loader))
    X_batch = batch_data[0] if isinstance(batch_data, list) else batch_data

    if n_visible is None:
        n_visible = X_batch.shape[1]
        print(f"\033[1mmodel.n_visible not specified in config.yaml. Inferred n_visible = {n_visible} from the data.\033[0m")
    else:
        if n_visible != X_batch.shape[1]:
            raise ValueError(f"n_visible in config ({n_visible}) does not match the size of the input data ({X_batch.shape[1]}). Please update config.yaml.")
    
    # Check if n_hidden is set in config, if not default to n_visible // 2 
    if n_hidden is None:
        n_hidden = 2 ** math.floor(math.log2(n_visible // 2))  # Default to close to half the number of visible units if not specified
        print(f"\033[1mmodel.n_hidden not specified in config. Defaulting to n_hidden = {n_hidden}.\033[0m")
        if n_hidden <= 0:
            raise ValueError(f"n_hidden infered from n_visible, n_hidden = {n_hidden}, is invalid. Please specify model.n_hidden in config.yaml")
    elif not (n_hidden > 0 and (n_hidden & (n_hidden - 1)) == 0):  # Check if power of 2
        raise ValueError(f"model.n_hidden must be a power of 2. Value specified is n_hidden={n_hidden}. Please update config.yaml.")

    # Initialize model    
    if model_type is None:
        raise ValueError("model.type must be specified in config.yaml. Please update config.yaml.")
    print(f"Using model type: {model_type}")

    if model_type != "binary" and mf is not None:  # Ensure checkpoint loading works properly.
        raise ValueError(f"Do not set model.mf, which is only available for binary rbms. Please update config.yaml.")

    if model_type == "binary":
        if mf is None:
            mf = True
            print(f"\033[1mmodel.mf not specified in config. Defaulting to n_hidden = {n_epochs}.\033[0m")
        print(f"Using mean-field: {mf}")
        print(f"Using binarize: {binarize}")
        from models.rbm_binary import RBM_binary
        rbm = RBM_binary(n_visible, n_hidden, mf=mf).to(device)
    elif model_type == "exponential":
        from models.rbm_exponential import RBM_exponential
        rbm = RBM_exponential(n_visible, n_hidden).to(device)
    elif model_type == "gaussian":
        from models.rbm_gaussian import RBM_gaussian
        rbm = RBM_gaussian(n_visible, n_hidden).to(device)
    elif model_type == "vonmises":
        from models.rbm_vonmises import RBM_vonmises
        rbm = RBM_vonmises(n_visible, n_hidden).to(device)
    elif model_type == "multinomial":
        from models.rbm_multinomial import RBM_multinomial
        rbm = RBM_multinomial(q, n_visible, n_hidden).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please update config.yaml with a valid model type.")
    
    # Check training parameters
    if n_epochs is None:
        n_epochs = 500
        print(f"\033[1mtraining.n_epochs not specified in config. Defaulting to n_epochs = {n_epochs}.\033[0m")
    
    if lr is None:
        lr = 0.01
        print(f"\033[1mtraining.lr not specified in config. Defaulting to lr = {lr}.\033[0m")

    if k is None:
        k = 10
        print(f"\033[1mtraining.k not specified in config. Defaulting to k = {k}.\033[0m")

    if pcd is None:
        pcd = True
        print(f"\033[1mtraining.pcd not specified in config. Defaulting to pcd = {pcd}.\033[0m")

    if sm is None:
        sm = False
        print(f"\033[1mtraining.sm not specified in config. Defaulting to sm = {sm}.\033[0m")

    if mc is None:
        mc = "gibbs"
        print(f"\033[1mtraining.mc not specified in config. Defaulting to mc = {mc}.\033[0m")
    elif mc != 'gibbs' or mc != 'gibbs':
        raise ValueError("sm needs to be either 'gibbs' or 'langevin'. Please update config.")

    if epsilon is None:
        epsilon = 1e-5
        print(f"\033[1mtraining.epsilon not specified in config. Defaulting to epsilon = {epsilon}.\033[0m")

    if sm == True and not model_type == "gaussian":
        sm = False
        print(f"\033[1mScore Matching is only available for Gaussian RBMs. Defaulting to sm = {sm}\033[0m")

    # Before Training the model, ensure output directory and run name is specified.
    # If unspecified, ask user if to run training anyways without an output.
    if (out_dir is None or run_name is None):
        print(f"\noutput.base_dir and/or output.run_name is unspecified in config.")
        print(f"There will be no outputs upon training.")
        while True:
            choice = input("Do you still want to run the training? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                # Logic for "yes"
                print("Continuing...")
                break
            elif choice in ['n', 'no']:
                # Logic for "no"
                print("Exiting...")
                return
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    # Train the model
    if sm == True and model_type == "gaussian":
        history = train_sm(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)
    else:
        history = train_cd(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)

    if (out_dir is None or run_name is None):
        return
    
    '''
    Specfiy output directory and the directory name.
    Resulting file structure will be:
    ├── out_dir
        ├── checkpoints
        |   └── dir_name
        ├── figures
        |   └── dir_name
        └── history
            └── dir_name
    '''
    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)
    figures_dir = path.join(out_dir, "figures", run_name)
    history_dir = path.join(out_dir, "history", run_name)
    samples_dir = path.join(out_dir, "samples", run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    print("")

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
    print(f"Checkpoint file, 'checkpoint.pt', saved to directory: {checkpoints_dir}")

    return

if __name__ == "__main__":
    main()