import argparse
import os
from os import path
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

from utils.device import get_device
from utils.config import load_config

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from the RBM model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=path.join("outputs", "checkpoints", "default_run", "checkpoint.pt"),
        help="Path to checkpoint file"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load device: Either CPU or CUDA
    device = get_device()

    # Get loaded model and checkpoint
    ckpt_path = args.checkpoint
    rbm, ckpt =  load_checkpoint(ckpt_path, device)

    if rbm is None:
        raise ValueError(f"The model was not loaded from the checkpoint file properly.")
    if ckpt is None:
        raise ValueError(f"The checkpoint file was not loaded from the checkpoint file properly.")

    # Get config file from checkpoint
    config = ckpt["config"]

    if "data" not in config:
        config["data"] = {}
    data_type = config["data"]["type"] if "type" in config["data"] else None
    data_dir = config["data"]["data_dir"] if "data_dir" in config["data"] else None
    data_filename = config["data"]["data_filename"] if "data_filename" in config["data"] else None
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

    if "training" not in config:
        config["training"] = {}
    n_epochs = config["training"]["n_epochs"] if "n_epochs" in config["training"] else None
    lr = config["training"]["lr"] if "lr" in config["training"] else None
    k = config["training"]["k"] if "k" in config["training"] else None
    pcd = config["training"]["pcd"] if "pcd" in config["training"] else None
    sm = config["training"]["sm"] if "sm" in config["training"] else None
    mf = config["training"]["mf"] if "mf" in config["training"] else None
    mc = config["training"]["mc"] if "mc" in config["training"] else None
    epsilon = config["training"]["epsilon"] if "epsilon" in config["training"] else None
    
    if "output" not in config:
        config["output"] = {}
    out_dir = config["output"]["base_dir"] if "base_dir" in config["output"] else None
    run_name = config["output"]["run_name"] if "run_name" in config["output"] else None

    if "sampling" not in config:
        config["sampling"] = {}
    n_sample = config["sampling"]["n_sample"] if "n_sample" in config["sampling"] else None
    k_gen = config["sampling"]["k_gen"] if "k_gen" in config["sampling"] else None

    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)
    figures_dir = path.join(out_dir, "figures", run_name)
    history_dir = path.join(out_dir, "history", run_name)
    samples_dir = path.join(out_dir, "samples", run_name)
    
    # Load dataset
    train_loader, test_loader = load_data(data_type, data_dir, data_filename, split, q, T, L, batch_size, binarize=binarize)
    
    '''
    Reconstruction
    '''
    # Reconstruction Using the entire test dataset
    X_test = test_loader.dataset.data
    X_test = X_test.to(device).float().view(-1, n_visible)
    X_test.shape

    # Use Gibbs sampling for generation
    with torch.no_grad():  # no graph, avoid GPU memory growth
        X_recon = rbm.forward(X_test, mc='gibbs', k=1).detach() 

    # Save Reconstructed values as a file in samples directory
    np.save(path.join(samples_dir, "recon.npy"), X_recon.cpu())
    print(f"Numpy file, 'recon.npy', saved to {samples_dir}")

    '''
    Generation
    '''

    if n_sample is None:
        n_sample = 1024
        print(f"\033[1msampling.n_sample not specified in config. Defaulting to n_sample = {n_sample}.\033[0m")
    
    if k_gen is None:
        k_gen = 1000
        print(f"\033[1msampling.k_gen not specified in config. Defaulting to k_gen = {k_gen}.\033[0m")

    # Generate starting from a random pixel distribution or a random hidden unit distribution.
    # Commented out lines are for different initializations.

    ph = 0.3  # Bernoulli probability
    h0 = torch.bernoulli(torch.ones(n_sample, n_hidden)*ph).float().to(device)  # bernoulli 0,1 with probability ph
    X0 = rbm.h_to_v(h0)

    # X0 = torch.randint(1, (n_sample, n_visible),requires_grad=False).float().to(device) # binary 0,1
    # X0 = torch.rand((n_sample, n_visible),requires_grad=False).float().to(device) # uniform [0,1]

    # Use Gibbs or Langevin for generation

    if mc == 'gibbs':
        with torch.no_grad():  # no graph, avoid GPU memory growth
            X_gen = rbm.forward(X0, mc='gibbs', k=k_gen).detach()  # gibbs or manual diff langevin does not need graph
    else:
        X_gen = rbm.forward(X0, mc='langevin', k=k_gen, epsilon=epsilon).detach()  # langevin autodiff must use graph, GPU memory cautious
    # X_gen = X_gen.clamp(0,1) 

    # Save Reconstructed values as a file in samples directory
    np.save(path.join(samples_dir, f"gen_n{n_sample}_k{k_gen}"), X_gen.cpu())
    print(f"Numpy file, 'gen_n{n_sample}_k{k_gen}', saved to {samples_dir}")

    return

if __name__ == "__main__":
    main()