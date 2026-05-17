import argparse
import os
from os import path
import numpy as np
import torch
import json
from datetime import datetime

from utils.device import get_device
from utils.config import load_config

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import load_checkpoint, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from the RBM model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=path.join("outputs", "checkpoints", "default_run", "checkpoint.pt"),
        help="Path to checkpoint file"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=8192,
        help="Number of samples to generate. Overwrites and updates config's value in checkpoint.pt"
    )

    parser.add_argument(
        "--k_gen",
        type=int,
        default=1000,
        help="Number of steps to generate the samples. Overwrites and updates config's value in checkpoint.pt"
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

    if "data" not in config:
        config["data"] = {}
    data_type = config["data"]["type"]
    data_dir = config["data"]["data_dir"]
    batch_size = config["data"]["batch_size"]
    split = config["data"]["split"]
    binarize = config["data"]["binarize"]
    q = config["data"]["q"]
    T = config["data"]["T"]
    L = config["data"]["L"]

    model_type = config["model"]["type"]
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

    n_samples = args.n_samples
    k_gen = args.k_gen

    # Get directories
    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)
    figures_dir = path.join(out_dir, "figures", run_name)
    history_dir = path.join(out_dir, "history", run_name)
    samples_dir = path.join(out_dir, "samples", run_name)
    
    # Load dataset
    train_loader, test_loader = load_data(data_type, data_dir, split, q, T, L, batch_size, binarize=binarize)
    
    '''
    Reconstruction
    '''
    # Reconstruction Using the entire test dataset
    batch = next(iter(test_loader))
    if isinstance(batch, (list, tuple)):
        X_test = torch.cat([x for x, *_ in test_loader], dim=0)
    else:
        X_test = torch.cat([x for x in test_loader], dim=0)

    X_test = X_test.to(device).float().view(-1, n_visible)

    # Use Gibbs sampling for generation
    with torch.no_grad():  # no graph, avoid GPU memory growth
        X_recon = rbm.forward(X_test, mc='gibbs', k=1).detach() 

    # Save Reconstructed values as a file in samples directory
    np.save(path.join(samples_dir, f"recon.npy"), X_recon.cpu())
    print(f"Numpy file, 'recon.npy', saved to {samples_dir}")

    '''
    Generation
    '''
    # Generate starting from a random pixel distribution or a random hidden unit distribution.
    # Commented out lines are for different initializations.

    ph = 0.3  # Bernoulli probability
    h0 = torch.bernoulli(torch.ones(n_samples, n_hidden)*ph).float().to(device)  # bernoulli 0,1 with probability ph
    X0 = rbm.h_to_v(h0)
    # X0 = torch.randint(1, (n_samples, n_visible),requires_grad=False).float().to(device) # binary 0,1
    # X0 = torch.rand((n_samples, n_visible),requires_grad=False).float().to(device) # uniform [0,1]

    # Use Gibbs or Langevin for generation
    if mc == 'gibbs':
        with torch.no_grad():  # no graph, avoid GPU memory growth
            X_gen = rbm.forward(X0, mc='gibbs', k=k_gen).detach()  # gibbs or manual diff langevin does not need graph
    else:
        X_gen = rbm.forward(X0, mc='langevin', k=k_gen, epsilon=epsilon).detach()  # langevin autodiff must use graph, GPU memory cautious
    # X_gen = X_gen.clamp(0,1) 

    # Save Reconstructed values as a file in samples directory
    np.save(path.join(samples_dir, f"gen_n{n_samples}_k{k_gen}.npy"), X_gen.cpu())
    print(f"Numpy file, 'gen_n{n_samples}_k{k_gen}.npy', saved to {samples_dir}")

    '''
    Create metadata for run_figures.py
    '''
    meta_path = path.join(samples_dir, f"metadata.json")

    existing = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
    
    existing.append({
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "k_gen":     args.k_gen,
        "recon_file": f"recon.npy",
        "gen_file":  f"gen_n{n_samples}_k{k_gen}.npy",
        "timestamp": datetime.now().isoformat(),
    })
    
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)

    return

if __name__ == "__main__":
    main()