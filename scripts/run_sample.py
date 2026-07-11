#!/usr/bin/env python3
import argparse
import os
from os import path
import numpy as np
import torch
import json
from datetime import datetime

from utils.device import get_device
import utils.config as cfg
import utils.sweep as swp
from utils.checkpoint import load_checkpoint
import utils.multinomial as multinomial

from data.data_loader import load_data

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from the RBM model")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--checkpoint", type=str, default=None,
        help="Path to checkpoint file"
    )
    group.add_argument("--sweep", type=str, default=None,
        help="Path to sweep configuration file"
    )

    parser.add_argument("--config", type=str, default=None,
        help="Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. \
            Use as an alternative to --checkpoint, or combine with --sweep to sample from all runs generated from that config/sweep combination."
    )

    parser.add_argument("--n_samples", type=int, default=8192,
        help="Number of samples to generate."
    )

    parser.add_argument("--k_gen", type=int, default=1000,
        help="Number of steps to generate the samples."
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

        failed_runs = []
        for ckpt_path in ckpt_paths:
            try:
                run_sample(device, ckpt_path, args.n_samples, args.k_gen)
            except Exception as e:
                failed_runs.append({"error": str(e)})
                print(f"Run failed. Error: {e}")
        
        if len(failed_runs) > 0:
            print("Failed runs in sweep:")

        for failed_run in failed_runs:
            print("\t", "Error: ", failed_run["error"])

    elif args.checkpoint:
        # Get checkpoint path
        ckpt_path = args.checkpoint
        run_sample(device, ckpt_path, args.n_samples, args.k_gen)

    elif args.config:
        # Get checkpoint path
        config = cfg.load_config(args.config)
        ckpt_path = cfg.get_checkpoint_path_from_config(config)
        run_sample(device, ckpt_path, args.n_samples, args.k_gen)
    
    else:
        argparse.error("Atleast one of --checkpoint, --config, and --sweep must be used.")

    return

def run_sample(device, ckpt_path, n_samples, k_gen):
    model, ckpt = load_checkpoint(ckpt_path, device)

    if model is None:
        raise ValueError(f"The model was not loaded from the checkpoint file properly.")
    if ckpt is None:
        raise ValueError(f"The checkpoint file was not loaded from the checkpoint file properly.")

    # Get config file from checkpoint and unpack values
    config = ckpt["config"]

    # Get values from configuration.
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    model_type = model_cfg.get("model_type")
    n_class = model_cfg.get("n_class")
    n_visible = model_cfg.get("n_visible")
    n_hidden = model_cfg.get("n_hidden")
    mf = model_cfg.get("mf")

    n_epochs = training_cfg.get("n_epochs")
    lr = training_cfg.get("lr")
    k = training_cfg.get("k")
    pcd = training_cfg.get("pcd")
    sm = training_cfg.get("sm")
    mc = training_cfg.get("mc")
    epsilon = training_cfg.get("epsilon")

    out_dir = output_cfg.get("base_dir")
    run_name = output_cfg.get("run_name")

    # Get directories
    paths = cfg.get_output_paths(out_dir, run_name)
    samples_dir = paths["samples"]
    os.makedirs(samples_dir, exist_ok=True)    

    # Load dataset
    train_loader, test_loader = load_data(data_cfg, model_type)
    
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
        X_recon = model.forward(X_test, mc='gibbs', k=1).detach() 

    # Save Reconstructed values as a file in samples directory
    np.save(path.join(samples_dir, f"recon.npy"), X_recon.cpu())
    print(f"Numpy file, 'recon.npy', saved to {samples_dir}")

    '''
    Generation
    '''
    # Generate starting from a random pixel distribution or a random hidden unit distribution.
    # Commented out lines are for different initializations.

    ph = 0.30  # Bernoulli probability
    h0 = torch.bernoulli(torch.ones(n_samples, n_hidden)*ph).float().to(device)  # bernoulli 0,1 with probability ph
    X0 = model.h_to_v(h0)
    
    # X0 = torch.randint(1, (n_samples, n_visible),requires_grad=False).float().to(device) # binary 0,1
    # X0 = torch.rand((n_samples, n_visible),requires_grad=False).float().to(device) # uniform [0,1]

    # Use Gibbs or Langevin for generation
    if mc == 'gibbs':
        with torch.no_grad():  # no graph, avoid GPU memory growth
            X_gen = model.forward(X0, mc='gibbs', k=k_gen).detach()  # gibbs or manual diff langevin does not need graph
    else:
        X_gen = model.forward(X0, mc='langevin', k=k_gen, epsilon=epsilon).detach()  # langevin autodiff must use graph, GPU memory cautious
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
        "ckpt_path": ckpt_path,
        "n_samples": n_samples,
        "k_gen":     k_gen,
        "recon_file": f"recon.npy",
        "gen_file":  f"gen_n{n_samples}_k{k_gen}.npy",
        "timestamp": datetime.now().isoformat(),
    })
    
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)

    return

if __name__ == "__main__":
    main()