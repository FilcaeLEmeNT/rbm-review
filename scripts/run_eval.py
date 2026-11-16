#!/usr/bin/env python3
import argparse
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import json
import pickle

from utils.device import get_device
import utils.config as cfg
import utils.sweep as swp
from utils.checkpoint import load_checkpoint

from data.data_loader import load_data

from utils.multinomial import onehot_to_categories, categories_to_grayscale
import utils.physics as physics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RBM model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )
    group.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Path to sweep configuration file",
    )

    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Set n_samples to specify which sample to use. If k_gen set to None, select the most recent sample with n_samples.",
    )

    parser.add_argument(
        "--k_gen",
        type=int,
        default=None,
        help="Set k_gen to specify which sample to use. If n_samples set to None, select the most recent sample with k_gen.",
    )

    parser.add_argument(
        "--ph",
        type=float,
        default=None,
        help="Select samples generated with this hidden-unit initialization probability.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Use the checkpoint saved after this epoch with --config or --sweep.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.sweep and not args.config:
        argparse.error("--config is required when using --sweep")

    if args.checkpoint and args.config:
        print("Warning: --config is ignored when --checkpoint is provided")

    # Load device: Either CPU or CUDA
    device = get_device(verbose=False)

    if args.sweep:
        config = cfg.load_config(args.config)
        sweep = cfg.load_config(args.sweep)
        trial_groups = swp.iter_sweep_configs(config, sweep)

        for trial_group in trial_groups:
            for trial_config in trial_group["configs"]:
                ckpt_path = cfg.get_checkpoint_path_from_config(
                    trial_config, args.epoch
                )
                run_eval(
                    device,
                    ckpt_path,
                    args.n_samples,
                    args.k_gen,
                    ph=args.ph,
                    config=trial_config,
                    epoch=args.epoch,
                )

    elif args.checkpoint:
        # Get checkpoint path
        ckpt_path = args.checkpoint
        run_eval(
            device,
            ckpt_path,
            args.n_samples,
            args.k_gen,
            ph=args.ph,
            epoch=args.epoch,
        )

    elif args.config:
        config = cfg.load_config(args.config)
        ckpt_path = cfg.get_checkpoint_path_from_config(config, args.epoch)
        run_eval(
            device,
            ckpt_path,
            args.n_samples,
            args.k_gen,
            ph=args.ph,
            config=config,
            epoch=args.epoch,
        )

    return


def run_eval(
    device,
    ckpt_path,
    n_samples,
    k_gen,
    config=None,
    epoch=None,
    ph=None,
):
    ckpt = load_checkpoint(ckpt_path, device, load_model=False)

    if ckpt is None:
        raise ValueError(
            f"The checkpoint file was not loaded from the checkpoint file properly."
        )

    checkpoint_config = ckpt.get("config", {}) or {}
    effective_config = dict(checkpoint_config)
    if isinstance(config, dict):
        output_override = config.get("output", {})
        if isinstance(output_override, dict):
            effective_config["output"] = {
                **effective_config.get("output", {}),
                **output_override,
            }

    # Get values from configuration.
    data_cfg = effective_config.get("data", {})
    model_cfg = effective_config.get("model", {})
    output_cfg = effective_config.get("output", {})

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

    out_dir = output_cfg.get("base_dir")
    run_name = output_cfg.get("run_name")

    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    # Load test data.
    train_loader, test_loader = load_data(data_cfg, model_type)

    if len(test_loader) == 0:
        if len(train_loader) == 0:
            raise ValueError(
                "Training and test loaders are both empty; check the dataset split configuration."
            )
        print(
            f"Warning: test split is empty for split={data_cfg.get('split')}. Using the train_loader as test batch."
        )
        batch = next(iter(train_loader))
        if isinstance(batch, (list, tuple)):
            X_test = torch.cat([x for x, *_ in train_loader], dim=0)
        else:
            X_test = torch.cat([x for x in train_loader], dim=0)
    else:
        batch = next(iter(test_loader))
        if isinstance(batch, (list, tuple)):
            X_test = torch.cat([x for x, *_ in test_loader], dim=0)
        else:
            X_test = torch.cat([x for x in test_loader], dim=0)

    if model_type == "multinomial":
        X_test = X_test.to(device).float().view(-1, n_visible * n_class)
    else:
        X_test = X_test.to(device).float().view(-1, n_visible)

    batch = next(iter(train_loader))
    if isinstance(batch, (list, tuple)):
        X_train = torch.cat([x for x, *_ in train_loader], dim=0)
    else:
        X_train = torch.cat([x for x in train_loader], dim=0)

    if model_type == "multinomial":
        X_train = X_train.to(device).float().view(-1, n_visible * n_class)
    else:
        X_train = X_train.to(device).float().view(-1, n_visible)

    # Get directories
    paths = cfg.get_output_paths(out_dir, run_name)
    samples_dir = paths["samples"]
    physics_dir = paths["physics"]
    os.makedirs(physics_dir, exist_ok=True)

    # Get samples using metadata. Import as torch tensor for consistency with X_test.
    meta_path = path.join(samples_dir, f"metadata.json")
    with open(meta_path) as f:
        all_meta = json.load(f)

    # Filter by args, fall back to most recent. Only honor the optional
    # epoch suffix when the caller explicitly supplied --epoch.
    matches = [
        m
        for m in all_meta
        if (n_samples is None or m["n_samples"] == n_samples)
        and (k_gen is None or m["k_gen"] == k_gen)
        and (ph is None or m.get("ph") == ph)
        and (
            epoch is None
            or (
                m.get("recon_file", "").endswith(f"{suffix}.npy")
                and m.get("gen_file", "").endswith(f"{suffix}.npy")
            )
        )
    ]

    if not matches:
        raise FileNotFoundError(
            f"No samples found for n_samples={n_samples}, k_gen={k_gen}, ph={ph}"
        )

    meta = matches[-1]  # most recent match

    n_samples = meta["n_samples"]
    k_gen = meta["k_gen"]

    X_recon = np.load(
        path.join(samples_dir, meta["recon_file"]), allow_pickle=True
    )
    X_gen = np.load(path.join(samples_dir, meta["gen_file"]), allow_pickle=True)
    X_recon = torch.tensor(X_recon)
    X_gen = torch.tensor(X_gen)

    # If model type is multinomial, extract categories from OneHot encoded data.
    # If image convert to grayscale in [0, 1].
    if model_type == "multinomial":
        X_test = onehot_to_categories(X_test, n_visible, n_class)
        X_train = onehot_to_categories(X_train, n_visible, n_class)
        X_recon = onehot_to_categories(X_recon, n_visible, n_class)
        X_gen = onehot_to_categories(X_gen, n_visible, n_class)

        if data_type in ["mnist", "cifar10", "stl10"]:
            X_test = categories_to_grayscale(X_test, n_class)
            X_train = categories_to_grayscale(X_train, n_class)
            X_recon = categories_to_grayscale(X_recon, n_class)
            X_gen = categories_to_grayscale(X_gen, n_class)

    # Wrap angles if model is VonMises
    if model_type == "vonmises":
        X_test = wrap(X_test, -np.pi, np.pi)
        X_recon = wrap(X_recon, -np.pi, np.pi)
        X_gen = wrap(X_gen, -np.pi, np.pi)

    """
    Calculate physical properties: Ising, Potts, XY
    """
    if data_type in ["ising", "xy"] or data_type.startswith("potts"):
        obs_train = physics.compute_observables(
            X_train, data_type, float(T), q=q
        )
        obs_test = physics.compute_observables(X_test, data_type, float(T), q=q)
        obs_recon = physics.compute_observables(
            X_recon, data_type, float(T), q=q
        )
        obs_gen = physics.compute_observables(X_gen, data_type, float(T), q=q)

        # Print Physical Properties as a table
        headers = ["Category", "E/N", "C/N", "M/N", "Chi/N"]
        data = [
            [
                "Test",
                obs_test.energy_mean_per_site,
                obs_test.heat_capacity_per_site,
                obs_test.magnetization_mean_per_site,
                obs_test.susceptibility_per_site,
            ],
            [
                "Recon",
                obs_recon.energy_mean_per_site,
                obs_recon.heat_capacity_per_site,
                obs_recon.magnetization_mean_per_site,
                obs_recon.susceptibility_per_site,
            ],
            [
                "Gen",
                obs_gen.energy_mean_per_site,
                obs_gen.heat_capacity_per_site,
                obs_gen.magnetization_mean_per_site,
                obs_gen.susceptibility_per_site,
            ],
        ]

        print("")
        print(
            f"{headers[0]:<10} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15}"
        )
        print("-" * 70)
        for row in data:
            print(
                f"{row[0]:<10} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15}"
            )
        # Save Observables to a dictionary for later use
        pickle_path = path.join(
            physics_dir,
            f"physical_properties{suffix}.pkl",
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(
                {
                    "T": T,
                    "observables_train": obs_train,
                    "observables_test": obs_test,
                    "observables_recon": obs_recon,
                    "observables_gen": obs_gen,
                },
                f,
            )

        print(
            f"File, 'physical_properties{suffix}.pkl', saved to {physics_dir}"
        )

        """
        # Save a dictionary
        np.savez(
            path.join(physics_dir, 'physical_properties.npz'),
            T=T,
            observables_train=obs_train,
            observables_test=obs_test,
            observables_recon=obs_recon,
            observables_gen=obs_gen,
        )
        print(f"File, 'physical_properties.npz', saved to {physics_dir}")
        """

    return


"""
    energy_test=obs_test.energy_mean_per_site,
    energy_recon=obs_recon.energy_mean_per_site,
    energy_gen=obs_gen.energy_mean_per_site,
    magnetization_test=obs_test.magnetization_mean_per_site,
    magnetization_recon=obs_recon.magnetization_mean_per_site,
    magnetization_gen=obs_gen.magnetization_mean_per_site,
    heat_capacity_test=obs_test.heat_capacity_per_site,
    heat_capacity_recon=obs_recon.heat_capacity_per_site,
    heat_capacity_gen=obs_gen.heat_capacity_per_site,
    susceptibility_test=obs_test.susceptibility_per_site,
    susceptibility_recon=obs_recon.susceptibility_per_site,
    susceptibility_gen=obs_gen.susceptibility_per_site,
    dominant_entropy_test=obs_test.dominant_entropy,
    dominant_entropy_recon=obs_recon.dominant_entropy,
    dominant_entropy_gen=obs_gen.dominant_entropy,
"""


def wrap(array, minim=-np.pi, maxim=np.pi):
    """Wrap angles in array to the range [minim, maxim).

    Args:
        array: Tensor of angles to wrap
        minim: Minimum angle (default -π)
        maxim: Maximum angle (default π).

    Returns:
        Wrapped angles in the range [minim, maxim)
    """
    width = maxim - minim  # = 2π
    return (array - minim) % width + minim


if __name__ == "__main__":
    main()
