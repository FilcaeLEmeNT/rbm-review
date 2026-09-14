#!/usr/bin/env python3
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import os
from os import path
import numpy as np
import math
import random
import torch
import json

from PIL import Image

from utils.device import get_device
import utils.config as cfg
import utils.sweep as swp
from utils.checkpoint import load_checkpoint

from data.data_loader import load_data

from utils.multinomial import onehot_to_categories, categories_to_grayscale
import utils.physics as physics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RBM model")

    group = parser.add_mutually_exclusive_group(required=False)
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
        "--config",
        type=str,
        default=None,
        help="Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. \
            Use as an alternative to --checkpoint, or combine with --sweep to create figures from all runs generated from that config/sweep combination.",
    )

    parser.add_argument(
        "--accumulate-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to skip runs with errors and accumulate error messages when running sweep.",
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
        "--epoch",
        type=int,
        default=None,
        help="Use the checkpoint saved after this epoch with --config or --sweep.",
    )

    parser.add_argument(
        "--skip-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip figures of weights.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check args
    if args.sweep and not args.config:
        argparse.error("--config is required when using --sweep")

    if args.checkpoint and args.config:
        print(
            "Warning: --config is ignored when --checkpoint is provide. Configuration file embedded in the checkpoint is used."
        )

    # Load device: Either CPU or CUDA
    device = get_device()

    if args.sweep:
        config = cfg.load_config(args.config)
        sweep = cfg.load_config(args.sweep)
        trial_groups = swp.iter_sweep_configs(config, sweep)

        failed_runs = []
        for trial_group in trial_groups:
            for trial_config in trial_group["configs"]:
                ckpt_path = cfg.get_checkpoint_path_from_config(
                    trial_config, args.epoch
                )
                if args.accumulate_errors:
                    try:
                        run_figures(
                            device,
                            ckpt_path,
                            args.n_samples,
                            args.k_gen,
                            args.skip_weights,
                            config=trial_config,
                            epoch=args.epoch,
                        )
                    except Exception as e:
                        failed_runs.append({"error": str(e)})
                        print(f"Run failed. Error: {e}")
                else:
                    run_figures(
                        device,
                        ckpt_path,
                        args.n_samples,
                        args.k_gen,
                        args.skip_weights,
                        config=trial_config,
                        epoch=args.epoch,
                    )

        if len(failed_runs) > 0:
            print("Failed runs in sweep:")
            for failed_run in failed_runs:
                print("\t", "Error: ", failed_run["error"])

    elif args.checkpoint:
        # Get checkpoint path
        ckpt_path = args.checkpoint
        run_figures(
            device,
            ckpt_path,
            args.n_samples,
            args.k_gen,
            args.skip_weights,
            epoch=args.epoch,
        )

    elif args.config:
        # Get checkpoint path
        config = cfg.load_config(args.config)
        ckpt_path = cfg.get_checkpoint_path_from_config(config, args.epoch)
        run_figures(
            device,
            ckpt_path,
            args.n_samples,
            args.k_gen,
            args.skip_weights,
            config=config,
            epoch=args.epoch,
        )

    else:
        argparse.error(
            "Atleast one of --checkpoint, --config, and --sweep must be used."
        )

    return


def run_figures(
    device,
    ckpt_path,
    n_samples,
    k_gen,
    skip_weights,
    config=None,
    epoch=None,
):
    model, ckpt = load_checkpoint(ckpt_path, device)

    if model is None:
        raise ValueError(
            f"The model was not loaded from the checkpoint file properly."
        )
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
    training_cfg = effective_config.get("training", {})
    output_cfg = effective_config.get("output", {})

    data_type = data_cfg.get("data_type")
    batch_size = data_cfg.get("batch_size")
    q = data_cfg.get("q")
    T = data_cfg.get("T")
    L = data_cfg.get("L")

    model_type = model_cfg.get("model_type")
    n_class = model_cfg.get("n_class")
    n_visible = model_cfg.get("n_visible")
    n_hidden = model_cfg.get("n_hidden")
    mf = model_cfg.get("mf")

    negative_phase_method = training_cfg.get("negative_phase_method")

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

    # Get directories
    paths = cfg.get_output_paths(out_dir, run_name)
    samples_dir = paths["samples"]
    figures_dir = paths["figures"]
    os.makedirs(figures_dir, exist_ok=True)

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
            f"No samples found for n_samples={n_samples}, k_gen={k_gen}"
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

    #  Get hidden unites corresponding to compare test vs recon. Needs to be done before any transformations for multinomial RBM
    with torch.no_grad():
        h_test = model.v_to_h(X_test.float().to(device)).detach()
        h_recon = model.v_to_h(X_recon.float().to(device)).detach()

    # Ensure image tensors are on CPU for plotting to avoid holding GPU memory
    try:
        X_test = X_test.cpu()
        X_recon = X_recon.cpu()
        X_gen = X_gen.cpu()
        torch.cuda.empty_cache()
    except Exception:
        pass

    # If model type is multinomial, extract categories from OneHot encoded data.
    # If image convert to grayscale in [0, 1].
    if model_type == "multinomial":
        X_test = onehot_to_categories(X_test, n_visible, n_class)
        X_recon = onehot_to_categories(X_recon, n_visible, n_class)
        X_gen = onehot_to_categories(X_gen, n_visible, n_class)

        if data_type in ["mnist", "cifar10", "stl10"]:
            X_test = categories_to_grayscale(X_test, n_class)
            X_recon = categories_to_grayscale(X_recon, n_class)
            X_gen = categories_to_grayscale(X_gen, n_class)

    # Ensure image tensors are on CPU for plotting to avoid holding GPU memory
    try:
        X_test = X_test.cpu()
        X_recon = X_recon.cpu()
        X_gen = X_gen.cpu()
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Wrap angles if model is VonMises
    if model_type == "vonmises":
        X_test = wrap(X_test, -np.pi, np.pi)
        X_recon = wrap(X_recon, -np.pi, np.pi)
        X_gen = wrap(X_gen, -np.pi, np.pi)

    """
    Make plots: History
    """
    # Get history from checkpoint and Plot curves
    history = ckpt["history"]
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

    axes[0].plot(history["E_data"], "rx-")
    axes[0].plot(history["E_model"], "bx-")

    axes[1].plot(history["E_diff"], "gx-")
    axes[2].plot(history["mse"], "kx-")
    axes[3].plot(history["loss"], "kx-")

    axes[0].set_title("visible energy: data/model")
    axes[1].set_title("visible energy diff")
    axes[2].set_title("reconstruction mse")
    axes[3].set_title("loss")
    fig.savefig(
        path.join(figures_dir, "training_curve.png"),
        dpi=600,
        bbox_inches="tight",
    )
    print(f"File, 'training_curve.png', saved to {figures_dir}")
    plt.close()

    """
    Make plot: Parallel Tempering Index history
    """
    if negative_phase_method == "PT":
        idx_history = history.get("pt_index_history")
        if isinstance(idx_history, (list, tuple)):
            if len(idx_history) > 0:
                keys = sorted(idx_history[0].keys(), key=lambda k: int(k))
                idx_history_cum = {k: [] for k in keys}
                for i in range(len(idx_history)):
                    for k in keys:
                        idx_history_cum[k].extend(idx_history[i][k])
            else:
                idx_history = {}

        if not idx_history:
            print("No pt_index_history found in checkpoint history")
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
            keys = sorted(idx_history[0].keys(), key=lambda k: int(k))
            data_series = {k: np.asarray(idx_history_cum[k]) for k in keys}

            # Basic sanity check: sum across replicas at each time (optional)
            length = data_series[keys[0]].shape[0]
            totals = []
            for t in range(length):
                totals.append(int(np.sum([data_series[k][t] for k in keys])))

            if any(t != totals[0] for t in totals):
                print(
                    "Warning: PT index totals vary over time; totals sample:",
                    totals[:10],
                )

            for k in keys:
                if int(k) in range(0, len(keys), max(int(len(keys) / 10), 1)):
                    ax.plot(data_series[k], label=f"idx_{k}")

            ax.set_xlabel("Step")
            ax.set_ylabel("Index")
            ax.set_title("Parallel Tempering index history per replica")
            # ax.legend(ncol=min(6, len(keys)), fontsize='small')

            fig.savefig(
                path.join(figures_dir, "training_pt_idx_curve.png"),
                dpi=600,
                bbox_inches="tight",
            )
            print(f"File, 'training_pt_idx_curve.png', saved to {figures_dir}")
            plt.close()

    """
    Make plots: Ising, Potts, XY
    """
    if data_type in ["ising", "xy", "potts", "potts_1"]:
        obs_test = physics.compute_observables(X_test, data_type, float(T), q=q)
        obs_recon = physics.compute_observables(
            X_recon, data_type, float(T), q=q
        )
        obs_gen = physics.compute_observables(X_gen, data_type, float(T), q=q)

        # Example data
        sets = ["E/N", "C/N", "M/N", "Chi/N"]

        # 4 observables for Test, Recon, and Gen. In order from E/N, C/N, M/N, Chi/N
        values = np.array(
            [
                [
                    obs_test.energy_per_site,
                    obs_recon.energy_per_site,
                    obs_gen.energy_per_site,
                ],  # E/N,
                [
                    obs_test.heat_capacity_per_site,
                    obs_recon.heat_capacity_per_site,
                    obs_gen.heat_capacity_per_site,
                ],  # C/N,
                [
                    obs_test.magnetization_per_site,
                    obs_recon.magnetization_per_site,
                    obs_gen.magnetization_per_site,
                ],  # M/N,
                [
                    obs_test.susceptibility_per_site,
                    obs_recon.susceptibility_per_site,
                    obs_gen.susceptibility_per_site,
                ],  # Chi/N
            ]
        )

        # Labels for the 4 bars
        categories = ["Test", "Reconstructed", "Generated"]

        # X positions
        x = np.arange(len(sets))

        # Width of each bar
        bar_width = 0.2

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        for i in range(3):
            bars = ax.bar(
                x + i * bar_width,
                values[:, i],
                width=bar_width,
                label=categories[i],
            )
            # Add values above bars
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(x + 1.5 * bar_width, sets)
        ax.set_ylabel("Value")
        ax.set_title("Bar Plot Example")
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            path.join(figures_dir, "physical_properties.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'physical_properties.png', saved to {figures_dir}")
        plt.close()

    """
    Make plots: wind_dir dataset
    """
    if data_type == "wind_dir":
        # Plot histogram distributions of test, recon, and gen
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))

        axes.hist(
            (
                X_test.cpu().flatten(),
                X_recon.cpu().flatten(),
                X_gen.cpu().flatten(),
            ),
            bins=50,
            color=((0.4, 0, 0, 0.6), (0, 0.4, 0, 0.6), (0, 0, 0.4, 0.6)),
            label=("Test", "Recon", "Gen"),
        )
        axes.legend()
        axes.set_xlim(-np.pi, np.pi)
        axes.set_title("Test vs Recon vs Gen Angles")

        fig.savefig(
            path.join(figures_dir, "Test_recon_gen_angles.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'Test_recon_gen_angles.png', saved to {figures_dir}")
        plt.close()

    """
    Make plots: protein dataset
    """
    if data_type == "protein":
        # Plot histogram distributions of phi and psi angles
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        fig.suptitle("Phi and psi Distributions")

        ax_test = axes[0]
        ax_recon = axes[1]
        ax_gen = axes[2]

        ax_test.hist(
            X_test.cpu(),
            bins=50,
            color=((0, 0, 0.4, 0.6), (0.4, 0, 0, 0.6)),
            label=("phi", "psi"),
        )
        ax_test.legend()
        ax_test.set_xlim(-np.pi, np.pi)
        ax_test.set_title("Test")

        ax_recon.hist(
            X_recon.cpu(),
            bins=50,
            color=((0, 0, 0.4, 0.6), (0.4, 0, 0, 0.6)),
            label=("phi", "psi"),
        )
        ax_recon.legend()
        ax_recon.set_xlim(-np.pi, np.pi)
        ax_recon.set_title("Recon")

        ax_gen.hist(
            X_gen.cpu(),
            bins=50,
            color=((0, 0, 0.4, 0.6), (0.4, 0, 0, 0.6)),
            label=("phi", "psi"),
        )
        ax_gen.legend()
        ax_gen.set_xlim(-np.pi, np.pi)
        ax_gen.set_title("Generated")

        fig.savefig(
            path.join(figures_dir, "phi_psi_angles.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'phi_psi_angles.png', saved to {figures_dir}")
        plt.close()

        # Plot Ramchadran of test, recon, and gen
        test_phi = X_test.cpu()[:, 0]
        test_psi = X_test.cpu()[:, 1]

        recon_phi = X_recon.cpu()[:, 0]
        recon_psi = X_recon.cpu()[:, 1]

        gen_phi = X_gen.cpu()[:, 0]
        gen_psi = X_gen.cpu()[:, 1]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        ax_test = axes[0]
        ax_recon = axes[1]
        ax_gen = axes[2]

        ax_test.scatter(test_phi, test_psi, alpha=0.5, s=1)
        ax_test.set_xlim(-np.pi, np.pi)
        ax_test.set_ylim(-np.pi, np.pi)
        ax_test.set_xlabel("Phi (rad)")
        ax_test.set_ylabel("Psi (rad)")
        ax_test.set_title("Ramachandran Plot | Test")

        ax_recon.scatter(recon_phi, recon_psi, alpha=0.5, s=1)
        ax_recon.set_xlim(-np.pi, np.pi)
        ax_recon.set_ylim(-np.pi, np.pi)
        ax_recon.set_xlabel("Phi (rad)")
        ax_recon.set_ylabel("Psi (rad)")
        ax_recon.set_title("Ramachandran Plot | Reconstructed")

        ax_gen.scatter(gen_phi, gen_psi, alpha=0.5, s=1)
        ax_gen.set_xlim(-np.pi, np.pi)
        ax_gen.set_ylim(-np.pi, np.pi)
        ax_gen.set_xlabel("Phi (rad)")
        ax_gen.set_ylabel("Psi (rad)")
        ax_gen.set_title("Ramachandran Plot | Generated")
        plt.grid(True)
        fig.savefig(
            path.join(figures_dir, "ramachandran.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'ramachandran.png', saved to {figures_dir}")
        plt.close()

    """
    Preparation for Image Datasets
    """
    # Check if dataset is an image dataset. Set dimensions if image dataset. Also specify cmap.
    if data_type in [
        "mnist",
        "cifar10",
        "stl10",
        "ising",
        "xy",
        "potts",
        "potts_1",
    ]:
        image = True
        # if dataset is an image dataset:
        # infer p from data if L is None
        if L is not None:
            p = L
            if p != int(np.sqrt(X_test.shape[1])):
                raise ValueError(
                    f"L={L} does not match the input data size of {X_test.shape[1]} visible units. "
                    f"Expected L={int(np.sqrt(X_test.shape[1]))}. Set L to null to infer automatically if not using data types 'ising', 'xy', or 'potts'."
                )
        else:
            p = int(np.sqrt(X_test.shape[1]))
            print(
                f"L not specified, inferred p={p} from n_visible={X_test.shape[1]}"
            )

        # Set cmap
        if model_type == "multinomial" and data_type in ["potts", "potts_1"]:
            cmap = "tab10"
        elif model_type == "vonmises":
            cmap = "hsv"
        elif model_type == "binary" and mf == False:
            cmap = "binary"
        else:
            cmap = "gray"

        # Set vmin and vmax for consistent tab10 colors
        if model_type == "multinomial" and data_type in ["potts", "potts_1"]:
            vmin, vmax = 0, n_class - 1
        else:
            vmin, vmax = None, None

        # Set range for gen histograms
        if model_type == "multinomial":
            range_ = [-0.5, n_class - 0.5]
        elif model_type == "vonmises":
            range_ = [-3.5, 3.5]
        else:
            range_ = None

        # Set x_ticks for gen histograms
        if model_type == "multinomial":
            x_ticks = range(n_class)
        elif model_type == "vonmises":
            x_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        else:
            x_ticks = []

    else:
        image = False
        p = None

    """
    Make plots: Image datasets - Reconstruction
    """
    if image:
        test_fig = plot_grid_of_images(
            dataset=X_test,
            indices=range(10),
            rows=1,
            cols=10,
            image_dim=(p, p),
            save=False,
            n_class=n_class,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            include_colorbar=False,
        )
        recon_fig = plot_grid_of_images(
            dataset=X_recon,
            indices=range(10),
            rows=1,
            cols=10,
            image_dim=(p, p),
            save=False,
            n_class=n_class,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            include_colorbar=False,
        )
        combine_figures(
            [test_fig, recon_fig],
            figures_dir,
            "test_vs_recon.png",
            rows=2,
            cols=1,
            labels=["Original", "Reconstructed"],
            n_class=n_class,
            dataset=X_test,
            include_colorbar=True,
        )
        """
        # Plot Test vs Recon Images
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            im = axes[0, i].imshow(
                X_test[i].cpu().view(p, p), cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].imshow(
                X_recon[i].cpu().view(p, p), cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        plt.tight_layout()
        cbar = categorical_colorbar(fig, im, axes, X_test, n_class, shrink=0.8)

        fig.savefig(
            path.join(figures_dir, "test_vs_recon.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'test_vs_recon.png', saved to {figures_dir}")
        plt.close()
        """

        # Plot Test vs Recon Histograms
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].hist(X_test[i].cpu().view(-1), range=range_)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].hist(X_recon[i].cpu().view(-1), range=range_)
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        plt.tight_layout()

        fig.savefig(
            path.join(figures_dir, "test_vs_recon_hist.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'test_vs_recon_hist.png', saved to {figures_dir}")
        plt.close()

        # Plot Test vs Recon Hidden Units Histograms
        fig, axes = plt.subplots(2, 10, figsize=(12, 3))
        for i in range(10):
            axes[0, i].hist(h_test[i].cpu().view(-1), range=[-0.5, 1.5])
            axes[0, i].set_xlim(0.0, 1.0)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

            axes[1, i].hist(h_recon[i].cpu().view(-1), range=[-0.5, 1.5])
            axes[1, i].set_xlim(0.0, 1.0)
            axes[1, i].set_yticks([])

        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
        plt.tight_layout()

        fig.savefig(
            path.join(figures_dir, "test_vs_recon_h_hist.png"),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, 'test_vs_recon_h_hist.png', saved to {figures_dir}")
        plt.close()

    """
    Make plots: Image datasets - Generation
    """
    if image:
        image_count = max(0, min(n_samples, 256))  # Limit to 256 images
        image_count = 2 ** math.floor(
            math.log2(image_count)
        )  # Round down to nearest power of 2
        # cols >= rows, cols = 2 * rows for powers of 2
        cols = int(2 ** math.ceil(math.log2(image_count) / 2 + 0.5))
        rows = image_count // cols

        rand_idxs_test = random.sample(range(len(X_test)), image_count)
        rand_idxs_gen = random.sample(range(len(X_gen)), image_count)

        plot_grid_of_images(
            dataset=X_test,
            indices=rand_idxs_test,
            rows=rows,
            cols=cols,
            image_dim=(p, p),
            n_class=n_class,
            dir=figures_dir,
            file_name="test_images.png",
            cmap=cmap,
            graph_title="Images of Test Dataset Samples",
            vmin=vmin,
            vmax=vmax,
        )

        plot_grid_of_hists(
            dataset=X_test,
            indices=rand_idxs_test,
            rows=rows,
            cols=cols,
            dir=figures_dir,
            file_name="test_hists.png",
            graph_title="Histograms of Test Dataset Samples",
            range_=range_,
            x_ticks=x_ticks,
        )

        plot_grid_of_images(
            dataset=X_gen,
            indices=rand_idxs_gen,
            rows=rows,
            cols=cols,
            image_dim=(p, p),
            n_class=n_class,
            dir=figures_dir,
            file_name="gen_images.png",
            cmap=cmap,
            graph_title="Images of Generated Dataset Samples",
            vmin=vmin,
            vmax=vmax,
        )

        plot_grid_of_hists(
            dataset=X_gen,
            indices=rand_idxs_gen,
            rows=rows,
            cols=cols,
            dir=figures_dir,
            file_name="gen_hists.png",
            graph_title="Histograms of Generated Dataset Samples",
            range_=range_,
            x_ticks=x_ticks,
        )

        test_fig = plot_grid_of_images(
            dataset=X_test,
            indices=random.sample(range(len(X_test)), 10),
            rows=1,
            cols=10,
            image_dim=(p, p),
            save=False,
            n_class=n_class,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            include_colorbar=False,
        )
        gen_fig = plot_grid_of_images(
            dataset=X_gen,
            indices=random.sample(range(len(X_gen)), 10),
            rows=1,
            cols=10,
            image_dim=(p, p),
            save=False,
            n_class=n_class,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            include_colorbar=False,
        )
        combine_figures(
            [test_fig, gen_fig],
            figures_dir,
            "dataset_vs_gen_images.png",
            rows=2,
            cols=1,
            labels=["Original", "Generated"],
            n_class=n_class,
            dataset=X_test,
            include_colorbar=True,
        )

    """
    Make plots: Image datasets - Persistent Batch
    """
    if negative_phase_method == "PCD":
        # Get persistent batch and apply transformations if necessary
        if model_type == "multinomial":
            persistent_batch = (
                model.persistent_v.to(device)
                .float()
                .view(-1, n_visible * n_class)
            )
        else:
            persistent_batch = (
                model.persistent_v.to(device).float().view(-1, n_visible)
            )

        if model_type == "multinomial":
            persistent_batch = onehot_to_categories(
                persistent_batch, n_visible, n_class
            )

            if data_type in ["mnist", "cifar10", "stl10"]:
                persistent_batch = categories_to_grayscale(
                    persistent_batch, n_class
                )

        # Wrap angles if model is VonMises
        if model_type == "vonmises":
            persistent_batch = wrap(persistent_batch, -np.pi, np.pi)

        if image:
            # Check persistent batch images
            image_count = max(0, min(batch_size, 256))  # Limit to 256 images
            image_count = 2 ** math.floor(
                math.log2(image_count)
            )  # Round down to nearest power of 2
            # cols >= rows, cols = 2 * rows for powers of 2
            cols = int(2 ** math.ceil(math.log2(image_count) / 2 + 0.5))
            rows = image_count // cols

            # See batch images
            plot_grid_of_images(
                dataset=persistent_batch.cpu(),
                indices=range(len(persistent_batch)),
                rows=rows,
                cols=cols,
                image_dim=(p, p),
                n_class=n_class,
                dir=figures_dir,
                file_name="persistent_batch.png",
                cmap=cmap,
                graph_title="Images of Persistent Batch",
                vmin=vmin,
                vmax=vmax,
            )

    """
    Make plots: Weights
    """
    if not skip_weights:
        # Get Weights and plot as filters
        # cols >= rows, cols = 2 * rows for powers of 2
        image_count = max(0, min(n_hidden, 256))  # Limit to 256 images
        cols = int(2 ** math.ceil(math.log2(image_count) / 2 + 0.5))
        rows = image_count // cols

        if model_type == "vonmises":
            Weight_A = model.A.detach().cpu()
            Weight_B = model.B.detach().cpu()

            # Check weight distribution
            plot_weight_hist(Weight_A, figures_dir, "weight_A_hist.png")
            plot_weight_hist(Weight_B, figures_dir, "weight_B_hist.png")

            # Plot the full weight matrix
            plot_weight(Weight_A, figures_dir, file_name="weight_A.png")
            plot_weight(Weight_B, figures_dir, file_name="weight_B.png")

            # Visualize all weight filters as images
            if image:
                plot_weight_as_images(
                    Weight_A,
                    rows,
                    cols,
                    (p, p),
                    figures_dir,
                    file_name="weight_A_images.png",
                )
                plot_weight_as_images(
                    Weight_B,
                    rows,
                    cols,
                    (p, p),
                    figures_dir,
                    file_name="weight_B_images.png",
                )

            # Plot histograms of weight matrix filters per hidden unit
            plot_weight_as_hists_per_h(
                Weight_A, rows, cols, figures_dir, "weight_A_hists"
            )
            plot_weight_as_hists_per_h(
                Weight_B, rows, cols, figures_dir, "weight_B_hists"
            )

        elif model_type == "multinomial":
            Weight = model.W.detach().cpu().contiguous()
            Weight = Weight.view(n_hidden, n_visible, n_class)
            Weight = Weight.transpose(1, 2)

            for i in range(n_class):
                Weight_i = Weight[:, i, :]

                plot_weight_hist(
                    Weight_i, figures_dir, f"weight_q={i}_hist.png"
                )
                plot_weight(
                    Weight_i, figures_dir, file_name=f"weight_q={i}.png"
                )

                if image:
                    plot_weight_as_images(
                        Weight_i,
                        rows,
                        cols,
                        (p, p),
                        figures_dir,
                        file_name=f"weight_q={i}_images.png",
                    )

                plot_weight_as_hists_per_h(
                    Weight_i, rows, cols, figures_dir, f"weight_q={i}_hists"
                )

        else:
            Weight = model.W.detach().cpu()

            # Check weight distribution
            plot_weight_hist(Weight, figures_dir, "weight_hist.png")

            # Plot the full weight matrix
            plot_weight(Weight, figures_dir, "weight.png")

            # Visualize all weight filters as images
            if image:
                plot_weight_as_images(
                    Weight, rows, cols, (p, p), figures_dir, "weight_images.png"
                )

            # Plot histograms of weight matrix filters per hidden unit
            plot_weight_as_hists_per_h(
                Weight, rows, cols, figures_dir, "weight_hists"
            )

    return


def combine_figures(
    figures: list,
    dir: str,
    file_name: str = "combined_figures.png",
    rows: int | None = None,
    cols: int | None = None,
    figsize: tuple | list | None = None,
    labels: list[str] | None = None,
    n_class: int | None = None,
    dataset: torch.Tensor | None = None,
    include_colorbar: bool = True,
):
    """Combine returned Matplotlib Figure objects directly from their image axes.

    The source objects are the real figures produced by ``plot_grid_of_images``.
    This routine walks each figure's true ``axes`` and ``images`` collections,
    then copies the image arrays into a brand-new composite figure without
    serializing through PIL or a PNG buffer.

    Args:
        figures: Iterable of matplotlib Figure objects or image file paths.
        dir: Directory where the combined image should be written.
        file_name: Output file name.
        rows: Grid rows. If omitted, inferred from len(figures).
        cols: Grid cols. If omitted, inferred from len(figures).
        figsize: Optional composite figure size.
        labels: Optional row labels for each source figure.
    """
    if not figures:
        raise ValueError("figures must not be empty.")

    n = len(figures)
    if rows is None or cols is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

    if rows * cols < n:
        raise ValueError(
            "rows * cols must be greater than or equal to the number of figures."
        )

    if labels is None:
        labels = [None] * n
    elif len(labels) != n:
        raise ValueError("labels must have the same length as figures.")

    # Infer the master canvas from the real source figure size so it remains
    # horizontally wide instead of being forced into a tall, squished stack.
    source_width = None
    source_height = None
    for item in figures:
        if isinstance(item, str):
            continue
        if hasattr(item, "get_size_inches"):
            item_width, item_height = item.get_size_inches()
            source_width = (
                item_width
                if source_width is None
                else max(source_width, item_width)
            )
            source_height = (
                item_height
                if source_height is None
                else max(source_height, item_height)
            )

    if figsize is None:
        if source_width is None or source_height is None:
            figsize = (max(8.0, 2.2 * cols), max(3.0, 1.6 * rows))
        else:
            # Derive the final grid from the source image-object size directly:
            # width follows the columns, height follows the rows, with a modest
            # reference shrink so the combined figure reads like a clean 2x10
            # test-vs-recon panel instead of being markedly taller than wide.
            fig_w = source_width * cols
            fig_h = source_height * rows
            figsize = (fig_w, fig_h)

    master_fig = plt.figure(figsize=figsize)
    gs = master_fig.add_gridspec(rows, cols)

    for idx, (item, label) in enumerate(zip(figures, labels)):
        row = idx // cols
        col = idx % cols
        subfig = master_fig.add_subfigure(gs[row, col])

        if isinstance(item, str):
            image = Image.open(item)
            ax = subfig.subplots(1, 1)
            ax.imshow(image)
            ax.set_axis_off()
            if label is not None:
                ax.text(
                    -0.20,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                )
            continue

        if not hasattr(item, "axes"):
            raise TypeError(
                "Each item must be either a matplotlib Figure or an image file path."
            )

        source_axes = [ax for ax in item.axes if len(ax.images) > 0]
        if not source_axes:
            raise TypeError(
                "The supplied matplotlib Figure contains no image axes."
            )

        # Preserve the source figure's own subplot grid. The image figure was
        # created by plot_grid_of_images(rows, cols), so a 2 x 10 source must
        # be copied to a 2 x 10 grid in the composite subfigure rather than to
        # a 1 x 20 row that silently collapses the user's intended layout.
        grid = source_axes[0].get_subplotspec().get_gridspec()
        src_rows = grid.nrows
        src_cols = grid.ncols
        sub_axes = subfig.subplots(src_rows, src_cols)
        for target_ax, src_ax in zip(sub_axes.flat, source_axes):
            src_im = src_ax.images[0]
            arr = src_im.get_array()
            cmap = src_im.get_cmap()
            vmin = (
                src_im.norm.vmin
                if hasattr(src_im, "norm") and hasattr(src_im.norm, "vmin")
                else None
            )
            vmax = (
                src_im.norm.vmax
                if hasattr(src_im, "norm") and hasattr(src_im.norm, "vmax")
                else None
            )

            target_ax.imshow(
                arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
            )
            target_ax.set_xticks([])
            target_ax.set_yticks([])
            target_ax.set_frame_on(True)
            for spine in target_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("black")

        if label is not None:
            sub_axes[0].text(
                -0.20,
                0.5,
                label,
                transform=sub_axes[0].transAxes,
                rotation=90,
                va="center",
                ha="center",
            )

    # Add one categorical colorbar to the master figure itself. This avoids
    # the mixed-subfigure parent problem entirely because the colorbar is made
    # on a dedicated axes that belongs to the same master figure, not on a list
    # of subfigure axes pulled from different parent figures.
    if include_colorbar and n_class is not None and dataset is not None:
        first_item = figures[0]
        if hasattr(first_item, "axes"):
            first_source_axes = [
                ax for ax in first_item.axes if len(ax.images) > 0
            ]
            if first_source_axes:
                first_im = first_source_axes[0].images[0]
                categories = used_categories(dataset, n_class)
                if first_im.get_cmap().name == "tab10" and len(categories) > 0:
                    base_cmap = plt.get_cmap("tab10", n_class)
                    cmap = ListedColormap(base_cmap(categories))
                    norm = BoundaryNorm(
                        np.arange(len(categories) + 1) - 0.5, len(categories)
                    )
                    mappable = matplotlib.cm.ScalarMappable(
                        norm=norm, cmap=cmap
                    )
                    # Place a dedicated cax directly in the master figure.
                    cax = master_fig.add_axes([0.935, 0.12, 0.025, 0.76])
                    cbar = master_fig.colorbar(
                        mappable, cax=cax, ticks=np.arange(len(categories))
                    )
                    cbar.ax.set_yticklabels(categories)

    out_path = path.join(dir, file_name)
    if not out_path.endswith(".png"):
        out_path = f"{out_path}.png"

    master_fig.savefig(out_path, dpi=600, bbox_inches="tight")
    print(f"File, '{file_name}', saved to {dir}")
    plt.close(master_fig)

    # The source figure objects are passed in by reference from
    # plot_grid_of_images(..., save=False). Use them only to copy image pixels,
    # then close them deterministically to avoid leaking live figures.
    for item in figures:
        if hasattr(item, "axes"):
            plt.close(item)


def plot_grid_of_images(
    dataset: torch.Tensor,
    indices: list,
    rows: int,
    cols: int,
    image_dim: tuple,
    n_class: int,
    dir: str = None,
    file_name: str = "grid_of_images.png",
    graph_title: str | None = None,
    save: bool = True,
    cmap: str = "gray",
    include_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot a grid of images from an image dataset and indices.

    Args:
        dataset:
        indices:
        rows:
        cols:
        image_dim:
        n_class:
        dir:
        file_name:
        graph_title:
        cmap:
        vmin:
        vmax:
        include_colorbar: Whether to add the categorical color bar.
    """
    if len(indices) < rows * cols:
        raise ValueError(
            "Length of arrays must be greater than or equal to the product of rows and cols."
        )

    fig = plt.figure(figsize=(cols, rows))
    if graph_title is not None:
        fig.suptitle(graph_title)

    for i in range(cols * rows):
        ax = fig.add_subplot(rows, cols, i + 1)
        im = ax.imshow(
            dataset[indices[i]].cpu().view(image_dim),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    if include_colorbar:
        categorical_colorbar(
            fig, im, fig.axes, dataset, n_class, fraction=0.046, pad=0.04
        )

    if save:
        if dir is None:
            raise ValueError(
                "'dir' must be specified when 'save' is set to True."
            )

        fig.savefig(
            path.join(dir, file_name),
            dpi=600,
            bbox_inches="tight",
        )
        print(f"File, '{file_name}', saved to {dir}")
        plt.close(fig)

    return fig


def plot_grid_of_hists(
    dataset: torch.Tensor,
    indices: list,
    rows: int,
    cols: int,
    dir: str,
    file_name: str = "grid_of_hists.png",
    graph_title: str | None = None,
    range_: tuple | list | None = None,
    x_ticks: tuple | list | None = None,
):
    """Plot a grid of histograms from an image dataset and indices.

    Args:
        dataset:
        indices:
        rows:
        cols:
        dir:
        file_name:
        graph_title:
        range_:
        x_ticks:
    """
    if len(indices) < rows * cols:
        raise ValueError(
            "Length of arrays must be greater than or equal to the product of rows and cols."
        )

    fig = plt.figure(figsize=(cols, rows))
    if graph_title is not None:
        fig.suptitle(graph_title)

    for i in range(cols * rows):  # grid
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(dataset[indices[i]].cpu().view(-1), range=range_)
        ax.set_xticks(x_ticks)
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    fig.savefig(
        path.join(dir, file_name),
        dpi=600,
        bbox_inches="tight",
    )
    print(f"File, '{file_name}', saved to {dir}")
    plt.close(fig)


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


def used_categories(samples, n_class):
    """Return the category values present in samples for a discrete colorbar."""
    categories = torch.unique(samples.detach().cpu().long()).numpy()
    return categories[(categories >= 0) & (categories < n_class)]


def categorical_colorbar(fig, image, axes, samples, n_class, **kwargs):
    """Add a colorbar containing only categories present in samples."""
    categories = used_categories(samples, n_class)
    if image.get_cmap().name != "tab10" or len(categories) == 0:
        return fig.colorbar(image, ax=axes, **kwargs)

    base_cmap = plt.get_cmap("tab10", n_class)
    cmap = ListedColormap(base_cmap(categories))
    norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        mappable,
        ax=axes,
        ticks=np.arange(len(categories)),
        **kwargs,
    )
    cbar.ax.set_yticklabels(categories)
    return cbar


def plot_weight_hist(weight, dir, file_name="weight_hist"):
    """Plot histogram of weight values.

    Args:
        weight: Tensor of weight values
        dir: Directory to save the plot
        file_name: Name of the file to save the plot
    """
    # Check weight distribution
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    axes.hist(weight.flatten().numpy(), bins=100)
    axes.set_xlabel("W_ij value")
    axes.set_yticks([])
    axes.set_title("Weight Distribution")

    fig.savefig(path.join(dir, file_name), dpi=600, bbox_inches="tight")
    print(f"File, '{file_name}', saved to {dir}")
    plt.close()


def plot_weight(weight, dir, file_name="weight.png", vmin=None, vmax=None):
    """Plot the full weight matrix.

    Args:
        weight: Tensor of weight values
        dir: Directory to save the plot
        file_name: Name of the file to save the plot
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
    """
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    img = axes.imshow(weight, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(img, location="left")
    fig.savefig(path.join(dir, file_name), dpi=600, bbox_inches="tight")
    print(f"File, '{file_name}', saved to {dir}")
    plt.close()


def plot_weight_as_images(
    weight,
    rows,
    cols,
    dim,
    dir,
    file_name="weight_images.png",
    vmin=None,
    vmax=None,
):
    """Visualize all weight filters as images.

    Args:
        weight: Tensor of weight values
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        dim: Tuple of (height, width) for each filter image
        dir: Directory to save the plot
        file_name: Name of the file to save the plot
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
    """
    fig = plt.figure(figsize=(cols, rows))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        im = ax.imshow(
            weight[i].reshape(dim),
            cmap="gray",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    # cbar = fig.colorbar(im, orientation='horizontal')  #, fraction=0.05, pad=0.02)
    fig.savefig(path.join(dir, file_name), dpi=600, bbox_inches="tight")
    print(f"File, '{file_name}', saved to {dir}")
    plt.close()


def plot_weight_as_hists_per_h(
    weight, rows, cols, dir, filename="weight_hists"
):
    """Plot histograms of weight matrix filters per hidden unit.

    Args:
        weight: Tensor of weight values
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        dir: Directory to save the plot
        filename: Name of the file to save the plot
    """
    fig = plt.figure(figsize=(cols, rows))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.hist(weight[i].flatten().numpy(), bins=50)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(path.join(dir, filename), dpi=600, bbox_inches="tight")
    print(f"File, '{filename}', saved to {dir}")
    plt.close()


if __name__ == "__main__":
    main()
