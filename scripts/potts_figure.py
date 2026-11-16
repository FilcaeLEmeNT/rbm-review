#!/usr/bin/env python3
import argparse
from os import path
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import json
import pandas as pd
import pickle


import utils.config as cfg
import utils.sweep as swp
from utils.checkpoint import load_checkpoint

from data.data_loader import load_data

from utils.multinomial import onehot_to_categories, categories_to_grayscale
import utils.physics as physics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a T vs Physical Properties plot from sweep and config after sampling."
    )

    group = parser.add_mutually_exclusive_group(required=True)
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
        "--epoch",
        type=int,
        default=None,
        help="Optional checkpoint epoch to scope the physics sample tree.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.sweep and not args.config:
        argparse.error("--config is required when using --sweep")

    config = cfg.load_config(args.config)
    sweep = cfg.load_config(args.sweep)
    trial_groups = swp.iter_sweep_configs(config, sweep)

    trial_results = []
    for trial_group in trial_groups:
        results, configs = analyze_trial(
            trial_group,
            config,
            epoch=args.epoch,
        )
        trial_results.append((results, configs))

    plot_all_trials(trial_results, config, epoch=args.epoch)


def analyze_trial(trial_group, config, epoch=None):
    trial_configs = trial_group["configs"]
    if not trial_configs:
        return

    output_dir = trial_configs[0]["output"]["base_dir"]
    results = []
    configs = []
    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    for trial_config in trial_configs:
        out_dir = trial_config["output"]["base_dir"]
        run_name = trial_config["output"]["run_name"]
        physics_dir = cfg.get_output_paths(out_dir, run_name)["physics"]
        physics_path = path.join(
            physics_dir,
            f"physical_properties{suffix}.pkl",
        )
        if not path.exists(physics_path):
            continue
        with open(physics_path, "rb") as f:
            results.append(pickle.load(f))
        configs.append(trial_config)

    if len(results) == 0:
        raise FileNotFoundError(
            f"No physical_properties.pkl files found for trial "
            f"{trial_group['trial_index']}. "
            f"basedir={trial_config["output"]["base_dir"]}. "
            f"runname={trial_config["output"]["run_name"]}"
        )

    Ts = [r["T"] for r in results]

    obs_train = [r["observables_train"] for r in results]
    obs_test = [r["observables_test"] for r in results]
    obs_recon = [r["observables_recon"] for r in results]
    obs_gen = [r["observables_gen"] for r in results]

    E_test = torch.tensor([obs.energy_mean_per_site for obs in obs_test])
    E_recon = torch.tensor([obs.energy_mean_per_site for obs in obs_recon])
    E_gen = torch.tensor([obs.energy_mean_per_site for obs in obs_gen])

    E_raw_test = torch.stack([obs.energy for obs in obs_test])
    E_raw_recon = torch.stack([obs.energy for obs in obs_recon])
    E_raw_gen = torch.stack([obs.energy for obs in obs_gen])

    M_test = torch.tensor([obs.magnetization_mean_per_site for obs in obs_test])
    M_recon = torch.tensor([obs.magnetization_mean_per_site for obs in obs_recon])
    M_gen = torch.tensor([obs.magnetization_mean_per_site for obs in obs_gen])

    M_raw_test = torch.stack([obs.magnetization for obs in obs_test])
    M_raw_recon = torch.stack([obs.magnetization for obs in obs_recon])
    M_raw_gen = torch.stack([obs.magnetization for obs in obs_gen])

    C_test = torch.tensor([obs.heat_capacity_per_site for obs in obs_test])
    C_recon = torch.tensor([obs.heat_capacity_per_site for obs in obs_recon])
    C_gen = torch.tensor([obs.heat_capacity_per_site for obs in obs_gen])

    Chi_test = torch.tensor([obs.susceptibility_per_site for obs in obs_test])
    Chi_recon = torch.tensor([obs.susceptibility_per_site for obs in obs_recon])
    Chi_gen = torch.tensor([obs.susceptibility_per_site for obs in obs_gen])

    dominant_entropy_test = torch.tensor(
        [obs.dominant_entropy if obs.dominant_entropy is not None else float("nan") for obs in obs_test]
    )
    dominant_entropy_recon = torch.tensor(
        [obs.dominant_entropy if obs.dominant_entropy is not None else float("nan") for obs in obs_recon]
    )
    dominant_entropy_gen = torch.tensor(
        [obs.dominant_entropy if obs.dominant_entropy is not None else float("nan") for obs in obs_gen]
    )

    E_MSE = torch.mean((E_gen - E_test) ** 2)
    M_MSE = torch.mean((M_gen - M_test) ** 2)
    C_MSE = torch.mean((C_gen - C_test) ** 2)
    Chi_MSE = torch.mean((Chi_gen - Chi_test) ** 2)

    E_MAE = torch.mean(torch.abs(E_gen - E_test))
    M_MAE = torch.mean(torch.abs(M_gen - M_test))
    C_MAE = torch.mean(torch.abs(C_gen - C_test))
    Chi_MAE = torch.mean(torch.abs(Chi_gen - Chi_test))

    E_RE = torch.mean(torch.abs(E_gen - E_test) / (torch.abs(E_test) + 1e-8))
    M_RE = torch.mean(torch.abs(M_gen - M_test) / (torch.abs(M_test) + 1e-8))
    C_RE = torch.mean(torch.abs(C_gen - C_test) / (torch.abs(C_test) + 1e-8))
    Chi_RE = torch.mean(
        torch.abs(Chi_gen - Chi_test) / (torch.abs(Chi_test) + 1e-8)
    )

    # Get Tc
    Tc = physics.potts_critical_T(config["data"]["q"])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    E_axes = axes[0][0]
    C_axes = axes[0][1]
    M_axes = axes[1][0]
    Chi_axes = axes[1][1]

    E_axes.plot(Ts, E_test, "kx-", label="Test")
    # E_axes.plot(Ts, E_recon, 'ro-', label="Recon")
    E_axes.plot(Ts, E_gen, "y^-", label="Gen")
    E_axes.axvline(x=Tc, color="red", linestyle="--")
    E_axes.legend()
    E_axes.set_xlabel("T")
    E_axes.set_ylabel("E/N")

    M_axes.plot(Ts, M_test, "kx-", label="Test")
    # M_axes.plot(Ts, M_recon, 'ro-', label="Recon")
    M_axes.plot(Ts, M_gen, "y^-", label="Gen")
    M_axes.axvline(x=Tc, color="red", linestyle="--")
    M_axes.legend()
    M_axes.set_xlabel("T")
    M_axes.set_ylabel("M/N")

    C_axes.plot(Ts, C_test, "kx-", label="Test")
    # C_axes.plot(Ts, C_recon, 'ro-', label="Recon")
    C_axes.plot(Ts, C_gen, "y^-", label="Gen")
    C_axes.axvline(x=Tc, color="red", linestyle="--")
    C_axes.legend()
    C_axes.set_xlabel("T")
    C_axes.set_ylabel("C/N")

    Chi_axes.plot(Ts, Chi_test, "kx-", label="Test")
    # Chi_axes.plot(Ts, Chi_recon, 'ro-', label="Recon")
    Chi_axes.plot(Ts, Chi_gen, "y^-", label="Gen")
    Chi_axes.axvline(x=Tc, color="red", linestyle="--")
    Chi_axes.legend()
    Chi_axes.set_xlabel("T")
    Chi_axes.set_ylabel("Chi/N")

    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    plt.tight_layout()
    plt.savefig(
        path.join(output_dir, f"physical_properties{suffix}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    print(f"File, 'physical_properties{suffix}.png', saved to {output_dir}")
    plt.close(fig)

    ### Make a table of physical poperties by temperature and save as a CSV
    df = pd.DataFrame(
        {
            "T": Ts,
            "batch_size": [
                cfg["data"].get("batch_size") for cfg in configs
            ],
            "n_hidden": [cfg["model"].get("n_hidden") for cfg in configs],
            "n_epochs": [
                cfg["training"].get("n_epochs") for cfg in configs
            ],
            "lr": [cfg["training"].get("lr") for cfg in configs],
            "weight_decay": [
                cfg["training"].get("weight_decay", 0.0)
                for cfg in configs
            ],
            "momentum": [
                cfg["training"].get("momentum", 0.0) for cfg in configs
            ],
            "k": [cfg["training"].get("k") for cfg in configs],
            "algorithm": [
                cfg["training"].get("algorithm") for cfg in configs
            ],
            "negative_phase_method": [
                cfg["training"].get("negative_phase_method")
                for cfg in configs
            ],
            "mc": [cfg["training"].get("mc") for cfg in configs],
            "epsilon": [
                cfg["training"].get("epsilon") for cfg in configs
            ],
            "pt_n_chains": [
                cfg["training"].get("pt_n_chains") for cfg in configs
            ],
            "pt_max_T": [
                cfg["training"].get("pt_max_T") for cfg in configs
            ],
            "E_MSE": E_MSE,
            "E_MAE": E_MAE,
            "E_RE": E_RE,
            "M_MSE": M_MSE,
            "M_MAE": M_MAE,
            "M_RE": M_RE,
            "C_MSE": C_MSE,
            "C_MAE": C_MAE,
            "C_RE": C_RE,
            "Chi_MSE": Chi_MSE,
            "Chi_MAE": Chi_MAE,
            "Chi_RE": Chi_RE,
            "E_test": E_test,
            "E_gen": E_gen,
            "M_test": M_test,
            "M_gen": M_gen,
            "C_test": C_test,
            "C_gen": C_gen,
            "Chi_test": Chi_test,
            "Chi_gen": Chi_gen,
            "Dominant_Entropy_test": dominant_entropy_test,
            "Dominant_Entropy_gen": dominant_entropy_gen,
        }
    )

    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    df.to_csv(
        path.join(output_dir, f"physical_properties{suffix}.csv"),
        index=False,
    )
    print(f"File, 'physical_properties{suffix}.csv', saved to {output_dir}")

    ### Make a scatter plot with all temperatures aggregated
    N = configs[0]["data"]["L"] ** 2
    colors = plt.cm.rainbow(np.linspace(0, 1, len(Ts)))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for idx_type, (label, energies, magnetizations) in enumerate(
        [
            ("Monte Carlo", E_raw_test, M_raw_test),
            ("RBM", E_raw_gen, M_raw_gen),
        ]
    ):
        for idx, T, color in zip(range(len(Ts)), Ts, colors):
            axes[idx_type].scatter(
                magnetizations[idx].flatten().cpu() / N,
                energies[idx].flatten().cpu() / N,
                s=10,
                alpha=0.4,
                color=color,
                edgecolors='black',
                linewidths=0.3,
                label=f"T = {T:.3f}",
            )
        
        axes[idx_type].set_title(label)
        axes[idx_type].set_ylabel("Energy per site (E/N)")
        axes[idx_type].set_xlabel("Magnetization per site (M/N)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        title="Temperature",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.savefig(
        path.join(output_dir, f"EM_scatter{suffix}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    print(f"File, 'EM_scatter{suffix}.png', saved to {output_dir}")
    plt.close(fig)

    ### Make scatter plots for each temperature
    fig, axes = plt.subplots(
        2,
        len(Ts),
        figsize=(4 * len(Ts), 8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for column, (temperature, color) in enumerate(zip(Ts, colors)):
        for row, energies, magnetizations in (
            (0, E_raw_test, M_raw_test),
            (1, E_raw_gen, M_raw_gen),
        ):
            axes[row, column].scatter(
                magnetizations[column].flatten().cpu() / N,
                energies[column].flatten().cpu() / N,
                s=10,
                alpha=0.4,
                color=color,
                edgecolors="black",
                linewidths=0.3,
            )
            axes[row, column].grid(True, alpha=0.3)
            if row == 0:
                axes[row, column].set_title(f"T = {temperature:.3f}")

    fig.text(0.01, 0.72, "Monte Carlo", rotation=90, va="center")
    fig.text(0.01, 0.28, "Generated", rotation=90, va="center")
    fig.supxlabel("Magnetization per site (M/N)")
    fig.supylabel("Energy per site (E/N)")
    fig.tight_layout(rect=(0.04, 0.04, 1, 1))
    plt.savefig(
        path.join(output_dir, f"EM_scatter_by_temperature{suffix}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    print(
        f"File, 'EM_scatter_by_temperature{suffix}.png', saved to {output_dir}"
    )
    plt.close(fig)

    ### Make pdf for E and M for each temperature
    fig, axes = plt.subplots(
        2,
        len(Ts),
        figsize=(4 * len(Ts), 8),
        squeeze=False,
    )
    for column, (temperature, color) in enumerate(zip(Ts, colors)):
        for row, property_name, property_gen, property_test in (
            (0, "energy (e)", E_raw_gen, E_raw_test),
            (1, "magnetization (m)", M_raw_gen, M_raw_test),
        ):
            axes[row, column].hist(property_gen[column].flatten().cpu() / N, bins=50, density=True, alpha=0.5, color="r", label="RBM")
            axes[row, column].hist(property_test[column].flatten().cpu() / N, bins=50, density=True, alpha=0.5, color="b", label="MC")
            axes[row, column].set_xlabel(property_name)
            axes[row, column].set_ylabel("PDF")
            axes[row, column].autoscale()
            axes[row, column].legend()
            
            if row == 0:
                axes[row, column].set_title(f"T = {temperature:.3f}")

    fig.tight_layout(rect=(0.04, 0.04, 1, 1))
    plt.savefig(
        path.join(output_dir, f"EM_pdf_by_temperature{suffix}.png"),
        dpi=600,
        bbox_inches="tight",
    )
    print(
        f"File, 'EM_pdf_by_temperature{suffix}.png', saved to {output_dir}"
    )
    plt.close(fig)

    ### Make a table of physical poperties by temperature and save as a CSV
    raw_observables = {
        "E_raw_test": E_raw_test,
        "E_raw_recon": E_raw_recon,
        "E_raw_gen": E_raw_gen,
        "M_raw_test": M_raw_test,
        "M_raw_recon": M_raw_recon,
        "M_raw_gen": M_raw_gen,
    }
    raw_rows = []
    for temperature_idx, temperature in enumerate(Ts):
        samples = {
            name: values[temperature_idx].detach().cpu().flatten().tolist()
            for name, values in raw_observables.items()
        }
        sample_count = max(map(len, samples.values()))
        for sample_idx in range(sample_count):
            raw_rows.append(
                {
                    "T": temperature,
                    "sample_idx": sample_idx,
                    **{
                        name: values[sample_idx]
                        if sample_idx < len(values)
                        else np.nan
                        for name, values in samples.items()
                    },
                }
            )
    df = pd.DataFrame(raw_rows)

    df.to_csv(
        path.join(output_dir, f"EM_raw{suffix}.csv"),
        index=False,
    )
    print(f"File, 'EM_raw{suffix}.csv', saved to {output_dir}")

    return results, configs


def plot_all_trials(trial_results, config, epoch=None):
    """Plot the mean and standard deviation of each property across trials."""
    results = [result for results, _ in trial_results for result in results]
    if not results:
        raise FileNotFoundError(
            "No physical_properties.pkl files found for the provided trials."
        )

    temperatures = sorted({result["T"] for result in results})
    grouped_results = {
        temperature: [
            result for result in results if result["T"] == temperature
        ]
        for temperature in temperatures
    }

    def property_values(source, observable_name):
        # Grouped results may contain a different number of trials per
        # temperature. Pad each temperature row with NaN so NumPy can build a
        # dense matrix for downstream mean/std computation.
        max_count = max(
            len(grouped_results[temperature]) for temperature in temperatures
        )
        rows = []
        for temperature in temperatures:
            values = []
            for result in grouped_results[temperature]:
                obs = result[f"observables_{source}"]
                value = getattr(obs, observable_name, np.nan)
                values.append(float(value) if value == value else np.nan)

            if len(values) < max_count:
                values.extend([np.nan] * (max_count - len(values)))
            rows.append(values)

        return np.asarray(rows, dtype=float)

    properties = {
        "E/N": "energy_per_site",
        "C/N": "heat_capacity_per_site",
        "M/N": "magnetization_per_site",
        "Chi/N": "susceptibility_per_site",
    }

    def mean_and_std(values):
        means = np.full(values.shape[0], np.nan)
        stds = np.full(values.shape[0], np.nan)
        for row_idx, row in enumerate(values):
            valid_values = row[~np.isnan(row)]
            if valid_values.size:
                means[row_idx] = np.mean(valid_values)
                stds[row_idx] = np.std(valid_values)
        return means, stds

    Tc = physics.potts_critical_T(config["data"]["q"])
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for axes_item, (label, observable_name) in zip(
        axes.flat, properties.items()
    ):
        test_values = property_values("test", observable_name)
        gen_values = property_values("gen", observable_name)
        test_mean, test_std = mean_and_std(test_values)
        gen_mean, gen_std = mean_and_std(gen_values)

        axes_item.errorbar(
            temperatures,
            test_mean,
            yerr=test_std,
            fmt="kx-",
            capsize=3,
            label="Test mean+/-std",
        )
        axes_item.errorbar(
            temperatures,
            gen_mean,
            yerr=gen_std,
            fmt="g^-",
            capsize=3,
            label="Gen mean+/-std",
        )
        axes_item.axvline(x=Tc, color="red", linestyle="--")
        axes_item.legend()
        axes_item.set_xlabel("T")
        axes_item.set_ylabel(label)

    suffix = f"_epoch_{epoch}" if epoch is not None else ""

    plt.tight_layout()
    output_path = path.join(
        config["output"]["base_dir"], f"physical_properties{suffix}.png"
    )
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(
        f"File, 'physical_properties{suffix}.png', saved to {config['output']['base_dir']}"
    )
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


if __name__ == "__main__":
    main()
