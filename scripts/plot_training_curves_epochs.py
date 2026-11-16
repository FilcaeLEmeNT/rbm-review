#!/usr/bin/env python3
"""Aggregate training histories from one epoch of a trial's checkpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch

from utils.checkpoint import load_checkpoint


PLOT_METRICS = (
    ("E_diff", "Visible energy diff"),
    ("mse", "Reconstruction MSE"),
    ("loss", "Reconstruction Loss (CE)"),
    ("PL", "PLL"),
    ("CE_recon_loss", "CE_loss"),
    ("BCE_recon_loss", "BCE_loss")
)


def plottable_metric(history: dict, key: str) -> list[float] | None:
    """Return a metric series if it contains at least one finite value."""
    values = history.get(key)
    if values is None:
        return None
    try:
        if len(values) == 0:
            return None
    except TypeError as exc:
        raise ValueError(f"History metric {key!r} is not a series") from exc

    normalized = []
    has_finite_value = False
    for value in values:
        if value is None:
            normalized.append(float("nan"))
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"History metric {key!r} contains a non-numeric value: "
                f"{value!r}"
            ) from exc
        normalized.append(numeric_value)
        has_finite_value |= math.isfinite(numeric_value)

    return normalized if has_finite_value else None


def load_trial_histories(trial_dir: Path, epoch: int) -> list[dict]:
    """Load histories from every run checkpoint saved at ``epoch``."""
    checkpoint_name = f"checkpoint_epoch_{epoch}.pt"
    checkpoint_paths = sorted(trial_dir.rglob(checkpoint_name))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No {checkpoint_name} checkpoints found under {trial_dir}"
        )

    runs = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = load_checkpoint(
            str(checkpoint_path), torch.device("cpu"), load_model=False
        )
        history = checkpoint.get("history")
        if not isinstance(history, dict):
            raise ValueError(f"No history dictionary in {checkpoint_path}")

        config = checkpoint.get("config") or {}
        data_config = config.get("data") or {}
        output_config = config.get("output") or {}
        temperature = data_config.get("T")
        run_name = output_config.get("run_name") or checkpoint_path.parent.name
        if temperature is None:
            label = str(run_name)
            sort_key = (1, 0.0, label)
        else:
            try:
                numeric_temperature = float(temperature)
            except (TypeError, ValueError):
                label = f"T={temperature}"
                sort_key = (1, 0.0, label)
            else:
                label = f"T={numeric_temperature:g}"
                sort_key = (0, numeric_temperature, label)

        runs.append(
            {
                "label": label,
                "sort_key": sort_key,
                "history": history,
                "checkpoint_path": checkpoint_path,
            }
        )

    return sorted(runs, key=lambda run: run["sort_key"])


def plot_histories(
    runs: list[dict], trial_dir: Path, epoch: int, output_path: Path
) -> None:
    """Plot only metrics with finite values in at least one run."""
    colors = colormaps["viridis"]
    energy_series = []
    metric_series = {key: [] for key, _ in PLOT_METRICS}
    recorded_runs = {"E_data": 0, "E_model": 0}
    recorded_runs.update({key: 0 for key, _ in PLOT_METRICS})

    for run_index, run in enumerate(runs):
        history = run["history"]
        color = colors(run_index / max(len(runs) - 1, 1))
        energy_data = plottable_metric(history, "E_data")
        energy_model = plottable_metric(history, "E_model")
        energy_series.append((run, color, energy_data, energy_model))
        if energy_data is not None:
            recorded_runs["E_data"] += 1
        if energy_model is not None:
            recorded_runs["E_model"] += 1

        for key, _ in PLOT_METRICS:
            metric = plottable_metric(history, key)
            metric_series[key].append((run, color, metric))
            if metric is not None:
                recorded_runs[key] += 1

    plotted_metrics = []
    if recorded_runs["E_data"] or recorded_runs["E_model"]:
        plotted_metrics.append(("energy", "Visible energy"))
    plotted_metrics.extend(
        (key, title)
        for key, title in PLOT_METRICS
        if recorded_runs[key]
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=max(len(plotted_metrics), 1),
        figsize=(max(5 * len(plotted_metrics), 8), 4),
        squeeze=False,
    )
    axes = axes[0]

    def availability(key: str) -> str:
        count = recorded_runs[key]
        if count < len(runs):
            return f"{count}/{len(runs)} runs"
        return ""

    if plotted_metrics:
        for axis, (metric_key, title) in zip(axes, plotted_metrics):
            if metric_key == "energy":
                for run, color, energy_data, energy_model in energy_series:
                    if energy_data is not None:
                        axis.plot(
                            range(1, len(energy_data) + 1),
                            energy_data,
                            color=color,
                            linestyle="-",
                            label=run["label"],
                        )
                    if energy_model is not None:
                        axis.plot(
                            range(1, len(energy_model) + 1),
                            energy_model,
                            color=color,
                            linestyle="--",
                            label="_nolegend_",
                        )
                energy_status = [
                    f"{name} {status}"
                    for name, key in (
                        ("data", "E_data"),
                        ("model", "E_model"),
                    )
                    if (status := availability(key))
                ]
                if energy_status:
                    title += f"\n{'; '.join(energy_status)}"
            else:
                for _, color, metric in metric_series[metric_key]:
                    if metric is not None:
                        axis.plot(
                            range(1, len(metric) + 1), metric, color=color
                        )
                status = availability(metric_key)
                if status:
                    title += f"\n{status}"
            axis.set_title(title)
    else:
        axes[0].set_axis_off()
        axes[0].text(
            0.5,
            0.5,
            "No finite metrics recorded",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    for axis in axes:
        if plotted_metrics:
            axis.set_xlabel("Training epoch")
            axis.grid(True, alpha=0.25)

    if recorded_runs["E_data"] or recorded_runs["E_model"]:
        fig.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="-", label="E_data"),
                Line2D([0], [0], color="black", linestyle="--", label="E_model"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.04, 0.88),
            ncol=2,
            fontsize=8,
        )
    fig.suptitle(f"Training curves through epoch {epoch}: {trial_dir.name}")
    if plotted_metrics:
        fig.subplots_adjust(
            left=0.06, right=0.86, bottom=0.17, top=0.72, wspace=0.38
        )
        colorbar_axis = fig.add_axes((0.89, 0.18, 0.018, 0.58))
        colorbar = fig.colorbar(
            ScalarMappable(
                norm=Normalize(vmin=0, vmax=max(len(runs) - 1, 1)),
                cmap=colors,
            ),
            cax=colorbar_axis,
            ticks=list(range(len(runs))),
        )
        colorbar.ax.set_yticklabels([run["label"] for run in runs])
        colorbar.ax.set_ylabel("Run temperature")
        colorbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate training curves from checkpoints in one trial directory."
        )
    )
    parser.add_argument(
        "trial_dir",
        type=Path,
        help="Trial directory containing per-run checkpoint folders.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="Checkpoint epoch to aggregate, such as 150.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: <trial_dir>/training_curve_epoch_<epoch>.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epoch < 1:
        raise SystemExit("--epoch must be at least 1")

    runs = load_trial_histories(args.trial_dir, args.epoch)
    output_path = args.output or args.trial_dir / (
        f"training_curve_epoch_{args.epoch}.png"
    )
    plot_histories(runs, args.trial_dir, args.epoch, output_path)
    print(f"Plotted {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()