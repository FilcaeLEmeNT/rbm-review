#!/usr/bin/env python3
"""Plot physical properties from multiple epoch CSVs in one trial directory."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D


PROPERTIES = {
    "Energy (E/N)": ("E_test", "E_gen"),
    "Magnetization (M/N)": ("M_test", "M_gen"),
    "Heat capacity (C/N)": ("C_test", "C_gen"),
    "Susceptibility (Chi/N)": ("Chi_test", "Chi_gen"),
}


def find_epoch_csvs(trial_dir: Path) -> list[tuple[int, Path]]:
    """Return physical-properties CSVs ordered by epoch number."""
    epoch_files = []
    pattern = re.compile(r"physical_properties_epoch_(\d+)\.csv")
    for csv_path in trial_dir.glob("physical_properties_epoch_*.csv"):
        match = pattern.fullmatch(csv_path.name)
        if match:
            epoch_files.append((int(match.group(1)), csv_path))
    return sorted(epoch_files)


def read_epoch_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read one epoch CSV and verify the columns needed for plotting."""
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"T"}
        for mcmc_column, generated_column in PROPERTIES.values():
            required.update((mcmc_column, generated_column))
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: "
                f"{', '.join(sorted(missing))}"
            )
        return list(reader)


def plot_epochs(trial_dir: Path, output_path: Path) -> int:
    """Plot MCMC and generated-property curves for each epoch."""
    epoch_files = find_epoch_csvs(trial_dir)
    if not epoch_files:
        raise FileNotFoundError(
            f"No physical_properties_epoch_*.csv files found in {trial_dir}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    flat_axes = list(axes.flat)
    colors = colormaps["viridis"]

    for color_index, (epoch, csv_path) in enumerate(epoch_files):
        rows = read_epoch_rows(csv_path)
        points = sorted(
            [
                (
                    float(row["T"]),
                    {
                        column: float(row[column])
                        for pair in PROPERTIES.values()
                        for column in pair
                    },
                )
                for row in rows
            ],
            key=lambda point: point[0],
        )
        if not points:
            continue

        temperatures = [point[0] for point in points]
        color = colors(color_index / max(len(epoch_files) - 1, 1))
        for axis, (label, columns) in zip(flat_axes, PROPERTIES.items()):
            mcmc_column, generated_column = columns
            axis.plot(
                temperatures,
                [point[1][mcmc_column] for point in points],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color="r",
                linestyle="--",
                label="_nolegend_",
            )
            axis.plot(
                temperatures,
                [point[1][generated_column] for point in points],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=f"Epoch {epoch}" if axis is flat_axes[0] else "_nolegend_",
            )
            axis.set_title(label)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)

    for axis in flat_axes:
        axis.set_xlabel("Temperature (T)")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].legend(
        handles=[
            Line2D([0], [0], color="red", linestyle="--", label="MCMC"),
            Line2D(
                [0], [0], color="black", linestyle="-", label="Generated (RBM)"
            ),
        ],
        loc="best",
        title="Series",
    )
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        title="Epoch",
    )


    fig.suptitle(f"Physical properties across epochs: {trial_dir.name}")
    fig.tight_layout(rect=(0, 0, 0.84, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return len(epoch_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay generated physical properties across epochs for one trial."
        )
    )
    parser.add_argument(
        "trial_dir",
        type=Path,
        help="Trial directory containing physical_properties_epoch_*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path (default: <trial_dir>/"
            "physical_properties_epoch_10.png)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.trial_dir / "physical_properties_per_epoch.png"
    epoch_count = plot_epochs(args.trial_dir, output_path)
    print(f"Plotted {epoch_count} epochs to {output_path}")


if __name__ == "__main__":
    main()