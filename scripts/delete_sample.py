#!/usr/bin/env python3
import argparse
import json
import os
from os import path

import torch

import utils.config as cfg
import utils.sweep as swp
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Permanently delete sample arrays from RBM output runs"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint whose embedded configuration identifies the run",
    )
    group.add_argument(
        "--sweep",
        type=str,
        help="Sweep configuration; requires --config",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Run configuration, or base configuration when used with --sweep",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Delete only samples generated from this checkpoint epoch",
    )
    parser.add_argument(
        "--accumulate-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue through a sweep and report runs that could not be cleaned",
    )

    return parser.parse_args()


def get_samples_dir(config):
    output = config.get("output")
    if not isinstance(output, dict):
        raise ValueError("Configuration is missing its output section.")

    out_dir = output.get("base_dir")
    run_name = output.get("run_name")
    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("Configuration output.base_dir must be a non-empty string.")
    if not isinstance(run_name, str) or not run_name:
        raise ValueError("Configuration output.run_name must be a non-empty string.")

    return cfg.get_output_paths(out_dir, run_name)["samples"]


def delete_samples(samples_dir, epoch=None):
    if not path.isdir(samples_dir):
        raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")

    suffix = f"_epoch_{epoch}.npy" if epoch is not None else None
    meta_path = path.join(samples_dir, "metadata.json")
    if path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        if not isinstance(metadata, list):
            raise ValueError(f"Expected a list in sample metadata: {meta_path}")
    else:
        metadata = []

    if suffix is None:
        remaining_metadata = []
    else:
        remaining_metadata = [
            entry
            for entry in metadata
            if not (
                isinstance(entry, dict)
                and isinstance(entry.get("recon_file"), str)
                and entry["recon_file"].endswith(suffix)
                and isinstance(entry.get("gen_file"), str)
                and entry["gen_file"].endswith(suffix)
            )
        ]

    deleted = 0
    with os.scandir(samples_dir) as entries:
        for entry in entries:
            is_sample_array = entry.name.endswith(".npy")
            matches_epoch = suffix is None or entry.name.endswith(suffix)
            if (
                entry.is_file(follow_symlinks=False)
                and is_sample_array
                and matches_epoch
            ):
                os.remove(entry.path)
                deleted += 1

    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(remaining_metadata, meta_file, indent=2)

    epoch_message = f" for epoch {epoch}" if epoch is not None else ""
    print(f"Deleted {deleted} sample array(s){epoch_message} from {samples_dir}")
    return deleted


def delete_samples_for_config(config, epoch=None):
    samples_dir = get_samples_dir(config)
    return delete_samples(samples_dir, epoch=epoch)


def main():
    args = parse_args()

    if args.sweep and not args.config:
        raise SystemExit("--config is required when using --sweep")
    if args.epoch is not None and args.epoch < 1:
        raise SystemExit("--epoch must be at least 1")

    if args.checkpoint and args.config:
        print(
            "Warning: --config is ignored when --checkpoint is provided. "
            "Configuration embedded in the checkpoint is used."
        )

    if args.sweep:
        config = cfg.load_config(args.config)
        sweep = cfg.load_config(args.sweep)
        failed_runs = []

        for trial_group in swp.iter_sweep_configs(config, sweep):
            for trial_config in trial_group["configs"]:
                if args.accumulate_errors:
                    try:
                        delete_samples_for_config(trial_config, epoch=args.epoch)
                    except (OSError, ValueError) as error:
                        failed_runs.append(
                            {
                                "run_name": trial_config.get("output", {}).get(
                                    "run_name", "<unknown>"
                                ),
                                "error": str(error),
                            }
                        )
                        print(f"Run failed. Error: {error}")
                else:
                    delete_samples_for_config(trial_config, epoch=args.epoch)

        if failed_runs:
            print("Failed runs in sweep:")
            for failed_run in failed_runs:
                print(f"\t{failed_run['run_name']}: {failed_run['error']}")
        return

    if args.checkpoint:
        checkpoint = load_checkpoint(
            args.checkpoint, torch.device("cpu"), load_model=False
        )
        checkpoint_config = checkpoint.get("config")
        if not isinstance(checkpoint_config, dict):
            raise ValueError(
                f"Checkpoint does not contain a valid configuration: {args.checkpoint}"
            )
        delete_samples_for_config(checkpoint_config, epoch=args.epoch)
        return

    if args.config:
        config = cfg.load_config(args.config)
        delete_samples_for_config(config, epoch=args.epoch)
        return

    raise SystemExit("One of --checkpoint, --config, or --sweep must be used.")


if __name__ == "__main__":
    main()
