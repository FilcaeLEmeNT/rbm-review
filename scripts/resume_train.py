#!/usr/bin/env python3
import argparse
import os
from os import path
import numpy as np
import math
import copy

from utils.device import get_device
import utils.config as cfg
import utils.sweep as swp

from data.data_loader import load_data

from training.training import train_cd, train_sm, train_ptt

from utils.checkpoint import save_checkpoint, load_checkpoint


def _merge_histories(
    previous_history, current_history, previous_epochs, current_epochs
):
    merged_history = {}
    for key in dict.fromkeys([*previous_history, *current_history]):
        previous_values = list(previous_history.get(key, []))
        current_values = list(current_history.get(key, []))
        if key != "pt_index_history":
            previous_values.extend(
                [None] * max(0, previous_epochs - len(previous_values))
            )
            current_values.extend(
                [None] * max(0, current_epochs - len(current_values))
            )
        merged_history[key] = previous_values + current_values
    return merged_history


def parse_args():
    parser = argparse.ArgumentParser(description="Continue Training RBM model")

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
        "--accumulate-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to skip runs with errors and accumulate error messages when running sweep.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file. Used to infer the checkpoint path from output.basedir and output.run_name. \
            Use as an alternative to --checkpoint, or combine with --sweep to resume all runs generated from that config/sweep combination.",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs to train, resuming from a checkpoint.",
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

    # Check args
    if args.sweep and not args.config:
        argparse.error("--config is required when using --sweep")

    if args.checkpoint and args.config:
        print(
            "Warning: --config is ignored when --checkpoint is provided. Configuration file embedded in the checkpoint is used."
        )

    # Load device: Either CPU or CUDA
    device = get_device()

    if args.sweep:
        config = cfg.load_config(args.config)
        sweep = cfg.load_config(args.sweep)
        trial_groups = swp.iter_sweep_configs(config, sweep)

        for trial_group in trial_groups:
            for trial_config in trial_group["configs"]:
                ckpt_path = cfg.get_checkpoint_path_from_config(
                    trial_config, args.epoch
                )
                resume_train(
                    device,
                    ckpt_path,
                    args.n_epochs,
                    config=trial_config,
                )

    elif args.checkpoint:
        # Get checkpoint path
        ckpt_path = args.checkpoint
        resume_train(device, ckpt_path, args.n_epochs)

    elif args.config:
        # Get checkpoint path
        config = cfg.load_config(args.config)
        ckpt_path = cfg.get_checkpoint_path_from_config(config, args.epoch)
        resume_train(device, ckpt_path, args.n_epochs, config=config)

    else:
        argparse.error(
            "Atleast one of --checkpoint, --config, and --sweep must be used."
        )

    return


def resume_train(device, ckpt_path, n_epochs_arg, config=None):
    # Get loaded model and checkpoint
    rbm, ckpt = load_checkpoint(ckpt_path, device)

    if rbm is None:
        raise ValueError(
            f"The model was not loaded from the checkpoint file properly."
        )
    if ckpt is None:
        raise ValueError(
            f"The checkpoint file was not loaded from the checkpoint file properly."
        )

    n_epochs_prev = ckpt["epoch"]
    history = ckpt["history"]

    checkpoint_config = ckpt.get("config", {}) or {}
    effective_config = dict(checkpoint_config)
    if isinstance(config, dict):
        output_override = config.get("output", {})
        if isinstance(output_override, dict):
            effective_config["output"] = {
                **effective_config.get("output", {}),
                **output_override,
            }

    updated_config = copy.deepcopy(effective_config)

    # Get values from configuration.
    data_cfg = effective_config.get("data", {})
    model_cfg = effective_config.get("model", {})
    training_cfg = effective_config.get("training", {})
    output_cfg = effective_config.get("output", {})

    model_type = model_cfg.get("model_type")

    # n_epochs = training_cfg.get("n_epochs")

    out_dir = output_cfg.get("base_dir")
    run_name = output_cfg.get("run_name")
    checkpoint_interval = training_cfg.get("checkpoint_interval")
    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)

    def save_periodic_checkpoint(epoch, session_history):
        if checkpoint_interval is None or epoch % checkpoint_interval != 0:
            return
        checkpoint_config = copy.deepcopy(updated_config)
        checkpoint_config["training"]["n_epochs"] = epoch
        os.makedirs(checkpoints_dir, exist_ok=True)
        save_checkpoint(
            model=rbm,
            optimizer=None,
            epoch=epoch,
            config=checkpoint_config,
            history=_merge_histories(
                history, session_history, n_epochs_prev, epoch - n_epochs_prev
            ),
            path=path.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt"),
        )
        print(
            f"Checkpoint file, 'checkpoint_epoch_{epoch}.pt', saved to directory: {checkpoints_dir}"
        )

    # Print configuration summary.
    cfg.print_cfg_summary(effective_config)
    print("")

    # Load data.
    train_loader, test_loader = load_data(data_cfg, model_type)

    # Print model information
    print(f"Using model type: {model_type}")

    if model_type == "binary":
        print(f"Using mean-field: {model_cfg['mf']}")
        print(f"Using binarize: {data_cfg['binarize']}")
    elif model_type == "multinomial":
        print(f"Number of categories: {model_cfg['n_class']}")

    # Get the history and n_epochs from the checkpoint.
    n_epochs = n_epochs_arg  # Overwrite n_epochs with how many epochs to resume training.

    print(f"\nResuming training, starting from {n_epochs_prev} epochs.")

    # Train the model
    rbm.train()
    if training_cfg["algorithm"] == "SM" and model_type == "gaussian":
        history_temp = train_sm(
            rbm,
            device,
            train_loader,
            training_cfg,
            n_epochs,
            n_epochs_prev,
            checkpoint_callback=save_periodic_checkpoint,
        )
    elif training_cfg["algorithm"] == "PTT":
        raise NotImplementedError(
            "PTT training is not implemented yet. Please use MLE or SM."
        )
        history_temp = train_ptt(
            rbm, device, train_loader, training_cfg, n_epochs, n_epochs_prev
        )
    else:
        history_temp = train_cd(
            rbm,
            device,
            train_loader,
            training_cfg,
            n_epochs,
            n_epochs_prev,
            checkpoint_callback=save_periodic_checkpoint,
        )

    history = _merge_histories(history, history_temp, n_epochs_prev, n_epochs)

    # Get total epochs
    n_epochs = n_epochs_prev + n_epochs
    updated_config["training"]["n_epochs"] = n_epochs

    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)

    save_checkpoint(
        model=rbm,
        optimizer=None,
        epoch=n_epochs,
        config=updated_config,
        history=history,
        path=path.join(checkpoints_dir, "checkpoint.pt"),
    )
    print(
        f"\nCheckpoint file, 'checkpoint.pt', saved to directory: {checkpoints_dir}"
    )
    print("Total Epochs:", n_epochs)

    return


if __name__ == "__main__":
    main()
