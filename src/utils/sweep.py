# SPDX-License-Identifier: MIT
# src/utils/sweep.py

"""Utilities for the sweep configuration.

This module provides utilities for validating sweep configurations
and building unique run_names for each run in a sweep run. It also
includes helper functions to easily access multiple outputs from
sweep runs collectively.
"""

import itertools
from os import path
import utils.config as cfg
import copy

SPECIAL_TRIAL_KEYS = {"training.trials", "training.trials_range"}


def resolve_trial_indices(sweep: dict) -> list[int]:
    """Return the concrete trial indices selected by the sweep config."""
    trial_key = None
    for key in SPECIAL_TRIAL_KEYS:
        if key in sweep:
            if trial_key is not None:
                raise ValueError(
                    "Only one of training.trials or training.trials_range may be specified in a sweep."
                )
            trial_key = key

    if trial_key is None:
        return [0]

    values = sweep[trial_key]
    if trial_key == "training.trials":
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                "training.trials must be a list or tuple of integer trial indices."
            )
        return [int(v) for v in values]

    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(
            "training.trials_range must be a two-item list or tuple of [start, stop]."
        )

    start, stop = values
    if int(stop) < int(start):
        raise ValueError("training.trials_range must satisfy start <= stop.")
    return list(range(int(start), int(stop)))


def iter_trial_configs(
    config: dict,
    trial_indices: list[int] | None = None,
) -> list[dict]:
    """Return configs for each trial, with a distinct output directory per trial."""
    if trial_indices is None:
        trial_indices = [0]

    base_dir = config.get("output", {}).get("base_dir")
    expanded = []
    for trial_idx in [int(v) for v in trial_indices]:
        trial_config = copy.deepcopy(config)
        trial_config.setdefault("training", {})["trial_index"] = trial_idx
        if len(trial_indices) > 0 and base_dir is not None:
            trial_config["output"]["base_dir"] = path.join(
                base_dir, f"trial={trial_idx}"
            )
        elif len(trial_indices) == 1 and base_dir is not None:
            trial_config["output"]["base_dir"] = base_dir
        expanded.append(trial_config)
    return expanded


def validate_sweep_keys(sweep: dict) -> None:
    """Validate a sweep configuration dictionary's keys.

    The keys in the dictionaries ``DATA_SCHEMA``, ``MODEL_SCHEMA``, ``TRAINING_SCHEMA``, ``OUTPUT_SCHEMA``
    from config.py are used to check the configuration dictionary.

    Args:
        sweep: Sweep configuration dictionary parsed from the YAML configuration file.

    Raises:
        ValueError: If a key in the sweep config does not follow the correct format or is invalid.
    """
    for key in sweep.keys():
        keys = key.split(".")
        if key in SPECIAL_TRIAL_KEYS or key == "training.n_trials":
            if key == "training.trials":
                values = sweep[key]
                if not isinstance(values, (list, tuple)):
                    raise ValueError(
                        "training.trials must be a list or tuple of integer trial indices."
                    )
                for value in values:
                    if not isinstance(value, int):
                        raise ValueError(
                            "training.trials entries must be integers."
                        )
            if key == "training.trials_range":
                values = sweep[key]
                if not isinstance(values, (list, tuple)) or len(values) != 2:
                    raise ValueError(
                        "training.trials_range must be a two-item list or tuple [start, stop]."
                    )
                if not all(isinstance(v, int) for v in values):
                    raise ValueError(
                        "training.trials_range entries must be integers."
                    )
            continue
        if keys[0] not in cfg.TOP_LEVEL_CONFIG_KEYS:
            raise ValueError(
                f"Unexpected key in sweep file: '{key}'."
                "Expected format: key1.key2: [value1, value2, ...]."
                f"Expected key1 are {', '.join(sorted(cfg.TOP_LEVEL_CONFIG_KEYS))}"
            )
        if (
            keys[0] == cfg.DATA_SECTION
            and keys[1] not in cfg.DATA_SCHEMA.keys()
        ):
            raise ValueError(
                f"Unexpected key in sweep file: '{key}'."
                "Expected format: key1.key2: [value1, value2, ...]."
                f"Expected key2 when key1 is '{cfg.DATA_SECTION}' are {list(cfg.DATA_SCHEMA.keys())}"
            )
        if (
            keys[0] == cfg.MODEL_SECTION
            and keys[1] not in cfg.MODEL_SCHEMA.keys()
        ):
            raise ValueError(
                f"Unexpected key in sweep file: '{key}'."
                "Expected format: key1.key2: [value1, value2, ...]."
                f"Expected key2 when key1 is '{cfg.MODEL_SECTION}' are {list(cfg.MODEL_SCHEMA.keys())}"
            )
        if (
            keys[0] == cfg.TRAINING_SECTION
            and keys[1] not in cfg.TRAINING_SCHEMA.keys()
        ):
            raise ValueError(
                f"Unexpected key in sweep file: '{key}'."
                "Expected format: key1.key2: [value1, value2, ...]."
                f"Expected key2 when key1 is '{cfg.TRAINING_SECTION}' are {list(cfg.TRAINING_SCHEMA.keys())}"
            )
        if (
            keys[0] == cfg.OUTPUT_SECTION
            and keys[1] not in cfg.OUTPUT_SCHEMA.keys()
        ):
            raise ValueError(
                f"Unexpected key in sweep file: '{key}'."
                "Expected format: key1.key2: [value1, value2, ...]."
                f"Expected key2 when key1 is '{cfg.OUTPUT_SECTION}' are {list(cfg.OUTPUT_SCHEMA.keys())}"
            )


def update_config_with_overwrites(
    config: dict, sweep: dict, overwrites: tuple
) -> dict:
    """Update the configuration dictionary with the values from the sweep configuration.

    Args:
        config: Original configuration dictionary.
        sweep: Sweep configuration dictionary.
        overwrites: Tuple of values to overwrite in the config.

    Returns:
        Updated configuration dictionary.
    """
    updated_config = copy.deepcopy(config)
    for key, value in zip(sweep.keys(), overwrites):
        keys = key.split(".")
        d = updated_config
        for k in keys[:-1]:  # Go to the second to last dictionary.
            d = d[k]
        d[keys[-1]] = value  # Update the actual value in the nested dictionary.

    run_name = build_run_name(config, sweep, overwrites)
    updated_config["output"]["run_name"] = run_name

    return updated_config


def iter_sweep_configs(config: dict, sweep: dict) -> list[dict]:
    """Return configs grouped by trial, with each trial containing all sweep points."""
    sweep_params = build_sweep_params(sweep)
    trial_indices = resolve_trial_indices(sweep)
    grouped = {trial_idx: [] for trial_idx in trial_indices}

    if not sweep_params:
        for trial_idx in trial_indices:
            grouped[trial_idx] = iter_trial_configs(config, [trial_idx])
        return [
            {"trial_index": trial_idx, "configs": grouped[trial_idx]}
            for trial_idx in trial_indices
        ]

    for overwrites in itertools.product(*sweep_params.values()):
        sweep_overwrites = {
            key: value for key, value in zip(sweep_params.keys(), overwrites)
        }
        config_overwrite = update_config_with_overwrites(
            config, sweep_overwrites, overwrites
        )
        for trial_config in iter_trial_configs(config_overwrite, trial_indices):
            trial_idx = int(
                trial_config.get("training", {}).get("trial_index", 0)
            )
            grouped.setdefault(trial_idx, []).append(trial_config)

    return [
        {"trial_index": trial_idx, "configs": grouped[trial_idx]}
        for trial_idx in trial_indices
    ]


def build_sweep_params(sweep: dict) -> dict:
    """Return a sweep dict without trial-selection keys."""
    return {
        key: value
        for key, value in sweep.items()
        if key not in SPECIAL_TRIAL_KEYS
    }


def build_run_name(config: dict, sweep: dict, combo: tuple) -> str:
    """Return a unique run_name for a run in sweep with overwritten values.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
            Used to get the base run_name, which serves as a prefix for the
            complete run_name.
        sweep: Sweep configuration dictionary parsed from the YAML configuration file.
            Used to get the keys the are being overwritten.
        combo: One of the tuples outputted from the itertools.product(...) function
            containing the values to be overwritten.
    """
    prefix = config[cfg.OUTPUT_SECTION]["run_name"]
    sweep_items = [
        (k, v)
        for k, v in zip(sweep.keys(), combo)
        if k not in SPECIAL_TRIAL_KEYS and k != "training.n_trials"
    ]
    if not sweep_items:
        return prefix
    suffix = "_".join(format_sweep_value(k, v) for k, v in sweep_items)
    run_name = f"{prefix}_{suffix}"

    return run_name


def format_sweep_value(key: str, value: float | int | str | bool | dict) -> str:
    """Return a part of the final run_name for a particular key, value pair in sweep overwrites.

    Args:
        key: One of the full key names specified in the YAML configuration file.
        value: Value that is used to overwrite the original configuration.
    """
    name = key.split(".")[-1]

    if key in SPECIAL_TRIAL_KEYS or key == "training.n_trials":
        return ""

    if key == f"{cfg.TRAINING_SECTION}.schedule":
        parts = []
        for node in value:
            start = node["start"]
            params = ",".join(
                f"{k}={v}" for k, v in node.items() if k != "start"
            )
            parts.append(f"{start}:{params}")

        return f"{name}=" + ";".join(parts)

    return f"{name}={value}"


def get_output_paths_from_sweep(config: dict, sweep: dict) -> dict:
    """Return a dictionary of lists of output paths from a sweep run.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
            Used to get the base_dir of outputs
        sweep: Sweep configuration dictionary parsed from the YAML configuration file.
            Used to get the updated run_name for each run in sweep.
    """
    paths_lists = {
        "checkpoints": [],
        "samples": [],
        "figures": [],
        "history": [],
        "physics": [],
    }

    for trial_group in iter_sweep_configs(config, sweep):
        for trial_config in trial_group["configs"]:
            out_dir = trial_config[cfg.OUTPUT_SECTION]["base_dir"]
            run_name = trial_config[cfg.OUTPUT_SECTION]["run_name"]
            paths = cfg.get_output_paths(out_dir, run_name)

            paths_lists["checkpoints"].append(paths["checkpoints"])
            paths_lists["figures"].append(paths["figures"])
            paths_lists["history"].append(paths["history"])
            paths_lists["samples"].append(paths["samples"])
            paths_lists["physics"].append(paths["physics"])

    return paths_lists


def get_checkpoints_from_sweep(
    config: dict, sweep: dict, epoch: int | None = None
) -> list:
    """Return a list of checkpoint paths from a sweep run.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
        sweep: Sweep configuration dictionary parsed from the YAML configuration file.
    """
    dir_paths_lists = get_output_paths_from_sweep(config, sweep)
    ckpt_dir_paths = dir_paths_lists["checkpoints"]

    filename = (
        "checkpoint.pt" if epoch is None else f"checkpoint_epoch_{epoch}.pt"
    )
    return [path.join(p, filename) for p in ckpt_dir_paths]
