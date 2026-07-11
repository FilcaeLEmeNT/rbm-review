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
    suffix = "_".join(
        format_sweep_value(k, v) for k, v in zip(sweep.keys(), combo)
    )
    run_name = f"{prefix}_{suffix}"

    return run_name


def format_sweep_value(key: str, value: float | int | str | bool | dict) -> str:
    """Return a part of the final run_name for a particular key, value pair in sweep overwrites.

    Args:
        key: One of the full key names specified in the YAML configuration file.
        value: Value that is used to overwrite the original configuration.
    """
    name = key.split(".")[-1]

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

    out_dir = config[cfg.OUTPUT_SECTION]["base_dir"]
    for overwrites in itertools.product(*sweep.values()):
        run_name = build_run_name(config, sweep, overwrites)
        paths = cfg.get_output_paths(out_dir, run_name)

        paths_lists["checkpoints"].append(paths["checkpoints"])
        paths_lists["figures"].append(paths["figures"])
        paths_lists["history"].append(paths["history"])
        paths_lists["samples"].append(paths["samples"])
        paths_lists["physics"].append(paths["physics"])

    return paths_lists


def get_checkpoints_from_sweep(config: dict, sweep: dict) -> list:
    """Return a list of checkpoint paths from a sweep run.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
        sweep: Sweep configuration dictionary parsed from the YAML configuration file.
    """
    dir_paths_lists = get_output_paths_from_sweep(config, sweep)
    ckpt_dir_paths = dir_paths_lists["checkpoints"]

    return [path.join(p, "checkpoint.pt") for p in ckpt_dir_paths]
