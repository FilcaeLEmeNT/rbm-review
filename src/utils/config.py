# SPDX-License-Identifier: MIT
# src/utils/config.py

"""Utilities for configuration.

This module provides utilities for loading, resolving, validating, and
summarizing experiment configurations. It also constructs output paths
and checkpoint locations used throughout the training, sampling,
and evaluation pipelines. Constant dictionaries define the expected schema
for each section of a valid configuration.

Attributes:
    DATA_SECTION: String specifying the data section key name.
    MODEL_SECTION: String specifying the model section key name.
    TRAINING_SECTION: String specifying the training section key name.
    OUTPUT_SECTION: String specifying the output section key name.
    TOP_LEVEL_CONFIG_KEYS: A frozen set of the valid top-level keys specified by
        the dictionaries ending with _SECTION above.
    DATA_SCHEMA: Schema for the data section of the configuration.
    MODEL_SCHEMA: Schema for the model section of the configuration.
    TRAINING_SCHEMA: Schema for the training section of the configuration.
    SCHEDULE_SCHEMA: Schema for an element of a schedule list inside the
        training section of a configuration.
    OUTPUT_SCHEMA: Schema for the output section of the configuration.

"""

import yaml
from os import path
import math

DATA_SECTION: str = "data"
MODEL_SECTION: str = "model"
TRAINING_SECTION: str = "training"
OUTPUT_SECTION: str = "output"
TOP_LEVEL_CONFIG_KEYS = frozenset(
    [DATA_SECTION, MODEL_SECTION, TRAINING_SECTION, OUTPUT_SECTION]
)

# update the dicitonaries below to add/remove values to be specified in the configuration file.
DATA_SCHEMA = {
    "data_type": {  # Choices are already handled in run_train.py with the VALID_MODEL_FOR_DATA constant.
        "type": str,
        "required": True,
    },
    "data_dir": {"type": str, "required": True},
    "batch_size": {"default": 64, "type": int, "min": 1},
    "split": {"default": None, "type": float, "min": 0.0, "max": 1.0},
    "binarize": {"default": False, "type": bool},
    "q": {"default": None, "type": int, "min": 1},
    "T": {"default": None, "type": (int, float)},
    "L": {"default": None, "type": int, "min": 1},
}

MODEL_SCHEMA = {
    "model_type": {  # Choices are already handled in run_train.py with the VALID_MODEL_FOR_DATA constant.
        "default": None,
        "type": str,
        "required": True,
    },
    "n_class": {"default": None, "type": int, "min": 1},
    "n_visible": {"default": None, "type": int, "min": 1},
    "n_hidden": {"default": None, "type": int, "min": 1},
    "mf": {"default": None, "type": bool},
}

TRAINING_SCHEMA = {
    "n_epochs": {"default": 500, "type": int, "min": 1},
    "lr": {"default": 0.01, "type": (int, float), "min": 0},
    "weight_decay": {"default": 0.0, "type": (int, float), "min": 0},
    "momentum": {"default": 0.0, "type": (int, float), "min": 0},
    "k": {"default": 10, "type": int, "min": 1},
    "pcd": {"default": True, "type": bool},
    "sm": {"default": False, "type": bool},
    "mc": {"default": "gibbs", "type": str, "choices": ["gibbs", "langevin"]},
    "epsilon": {"default": 0.05, "type": (int, float), "min": 0},
    "schedule": {"default": [{"start": 0}], "type": list},
}

SCHEDULE_SCHEMA = {
    "start": {"default": None, "type": int, "min": 0, "required": True},
    "lr": {"default": None, "type": (int, float), "min": 0},
    "weight_decay": {"default": None, "type": (int, float), "min": 0},
    "momentum": {"default": None, "type": (int, float), "min": 0},
    "k": {"default": None, "type": int, "min": 1},
    "pcd": {"default": None, "type": bool},
    "mc": {"default": None, "type": str, "choices": ["gibbs", "langevin"]},
    "epsilon": {"default": None, "type": (int, float), "min": 0},
}

OUTPUT_SCHEMA = {
    "base_dir": {"default": None, "type": str},
    "run_name": {"default": None, "type": str},
}


def load_config(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_config_keys(config: dict) -> None:
    """Validate a configuration dictionary's keys based on the constant dictionaries.

    The keys in the dictionaries ``DATA_SCHEMA``, ``MODEL_SCHEMA``, ``TRAINING_SCHEMA``, ``OUTPUT_SCHEMA``
    are used to check the configuration dictionary.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.

    Raises:
        ValueError: If a key in the config does not follow the format specified with the constant dictionaries.
    """
    # Check keys in the configuration file.
    for k in config.keys():
        if k not in TOP_LEVEL_CONFIG_KEYS:
            raise ValueError(
                f"Unexpected key in config file: '{k}'. Expected keys are {', '.join(sorted(TOP_LEVEL_CONFIG_KEYS))}."
            )

    # Check keys in each respective category of keys.
    data_cfg = config.get(DATA_SECTION, {})
    model_cfg = config.get(MODEL_SECTION, {})
    training_cfg = config.get(TRAINING_SECTION, {})
    output_cfg = config.get(OUTPUT_SECTION, {})

    for k in data_cfg.keys():
        if k not in DATA_SCHEMA.keys():
            raise ValueError(
                f"Unexpected key in config file: '{k}'. Expected keys for data are {list(DATA_SCHEMA.keys())}."
            )

    for k in model_cfg.keys():
        if k not in MODEL_SCHEMA.keys():
            raise ValueError(
                f"Unexpected key in config file: '{k}'. Expected keys for data are {list(MODEL_SCHEMA.keys())}."
            )

    for k in training_cfg.keys():
        if k not in TRAINING_SCHEMA.keys():
            raise ValueError(
                f"Unexpected key in config file: '{k}'. Expected keys for data are {list(TRAINING_SCHEMA.keys())}."
            )

    for k in output_cfg.keys():
        if k not in OUTPUT_SCHEMA.keys():
            raise ValueError(
                f"Unexpected key in config file: '{k}'. Expected keys for data are {list(OUTPUT_SCHEMA.keys())}."
            )


def resolve_section(config: dict, section: str, section_schema: dict) -> dict:
    """Resolve a configuration dictionary based on one of the constant dictionaries.

    The keys and rules in the dictionaries ``DATA_SCHEMA``, ``MODEL_SCHEMA``, ``TRAINING_SCHEMA``, ``OUTPUT_SCHEMA``
    are used to apply defaults and check values in the configuration dictionary.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
        section: String to specify which section/key to check in the config dictionary,
            e.g., DATA_SECTION, MODEL_SECTION, etc.
        section_schema: Dictionary that specifies valid keys, default values,
            min, max, choices, and required,
            e.g., ``DATA_SCHEMA``, ``MODEL_SCHEMA``, ``TRAINING_SCHEMA``, ``OUTPUT_SCHEMA``.

    Returns:
        Resolved configuration dictionary with defaults applied
            if values are not set in the original configuration dictionary.

    Raises:
        ValueError: If a required value is not given or if a given value does not follow constraints
            specified by the section_schema.
        TypeError: If a given value does not follow the type specified by the section_schema.
    """
    # Get the configuration
    raw_cfg = config.get(section, {})

    # Construct resolved configuration using the schema, replacing None values with defaults.
    resolved_cfg = {}
    for k, rule in section_schema.items():
        if k in raw_cfg and raw_cfg[k] is not None:
            resolved_cfg[k] = raw_cfg[k]
        else:
            resolved_cfg[k] = rule.get("default")
            if not rule.get("default") == None:
                print(
                    f"\033[1m{section}.{k} not specified in config. Defaulting to {rule.get('default')}\033[0m"
                )

    # Validate using schema
    for k, rule in section_schema.items():
        value = resolved_cfg[k]

        # required
        if rule.get("required") and value is None:
            raise ValueError(
                f"{section}.{k} is a required value. Update configuration file."
            )

        # type
        expected = rule.get("type")
        if expected and value is not None and not isinstance(value, expected):
            expected_name = (
                expected.__name__
                if isinstance(expected, type)
                else tuple(t.__name__ for t in expected)
            )

            raise TypeError(
                f"{section}.{k} needs to be of type {expected_name}, but instead got {type(value).__name__}"
            )

        # constraints
        if value is not None:
            if "min" in rule and value < rule["min"]:
                raise ValueError(
                    f"{section}.{k} needs to be greater than or equal to {rule['min']}, but got {value}"
                )
            if "max" in rule and value > rule["max"]:
                raise ValueError(
                    f"{section}.{k} needs to be less than or equal to {rule['max']}, but got {value}"
                )
            if "choices" in rule and value not in rule["choices"]:
                raise ValueError(
                    f"{section}.{k} needs to be one of {rule['choices']}, but got {value}"
                )

    return resolved_cfg


def resolve_data_cfg(config: dict) -> dict:
    """Resolve a configuration dictionary's data section based on ``DATA_SCHEMA``.

    The function ``resolve_section(...)`` is used to apply defaults and check values
    in the configuration dictionary using the constant dictionary, ``DATA_SCHEMA``.
    Also, defaulting with cross dependencies between different values are addressed.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.

    Returns:
        Resolved configuration dictionary with defaults applied
            if values are not set in the original configuration dictionary.

    Raises:
        ValueError: If a required value is not given or if a given value does not follow constraints
            specified by ``DATA_SCHEMA``.
        TypeError: If a given value does not follow the type specified by ``DATA_SCHEMA``.
    """
    # Infer the values with default and check if they follow the schema.
    data_cfg = resolve_section(config, DATA_SECTION, DATA_SCHEMA)

    # Handle inferring with dependencies.
    if data_cfg["split"] is None and data_cfg["data_type"] not in [
        "mnist",
        "cifar10",
        "stl10",
    ]:
        data_cfg["split"] = (
            0.8  # Default to 80% train, 20% test if not specified
        )
        print(
            f"\033[1mdata.split not specified in config. Defaulting to split = {data_cfg['split']}.\033[0m"
        )

    if data_cfg["binarize"] is None and data_cfg["data_type"] in [
        "mnist",
        "cifar10",
        "stl10",
    ]:
        data_cfg["binarize"] = False  # Default to False if not specified
        print(
            f"\033[1mdata.binarize not specified in config. Defaulting to binarize = {data_cfg['binarize']}.\033[0m"
        )

    config[DATA_SECTION] = data_cfg

    return config


def validate_config_preload(data_cfg: dict, model_type: str) -> None:
    """Validate a configuration dictionary's values before loading the data.

    The values in data_cfg and the model_type is used to check for invalid
    combinations of values.

    Args:
        data_cfg: Data section/dictionary of the configuration dictionary parsed from the
            YAML configuration file.
        model_type: The model_type specified in the configuration dictionary parsed from the
            YAML configuration file.

    Raises:
        ValueError: If a value in data_cfg is incompatible with a specific model_type
            or if a necessary value is missing.
    """
    # Validate the values
    if data_cfg["binarize"] and model_type == "multinomial":
        raise ValueError(
            f"binarize was set to {data_cfg['binarize']} but is not compatible with multinomial RBMs."
        )

    if ((data_cfg["T"] is None) or (data_cfg["L"] is None)) and data_cfg[
        "data_type"
    ] in ["ising", "xy", "potts"]:
        raise ValueError(
            f"data.T and data.L must be specified in config.yaml when data.data_type is '{data_cfg['data_type']}'. Please update config.yaml."
        )

    if (data_cfg["q"] is None) and model_type == "multinomial":
        raise ValueError(
            f"data.q must be specified in config.yaml when data.model_type is 'multinomial'. Please update config.yaml."
        )


def resolve_model_cfg(
    config: dict, default_n_visible: int, q: int | None
) -> dict:
    """Resolve a configuration dictionary's model section based on ``MODEL_SCHEMA``.

    The function ``resolve_section(...)`` is used to apply defaults and check values
    in the configuration dictionary using the constant dictionary, ``MODEL_SCHEMA``.
    The configuration is then updated to account for dependencies between model
    parameters, the inferred ``default_n_visible``, and the
    state/category count ``q`` for multinomial RBMs.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
        default_n_visible: A default value of n_visible inferred from the data
            with the ``infer_n_visible(...)`` function in ``run_train.py``.
        q: The number of q spins states (or categories) for the visisble units (data)
            specified in the YAML configuration file if multinomial RBM is used.

    Returns:
        Resolved configuration dictionary with defaults applied
            if values are not set in the original configuration dictionary.

    Raises:
        ValueError: If a required value is not given or if a given value does not follow constraints
            specified by ``MODEL_SCHEMA``.
        TypeError: If a given value does not follow the type specified by ``MODEL_SCHEMA``.
    """
    # Infer the values with default and check if they follow the schema.
    model_cfg = resolve_section(config, MODEL_SECTION, MODEL_SCHEMA)

    # Infer n_visible.
    if model_cfg["n_visible"] is None:
        model_cfg["n_visible"] = default_n_visible
        print(
            f"\033[1mmodel.n_visible not specified in config.yaml. Inferred n_visible = {model_cfg['n_visible']} from the data.\033[0m"
        )

    # Check if n_hidden is set in config, if not default to a value close to n_visible // 2
    if model_cfg["n_hidden"] is None:
        model_cfg["n_hidden"] = 2 ** math.floor(
            math.log2(max(1, model_cfg["n_visible"] // 2))
        )
        print(
            f"\033[1mmodel.n_hidden not specified in config. Defaulting to n_hidden = {model_cfg['n_hidden']}.\033[0m"
        )

    # Infer mf for binary RBMs
    if model_cfg["model_type"] == "binary" and model_cfg["mf"] is None:
        model_cfg["mf"] = True
        print(
            f"\033[1mmodel.mf not specified in config. Defaulting to mf = {model_cfg['mf']}.\033[0m"
        )

    # handle n_class
    if model_cfg["model_type"] == "multinomial":
        if model_cfg["n_class"] is None and q is not None:
            model_cfg["n_class"] = q
            print(
                f"\033[1mmodel.n_class not specified in config. Defaulting to n_class = data.q = {q}.\033[0m"
            )

    config["model"] = model_cfg

    return config


def validate_config_postload(
    model_cfg: dict, default_n_visible: int, q: int | None
) -> None:
    """Validate a configuration dictionary's values after loading the data.

    The values in ``model_cfg``, ``default_n_visible``, and the
    state/category count ``q`` is used to check for invalid combinations of values.

    Args:
        model_cfg: Model section/dictionary of the configuration dictionary parsed from the
            YAML configuration file.
        default_n_visible: A default value of n_visible inferred from the data
            with the ``infer_n_visible(...)`` function in ``run_train.py``.
        q: The number of q spins states (or categories) for the visisble units (data)
            specified in the YAML configuration file if multinomial RBM is used.

    Raises:
        ValueError: If a value in model_cfg is incompatible another value
            such as ``default_n_visible`` or ``q`` or if ``model.n_hidden`` is
            not a power of 2.
    """
    # Verify n_visible matches data
    if model_cfg["n_visible"] != default_n_visible:
        raise ValueError(
            f"n_visible in config ({model_cfg['n_visible']}) does not match the size of the input data ({default_n_visible}). Please update config.yaml."
        )

    # Verify n_hidden power of 2 rule
    if not (
        model_cfg["n_hidden"] > 0
        and (model_cfg["n_hidden"] & (model_cfg["n_hidden"] - 1)) == 0
    ):  # Check if power of 2
        raise ValueError(
            f"model.n_hidden must be a power of 2. Value specified is n_hidden={model_cfg['n_hidden']}. Please update config.yaml."
        )

    # mf compatibility
    if model_cfg["mf"] is not None and not model_cfg["model_type"] == "binary":
        raise ValueError(
            f"mf was set to {model_cfg.get('mf')} but is not compatible with model_type = {model_cfg['model_type']}. Please set mf to null."
        )

    # n_class
    if model_cfg["model_type"] == "multinomial":
        if model_cfg["n_class"] is None:
            if q is None:
                raise ValueError(
                    "model.n_class not specified in config. n_class's default value, data.q, is also not specified in config. Please update config.yaml."
                )

        if model_cfg["n_class"] != q and q is not None:
            raise ValueError(
                f"model.n_class={model_cfg['n_class']} and data.q={q} are both specified but do not match. They must be equal for multinomial RBM."
            )


def resolve_training_cfg(config: dict) -> dict:
    """Resolve a configuration dictionary's training section based on ``TRAINING_SCHEMA``.

    The function ``resolve_section(...)`` is used to apply defaults and check values
    in the configuration dictionary using the constant dictionary, ``TRAINING_SCHEMA``.
    Also, defaulting with cross dependencies between different values are addressed.

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.

    Returns:
        Resolved configuration dictionary with defaults applied
            if values are not set in the original configuration dictionary.

    Raises:
        ValueError: If a required value is not given or if a given value does not follow constraints
            specified by ``TRAINING_SCHEMA``.
        TypeError: If a given value does not follow the type specified by ``TRAINING_SCHEMA``.
    """
    # Infer the values with default and check if they follow the schema.
    training_cfg = resolve_section(config, TRAINING_SECTION, TRAINING_SCHEMA)

    config[TRAINING_SECTION] = training_cfg

    return config


def validate_schedule(training_cfg: dict) -> None:
    """Validate the schedule in the configuration dictionary's training section.

    Validates that the schedule is a list of dicts and that the keys and values of the dicts
    are valid using the function ``resolve_section(...)`` and the constant dictionary, ``SCHEDULE_SCHEMA``.

    Args:
        training_cfg: Training section/dictionary of the configuration dictionary parsed from the
            YAML configuration file.

    Raises:
        ValueError: If a required value is not given or if a given value does not follow constraints
            specified by ``SCHEDULE_SCHEMA`` or if the schedule's first ``start`` value
            is not 0 or if the schedule's ``start`` values are not in ascending order.
        TypeError: If the schedule is not a list of dicts or if a given value does
            not follow the type specified by ``SCHEDULE_SCHEMA``.
    """
    schedule = training_cfg.get("schedule", None)

    # Check if schedule is a list of dicts
    if not isinstance(schedule, list):
        raise TypeError(
            f"Scheudle needs to be formatted as a list of dictionaries. Expected type list but instead got {type(schedule).__name__}"
        )

    for i in range(len(schedule)):
        if not isinstance(schedule[i], dict):
            raise TypeError(
                f"Scheudle needs to be formatted as a list of dictionaries. Expected type dict but instead got {type(schedule[i]).__name__}"
            )

    # Check schedule schema and types. Create a dictionary with a custom section name
    # to use the resolve_section(...) function.
    schedule_resolved = []
    for idx, node_cfg in enumerate(schedule):
        node_cfg_resolved = resolve_section(
            {f"{TRAINING_SECTION}.schedule.{idx}": node_cfg},
            f"{TRAINING_SECTION}.schedule.{idx}",
            SCHEDULE_SCHEMA,
        )
        schedule_resolved.append(node_cfg_resolved)

    # Check start order.
    if not schedule_resolved[0]["start"] == 0:
        raise ValueError(
            f"{TRAINING_SECTION}.schedule.0.start must be 0 but instead got {schedule_resolved[0]['start']}"
        )

    for i in range(len(schedule)):
        # [i]start must be less than [i+1]start
        if i + 1 < len(schedule):
            if (
                not schedule_resolved[i]["start"]
                < schedule_resolved[i + 1]["start"]
            ):
                raise ValueError(
                    f"{TRAINING_SECTION}.schedule.{i + 1}.start must be greater than {TRAINING_SECTION}.schedule.{i}.start."
                    f"Got values {schedule_resolved[i + 1]['start']} and {schedule_resolved[i]['start']}."
                )

    return


def print_cfg_summary(config: dict, *, verbose: bool = True) -> None:
    """Prints a summary of the configuration file.

    Prints the contents of the configuration file in the format,

        {section1_name} Parameters:
            {key1}={value1}

            {key2}={value2}
            ...
        {section2_name} Parameters:
            {key1}={value1}

            {key2}={value2}
            ...
        ...

    Args:
        config: Configuration dictionary parsed from the YAML configuration file.
        verbose: If ``True``, prints every possible parameter even if they were
            not specified in the configuration dictionary. The value printed
            will be None if it was unspecified in the configuration dictionary.
    """
    if verbose:
        # Write config with that containing None values to print out every possible value in print_cfg_summary.
        data_cfg = {
            k: config.get(DATA_SECTION, {}).get(k, None) for k in DATA_SCHEMA
        }
        model_cfg = {
            k: config.get(MODEL_SECTION, {}).get(k, None) for k in MODEL_SCHEMA
        }
        training_cfg = {
            k: config.get(TRAINING_SECTION, {}).get(k, None)
            for k in TRAINING_SCHEMA
        }
        output_cfg = {
            k: config.get(OUTPUT_SECTION, {}).get(k, None)
            for k in OUTPUT_SCHEMA
        }

    else:
        # Get dictionaries from configuration.
        data_cfg = config.get(DATA_SECTION, {})
        model_cfg = config.get(MODEL_SECTION, {})
        training_cfg = config.get(TRAINING_SECTION, {})
        output_cfg = config.get(OUTPUT_SECTION, {})

    # Print config summary
    print(f"Config summary:")
    print("Data parameters:")
    for k, v in data_cfg.items():
        print(f"\t{k}={v}")

    print("Model parameters:")
    for k, v in model_cfg.items():
        print(f"\t{k}={v}")

    print("Training parameters:")
    for k, v in training_cfg.items():
        print(f"\t{k}={v}")

    print("Output parameters:")
    for k, v in output_cfg.items():
        print(f"\t{k}={v}")

    return


def get_output_paths(out_dir: str, run_name: str) -> dict:
    """Construct the output directory paths for a training run.

    Args:
        out_dir: Base output directory.
        run_name: Name of the training run.

    Returns: A dictionary mapping output categories to their corresponding directory paths.
        The returned dictionary contains the following keys:
        - ``checkpoints``
        - ``samples``
        - ``figures``
        - ``history``
        - ``physics``
    """
    return {
        "checkpoints": path.join(out_dir, "checkpoints", run_name),
        "samples": path.join(out_dir, "samples", run_name),
        "figures": path.join(out_dir, "figures", run_name),
        "history": path.join(out_dir, "history", run_name),
        "physics": path.join(out_dir, "physics", run_name),
    }


def get_checkpoint_path_from_config(config: dict) -> str:
    """Construct the checkpoint file path from the configuration.

    Args:
        config: Configuration dictionary containing the output settings.

    Returns: The full path to ``checkpoint.pt`` for the configured training run.

    Raises:
        KeyError: If the required output configuration entries are missing.
    """
    out_dir = config[OUTPUT_SECTION]["base_dir"]
    run_name = config[OUTPUT_SECTION]["run_name"]
    dir_paths_list = get_output_paths(out_dir=out_dir, run_name=run_name)
    ckpt_dir = dir_paths_list["checkpoints"]

    return path.join(ckpt_dir, "checkpoint.pt")
