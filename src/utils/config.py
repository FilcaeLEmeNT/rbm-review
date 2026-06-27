import yaml
from os import path
import math

# update the dicitonaries below to add/remove values to be specified in the configuration file.
DATA_CONFIG = {
    "data_type": {  # Choices are already handled in run_train.py with the VALID_MODEL_FOR_DATA constant.
        "default": None,
        "type": str,
        "required": True,
    },
    "data_dir": {
        "default": None,
        "type": str,
        "required": True,
    },
    "batch_size": {
        "default": 64,
        "type": int,
        "min": 1,
    },
    "split": {
        "default": None,
        "type": float,
        "min": 0.0,
        "max": 1.0,
    },
    "binarize": {
        "default": False,
        "type": bool,
    },
    "q": {
        "default": None,
        "type": int,
        "min": 1,
    },
    "T": {
        "default": None,
        "type": (int, float),
    },
    "L": {
        "default": None,
        "type": int,
        "min": 1,
    }
}

MODEL_CONFIG = {
    "model_type": {  # Choices are already handled in run_train.py with the VALID_MODEL_FOR_DATA constant.
        "default": None,
        "type": str,
        "required": True,
    },
    "n_class": {
        "default": None,
        "type": int,
        "min": 1
    },
    "n_visible": {
        "default": None,
        "type": int,
        "min": 1
    },
    "n_hidden": {
        "default": None,
        "type": int,
        "min": 1
    },
    "mf": {
        "default": None,
        "type": bool
    }
}

TRAINING_CONFIG = {
    "n_epochs": {
        "default": 500,
        "type": int,
        "min": 1
    },
    "lr": {
        "default": 0.01,
        "type": (int, float),
        "min": 0
    },
    "weight_decay": {
        "default": 0.,
        "type": (int, float),
        "min": 0
    },
    "momentum": {
        "default": 0.,
        "type": (int, float),
        "min": 0
    },
    "k": {
        "default": 10,
        "type": int,
        "min": 1
    },
    "pcd": {
        "default": True,
        "type": bool,
    },
    "sm": {
        "default": False,
        "type": bool,
    },
    "mc": {
        "default": "gibbs",
        "type": str,
        "choices": ["gibbs", "langevin"]
    },
    "epsilon": {
        "default": 0.05,
        "type": (int, float),
        "min": 0
    },
    "schedule": {
        "default": [{"start": 0}],
        "type": list
    }
}

SCHEDULE_CONFIG = {
    "start": {
        "default": None,
        "type": int,
        "min":0,
        "required": True
    },
    "lr": {
        "default": None,
        "type": (int, float),
        "min": 0
    },
    "weight_decay": {
        "default": None,
        "type": (int, float),
        "min": 0
    },
    "momentum": {
        "default": None,
        "type": (int, float),
        "min": 0
    },
    "k": {
        "default": None,
        "type": int,
        "min": 1
    },
    "pcd": {
        "default": None,
        "type": bool,
    },
    "mc": {
        "default": None,
        "type": str,
        "choices": ["gibbs", "langevin"]
    },
    "epsilon": {
        "default": None,
        "type": (int, float),
        "min": 0
    },
}

OUTPUT_DEFAULTS = {
    "base_dir": {
        "default": None,
        "type": str
    },
    "run_name": {
        "default": None,
        "type": str
    }
}

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def validate_config_keys(config: dict) -> None:
    # Check keys in the configuration file.
    for k in config.keys():
        if k not in ["data", "model", "training", "output"]:
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys are 'data', 'model', 'training', and 'output'.")
    
    # Check keys in each respective category of keys.
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    for k in data_cfg.keys():
        if k not in DATA_CONFIG.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(DATA_CONFIG.keys())}.")
        
    for k in model_cfg.keys():
        if k not in MODEL_CONFIG.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(MODEL_CONFIG.keys())}.")
        
    for k in train_cfg.keys():
        if k not in TRAINING_CONFIG.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(TRAINING_DEFAULTS.keys())}.")
        
    for k in output_cfg.keys():
        if k not in OUTPUT_DEFAULTS.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(OUTPUT_DEFAULTS.keys())}.")

def resolve_section(config: dict, section: str, section_schema: dict) -> dict:
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
                print(f"\033[1m{section}.{k} not specified in config. Defaulting to {rule.get("default")}\033[0m")

    # Validate using schema
    for k, rule in section_schema.items():
        value = resolved_cfg[k]

        # required
        if rule.get("required") and value is None:
            raise ValueError(f"{section}.{k} is a required value. Update configuration file.")

        # type
        expected = rule.get("type")
        if expected and value is not None and not isinstance(value, expected):
            expected_name = (
                expected.__name__ if isinstance(expected, type)
                else tuple(t.__name__ for t in expected)
            )

            raise TypeError(f"{section}.{k} needs to be of type {expected_name}, but instead got {type(value).__name__}")
        
        # constraints
        if value is not None:
            if "min" in rule and value < rule["min"]:
                raise ValueError(f"{section}.{k} needs to be greater than or equal to {rule["min"]}, but got {value}")
            if "max" in rule and value > rule["max"]:
                raise ValueError(f"{section}.{k} needs to be less than or equal to {rule["max"]}, but got {value}")
            if "choices" in rule and value not in rule["choices"]:
                raise ValueError(f"{section}.{k} needs to be one of {rule["choices"]}, but got {value}")
            
    return resolved_cfg

def resolve_data_cfg(config: dict) -> dict:
    # Infer the values with default and check if they follow the schema.
    data_cfg = resolve_section(config, "data", DATA_CONFIG)

    # Handle inferring with dependencies.
    if data_cfg["split"] is None and data_cfg["data_type"] not in ["mnist", "cifar10", "stl10"]:
        data_cfg["split"] = 0.8  # Default to 80% train, 20% test if not specified
        print(f"\033[1mdata.split not specified in config. Defaulting to split = {data_cfg["split"]}.\033[0m")

    if data_cfg["binarize"] is None and data_cfg["data_type"] in ["mnist", "cifar10", "stl10"]:
        data_cfg["binarize"] = False  # Default to False if not specified
        print(f"\033[1mdata.binarize not specified in config. Defaulting to binarize = {data_cfg["binarize"]}.\033[0m")

    config["data"] = data_cfg

    return config

def validate_config_preload(data_cfg, model_type):
    # Validate the values
    if data_cfg["binarize"] and model_type == 'multinomial':
        raise ValueError(f"binarize was set to {data_cfg["binarize"]} but is not compatible with multinomial RBMs.")
    
    if ((data_cfg["T"] is None) or (data_cfg["L"] is None)) and data_cfg["data_type"] in ["ising", "xy", "potts"]:
        raise ValueError(f"data.T and data.L must be specified in config.yaml when data.data_type is '{data_cfg["data_type"]}'. Please update config.yaml.")

    if (data_cfg["q"] is None) and model_type == "multinomial":
        raise ValueError(f"data.q must be specified in config.yaml when data.model_type is 'multinomial'. Please update config.yaml.")

def resolve_model_cfg(config: dict, default_n_visible, q) -> dict:
    # Infer the values with default and check if they follow the schema.
    model_cfg = resolve_section(config, "model", MODEL_CONFIG)

    # Infer n_visible.
    if model_cfg["n_visible"] is None:
        model_cfg["n_visible"] = default_n_visible
        print(f"\033[1mmodel.n_visible not specified in config.yaml. Inferred n_visible = {model_cfg["n_visible"]} from the data.\033[0m")
    
    # Check if n_hidden is set in config, if not default to a value close to n_visible // 2 
    if model_cfg["n_hidden"] is None:
        model_cfg["n_hidden"] = 2 ** math.floor(math.log2(max(1, model_cfg["n_visible"] // 2)))
        print(f"\033[1mmodel.n_hidden not specified in config. Defaulting to n_hidden = {model_cfg["n_hidden"]}.\033[0m")
    
    # Infer mf for binary RBMs
    if model_cfg["model_type"] == "binary" and model_cfg["mf"] is None:
        model_cfg["mf"] = True
        print(f"\033[1mmodel.mf not specified in config. Defaulting to mf = {model_cfg["mf"]}.\033[0m")

    # handle n_class
    if model_cfg["model_type"] == "multinomial":
        if model_cfg["n_class"] is None and q is not None:
            model_cfg["n_class"] = q
            print(f"\033[1mmodel.n_class not specified in config. Defaulting to n_class = data.q = {q}.\033[0m")

    config["model"] = model_cfg

    return config

def validate_config_postload(model_cfg, default_n_visible, q):
    # Verify n_visible matches data
    if model_cfg["n_visible"] != default_n_visible:
        raise ValueError(f"n_visible in config ({model_cfg["n_visible"]}) does not match the size of the input data ({default_n_visible}). Please update config.yaml.")
    
    # Verify n_hidden power of 2 rule
    if not (model_cfg["n_hidden"] > 0 and (model_cfg["n_hidden"] & (model_cfg["n_hidden"] - 1)) == 0):  # Check if power of 2
        raise ValueError(f"model.n_hidden must be a power of 2. Value specified is n_hidden={model_cfg["n_hidden"]}. Please update config.yaml.")
    
    # mf compatibility
    if model_cfg["mf"] is not None and not model_cfg["model_type"] == "binary":
        raise ValueError(f"mf was set to {model_cfg.get("mf")} but is not compatible with model_type = {model_cfg["model_type"]}. Please set mf to null.")
    
    # n_class
    if model_cfg["model_type"] == "multinomial":
        if model_cfg["n_class"] is None:
            if q is None:
                raise ValueError("model.n_class not specified in config. n_class's default value, data.q, is also not specified in config. Please update config.yaml.")

        if model_cfg["n_class"] != q and q is not None:
            raise ValueError(f"model.n_class={model_cfg["n_class"]} and data.q={q} are both specified but do not match. They must be equal for multinomial RBM.")

def resolve_train_cfg(config: dict) -> dict:
    # Infer the values with default and check if they follow the schema.
    train_cfg = resolve_section(config, "training", TRAINING_CONFIG)

    config["training"] = train_cfg

    return config

def validate_schedule(train_cfg: dict):
    schedule = train_cfg.get("schedule", None)
    
    # Check if schedule is a list of dicts
    for i in range(len(schedule)):
        if not isinstance(schedule[i], dict):
            raise TypeError(f"Scheudle needs to be formatted as a list of dictionaries. Expected type dict but instead got {type(schedule[i]).__name__}")
    
    # Check schedule schema and types.
    schedule_resolved = []
    for idx, node_cfg in enumerate(schedule):
        node_cfg_resolved = resolve_section({f"training.schedule.{idx}": node_cfg}, f"training.schedule.{idx}", SCHEDULE_CONFIG)
        schedule_resolved.append(node_cfg_resolved)
    
    # Check start order.
    if not schedule_resolved[0]["start"] == 0:
        raise ValueError(f"training.schedule.0.start must be 0 but instead got {schedule_resolved[0]["start"]}")
    
    for i in range(len(schedule)):
        # [i]start must be less than [i+1]start
        if i+1 < len(schedule):
            if not schedule_resolved[i]["start"] < schedule_resolved[i+1]["start"]:
                raise ValueError(f"training.schedule.{i+1}.start must be greater than training.schedule.{i}.start. Got training.schedule.{i+1}.start={schedule_resolved[i+1]["start"]}. training.schedule.{i}.start={schedule_resolved[i]["start"]}.")
    
    return

def print_cfg_summary(config: dict, verbose: bool = True):
    '''
    Prints a summary of the configuration file.
    '''
    if verbose:
        # Write config with that containing None values to print out every possible value in print_cfg_summary.
        data_cfg = {k: config.get("data", {}).get(k, None) for k in DATA_CONFIG}
        model_cfg = {k: config.get("model", {}).get(k, None) for k in MODEL_CONFIG}
        train_cfg = {k: config.get("training", {}).get(k, None) for k in TRAINING_CONFIG}
        output_cfg = {k: config.get("output", {}).get(k, None) for k in OUTPUT_DEFAULTS}

    else:
        # Get dictionaries from configuration.
        data_cfg = config.get("data", {})
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        output_cfg = config.get("output", {})

    # Print config summary
    print(f"Config summary:")
    print("Data parameters:")
    for k, v in data_cfg.items():
        print(f"\t{k}={v}")
    
    print("Model parameters:")
    for k, v in model_cfg.items():
        print(f"\t{k}={v}")
        
    print("Training parameters:")
    for k, v in train_cfg.items():
        print(f"\t{k}={v}")

    print("Output parameters:")
    for k, v in output_cfg.items():
        print(f"\t{k}={v}")

    return

def get_output_paths(out_dir: str, run_name: str):
    return {
        "checkpoints": path.join(out_dir, "checkpoints", run_name),
        "samples": path.join(out_dir, "samples", run_name),
        "figures": path.join(out_dir, "figures", run_name),
        "history": path.join(out_dir, "history", run_name),
        "physics": path.join(out_dir, "physics", run_name),
    }

def get_checkpoint_from_config(config: dict):
    out_dir = config["output"]["base_dir"]
    run_name = config["output"]["run_name"]
    dir_paths_list = get_output_paths(out_dir=out_dir, run_name=run_name)
    ckpt_dir = dir_paths_list["checkpoints"]

    return path.join(ckpt_dir, "checkpoint.pt")