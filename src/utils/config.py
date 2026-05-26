import yaml
from os import path

# update the dicitonaries below to add/remove values to be specified in the configuration file.
DATA_DEFAULTS = {
    "data_type": None,
    "data_dir": None,
    "batch_size": None,
    "split": None,
    "binarize": None,
    "q": None,
    "T": None,
    "L": None
}

MODEL_DEFAULTS = {
    "model_type": None,
    "n_class": None,
    "n_visible": None,
    "n_hidden": None,
    "mf": None
}

TRAINING_DEFAULTS = {
    "n_epochs": None,
    "lr": None,
    "k": None,
    "pcd": None,
    "sm": None,
    "mc": None,
    "epsilon": None
}

OUTPUT_DEFAULTS = {
    "base_dir": None,
    "run_name": None
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
        if k not in DATA_DEFAULTS.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(DATA_DEFAULTS.keys())}.")
        
    for k in model_cfg.keys():
        if k not in MODEL_DEFAULTS.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(MODEL_DEFAULTS.keys())}.")
        
    for k in train_cfg.keys():
        if k not in TRAINING_DEFAULTS.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(TRAINING_DEFAULTS.keys())}.")
        
    for k in output_cfg.keys():
        if k not in OUTPUT_DEFAULTS.keys():
            raise ValueError(f"Unexpected key in config file: '{k}'. Expected keys for data are {list(OUTPUT_DEFAULTS.keys())}.")
        
def print_cfg_summary(config: dict, verbose: bool = True):
    '''
    Prints a summary of the configuration file.
    '''
    if verbose:
        # Write config with that containing None values to print out every possible value in print_cfg_summary.
        data_cfg = {
            k: config.get("data", {}).get(k, v)
            for k, v in DATA_DEFAULTS.items()
        }
        model_cfg = {
            k: config.get("model", {}).get(k, v)
            for k, v in MODEL_DEFAULTS.items()
        }
        train_cfg = {
            k: config.get("training", {}).get(k, v)
            for k, v in TRAINING_DEFAULTS.items()
        }
        output_cfg = {
            k: config.get("output", {}).get(k, v)
            for k, v in OUTPUT_DEFAULTS.items()
        }
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

