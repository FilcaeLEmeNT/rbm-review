import itertools
from os import path
import utils.config as cfg

def validate_sweep_keys(sweep: dict):
    for key in sweep.keys():
                keys = key.split(".")
                if keys[0] not in ["data", "model", "training", "output"]:
                    raise ValueError(f"Unexpected key in sweep file: '{key}'. \
                                    Expected format: key1.key2: [value1, value2, ...]. \
                                    Expected key1 are 'data', 'model', 'training', and 'output'")
                if keys[0] == "data" and keys[1] not in cfg.DATA_DEFAULTS.keys():
                    raise ValueError(f"Unexpected key in sweep file: '{key}'. \
                                    Expected format: key1.key2: [value1, value2, ...]. \
                                    Expected key2 when key1 is 'data' are {list(cfg.DATA_DEFAULTS.keys())}")
                if keys[0] == "model" and keys[1] not in cfg.MODEL_DEFAULTS.keys():
                    raise ValueError(f"Unexpected key in sweep file: '{key}'. \
                                    Expected format: key1.key2: [value1, value2, ...]. \
                                    Expected key2 when key1 is 'model' are {list(cfg.MODEL_DEFAULTS.keys())}")
                if keys[0] == "training" and keys[1] not in cfg.TRAINING_DEFAULTS.keys():
                    raise ValueError(f"Unexpected key in sweep file: '{key}'. \
                                    Expected format: key1.key2: [value1, value2, ...]. \
                                    Expected key2 when key1 is 'training' are {list(cfg.TRAINING_DEFAULTS.keys())}")
                if keys[0] == "output" and keys[1] not in cfg.DATA_DEFAULTS.keys():
                    raise ValueError(f"Unexpected key in sweep file: '{key}'. \
                                    Expected format: key1.key2: [value1, value2, ...]. \
                                    Expected key2 when key1 is 'model' are {list(cfg.DATA_DEFAULTS.keys())}")

def build_run_name(config, sweep, combo):
    prefix = config["output"]["run_name"]
    suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(sweep.keys(), combo))
    run_name = f"{prefix}_{suffix}"
    
    return run_name

def get_output_paths_from_sweep(config, sweep):
    paths_lists = {
        "checkpoints": [],
        "samples": [],
        "figures": [],
        "history": [],
        "physics": []
    }

    out_dir = config["output"]["base_dir"]
    for overwrites in itertools.product(*sweep.values()):
        run_name = build_run_name(config, sweep, overwrites)
        paths = cfg.get_output_paths(out_dir, run_name)

        paths_lists["checkpoints"].append(paths["checkpoints"])
        paths_lists["figures"].append(paths["figures"])
        paths_lists["history"].append(paths["history"])
        paths_lists["samples"].append(paths["samples"])
        paths_lists["physics"].append(paths["physics"])
        
    return paths_lists

def get_checkpoints_from_sweep(config, sweep):
    dir_paths_lists = get_output_paths_from_sweep(config, sweep)
    ckpt_dir_paths = dir_paths_lists["checkpoints"]
                          
    return [path.join(p, "checkpoint.pt") for p in ckpt_dir_paths]