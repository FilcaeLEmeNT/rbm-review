import itertools
from os import path
def build_run_name(config, sweep, combo):
    prefix = config["output"]["run_name"]
    suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(sweep.keys(), combo))
    run_name = f"{prefix}_{suffix}"
    
    return run_name