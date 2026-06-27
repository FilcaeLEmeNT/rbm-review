import argparse
import os
from os import path
import math
import itertools
import copy

from utils.device import get_device
import utils.config as cfg
import utils.sweep as swp

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import save_checkpoint

# model-data compatibility
VALID_MODEL_FOR_DATA = {
    "mnist": ["binary", "exponential", "gaussian", "multinomial"],
    "cifar10": ["binary", "exponential", "gaussian", "multinomial"],
    "stl10": ["binary", "exponential", "gaussian", "multinomial"],
    "ising": ["binary"],
    "potts": ["multinomial"],
    "xy": ["vonmises"],
    "wind_dir": ["vonmises"],
    "protein": ["vonmises"],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train RBM model")

    parser.add_argument(
        "--config",
        type=str,
        default=path.join("configs", "default.yaml"),
        help="Path to config file"
    )

    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Path to sweep configuration file"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load device: Either CPU or CUDA
    device = get_device()

    # Load configuration file
    config_dict = cfg.load_config(args.config)
    print(f"Using config file: {args.config}")

    # Validate config keys
    cfg.validate_config_keys(config_dict)

    # Print configuration summary.
    cfg.print_cfg_summary(config_dict)
    print("")

    # Load sweep if available.
    if args.sweep:
        sweep_dict = cfg.load_config(args.sweep)
        print(f"Using sweep configuration file: {args.sweep}")

        # Check if sweep config is valid.
        swp.validate_sweep_keys(sweep_dict)

        # Print sweep configuration summary.
        print("Sweep configuration summary:")
        for k, v in sweep_dict.items():
            print(f"\t{k}={v}")\
            
        print("")

        for overwrites in itertools.product(*sweep_dict.values()):
            config_overwrite = copy.deepcopy(config_dict)

            print("Sweep running with overwrites:")
            for key, value in zip(sweep_dict.keys(), overwrites):
                keys = key.split(".")
                d = config_overwrite
                for k in keys[:-1]:  # Go to the second to last dictionary.
                    d = d[k]
                d[keys[-1]] = value  # Update the actual value in the nested dictionary.
                print(f"\t{key}={value}")
            
            # Change run_name for each run:
            run_name = swp.build_run_name(config_dict, sweep_dict, overwrites)
            config_overwrite["output"]["run_name"] = run_name
            print(f"\trun_name={run_name}")
            print("")

            run_train(device, config_overwrite)

    else:
        run_train(device, config_dict)

def run_train(device, config: dict):
    '''
    Given a configuration, run training and ouput a checkpoint in the directory
    specified by output.base_dir and output.run_name in the configuration.

    Parameters:
    - device : device outputted by get_device()
    - config : dictionary containing settings for training and output.

    Returns: None
    
    Outputs:
    - checkpoint.pt file: Upon running training, a checkpoint file is saved
    which contains everything needed to reconstruct the model without any additional arguments.

    Input size: [batch_size, p * p]
    Output size: [batch_size, ]
    '''
    # Modify config along the way with defaults, etc.
    updated_config = copy.deepcopy(config)
    
    # Validate datasets and model compatibility:
    data_type = updated_config.get("data", {}).get("data_type")
    model_type = updated_config.get("model", {}).get("model_type")
    valid_models = VALID_MODEL_FOR_DATA.get(data_type)
    if valid_models is None:
        raise ValueError(f"Unknown data type: {data_type}. Please update data.data_type in config.yaml. Refer to default.yaml for supported types.")
    if model_type not in valid_models:
        raise ValueError(
            f"Model '{model_type}' is incompatible with data type '{data_type}'. Please update model.model_type in config.yaml. "
            f"Valid models: {valid_models}"
        )
    
    # Check data values before passing it to load_data.
    updated_config = cfg.resolve_data_cfg(updated_config)
    data_cfg = updated_config.get("data", {})
        
    cfg.validate_config_preload(data_cfg, model_type)

    # Load data.
    train_loader, test_loader = load_data(data_cfg, model_type)
    
    # Check if n_visible is set in config, if not infer from data. If set, check if it matches the data.
    default_n_visible = infer_n_visible(test_loader, model_type, data_cfg["q"])

    updated_config = cfg.resolve_model_cfg(updated_config, default_n_visible, data_cfg["q"])
    model_cfg = updated_config.get("model", {})

    cfg.validate_config_postload(model_cfg, default_n_visible, data_cfg["q"])
        
    # Initialize model    
    print(f"Using model type: {model_type}")
    
    if model_type == "binary":
        print(f"Using mean-field: {model_cfg["mf"]}")
        print(f"Using binarize: {data_cfg["binarize"]}")
        from models.rbm_binary import RBM_binary
        rbm = RBM_binary(model_cfg.get("n_visible"), model_cfg["n_hidden"], mf=model_cfg.get("mf")).to(device)
    elif model_type == "exponential":
        from models.rbm_exponential import RBM_exponential
        rbm = RBM_exponential(model_cfg.get("n_visible"), model_cfg["n_hidden"]).to(device)
    elif model_type == "gaussian":
        from models.rbm_gaussian import RBM_gaussian
        rbm = RBM_gaussian(model_cfg.get("n_visible"), model_cfg["n_hidden"]).to(device)
    elif model_type == "vonmises":
        from models.rbm_vonmises import RBM_vonmises
        rbm = RBM_vonmises(model_cfg.get("n_visible"), model_cfg["n_hidden"]).to(device)
    elif model_type == "multinomial":
        print(f"Number of categories: {model_cfg["n_class"]}")
        from models.rbm_multinomial import RBM_multinomial
        rbm = RBM_multinomial(model_cfg["n_class"], model_cfg.get("n_visible"), model_cfg["n_hidden"]).to(device)
    print("")
    
    # Check training parameters
    updated_config = cfg.resolve_train_cfg(updated_config)
    train_cfg = updated_config.get("training", {})
    n_epochs = train_cfg.get("n_epochs")
    sm = train_cfg.get("sm")

    if sm == True and not model_type == "gaussian":
        raise ValueError(f"Score Matching is only available for Gaussian RBMs. Set training.sm to False.")

    # Validate schedule:
    cfg.validate_schedule(train_cfg=train_cfg)
    
    # Before Training the model, ensure output directory and run name is specified.
    # If unspecified, ask user if to run training anyways without an output.
    output_cfg = updated_config.get("output", {})
    out_dir = output_cfg.get("base_dir")
    run_name = output_cfg.get("run_name")

    if (out_dir is None or run_name is None):
        print(f"\noutput.base_dir and/or output.run_name is unspecified in config.")
        print(f"There will be no outputs upon training.")
        while True:
            choice = input("Do you still want to run the training? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                # Logic for "yes"
                print("Continuing...")
                break
            elif choice in ['n', 'no']:
                # Logic for "no"
                print("Exiting...")
                return
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    # Train the model
    if sm == True and model_type == "gaussian":
        history = train_sm(rbm, device, train_loader, train_cfg, n_epochs)
    else:
        history = train_cd(rbm, device, train_loader, train_cfg, n_epochs)

    if out_dir is None or run_name is None:
        return
    
    '''
    Specfiy output directory and the directory name.
    Resulting file structure will be:
    ├── out_dir
        ├── checkpoints
        |   └── run_name
        ├── figures
        |   └── run_name
        └── physics
            └── run_name
    '''
    paths = cfg.get_output_paths(out_dir, run_name)
    checkpoints_dir = paths["checkpoints"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    print("")

    save_checkpoint(model=rbm, optimizer=None, persistent_v=rbm.persistent_v, epoch=n_epochs, config=updated_config, history=history, path=path.join(checkpoints_dir, "checkpoint.pt"))
    print(f"Checkpoint file, 'checkpoint.pt', saved to directory: {checkpoints_dir}")
    print("")

    return

def infer_n_visible(test_loader, model_type, q):
    # Get a batch in data
    batch_data = next(iter(test_loader))
    X_batch = batch_data[0] if isinstance(batch_data, list) else batch_data
    if X_batch.dim() == 1:
        X_batch = X_batch.unsqueeze(0)  # Add batch dimension if input is a single configuration

    # default n_visible different for multinomial. For multinomial, shape would be image size * number of categories due to OneHot encoding.
    if model_type == "multinomial":
        return int(X_batch.shape[1] / q)
    else:
        return X_batch.shape[1]

if __name__ == "__main__":
    main()