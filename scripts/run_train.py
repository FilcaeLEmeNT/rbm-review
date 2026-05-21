import argparse
import os
from os import path
import math

from utils.device import get_device
from utils.config import load_config

from data.data_loader import load_data

from training.training import train_cd, train_sm

from utils.checkpoint import save_checkpoint

# update the dicitonaries below to add/remove values to be specified in the configuration file.
DATA_DEFAULTS = {
    "type": None,
    "data_dir": None,
    "batch_size": None,
    "split": None,
    "binarize": None,
    "q": None,
    "T": None,
    "L": None
}

MODEL_DEFAULTS = {
    "type": None,
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

    return parser.parse_args()

def main():
    args = parse_args()

    # Load device: Either CPU or CUDA
    device = get_device()

    # Load configuration file
    config = load_config(args.config)
    print(f"Using config file: {args.config}")

    data_cfg = {
        k: config.get("data", {}).get(k, v)
        for k, v in DATA_DEFAULTS.items()
    }
    model_cfg = {
        k: config.get("model", {}).get(k, v)
        for k, v in MODEL_DEFAULTS.items()
    }
    training_cfg = {
        k: config.get("training", {}).get(k, v)
        for k, v in TRAINING_DEFAULTS.items()
    }
    output_cfg = {
        k: config.get("output", {}).get(k, v)
        for k, v in OUTPUT_DEFAULTS.items()
    }

    # Overwrite config with that containing None values.
    config = {
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "output": output_cfg
    }

    # Print configuration summary.
    print_cfg_summary(config)

    run_training(device, config)

def print_cfg_summary(config):
    '''
    Prints a summary of the configuration file.
    '''
    # Get dictionaries from configuration.
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    # Print config summary
    print("Config summary:")
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

    print("")
    return

def run_training(device, config):
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
    # Get values from configuration.
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    data_type = data_cfg.get("type")
    data_dir = data_cfg.get("data_dir")
    batch_size = data_cfg.get("batch_size")
    split = data_cfg.get("split")
    binarize = data_cfg.get("binarize")
    q = data_cfg.get("q")
    T = data_cfg.get("T")
    L = data_cfg.get("L")

    model_type = model_cfg.get("type")
    n_class = model_cfg.get("n_class")
    n_visible = model_cfg.get("n_visible")
    n_hidden = model_cfg.get("n_hidden")
    mf = model_cfg.get("mf")

    n_epochs = training_cfg.get("n_epochs")
    lr = training_cfg.get("lr")
    k = training_cfg.get("k")
    pcd = training_cfg.get("pcd")
    sm = training_cfg.get("sm")
    mc = training_cfg.get("mc")
    epsilon = training_cfg.get("epsilon")

    out_dir = output_cfg.get("base_dir")
    run_name = output_cfg.get("run_name")

    # Validate datasets and model compatibility:
    valid_models = VALID_MODEL_FOR_DATA.get(data_type)
    if valid_models is None:
        raise ValueError(f"Unknown data type: {data_type}. Please update data.type in config.yaml. Refer to default.yaml for supported types.")
    if model_type not in valid_models:
        raise ValueError(
            f"Model '{model_type}' is incompatible with data type '{data_type}'. Please update model.type in config.yaml. "
            f"Valid models: {valid_models}"
        )
    
    # Check data values before passing it to load_data.
    if data_dir is None:
        raise ValueError("data.data_dir must be specified in config.yaml. Please update config.yaml.")
    
    if batch_size is None:
        batch_size = 64  # Default batch size if not specified
        print(f"\033[1mdata.batch_size not specified in config. Defaulting to batch_size = {batch_size}.\033[0m")

    if split is None and data_type not in ["mnist", "cifar10", "stl10"]:
        split = 0.8  # Default to 80% train, 20% test if not specified
        print(f"\033[1mdata.split not specified in config. Defaulting to split = {split}.\033[0m")

    if binarize is None and data_type in ["mnist", "cifar10", "stl10"]:
        binarize = False  # Default to False if not specified
        print(f"\033[1mdata.binarize not specified in config. Defaulting to binarize = {binarize}.\033[0m")
    
    if binarize is True and model_type == 'multinomial':
        binarize = False  # binarize is not compatible with multinomial. Set to false and print a warning.
        print(f"binarize is not compatible with multinomial RBMs. Setting binarize to {binarize}.")

    if ((T is None) or (L is None)) and data_type in ["ising", "xy", "potts"]:
        raise ValueError(f"data.T and data.L must be specified in config.yaml when data.type is '{data_type}'. Please update config.yaml.")
    
    if q is None and model_type == 'multinomial':
        raise ValueError("data.q must be specified in config.yaml when model.type is 'multinomial'. Please update config.yaml.")
    
    # Load data.
    train_loader, test_loader = load_data(data_type, data_dir, split, q, T, L, batch_size, binarize, model_type)
    
    # Check if n_visible is set in config, if not infer from data. If set, check if it matches the data.
    # Get a batch in data
    batch_data = next(iter(test_loader))
    X_batch = batch_data[0] if isinstance(batch_data, list) else batch_data
    if X_batch.dim() == 1:
        X_batch = X_batch.unsqueeze(0)  # Add batch dimension if input is a single configuration

    # default n_visible different for multinomial. For multinomial, shape would be image size * number of categories due to OneHot encoding.
    if model_type == "multinomial":
        default_n_visible = int(X_batch.shape[1] / q)
    else:
        default_n_visible = X_batch.shape[1]

    if n_visible is None:
        n_visible = default_n_visible
        print(f"\033[1mmodel.n_visible not specified in config.yaml. Inferred n_visible = {n_visible} from the data.\033[0m")
    else:
        if n_visible != default_n_visible:
            raise ValueError(f"n_visible in config ({n_visible}) does not match the size of the input data ({default_n_visible}). Please update config.yaml.")
    
    # Check if n_hidden is set in config, if not default to n_visible // 2 
    if n_hidden is None:
        n_hidden = 2 ** math.floor(math.log2(n_visible // 2))  # Default to close to half the number of visible units if not specified
        print(f"\033[1mmodel.n_hidden not specified in config. Defaulting to n_hidden = {n_hidden}.\033[0m")
        if n_hidden <= 0:
            raise ValueError(f"n_hidden infered from n_visible, n_hidden = {n_hidden}, is invalid. Please specify model.n_hidden in config.yaml")
    elif not (n_hidden > 0 and (n_hidden & (n_hidden - 1)) == 0):  # Check if power of 2
        raise ValueError(f"model.n_hidden must be a power of 2. Value specified is n_hidden={n_hidden}. Please update config.yaml.")

    # Initialize model    
    print(f"Using model type: {model_type}")
    
    if model_type == "binary":
        if mf is None:
            mf = True
            print(f"\033[1mmodel.mf not specified in config. Defaulting to n_hidden = {n_epochs}.\033[0m")
        print(f"Using mean-field: {mf}")
        print(f"Using binarize: {binarize}")
        from models.rbm_binary import RBM_binary
        rbm = RBM_binary(n_visible, n_hidden, mf=mf).to(device)
    elif model_type == "exponential":
        from models.rbm_exponential import RBM_exponential
        rbm = RBM_exponential(n_visible, n_hidden).to(device)
    elif model_type == "gaussian":
        from models.rbm_gaussian import RBM_gaussian
        rbm = RBM_gaussian(n_visible, n_hidden).to(device)
    elif model_type == "vonmises":
        from models.rbm_vonmises import RBM_vonmises
        rbm = RBM_vonmises(n_visible, n_hidden).to(device)
    elif model_type == "multinomial":
        if n_class is None:
            if q is None:
                raise ValueError("model.n_class not specified in config. n_class's default value, data.q, is also not specified in config. Please update config.yaml.")
            else:
                n_class = q
                print(f"\033[1mmodel.n_class not specified in config. Defaulting to n_class = data.q = {q}.\033[0m")

        if n_class != q and q is not None:
            raise ValueError(f"model.n_class={n_class} and data.q={q} are both specified but do not match. They must be equal for multinomial RBM.")
        print(f"Number of categories: {n_class}")
        from models.rbm_multinomial import RBM_multinomial
        rbm = RBM_multinomial(n_class, n_visible, n_hidden).to(device)
    
    # Check training parameters
    if n_epochs is None:
        n_epochs = 500
        print(f"\033[1mtraining.n_epochs not specified in config. Defaulting to n_epochs = {n_epochs}.\033[0m")
    
    if lr is None:
        lr = 0.01
        print(f"\033[1mtraining.lr not specified in config. Defaulting to lr = {lr}.\033[0m")

    if k is None:
        k = 10
        print(f"\033[1mtraining.k not specified in config. Defaulting to k = {k}.\033[0m")

    if pcd is None:
        pcd = True
        print(f"\033[1mtraining.pcd not specified in config. Defaulting to pcd = {pcd}.\033[0m")

    if sm is None:
        sm = False
        print(f"\033[1mtraining.sm not specified in config. Defaulting to sm = {sm}.\033[0m")

    if mc is None:
        mc = "gibbs"
        print(f"\033[1mtraining.mc not specified in config. Defaulting to mc = {mc}.\033[0m")
    elif mc not in ["gibbs", "langevin"]:
        raise ValueError("mc needs to be either 'gibbs' or 'langevin'. Please update config.")

    if epsilon is None:
        epsilon = 0.05
        print(f"\033[1mtraining.epsilon not specified in config. Defaulting to epsilon = {epsilon}.\033[0m")

    if sm == True and not model_type == "gaussian":
        sm = False
        print(f"\033[1mScore Matching is only available for Gaussian RBMs. Defaulting to sm = {sm}\033[0m")

    # Before Training the model, ensure output directory and run name is specified.
    # If unspecified, ask user if to run training anyways without an output.
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
        history = train_sm(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)
    else:
        history = train_cd(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)

    if (out_dir is None or run_name is None):
        return
    
    '''
    Specfiy output directory and the directory name.
    Resulting file structure will be:
    ├── out_dir
        ├── checkpoints
        |   └── run_name
        ├── figures
        |   └── run_name
        └── history
            └── run_name
    '''
    checkpoints_dir = path.join(out_dir, "checkpoints", run_name)
    figures_dir = path.join(out_dir, "figures", run_name)
    history_dir = path.join(out_dir, "history", run_name)
    samples_dir = path.join(out_dir, "samples", run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    print("")

    # Create new config with all parameters for saving
    new_config = {
        "data": {
            "type": data_type,
            "data_dir": data_dir,
            "batch_size": batch_size,
            "split": split,
            "binarize": binarize,
            "q": q,
            "T": T,
            "L": L,
        },
        "model": {
            "type": model_type,
            "n_class": n_class,
            "n_visible": n_visible,
            "n_hidden": n_hidden,
            "mf": mf,
        },
        "training": {
            "n_epochs": n_epochs,
            "lr": lr,
            "k": k,
            "pcd": pcd,
            "sm": sm,
            "mc": mc,
            "epsilon": epsilon,
        },
        "output": {
            "base_dir": out_dir,
            "run_name": run_name,
        }
    }

    save_checkpoint(model=rbm, optimizer=None, epoch=n_epochs, config=new_config, history=history, path=path.join(checkpoints_dir, "checkpoint.pt"))
    print(f"Checkpoint file, 'checkpoint.pt', saved to directory: {checkpoints_dir}")

    return

if __name__ == "__main__":
    main()