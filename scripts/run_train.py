import argparse

from rbm_review.utils.device import get_device
from rbm_review.utils.config import load_config

from rbm_review.data.data_loader import load_data

from rbm_review.training.training import train_cd

def parse_args():
    parser = argparse.ArgumentParser(description="Train RBM model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    device = get_device()

    # Load config
    config = load_config(args.config)
    print(f"Using config file: {args.config}")

    batch_size = config["training"]["batch_size"] if "batch_size" in config["training"] else None
    n_epochs = config["training"]["n_epochs"] if "n_epochs" in config["training"] else None
    lr = config["training"]["lr"] if "lr" in config["training"] else None
    k = config["training"]["k"] if "k" in config["training"] else None
    pcd = config["training"]["pcd"] if "pcd" in config["training"] else False
    mf = config["training"]["mf"] if "mf" in config["training"] else False
    mc = config["training"]["mc"] if "mc" in config["training"] else False
    epsilon = config["training"]["epsilon"] if "epsilon" in config["training"] else None

    data_type = config["data"]["type"] if "type" in config["data"] else None
    data_dir = config["data"]["data_dir"] if "data_dir" in config["data"] else None
    data_filename = config["data"]["data_filename"] if "data_filename" in config["data"] else None
    split = config["data"]["split"] if "split" in config["data"] else None
    binarize = config["data"]["binarize"] if "binarize" in config["data"] else False
    q = config["data"]["q"] if "q" in config["data"] else None
    T = config["data"]["T"] if "T" in config["data"] else None
    L = config["data"]["L"] if "L" in config["data"] else None

    model_type = config["model"]["type"] if "type" in config["model"] else None
    n_visible = config["model"]["n_visible"] if "n_visible" in config["model"] else None
    n_hidden = config["model"]["n_hidden"] if "n_hidden" in config["model"] else None

    output_dir = config["output_dir"] if "output_dir" in config else None

    # Print config summary
    print("Config summary:")
    print("Training parameters:")
    print(f"\tbatch_size={batch_size}", f"n_epochs={n_epochs}", f"lr={lr}", f"k={k}", f"pcd={pcd}", f"mf={mf}", f"mc={mc}", f"epsilon={epsilon}", sep="\n\t")
    print("Data parameters:")
    print(f"\ttype={data_type}", f"data_dir={data_dir}", f"data_filename={data_filename}", f"split={split}", f"binarize={binarize}", f"q={q}", f"T={T}", f"L={L}", sep="\n\t")
    print("Model parameters:")
    print(f"\ttype={model_type}", f"n_visible={n_visible}", f"n_hidden={n_hidden}", sep="\n\t")
    print(f"Output directory: {output_dir}")
    print("")

    # Load data
    train_loader, test_loader = load_data(data_type, data_dir, data_filename, split, q, T, L, batch_size, binarize=binarize)
    
    # Check if n_visible is set in config, if not infer from data. If set, check if it matches the data.
    if n_visible is None:
            if data_type in ["mnist", "cifar10", "stl10"]:
                n_visible = train_loader.dataset[0][0].shape[0]  # For image datasets, infer from the shape of each image
            else:
                n_visible = train_loader.dataset[0].shape[0]
            print(f"n_visible not specified in config. Inferred n_visible = {n_visible} from the data.")
    else:
        if data_type in ["mnist", "cifar10", "stl10"]:
            if n_visible != train_loader.dataset[0][0].shape[0]:
                raise ValueError(f"n_visible in config ({n_visible}) does not match the size of the input data ({train_loader.dataset[0][0].shape[0]}). Please update config.yaml.")
        else:
            if n_visible != train_loader.dataset[0].shape[0]:
                raise ValueError(f"n_visible in config ({n_visible}) does not match the size of the input data ({train_loader.dataset[0].shape[0]}). Please update config.yaml.")
    
    # Check if n_hidden is set in config, if not default to n_visible // 2 
    if n_hidden is None:
        n_hidden = n_visible // 2  # Default to half the number of visible units if not specified
        print(f"n_hidden not specified in config. Defaulting to n_hidden = {n_hidden}.")

    # Initialize model
    if model_type == None:
        raise ValueError("model.type must be specified in config.yaml. Please update config.yaml.")
    
    print(f"Using model type: {model_type}")
    if model_type == "binary":
        print(f"Using mean-field: {mf}")
        print(f"Using binarize: {binarize}")
        from rbm_review.models.rbm_binary import RBM_binary
        rbm = RBM_binary(n_visible, n_hidden, mf=mf).to(device)
    elif model_type == "exponential":
        from rbm_review.models.rbm_exponential import RBM_exponential
        rbm = RBM_exponential(n_visible, n_hidden).to(device)
    elif model_type == "gaussian":
        from rbm_review.models.rbm_gaussian import RBM_gaussian
        rbm = RBM_gaussian(n_visible, n_hidden).to(device)
    elif model_type == "vonmises":
        from rbm_review.models.rbm_vonmises import RBM_vonmises
        rbm = RBM_vonmises(n_visible, n_hidden).to(device)
    elif model_type == "multinomial":
        from rbm_review.models.rbm_multinomial import RBM_multinomial
        rbm = RBM_multinomial(q, n_visible, n_hidden).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please update config.yaml with a valid model type.")
    
    # Unimplemented: add code for checking parameters. Also add train_sm for score matching.

    history = train_cd(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)

if __name__ == "__main__":
    main()