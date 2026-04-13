import argparse

from utils.device import get_device
from utils.config import load_config

from data.mnist import get_mnist_loaders
from data.cifar10 import get_cifar10_loaders
from data.stl10 import get_stl10_loaders

from models.rbm_binary import RBM_binary

from training.training import train

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

    config = load_config(args.config)
    print(f"Using config file: {args.config}")
    print("Configuration:", config)
    print("")

    batch_size = config["training"]["batch_size"]
    n_epochs = config["training"]["n_epochs"]
    lr = config["training"]["lr"]
    k = config["training"]["k"]
    pcd = config["training"]["pcd"]
    mc = config["training"]["mc"]
    epsilon = config["training"]["epsilon"]

    type = config["data"]["type"]
    path = config["data"]["path"]

    n_visible = config["model"]["n_visible"]
    n_hidden = config["model"]["n_hidden"]

    output_dir = config["output_dir"]

    # Load data
    if type == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, path=path, verbose=True)
    elif type == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, path=path, verbose=True)
    elif type == "stl10":
        train_loader, test_loader = get_stl10_loaders(batch_size=batch_size, path=path, verbose=True)
    else:
        raise ValueError(f"Unsupported dataset type: {type}. Refer to config.yaml for supported types.")
    
    # Check if n_visible is set in config, if not infer from data. If set, check if it matches the data.
    if n_visible is None:
            n_visible = train_loader.dataset[0][0].shape[0]
    elif n_visible != train_loader.dataset[0][0].shape[0]:
        raise ValueError(f"n_visible in config ({n_visible}) does not match the size of the input data ({train_loader.dataset[0][0].shape[0]}). Please update config.yaml.")

    rbm = RBM_binary(n_visible, n_hidden).to(device)
    
    history = train(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)

if __name__ == "__main__":
    main()