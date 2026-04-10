from utils.device import get_device
from utils.config import load_config
from data.mnist import get_mnist_loaders
from models.rbm_binary import RBM_binary
from training.training import train

def main():
    device = get_device()
    config = load_config("configs/default.yaml")

    print("Configuration:", config)

    batch_size = config["training"]["batch_size"]
    n_epochs = config["training"]["n_epochs"]
    lr = config["training"]["lr"]
    k = config["training"]["k"]
    pcd = config["training"]["pcd"]
    mc = config["training"]["mc"]
    epsilon = config["training"]["epsilon"]

    type = config["data"]["type"]
    path = config["data"]["path"]
    if type == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size, verbose=True)

    n_visible = config["model"]["n_visible"]
    n_hidden = config["model"]["n_hidden"]

    rbm = RBM_binary(n_visible, n_hidden).to(device)
    
    history = train(rbm, device, train_loader, pcd, mc, k, epsilon, lr, n_epochs)

if __name__ == "__main__":
    main()