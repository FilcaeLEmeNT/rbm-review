import torch

def get_device(verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())

        if torch.cuda.is_available():
            print("GPU device name:", torch.cuda.get_device_name(0))
        print("Using device:", device)
        print("")
    return device