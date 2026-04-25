import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import numpy as np

def load_data(type, data_dir, data_filename, split, q, T, L, batch_size, binarize=False, verbose=True):
    if type is None:
        raise ValueError("data.type must be specified in config.yaml. Refer to config.yaml for supported types.")
    
    if data_dir is None:
        raise ValueError("data.data_dir must be specified in config.yaml. Please update config.yaml.")
    
    if type == "custom" and data_filename is None:
        raise ValueError("data.data_filename must be specified in config.yaml when data.type is 'custom'. Please update config.yaml.")
    
    if type not in ["mnist", "cifar10", "stl10"] and split is None:
        split = 0.8  # Default to 80% train, 20% test if not specified
        print(f"data.split not specified in config. Defaulting to split = {split}.")

    if type in ["ising", "xy", "potts"] and (T is None or L is None):
        raise ValueError(f"data.T and data.L must be specified in config.yaml when data.type is '{type}'. Please update config.yaml.")

    if type == "potts" and q is None:
        raise ValueError("data.q must be specified in config.yaml when data.type is 'potts'. Please update config.yaml.")
    
    if batch_size is None:
        batch_size = 64  # Default batch size if not specified
        print(f"batch_size not specified in config. Defaulting to batch_size = {batch_size}.") 

    if type == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten 28x28 -> 784
            transforms.Lambda(lambda x: torch.round(x) if binarize else x)
        ])

        train_data = datasets.MNIST(
            root=data_dir,
            train=True,
            transform=transform,
            download=True
        )

        test_data = datasets.MNIST(
            root=data_dir,
            train=False,
            transform=transform,
            download=True
        )

    elif type == "cifar10":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten 32x32 -> 1024
            transforms.Lambda(lambda x: torch.round(x) if binarize else x)
        ])

        train_data = datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=transform,
            download=True
        )

        test_data = datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=transform,
            download=True
        )

    elif type == "stl10":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # convert to grayscale
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR), # downsampling reduce size from 96x96 to 32x32
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten 32x32 -> 1024
            transforms.Lambda(lambda x: torch.round(x) if binarize else x)
        ])

        train_data = datasets.STL10(
            root=data_dir,
            split='train',
            transform=transform,
            download=True
        )

        test_data = datasets.STL10(
            root=data_dir,
            split='test',
            transform=transform,
            download=True
        )
        
    elif type == "ising":
        path = os.path.join(data_dir, f"2dIsing_L{L}", f"L{L}T{T:.2f}.npy")

        dataset = np.load(path, allow_pickle=True)
        dataset_tensor = torch.Tensor(dataset).float()
        train_data, test_data = torch.split(dataset_tensor, int(len(dataset_tensor) * split))

    elif type == "xy":
        path = os.path.join(data_dir, f"XY_L{L}", f"XYconfigsT{T:.1f}.npy")

        dataset = np.load(path, allow_pickle=True)
        dataset_tensor = torch.Tensor(dataset).float()
        train_data, test_data = torch.split(dataset_tensor, int(len(dataset_tensor) * split))

    elif type == "potts":
        path = os.path.join(data_dir, f"2dPotts_L{L}", f"potts_configs_q{q}L{L}T{T:.3f}.npy")

        dataset = np.load(path, allow_pickle=True)
        dataset_tensor = torch.Tensor(dataset).float()
        train_data, test_data = torch.split(dataset_tensor, int(len(dataset_tensor) * split))

    elif type == "custom":
        if data_filename is None:
            raise ValueError("data.filename must be specified in config.yaml when data.type is 'custom'.")
        # Implement get_custom_loaders in src/data/custom.py to load your custom dataset
        dataset_path = os.path.join(data_dir, data_filename)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Custom dataset file not found: {dataset_path}")
        
        dataset = np.load(dataset_path, allow_pickle=True)
        dataset_tensor = torch.Tensor(dataset).float()
        train_data, test_data = torch.split(dataset_tensor, int(len(dataset_tensor) * split))

    else:
        raise ValueError(f"Unsupported dataset type: {type}. Refer to config.yaml for supported types.")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"Train dataset size: {len(train_data)}")
        print(f"Test dataset size: {len(test_data)}")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        if type in ["mnist", "cifar10", "stl10"]:
            print(f"Shape of each image: {train_loader.dataset[0][0].shape}")
            print(f"Size of each image (flattened): {train_loader.dataset[0][0].shape[0]} \n")
        else:
            print(f"Shape of each sample: {train_loader.dataset[0].shape}")
            print(f"Size of each sample (flattened): {train_loader.dataset[0].size()} \n")

    return train_loader, test_loader