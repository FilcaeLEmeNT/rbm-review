import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd

def load_data(type, data_dir, split, q, T, L, batch_size, binarize, verbose=True):
    '''
    Load dataset based on the specified type and parameters in config.yaml.
    Supported types: "mnist", "cifar10", "stl10", "ising", "xy", "potts", "custom"
    For "custom" type, data_filename must be provided and should point to a .npy file containing the dataset.
    For "ising", "xy", and "potts" types, T and L must be provided to locate the correct dataset file. For "potts", q must also be provided.

    Parameters:
    - type: Dataset type (str)
    - data_dir: Directory where the dataset is located (str)
    - data_filename: Filename for custom dataset (str, required if type is "custom")
    - split: Fraction of data to use for training (float, between 0 and 1)
    - q: Number of states for Potts model (int, required if type is "potts")
    - T: Temperature for Ising, XY, or Potts models (float, required if type is "ising", "xy", or "potts")
    - L: System size for Ising, XY, or Potts models (int, required if type is "ising", "xy", or "potts")
    - batch_size: Batch size for DataLoader (int)
    - binarize: Whether to binarize the data (bool, only applicable for image datasets)
    - verbose: Whether to print dataset information (bool)
    '''
    if type is None:
        raise ValueError("data.type must be specified in config.yaml. Refer to config.yaml for supported types.")
    
    if data_dir is None:
        raise ValueError("data.data_dir must be specified in config.yaml. Please update config.yaml.")
    
    if batch_size is None:
        batch_size = 64  # Default batch size if not specified
        print(f"\033[1mdata.batch_size not specified in config. Defaulting to batch_size = {batch_size}.\033[0m")

    if split is None and type not in ["mnist", "cifar10", "stl10"]:
        split = 0.8  # Default to 80% train, 20% test if not specified
        print(f"\033[1mdata.split not specified in config. Defaulting to split = {split}.\033[0m")

    if binarize is None and type in ["mnist", "cifar10", "stl10"]:
        binarize = False  # Default to False if not specified
        print(f"\033[1mdata.binarize not specified in config. Defaulting to binarize = {binarize}.\033[0m")

    if ((T is None) or (L is None)) and type in ["ising", "xy", "potts"]:
        raise ValueError(f"data.T and data.L must be specified in config.yaml when data.type is '{type}'. Please update config.yaml.")

    if q is None and type == "potts":
        raise ValueError("data.q must be specified in config.yaml when data.type is 'potts'. Please update config.yaml.")

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
    
    elif type == "wind_dir":
        path = os.path.join(data_dir, '42503e2023.txt')
        dataset = pd.read_csv(path, sep='\s+', header=[0, 1])
        wdir = dataset.iloc[:, 6]
        wdir = wdir.to_numpy()

        # remove missing values
        wdir = wdir[wdir < 999]

        # wrap and convert to rad
        wdir = np.mod(wdir, 360)
        wdir = np.deg2rad(wdir.astype(np.float32))
        wdir_tensor = torch.from_numpy(wdir)

        train_data, test_data = torch.split(wdir_tensor, int(len(wdir_tensor) * split))
    
    elif type == "protein":
        import sidechainnet as scn

        data = scn.load(casp_version=12, casp_thinning=30, scn_dir=os.path.join(data_dir, "sidechainnet_data"))

        # Generate an array of phi and psi angles from the dataset of shape [M, 2], where M is the amount of residues considered.
        row = 0
        M = 200000

        # Shuffle data
        data_list = list(data)
        np.random.seed(42)  # Set seed for consistency
        np.random.shuffle(data_list)

        data_array = [[], []]
        for protein in data_list:
            if row >= M:
                break
            
            phi_angles = protein.angles[:, 0]
            psi_angles = protein.angles[:, 1]

            # Use mask to filter missing residues.
            mask = [True if c == '+' else False for c in protein.mask]
            phi_angles = phi_angles[mask]
            psi_angles = psi_angles[mask]

            # Ensure there are no nan values passed to the data array.
            # Search for indices where either one of phi or psi is nan.
            nan_indices = np.isnan(phi_angles) | np.isnan(psi_angles)
            phi_angles = phi_angles[~nan_indices]
            psi_angles = psi_angles[~nan_indices]

            data_array[0].extend(phi_angles)
            data_array[1].extend(psi_angles)

            row += len(phi_angles)

        data_array = np.array(data_array).T
        dataset_tensor = torch.from_numpy(data_array).float()
        train_data, test_data = torch.split(dataset_tensor, int(len(dataset_tensor) * split))

    else:
        raise ValueError(f"Unsupported dataset type: {type}. Refer to config.yaml for supported types.")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Get a batch in data
    batch_data = next(iter(test_loader))
    X_batch = batch_data[0] if isinstance(batch_data, list) else batch_data
    
    if verbose:
        print(f"Train dataset size: {len(train_data)}")
        print(f"Test dataset size: {len(test_data)}")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Shape of each batch: {X_batch.shape}\n")
        
    return train_loader, test_loader