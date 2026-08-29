# SPDX-License-Identifier: MIT
# src/data/data_loader.py

"""Module for loading the data from data configurations."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd

from data.datasets import PottsDataset
from utils.multinomial import OneHotTransform, DiscretizeTransform


def load_data(
    data_cfg: dict, model_type: str, verbose: bool = False
) -> torch.utils.data.DataLoader:
    """Load dataset based on the specified type and parameters in config.yaml.

    Supported types: "mnist", "cifar10", "stl10", "ising", "xy", "potts".
    For "ising", "xy", and "potts" types, T and L must be provided to locate the
    correct dataset file. For "potts", q must also be provided.

    Args:
        data_cfg: Data section/dictionary of the configuration dictionary parsed from the
            YAML configuration file.
        model_type: The model_type specified in the configuration dictionary parsed from the
            YAML configuration file.
        verbose: If ``True`` print the dataset information.

    Return:
        A DataLoader class from torch which provides an iterable over the dataset.
    """
    data_type = data_cfg.get("data_type")
    data_dir = data_cfg.get("data_dir")
    batch_size = data_cfg.get("batch_size")
    split = data_cfg.get("split")
    binarize = data_cfg.get("binarize")
    q = data_cfg.get("q")
    T = data_cfg.get("T")
    L = data_cfg.get("L")

    # Check if multinomial
    is_multinomial = model_type == "multinomial"

    # Build transforms
    if data_type == "mnist":
        transform_list = [transforms.ToTensor()]
    elif data_type == "cifar10":
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    elif data_type == "stl10":
        transform_list = [
            transforms.Grayscale(num_output_channels=1),  # convert to grayscale
            transforms.Resize(
                (32, 32), interpolation=transforms.InterpolationMode.BILINEAR
            ),  # downsampling reduce size from 96x96 to 32x32
            transforms.ToTensor(),
        ]
    else:
        transform_list = []

    if is_multinomial:
        if data_type in [
            "mnist",
            "cifar10",
            "stl10",
        ]:  #  need to discretize first before one_hot transform
            transform_list.append(DiscretizeTransform(q))

        transform_list.append(OneHotTransform(q))

    elif binarize:
        transform_list.append(transforms.Lambda(lambda x: torch.round(x)))

    # lastly flatten
    transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)

    if data_type == "mnist":
        train_data = datasets.MNIST(
            root=data_dir, train=True, transform=transform, download=True
        )

        test_data = datasets.MNIST(
            root=data_dir, train=False, transform=transform, download=True
        )

    elif data_type == "cifar10":
        train_data = datasets.CIFAR10(
            root=data_dir, train=True, transform=transform, download=True
        )

        test_data = datasets.CIFAR10(
            root=data_dir, train=False, transform=transform, download=True
        )

    elif data_type == "stl10":
        train_data = datasets.STL10(
            root=data_dir, split="train", transform=transform, download=True
        )

        test_data = datasets.STL10(
            root=data_dir, split="test", transform=transform, download=True
        )

    elif data_type == "ising":
        path = os.path.join(data_dir, f"2dIsing_L{L}", f"L{L}T{T:.2f}.npy")

        dataset = np.load(path, allow_pickle=True)
        dataset_tensor = torch.Tensor(dataset).float()
        train_data, test_data = torch.split(
            dataset_tensor, int(len(dataset_tensor) * split)
        )

    elif data_type == "xy":
        path = os.path.join(data_dir, f"XY_L{L}", f"XYconfigsT{T:.1f}.npy")

        dataset = np.load(path, allow_pickle=True)
        dataset_tensor = torch.Tensor(dataset).float()
        train_data, test_data = torch.split(
            dataset_tensor, int(len(dataset_tensor) * split)
        )

    elif data_type == "potts":
        path = os.path.join(
            data_dir, f"2dPotts_L{L}", f"potts_configs_q{q}L{L}T{T:.3f}.npy"
        )

        dataset_tensor = PottsDataset(path, q, transform)

        train_size = int(len(dataset_tensor) * split)
        test_size = len(dataset_tensor) - train_size
        train_data, test_data = torch.utils.data.random_split(
            dataset_tensor,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    elif data_type == "wind_dir":
        path = os.path.join(data_dir, "42503e2023.txt")
        dataset = pd.read_csv(path, sep=r"\s+", header=[0, 1])
        wdir = dataset.iloc[:, 6]
        wdir = wdir.to_numpy()

        # remove missing values
        wdir = wdir[wdir < 999]

        # wrap and convert to rad
        wdir = np.mod(wdir, 360)
        wdir = np.deg2rad(wdir.astype(np.float32))
        wdir_tensor = torch.from_numpy(wdir)

        train_data, test_data = torch.split(
            wdir_tensor, int(len(wdir_tensor) * split)
        )

    elif data_type == "protein":
        import sidechainnet as scn

        data = scn.load(
            casp_version=12,
            casp_thinning=30,
            scn_dir=os.path.join(data_dir, "sidechainnet_data"),
        )

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
            mask = [True if c == "+" else False for c in protein.mask]
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
        train_data, test_data = torch.split(
            dataset_tensor, int(len(dataset_tensor) * split)
        )

    else:
        raise ValueError(
            f"Unsupported dataset type: {data_type}. Refer to config.yaml for supported types."
        )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Get a batch in data
    batch_data = next(iter(train_loader))
    X_batch = batch_data[0] if isinstance(batch_data, list) else batch_data

    if verbose:
        print(f"Train dataset size: {len(train_data)}")
        print(f"Test dataset size: {len(test_data)}")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Shape of each batch: {X_batch.shape}")
        print("")

    return train_loader, test_loader
