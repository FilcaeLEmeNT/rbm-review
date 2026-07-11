# SPDX-License-Identifier: MIT
# src/data/datasets.py

"""Module that defines classes for particular types of datasets."""

import torch
from torch.utils.data import Dataset

import os
import numpy as np


class PottsDataset(Dataset):
    """Dataset for Potts model spin configurations stored in NumPy files.

    Each sample consists of a spin configuration together with its
    corresponding temperature and phase labels. The temperature is inferred
    from the filename, which is expected to contain a substring of the form
    ``T<temperature>.npy`` (e.g., ``potts_T0.75.npy``).

    Attributes:
        data: Tensor containing the spin configurations.
        T_label: Temperature associated with every sample in the dataset.
        phase_label: Phase label for every sample, where 0 denotes the
            ordered phase and 1 denotes the disordered phase.
        q: Number of Potts spin states.
        transform: Optional transform applied to each sample.
    """

    def __init__(self, npy_path: str = "./", q: int = 4, transform=None):
        """Initialize the dataset.

        Args:
            npy_path: Path to the NumPy file containing Potts model samples.
            q: Number of Potts spin states.
            transform: Optional transform applied to each sample when it is
                retrieved.
        """
        self.data = np.load(npy_path)  # numpy array (n_samples, L*L)
        self.data = torch.from_numpy(
            self.data
        ).long()  # convert to tensor (int64)

        filename = os.path.basename(npy_path)
        T = float(filename.split("T")[1].split(".npy")[0])
        self.T_label = torch.tensor(T).float()  # regression label
        Tc = 1.0 / np.log(1.0 + np.sqrt(q))

        self.phase_label = torch.tensor(
            0 if T < Tc else 1
        ).long()  # Phase classification label

        self.q = q
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Retrieve a sample and its labels.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - **x**: Spin configuration, optionally transformed.
                - **T_label**: Temperature associated with the sample.
                - **phase_label**: phase classification label, with ``0`` for the
                    ordered phase and ``1`` for the disordered phase.
        """
        x = self.data[idx]  # shape (L*L,)
        if self.transform:
            x = self.transform(x)
        return x, self.T_label, self.phase_label
