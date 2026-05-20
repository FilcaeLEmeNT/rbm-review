import torch
from torch.utils.data import Dataset

import os
import numpy as np

class PottsDataset(Dataset):
    def __init__(self, npy_path='./', q=4, transform=None):
        self.data = np.load(npy_path)  # numpy array (n_samples, L*L)
        self.data = torch.from_numpy(self.data).long()  # convert to tensor (int64)

        filename = os.path.basename(npy_path)
        T = float(filename.split("T")[1].split(".npy")[0])
        self.T_label = torch.tensor(T).float()  # regression label
        Tc = 1.0 / np.log(1.0 + np.sqrt(q))

        self.phase_label = torch.tensor(0 if T < Tc else 1).long()  # Phase classification label

        self.q = q
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]  # shape (L*L,)
        if self.transform:
            x = self.transform(x)
        return x, self.T_label, self.phase_label