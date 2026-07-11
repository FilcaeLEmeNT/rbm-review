# SPDX-License-Identifier: MIT
# src/utils/device.py

"""Utilities for selecting the PyTorch compute device.

Provides helper functions for detecting CUDA availability, selecting the
appropriate compute device, and optionally printing diagnostic information
about the current PyTorch and CUDA environment.
"""

import torch


def get_device(*, verbose: bool = True) -> torch.device:
    """Return the ``torch.device`` class. The device will be either 'cuda' or 'cpu'.

    Args:
        verbose: If ``True``, prints diagnostic information.

            Specifically, this includes:
                - PyTorch version: ``torch.__version__``
                - CUDA available: ``torch.cuda.is_available()``
                - CUDA device count: ``torch.cuda.device_count()``
                - GPU device name: ``torch.cuda.get_device_name(0)`` if cuda is available.
                - Using device: either 'cuda' or 'cpu'

    """
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
