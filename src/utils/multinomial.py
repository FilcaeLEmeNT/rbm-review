# SPDX-License-Identifier: MIT
# src/utils/multinomial.py

"""Utilities for manipulating tensors when using multinomial RBMs."""

import numpy as np
import math
import torch
import torch.nn.functional as F


class OneHotTransform:
    """Convert integer categories to one-hot encoded vectors.

    Attributes:
        q: Number of discrete categories.
    """

    def __init__(self, q: int):
        """Initialize the transform.

        Args:
            q: Number of discrete categories.
        """
        self.q = q

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Convert integer-valued tensor to one-hot encoding.

        Args:
            x: One-dimensional tensor of integer category labels with shape
                `(n_visible,)`.

        Returns:
            A float tensor of shape `(n_visible, q)` containing the one-hot
            representation of `x`.
        """
        return F.one_hot(
            x.long(), num_classes=self.q
        ).float()  # .transpose(0, 1)


class DiscretizeTransform:
    """Discretize grayscale values into integer categories.

    Attributes:
        bins: Number of discretization bins.
    """

    def __init__(self, bins: int):
        """Initialize the transform.

        Args:
            bins: Number of discretization bins.
        """
        self.bins = bins

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Discretize grayscale values.

        Args:
            x: Tensor of grayscale values in the interval `[0, 1]`.
                Shape e.g. [B, 32, 32] in [0,1]

        Returns:
            An long tensor with the same shape as `x` whose values lie in
            `[0, bins - 1]`.
        """
        x_bin = (x * self.bins).floor().clamp(max=self.bins - 1).long()
        return x_bin


def onehot_to_categories(
    x: torch.Tensor, data_size: int, bins: int = 10
) -> torch.Tensor:
    """Convert one-hot encoded vectors to integer category labels.

    Args:
        x: One-hot encoded tensor of shape `(batch_size, data_size * bins)`,
            or any tensor that can be reshaped to
            `(-1, data_size, bins)`.
        data_size: Number of categorical variables.
        bins: Number of categories per variable.

    Returns:
        An integer tensor of shape `(batch_size, data_size)` containing the
        category indices.
    """
    return x.view(-1, data_size, bins).argmax(dim=-1)


def categories_to_grayscale(
    x_cat: torch.Tensor, bins: int = 10
) -> torch.Tensor:
    """Convert discretized categories to grayscale values.

    Each category is mapped to the midpoint of its corresponding interval in
    `[0, 1]`.

    Args:
        x_cat: Integer tensor containing category labels.
        bins: Number of discrete categories.

    Returns:
        A float tensor with values in `[0, 1]` representing the midpoint of
        each discretization bin.
    """
    x_float = (x_cat.float() + 0.5) / bins  # map bins -> midpoints in [0,1]
    return x_float
