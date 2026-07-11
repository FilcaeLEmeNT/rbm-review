# SPDX-License-Identifier: MIT
# # src/utils/physics.py

"""Utilities for calculating physical properties for ising, xy, and potts models.

Typical usage example:
    # tensor: A batch of configurations, shape [batch_size, number_of_spins]
    observables = compute_observables(tensor, data_type="potts", T=T, q=q)
    e = observables.energy_per_site
    c = observables.heat_capacity_per_site
    m = observables.magnetization_per_site
    chi = observables.susceptibility_per_site
"""

import numpy as np
import math
import torch
from dataclasses import dataclass


@dataclass
class Observables:
    """Data class containing physical observables for ising, xy, and potts models.

    Attributes:
        energy (Tensor): Total Mean Energy, E
        energy_per_site (Tensor): Mean Energy per site, E / N
        heat_capacity (Tensor): Heat Capacity, C
        heat_capacity_per_site (Tensor): Heat Capacity per site, C / N
        magnetization (Tensor): Total Mean Magnetization. M
        magnetization_per_site (Tensor): Mean Magnetization per site. M / N
        susceptibility (Tensor): Susceptibility, Chi
        susceptibility_per_site (Tensor): Susceptibility per site, Chi / N
    """

    energy: torch.Tensor
    energy_per_site: torch.Tensor
    heat_capacity: torch.Tensor
    heat_capacity_per_site: torch.Tensor
    magnetization: torch.Tensor
    magnetization_per_site: torch.Tensor
    susceptibility: torch.Tensor
    susceptibility_per_site: torch.Tensor


def compute_observables(
    tensor: torch.Tensor,
    data_type: str,
    T: float,
    dimension: tuple | None = None,
    q: int = None,
    J: float = 1.0,
    k_B: float = 1.0,
) -> Observables:
    """Compute the physical ovservables for a particular tensor configuration.

    Calculate physical properties for an ising, xy, or potts configuration and
    return the Observables data class.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins] or [number_of_spins].
        data_type: String specifying the type of model, 'ising', 'xy', or 'potts'
        T: Temperature.
        dimension: Dimensions of the grid. If None, it will be inferred
            from the tensor assuming a square grid.
        q: Number of states for a q-state Potts model.
        J: Coupling constant.
        k_B: Boltzmann constant.

    Returns:
        Observables data class containing the physical properties computed by the function.

    Raises:
        ValueError: If tensor has more than 2 dimensions, if data_type is 'potts'
            and ``q`` is None, or if ``dimension`` is None but the number_of_spins
            detected from the tensor is not a perfect square.
    """
    tensor = ensure_batch_dim(tensor)
    size = tensor.shape[1]

    # If dimension is not provided, assume it's a square grid and infer dimensions from size.
    if dimension == None:
        dimension = infer_square_dim(size=size)

    if data_type == "ising":
        E = ising_energy(tensor.cpu(), dimension=dimension, J=J)
        M = ising_magnetization(tensor.cpu())

    elif data_type == "xy":
        E = xy_energy(tensor.cpu(), dimension=dimension, J=J)
        M = xy_magnetization(tensor.cpu())

    elif data_type == "potts":
        if q == None:
            raise ValueError(
                "q needs to be defined when the data_type is 'potts' to calculate the magnetization."
            )
        E = potts_energy(tensor.cpu().long(), dimension=dimension, J=J)
        M = potts_magnetization(tensor.cpu().long(), q)
    else:
        ValueError(
            f"Invalid data_type {data_type}. Valid data_types are: 'ising', 'xy', and 'potts'"
        )

    C = heat_capacity(E, T, k_B=k_B)
    Chi = susceptibility(M, T, k_B=k_B)

    M_mean = M.abs().mean()
    E_mean = E.mean()

    N = tensor.flatten(start_dim=1).shape[1]

    return Observables(
        energy=E_mean,
        energy_per_site=E_mean / N,
        heat_capacity=C,
        heat_capacity_per_site=C / N,
        magnetization=M_mean,
        magnetization_per_site=M_mean / N,
        susceptibility=Chi,
        susceptibility_per_site=Chi / N,
    )


def ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor has a batch dimension.

    If `tensor` is one-dimensional, a leading batch dimension is added.
    Two-dimensional tensors are returned unchanged.

    Args:
        tensor: Input tensor.

    Returns:
        A tensor with shape `[batch_size, number_of_spins]`.

    Raises:
        ValueError: If `tensor` is not one- or two-dimensional.
    """
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)

    if tensor.dim() != 2:
        raise ValueError(
            f"Tensor must be 1D or 2D. Got {tensor.dim()} dimensions."
        )

    return tensor


def infer_square_dim(size: int) -> tuple[int, int]:
    """Return the 2D square dimensions based on input as a tuple.

    Raises:
        ValueError: If input is not a perfect squre.
    """
    sqrt = math.isqrt(size)
    if not sqrt * sqrt == size:
        raise ValueError(
            "Size needs to be a perfect square if dimension is set to None. "
            f"Size given: {size}."
        )

    return (int(sqrt), int(sqrt))


def uses_zero_one_encoding(tensor: torch.Tensor) -> bool:
    """Return True if the tensor appears to use {0, 1} encoding.

    The function looks for any 0 values in the tensor.
    """
    mask = tensor == 0
    return mask.any()


def potts_critical_T(q: int) -> float:
    """Return the critical temperature of a Potts model based on q."""
    return 1.0 / np.log(1.0 + np.sqrt(q))  # exact Tc with 2D Potts


def ising_energy(
    tensor: torch.Tensor, dimension: tuple | None = None, J: float = 1.0
) -> torch.Tensor:
    """Given a batch of configurations, compute the Ising energy for each configuration.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins].
            If tensor in binary format {0, 1}, convert to {-1, 1} if there are 0
            values in the tensor.
        dimension: The dimensions of the grid. If None, it will be inferred
            from the tensor assuming a square grid.
        J: The coupling constant.

    Returns:
        A tensor of energies for each configuration, shape [batch_size, ]

    Raises:
        ValueError: If tensor has more than 2 dimensions or if ``dimension`` is None
            but the number_of_spins detected from the tensor is not a perfect square.
    """
    tensor = ensure_batch_dim(tensor)
    batch_size = tensor.shape[0]
    size = tensor.shape[1]

    # If dimension is not provided, assume it's a square grid and infer dimensions from size.
    if dimension == None:
        dimension = infer_square_dim(size=size)

    # Check if there are any zeros. If so convert them to -1.
    if uses_zero_one_encoding(tensor):
        tensor = 2 * tensor - 1

    dim1, dim2 = dimension
    grid = tensor.view(batch_size, dim1, dim2)
    right = grid.roll(shifts=-1, dims=2)  # Horizontal Neighbors
    down = grid.roll(shifts=-1, dims=1)  # Vertical Neighbors

    E_h = grid * right  # Horizontal energy
    E_v = grid * down  # Vertical energy

    sums = E_h.sum(dim=(1, 2)) + E_v.sum(dim=(1, 2))
    E = -J * sums

    return E


def ising_magnetization(tensor: torch.Tensor) -> torch.Tensor:
    """Given a batch of configurations, compute the Ising magnetization for each configuration.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins]
            If tensor in binary format {0, 1}, convert to {-1, 1} if there are 0
            values in the tensor.

    Returns:
        torch.Tensor, shape [batch_size,]
        Raw signed extensive magnetization M = sum_i s_i for each
        configuration. To get per-spin magnetization divide by N = number_of_spins.
        To get the order parameter take abs() and divide by N.

    Raises:
        ValueError: If tensor has more than 2 dimensions.
    """
    tensor = ensure_batch_dim(tensor)

    # Check if there are any zeros. If so convert them to -1.
    if uses_zero_one_encoding(tensor):
        tensor = 2 * tensor - 1

    M = tensor.sum(dim=1)  # Raw, Signed, Magnetization

    return M


def potts_energy(
    tensor: torch.Tensor, dimension: tuple | None = None, J: float = 1.0
) -> torch.Tensor:
    # E = -J * sum_{<i,j>} delta(s_i, s_j)
    # config: (N, L, L) with s in {0, ..., q-1}
    """Given a batch of configurations, compute the Potts energy for each configuration.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins] or [number_of_spins]
            Spin configurations in {0, ..., q-1} format.
        dimension: The dimensions of the grid. If None, it will be inferred from the tensor's second dimension assuming a square grid.
        J: The coupling constant.

    Returns:
        E: A tensor of energies for each configuration, shape [batch_size, ]

    Raises:
        ValueError: If tensor has more than 2 dimensions or if ``dimension`` is None
            but the number_of_spins detected from the tensor is not a perfect square.
    """
    tensor = ensure_batch_dim(tensor)
    batch_size = tensor.shape[0]
    size = tensor.shape[1]

    # If dimension is not provided, assume it's a square grid and infer dimensions from size
    if dimension == None:
        dimension = infer_square_dim(size=size)

    dim1, dim2 = dimension
    grid = tensor.view(batch_size, dim1, dim2)
    right = grid.roll(shifts=-1, dims=2)  # Horizontal Neighbors
    down = grid.roll(shifts=-1, dims=1)  # Vertical Neighbors

    E_h = (grid == right).float()
    E_v = (grid == down).float()

    sums = E_h.sum(dim=(1, 2)) + E_v.sum(dim=(1, 2))
    E = -J * sums

    return E


def potts_magnetization(tensor: torch.Tensor, q: int) -> torch.Tensor:
    # Order parameter: fraction in majority state
    """Given a batch of configurations, compute the Potts magnetization for each configuration.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins] or [number_of_spins]
            Spin configurations in {0, ..., q-1} format.
        q: Number of states for a q-state Potts model.

    Returns:
        torch.Tensor, shape [batch_size,]
        Extensive magnetization for each
        configuration. To get per-spin magnetization divide by N = number_of_spins.

    Raises:
        ValueError: If tensor has more than 2 dimensions.
    """
    tensor = ensure_batch_dim(tensor)
    N = tensor.shape[1]

    # count occurrences of each Potts state
    counts = torch.stack([(tensor == s).sum(dim=1) for s in range(q)], dim=1)

    # largest occupation number
    Nmax = counts.max(dim=1).values

    # Extensive magnetization
    M = (q * Nmax.float() - N) / (q - 1)

    return M


def xy_energy(
    tensor: torch.Tensor, dimension: tuple | None = None, J: float = 1.0
) -> torch.Tensor:
    """Given a batch of configurations, compute the XY energy for each configuration.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins] or
            [number_of_spins] for a single configuration.
        dimension: The dimensions of the grid.
        J: The coupling constant.

    Returns:
        A tensor of energies for each configuration, shape [batch_size, ]

    Raises:
        ValueError: If tensor has more than 2 dimensions or if ``dimension`` is None
            but the number_of_spins detected from the tensor is not a perfect square.
    """
    tensor = ensure_batch_dim(tensor)
    batch_size = tensor.shape[0]
    size = tensor.shape[1]

    # If dimension is not provided, assume it's a square grid and infer dimensions from size
    if dimension == None:
        dimension = infer_square_dim(size=size)

    dim1, dim2 = dimension
    grid = tensor.view(batch_size, dim1, dim2)
    right = grid.roll(shifts=-1, dims=2)  # Horizontal Neighbors
    down = grid.roll(shifts=-1, dims=1)  # Vertical Neighbors

    E_h = torch.cos(grid - right)
    E_v = torch.cos(grid - down)

    sums = E_h.sum(dim=(1, 2)) + E_v.sum(dim=(1, 2))
    E = -J * sums

    return E


def xy_magnetization(tensor: torch.Tensor) -> torch.Tensor:
    """Given a batch of configurations, compute the XY magnetization for each configuration.

    Args:
        tensor: Tensor of shape [batch_size, number_of_spins] or [number_of_spins]

    Returns:
        torch.Tensor, shape [batch_size,]
        Extensive magnetization for each
        configuration. To get per-spin magnetization divide by N = number_of_spins.
        To get the order parameter take abs() and divide by N.

    Raises:
        ValueError: If tensor has more than 2 dimensions.
    """
    tensor = ensure_batch_dim(tensor)

    cos = torch.cos(tensor)
    sin = torch.sin(tensor)

    Mx = cos.sum(dim=1)
    My = sin.sum(dim=1)

    M = np.sqrt(Mx**2 + My**2)

    return M


def heat_capacity(Energies: torch.Tensor, T: float, k_B: float = 1.0):
    """Given a batch of energies given by the previous function, compute the heat capacity.

    Args:
        Energies: A batch of Energies, shape [batch_size, ]
        T: The temperature.
        k_B: The Boltzmann constant.

    Returns:
        C: The heat capacity, shape [1, ]
    """
    E_mean = Energies.mean()
    E2_mean = (Energies**2).mean()

    C = (E2_mean - E_mean**2) / (k_B * T**2)

    return C


def susceptibility(Magnetizations: torch.Tensor, T: float, k_B: float = 1.0):
    """Given a batch of magnetizations given by the previous function, compute the magnetic susceptibiility.

    Args:
        Magnetizations: A batch of magnetizations, shape [batch_size, ]
        T: The temperature.
        k_B: The Boltzmann constant.

    Returns:
        Chi: The magnetic susceptibility, shape [1, ]
    """
    m_mean = Magnetizations.mean()
    m2_mean = (Magnetizations**2).mean()

    Chi = (m2_mean - m_mean**2) / (k_B * T)

    return Chi
