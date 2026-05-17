import numpy as np
import math
import torch

def ising_energy(tensor, dimension=None, J=1.0):
    '''
    Given a batch of configurations, compute the Ising energy for each configuration.

    Parameters:
    - tensor : torch.Tensor, shape [batch_size, p * p] or [p * p]
        Spin configurations in {-1, +1} format (NOT binary {0, 1}).
    - dimension: The dimensions of the grid. If None, it will be inferred from the tensor's second dimension assuming a square grid.
    - J: The coupling constant.

    Returns:
    - E: A tensor of energies for each configuration, shape [batch_size, ]

    Input size: [batch_size, p * p]
    Output size: [batch_size, ]
    '''
    if tensor.dim() == 2:
        batch_size = tensor.shape[0]
        size = tensor.shape[1]
        
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
        batch_size = 1
        size = tensor.shape[0]
    else:
        raise ValueError("Tensor must be 1D or 2D. \n", f"Usage: ising_energy(Tensor[batch_size, p * p], dimension=[p, p], J = 1.0).")
        
    # If dimension is not provided, assume it's a square grid and infer dimensions from size
    if dimension == None:
        if not math.isqrt(size):
            raise ValueError(f"Tensor's 2nd dimension needs to be a perfect square if dimension is set to None. Tensor shape is: {tensor.shape}. Suggestion: set dimension=[p, p] where p is the grid size.")
        sqrt = np.sqrt(size)
        dimension = [int(sqrt), int(sqrt)]

    dim1, dim2 = dimension
    grid = tensor.view(batch_size, dim1, dim2)
    right = grid.roll(shifts=-1, dims=2) # Horizontal Neighbors
    down = grid.roll(shifts=-1, dims=1) # Vertical Neighbors

    E_h = (grid * right)  # Horizontal energy
    E_v = (grid * down)   # Vertical energy

    sums = E_h.sum(dim=(1,2)) + E_v.sum(dim=(1,2))
    E = -J * sums

    return E

def ising_magnetization(tensor):
    '''
    Given a batch of configurations, compute the 
    magnetization and susceptibility.

    Parameters:
    - tensor : torch.Tensor, shape [batch_size, p * p] or [p * p]
        Spin configurations in {-1, +1} format (NOT binary {0, 1}).

    Returns:
    - M : torch.Tensor, shape [batch_size,]
        Raw signed extensive magnetization M = sum_i s_i for each
        configuration. To get per-spin magnetization divide by N = p*p.
        To get the order parameter take abs() and divide by N.

    Input size: [batch_size, p * p]
    Output size: [batch_size, ]
    '''
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # Add batch dimension if input is a single configuration

    M = tensor.sum(dim=1)  # Raw, Signed, Magnetization

    return M

def potts_energy(config, J=1.0):
    # E = -J * sum_{<i,j>} delta(s_i, s_j)
    # config: (N, L, L) with s in {0, ..., q-1}
    raise NotImplementedError("This method is not yet implemented.")
    return

def potts_magnetization(config, q):
    # Order parameter: fraction in majority state
    raise NotImplementedError("This method is not yet implemented.")
    return

def xy_energy(tensor, dimension=None, J=1.0):
    '''
    Given a batch of configurations, compute the XY energy for each configuration.

    Parameters:
    - tensor: A batch of configurations, shape [batch_size, p * p] or
    [p * p] for a single configuration.
    - dimension: The dimensions of the grid.
    - J: The coupling constant.

    Returns:
    - E: A tensor of energies for each configuration, shape [batch_size, ]

    Input size: [batch_size, p * p]
    Output size: [batch_size, ]
    '''
    if tensor.dim() == 2:
        batch_size = tensor.shape[0]
        size = tensor.shape[1]
        
    elif tensor.dim() == 1:
        batch_size = 1
        size = tensor.shape[0]
    else:
        raise ValueError(f"Usage: XY_energy(Tensor[batch_size, p * p], dimension=[p * p], J = 1.0).")
        
    # If dimension is not provided, assume it's a square grid and infer dimensions from size
    if dimension == None:
        if not math.isqrt(size):
            raise ValueError(f"Tensor's 2nd dimension needs to be a perfect square if dim is set to None. Tensor shape is: {tensor.shape}")
        sqrt = np.sqrt(size)
        dimension = [int(sqrt), int(sqrt)]

    dim1, dim2 = dimension
    grid = tensor.view(batch_size, dim1, dim2)
    right = grid.roll(shifts=-1, dims=2) # Horizontal Neighbors
    down = grid.roll(shifts=-1, dims=1) # Vertical Neighbors

    E_h = torch.cos(grid - right)
    E_v = torch.cos(grid - down)

    sums = E_h.sum(dim=(1,2)) + E_v.sum(dim=(1,2))
    E = -J * sums

    return E

def xy_magnetization(tensor):
    '''
    Given a batch of configurations, compute the 
    magnetization and susceptibility.

    Parameters:
    - tensor : torch.Tensor, shape [batch_size, p * p] or [p * p]

    Returns:
    - M : torch.Tensor, shape [batch_size,]
        Extensive magnetization for each
        configuration. To get per-spin magnetization divide by N = p*p.
        To get the order parameter take abs() and divide by N.

    Input size: [batch_size, p * p]
    Output size: [batch_size, ]
    '''
    cos = torch.cos(tensor)
    sin = torch.sin(tensor)

    Mx = cos.sum(dim=1)
    My = sin.sum(dim=1)

    M = np.sqrt(Mx**2 + My**2)

    return M

def heat_capacity(energies, T, k_B = 1.0):
    '''
    Given a batch of energies given by the previous function, 
    compute the heat capacity.

    Parameters:
    - energies: A batch of energies, shape [batch_size, ]
    - T: The temperature.
    - k_B: The Boltzmann constant.

    Returns:
    - C: The heat capacity, shape [1, ]

    Input size: [batch_size, ] of Energies
    Output size: [1, ]
    '''

    E_mean = energies.mean()
    E2_mean = (energies**2).mean()

    C = (E2_mean - E_mean**2) / (k_B * T**2)
    
    return C

def susceptibility(magnetizations, T, k_B = 1.0):
    '''
    Given a batch of magnetizations given by the previous function, 
    compute the magnetic susceptibiility.

    Parameters:
    - magnetizations: A batch of magnetizations, shape [batch_size, ]
    - T: The temperature.
    - k_B: The Boltzmann constant.

    Returns:
    - Chi: The magnetic susceptibility, shape [1, ]

    Input size: [batch_size, ] of Magnetizations
    Output size: [1, ]
    '''
    m_mean = magnetizations.mean()
    m2_mean = (magnetizations**2).mean()

    Chi = (m2_mean - m_mean**2) / (k_B * T)

    return Chi

'''
### Delete after testing: ###
'''
def main():
    # Test ising energy function with a simple example
    T = 3.00
    example_config = torch.ones(1, 4)

    E = ising_energy(example_config, [2, 2])
    C = heat_capacity(E, float(T))
    M = ising_magnetization(example_config)
    Chi = susceptibility(M, float(T))
    M = M.abs().mean()
    E = E.mean()

    N = example_config.shape[1]

    print("Total Energy (E): ", E)
    print("Energy per site (E/N): ", E / N)
    print("Total Heat Capacity (C): ", C)
    print("Heat Capacity per site (C/N): ", C / N)
    print("Total Magnetization (m): ", M)
    print("Magnetization per site (m/N): ", M / N)
    print("Total Susceptibility (chi): ", Chi)
    print("Susceptibility per site (chi/N): ", Chi / N)

if __name__ == "__main__":
    main()