"""Synthetic matrix generation and perturbation utilities."""

import torch


def make_initial_matrix(m, n, rank, decay=0.1, device='cpu', dtype=torch.float32, seed=None):
    """
    Generate a low-rank matrix with controlled singular value decay.
    
    Creates S = U @ diag(s) @ V.T where singular values decay exponentially.
    
    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    rank : int
        True rank of the matrix.
    decay : float, optional
        Decay rate for singular values. Default is 0.1.
        s_i = exp(-decay * i) for i in [0, rank).
    device : str or torch.device, optional
        Device for computation. Default is 'cpu'.
    dtype : torch.dtype, optional
        Data type. Default is torch.float32.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    S : torch.Tensor
        Low-rank matrix of shape (m, n).
    U : torch.Tensor
        Left singular vectors of shape (m, rank).
    s : torch.Tensor
        Singular values of shape (rank,).
    Vt : torch.Tensor
        Right singular vectors of shape (rank, n).
    """
    if rank > min(m, n):
        raise ValueError(f"rank={rank} must be <= min(m,n)={min(m,n)}")
    
    device = torch.device(device)
    
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    # Generate random orthonormal matrices
    U_full = torch.randn(m, rank, dtype=dtype, device=device, generator=generator)
    U, _ = torch.linalg.qr(U_full, mode='reduced')
    
    V_full = torch.randn(n, rank, dtype=dtype, device=device, generator=generator)
    V, _ = torch.linalg.qr(V_full, mode='reduced')
    Vt = V.T
    
    # Generate decaying singular values
    indices = torch.arange(rank, dtype=dtype, device=device)
    s = torch.exp(-decay * indices)
    
    # Construct S = U @ diag(s) @ Vt
    S = U @ torch.diag(s) @ Vt
    
    return S, U, s, Vt


def perturb_step(S_prev, eta, noise_rank=None, device=None, dtype=None, seed=None):
    """
    Add a perturbation to the matrix: S_new = S_prev + E.
    
    The perturbation E has Frobenius norm controlled by eta.
    
    Parameters
    ----------
    S_prev : torch.Tensor
        Previous matrix of shape (m, n).
    eta : float
        Perturbation magnitude (Frobenius norm of E will be approximately eta * ||S_prev||_F).
    noise_rank : int, optional
        Rank of the noise perturbation. If None, uses full rank noise.
        Low-rank noise can simulate structured perturbations.
    device : str or torch.device, optional
        Device for computation. If None, uses S_prev's device.
    dtype : torch.dtype, optional
        Data type. If None, uses S_prev's dtype.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    S_new : torch.Tensor
        Perturbed matrix of shape (m, n).
    E : torch.Tensor
        Perturbation matrix of shape (m, n).
    """
    m, n = S_prev.shape
    
    if device is None:
        device = S_prev.device
    if dtype is None:
        dtype = S_prev.dtype
    
    device = torch.device(device)
    
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    # Generate perturbation
    if noise_rank is None or noise_rank >= min(m, n):
        # Full-rank noise
        E = torch.randn(m, n, dtype=dtype, device=device, generator=generator)
    else:
        # Low-rank noise
        if noise_rank <= 0:
            raise ValueError(f"noise_rank={noise_rank} must be positive")
        
        U_noise = torch.randn(m, noise_rank, dtype=dtype, device=device, generator=generator)
        V_noise = torch.randn(n, noise_rank, dtype=dtype, device=device, generator=generator)
        E = U_noise @ V_noise.T
    
    # Scale perturbation to desired magnitude
    S_norm = torch.linalg.norm(S_prev, ord='fro')
    E_norm = torch.linalg.norm(E, ord='fro')
    
    target_norm = eta * S_norm
    E = E * (target_norm / E_norm)
    
    # Add perturbation
    S_new = S_prev + E
    
    return S_new, E
