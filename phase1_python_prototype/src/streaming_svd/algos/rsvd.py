"""Cold-start randomized SVD implementation."""

import time
from typing import Dict, Optional, Tuple, Union

import torch


def rsvd(
    A: torch.Tensor,
    k: int,
    p: int = 10,
    q: int = 0,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    return_stats: bool = True,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
]:
    """
    Randomized SVD (cold-start) using the Halko et al. algorithm.
    
    Computes a rank-k approximation: A ≈ U @ diag(s) @ Vt
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (m, n).
    k : int
        Target rank.
    p : int, optional
        Oversampling parameter. Default is 10.
    q : int, optional
        Number of power iterations (subspace iterations). Default is 0.
    device : str or torch.device, optional
        Device to perform computation on. If None, uses A's device.
    dtype : torch.dtype, optional
        Data type for computation. Default is torch.float32.
    seed : int, optional
        Random seed for reproducibility.
    return_stats : bool, optional
        Whether to return statistics dictionary. Default is True.
    
    Returns
    -------
    U : torch.Tensor
        Left singular vectors of shape (m, k).
    s : torch.Tensor
        Singular values of shape (k,).
    Vt : torch.Tensor
        Right singular vectors of shape (k, n).
    stats : dict, optional
        Statistics including timings and matmul counts.
        Only returned if return_stats=True.
    
    Algorithm
    ---------
    1. Draw random Gaussian matrix Omega ~ N(0,1) of shape (n, k+p)
    2. Compute Y = A @ Omega
    3. Perform q power iterations: Y = A @ (A.T @ Y)
    4. Orthonormalize: Q = orth(Y) via QR decomposition
    5. Project: B = Q.T @ A
    6. Compute thin SVD: B = Uhat @ diag(s) @ Vt
    7. Lift: U = Q @ Uhat
    8. Truncate to rank k
    
    References
    ----------
    Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).
    Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
    """
    # Input validation
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    
    m, n = A.shape
    if k > min(m, n):
        raise ValueError(f"k={k} must be <= min(m,n)={min(m,n)}")
    if p < 0:
        raise ValueError(f"p={p} must be non-negative")
    if q < 0:
        raise ValueError(f"q={q} must be non-negative")
    
    # Device and dtype handling
    if device is None:
        device = A.device
    device = torch.device(device)
    
    A = A.to(device=device, dtype=dtype)
    
    # Statistics tracking
    stats = {
        'params': {'k': k, 'p': p, 'q': q, 'm': m, 'n': n, 'device': str(device)},
        'timings': {},
        'matmul_counts': {'A@X': 0, 'AT@X': 0},
    }
    
    t_start = time.perf_counter()
    
    # Step 1: Draw random Gaussian matrix Omega
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    t0 = time.perf_counter()
    Omega = torch.randn(n, k + p, dtype=dtype, device=device, generator=generator)
    stats['timings']['omega_gen'] = time.perf_counter() - t0
    
    # Step 2: Y = A @ Omega
    t0 = time.perf_counter()
    Y = A @ Omega
    stats['matmul_counts']['A@X'] += 1
    stats['timings']['initial_matmul'] = time.perf_counter() - t0
    
    # Step 3: Power iterations
    t0 = time.perf_counter()
    for _ in range(q):
        Y = A.T @ Y
        stats['matmul_counts']['AT@X'] += 1
        Y = A @ Y
        stats['matmul_counts']['A@X'] += 1
    stats['timings']['power_iterations'] = time.perf_counter() - t0
    
    # Step 4: Orthonormalization via QR
    t0 = time.perf_counter()
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    stats['timings']['qr'] = time.perf_counter() - t0
    
    # Step 5: Project B = Q.T @ A  (equivalent to A.T @ Q, counts as AT@X)
    t0 = time.perf_counter()
    B = Q.T @ A
    stats['matmul_counts']['AT@X'] += 1
    stats['timings']['projection'] = time.perf_counter() - t0
    
    # Step 6: Small SVD of B
    t0 = time.perf_counter()
    Uhat, s, Vt = torch.linalg.svd(B, full_matrices=False)
    stats['timings']['small_svd'] = time.perf_counter() - t0
    
    # Step 7: Lift U = Q @ Uhat
    t0 = time.perf_counter()
    U = Q @ Uhat
    stats['timings']['lift'] = time.perf_counter() - t0
    
    # Step 8: Truncate to rank k
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    
    stats['timings']['total'] = time.perf_counter() - t_start
    
    if return_stats:
        return U, s, Vt, stats
    else:
        return U, s, Vt
