"""Warm-start randomized SVD implementation."""

import time
import torch
from .rsvd import rsvd


def warm_rsvd(
    A,
    U_prev,
    k,
    p=5,
    q=0,
    *,
    device=None,
    dtype=torch.float32,
    seed=None,
    return_stats=True,
):
    """
    Warm-start randomized SVD using previous subspace information.
    
    Uses the subspace U_prev from the previous snapshot to initialize
    the range finding step, requiring fewer new random vectors.
    
    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (m, n).
    U_prev : torch.Tensor or None
        Orthonormal basis from previous snapshot, shape (m, r).
        If None, falls back to cold-start rSVD.
    k : int
        Target rank.
    p : int, optional
        Oversampling parameter (typically smaller than cold-start). Default is 5.
    q : int, optional
        Number of power iterations. Default is 0.
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
    If U_prev is None, falls back to cold rsvd(A, k, p, q).
    Otherwise:
    1. Compute G = A.T @ U_prev (warm-start component)
    2. Y1 = A @ G (equals (A @ A.T) @ U_prev via subspace iteration)
    3. Draw random matrix Omega ~ N(0,1) of shape (n, p)
    4. Y2 = A @ Omega (exploration component)
    5. Y = [Y1, Y2] (concatenate along columns)
    6. Perform q power iterations: Y = A @ (A.T @ Y)
    7. Orthonormalize: Q = orth(Y) via QR
    8. Project: B = Q.T @ A
    9. Compute thin SVD: B = Uhat @ diag(s) @ Vt
    10. Lift: U = Q @ Uhat
    11. Truncate to rank k
    
    References
    ----------
    Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition.
    Linear algebra and its applications, 415(1), 20-30.
    """
    # Fallback to cold-start if no previous basis
    if U_prev is None:
        return rsvd(A, k, p=p, q=q, device=device, dtype=dtype, seed=seed, return_stats=return_stats)
    
    # Input validation
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if U_prev.ndim != 2:
        raise ValueError(f"U_prev must be 2D, got shape {U_prev.shape}")
    
    m, n = A.shape
    m_prev, r_prev = U_prev.shape
    
    if m != m_prev:
        raise ValueError(f"A has {m} rows but U_prev has {m_prev} rows")
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
    U_prev = U_prev.to(device=device, dtype=dtype)
    
    # Orthonormalize U_prev if needed (for numerical stability)
    U_prev_norm = torch.linalg.norm(U_prev.T @ U_prev - torch.eye(r_prev, device=device, dtype=dtype))
    if U_prev_norm > 1e-6:
        U_prev, _ = torch.linalg.qr(U_prev, mode='reduced')
    
    # Statistics tracking
    stats = {
        'params': {
            'k': k, 'p': p, 'q': q, 'm': m, 'n': n,
            'r_prev': r_prev, 'device': str(device),
            'warm_start': True,
        },
        'timings': {},
        'matmul_counts': {'A@X': 0, 'AT@X': 0},
    }
    
    t_start = time.perf_counter()
    
    # Step 1: Compute G = A.T @ U_prev
    t0 = time.perf_counter()
    G = A.T @ U_prev
    stats['matmul_counts']['AT@X'] += 1
    stats['timings']['warm_proj'] = time.perf_counter() - t0
    
    # Step 2: Y1 = A @ G (warm-start component)
    t0 = time.perf_counter()
    Y1 = A @ G
    stats['matmul_counts']['A@X'] += 1
    stats['timings']['warm_matmul'] = time.perf_counter() - t0
    
    # Step 3: Draw random matrix Omega
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None
    
    t0 = time.perf_counter()
    Omega = torch.randn(n, p, dtype=dtype, device=device, generator=generator)
    stats['timings']['omega_gen'] = time.perf_counter() - t0
    
    # Step 4: Y2 = A @ Omega (exploration component)
    t0 = time.perf_counter()
    Y2 = A @ Omega
    stats['matmul_counts']['A@X'] += 1
    stats['timings']['random_matmul'] = time.perf_counter() - t0
    
    # Step 5: Concatenate Y = [Y1, Y2]
    t0 = time.perf_counter()
    Y = torch.cat([Y1, Y2], dim=1)
    stats['timings']['concat'] = time.perf_counter() - t0
    
    # Step 6: Power iterations
    t0 = time.perf_counter()
    for _ in range(q):
        Y = A.T @ Y
        stats['matmul_counts']['AT@X'] += 1
        Y = A @ Y
        stats['matmul_counts']['A@X'] += 1
    stats['timings']['power_iterations'] = time.perf_counter() - t0
    
    # Step 7: Orthonormalization via QR
    t0 = time.perf_counter()
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    stats['timings']['qr'] = time.perf_counter() - t0
    
    # Step 8: Project B = Q.T @ A
    t0 = time.perf_counter()
    B = Q.T @ A
    stats['timings']['projection'] = time.perf_counter() - t0
    
    # Step 9: Small SVD of B
    t0 = time.perf_counter()
    Uhat, s, Vt = torch.linalg.svd(B, full_matrices=False)
    stats['timings']['small_svd'] = time.perf_counter() - t0
    
    # Step 10: Lift U = Q @ Uhat
    t0 = time.perf_counter()
    U = Q @ Uhat
    stats['timings']['lift'] = time.perf_counter() - t0
    
    # Step 11: Truncate to rank k
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    
    stats['timings']['total'] = time.perf_counter() - t_start
    
    if return_stats:
        return U, s, Vt, stats
    else:
        return U, s, Vt
