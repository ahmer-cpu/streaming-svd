"""Independent random matrix series for control experiments."""

import torch


def make_random_matrix(m, n, rank=None, decay=0.1, model="lowrank", device="cpu", dtype=torch.float32, seed=None):
    """
    Generate a single independent random matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    rank : int, optional
        Rank for lowrank model. If None, defaults to 2*k (caller's responsibility to pass).
        Ignored for gaussian model.
    decay : float, optional
        Decay rate for singular values in lowrank model. Default is 0.1.
        s_i = exp(-decay * i) for i in [0, rank).
    model : str, optional
        Model type: 'lowrank' (default) or 'gaussian'.
        - 'lowrank': S = U @ diag(s) @ V.T with decaying singular values
        - 'gaussian': iid N(0,1) entries, optionally scaled
    device : str or torch.device, optional
        Device for computation. Default is 'cpu'.
    dtype : torch.dtype, optional
        Data type. Default is torch.float32.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    S : torch.Tensor
        Random matrix of shape (m, n).
    """
    device = torch.device(device)

    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    if model == "lowrank":
        if rank is None:
            raise ValueError("rank must be provided for lowrank model")
        if rank > min(m, n):
            rank = min(m, n)

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

        return S

    elif model == "gaussian":
        # iid N(0,1) entries, scaled by 1/sqrt(n) for stability
        S = torch.randn(m, n, dtype=dtype, device=device, generator=generator) / torch.sqrt(torch.tensor(n, dtype=dtype))
        return S

    else:
        raise ValueError(f"Unknown model: {model}. Choose 'lowrank' or 'gaussian'.")


def sample_independent_series(m, n, T, *, rank=None, decay=0.1, model="lowrank", device="cpu", dtype=torch.float32, seed=None):
    """
    Generator that yields T independent random matrices.

    Each matrix is generated with a reproducible but independent seed (seed + t).
    This ensures reproducibility while maintaining independence across timesteps.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    T : int
        Number of timesteps to generate.
    rank : int, optional
        Rank for lowrank model.
    decay : float, optional
        Decay rate for lowrank model. Default is 0.1.
    model : str, optional
        Model type: 'lowrank' (default) or 'gaussian'.
    device : str or torch.device, optional
        Device for computation. Default is 'cpu'.
    dtype : torch.dtype, optional
        Data type. Default is torch.float32.
    seed : int, optional
        Base random seed. If None, uses non-deterministic generation.

    Yields
    ------
    S_t : torch.Tensor
        Random matrix of shape (m, n) for timestep t.
    """
    for t in range(1, T + 1):
        # Seed each timestep independently but reproducibly
        if seed is not None:
            t_seed = seed + t
        else:
            t_seed = None

        S_t = make_random_matrix(
            m,
            n,
            rank=rank,
            decay=decay,
            model=model,
            device=device,
            dtype=dtype,
            seed=t_seed,
        )

        yield S_t
