"""Rotating-subspace synthetic streaming simulation utilities."""

import torch

from streaming_svd.sims.perturbation import make_initial_matrix


def make_initial_matrix_rotating(
    m,
    n,
    rank,
    decay=0.1,
    seed=None,
    device='cpu',
    dtype=torch.float32,
):
    """
    Generate initial low-rank matrix for rotating-subspace experiments.

    Returns
    -------
    S0 : torch.Tensor
        Initial matrix of shape (m, n).
    U : torch.Tensor
        Left singular vectors of shape (m, rank).
    s : torch.Tensor
        Singular values of shape (rank,).
    Vt : torch.Tensor
        Right singular vectors of shape (rank, n).
    """
    return make_initial_matrix(
        m=m,
        n=n,
        rank=rank,
        decay=decay,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def _random_rotation(rank, angle, device='cpu', dtype=torch.float32, seed=None):
    """Generate a small random orthogonal rotation matrix via matrix exponential."""
    if rank <= 0:
        raise ValueError("rank must be positive")

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    z = torch.randn(rank, rank, device=device, dtype=dtype, generator=generator)
    k = z - z.T
    r = torch.matrix_exp(angle * k)
    return r


def rotate_step(
    U,
    V,
    s,
    angle,
    device=None,
    dtype=None,
    seed=None,
    rotate_both=True,
    reorthonormalize=True,
):
    """
    Apply a small random orthogonal rotation to latent factors.

    Parameters
    ----------
    U : torch.Tensor
        Current left singular vectors of shape (m, rank).
    V : torch.Tensor
        Current right singular vectors of shape (n, rank).
    s : torch.Tensor
        Fixed singular values of shape (rank,).
    angle : float
        Rotation magnitude.
    rotate_both : bool, optional
        Whether to rotate both U and V. If False, rotates U only.

    Returns
    -------
    S_new : torch.Tensor
        Updated matrix U_new @ diag(s) @ V_new.T.
    U_new : torch.Tensor
        Updated left basis.
    V_new : torch.Tensor
        Updated right basis.
    """
    if device is None:
        device = U.device
    if dtype is None:
        dtype = U.dtype

    rank = U.shape[1]
    r_u = _random_rotation(rank, angle, device=device, dtype=dtype, seed=seed)
    U_new = U @ r_u

    if rotate_both:
        seed_v = None if seed is None else seed + 1
        r_v = _random_rotation(rank, angle, device=device, dtype=dtype, seed=seed_v)
        V_new = V @ r_v
    else:
        V_new = V

    if reorthonormalize:
        U_new, _ = torch.linalg.qr(U_new, mode='reduced')
        V_new, _ = torch.linalg.qr(V_new, mode='reduced')

    S_new = U_new @ torch.diag(s) @ V_new.T
    return S_new, U_new, V_new
