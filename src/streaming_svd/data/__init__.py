"""Data loading and preprocessing for the Hurricane Isabel dataset.

Real-data loaders for the IEEE Visualization 2004 Contest hurricane simulation.
Data consists of 13 atmospheric variables × 48 hourly timesteps.

Each binary file contains a 100×500×500 float32 volume (z-levels, y, x),
reshaped to a matrix of shape (250000, 100): rows are spatial (x, y) points,
columns are z-levels.

File layout on disk::

    data/raw/{VAR}/{VAR}{T:02d}.bin     T in 01..48

Example::

    from streaming_svd.data import (
        HURRICANE_VARIABLES,
        load_weather_matrix,
        discover_variable_files,
        optimal_rank_k_rel_fro_error_from_gram,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HURRICANE_VARIABLES: Tuple[str, ...] = (
    "CLOUDf",
    "Pf",
    "PRECIPf",
    "QCLOUDf",
    "QGRAUPf",
    "QICEf",
    "QRAINf",
    "QSNOWf",
    "QVAPORf",
    "TCf",
    "Uf",
    "Vf",
    "Wf",
)
"""All 13 atmospheric variables in the Hurricane Isabel dataset."""

# Shape of the raw volumetric data: (z-levels, y, x)
_VOLUME_SHAPE: Tuple[int, int, int] = (100, 500, 500)

# Matrix shape after unfolding: (n_spatial, n_z) = (500*500, 100)
MATRIX_SHAPE: Tuple[int, int] = (250_000, 100)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_variable_files(
    data_dir: Path,
    var: str,
    start: int = 1,
    end: int = 48,
) -> List[Tuple[int, Path]]:
    """Return sorted ``(timestep, path)`` pairs for files that exist on disk.

    Silently omits missing timesteps; the caller is responsible for warning
    about gaps if needed.

    Parameters
    ----------
    data_dir:
        Root data directory (e.g. ``Path("data/raw")``).
    var:
        Variable name prefix, e.g. ``"Uf"``.
    start, end:
        Inclusive timestep range to search (1-indexed).

    Returns
    -------
    list of (timestep, path) tuples, sorted by timestep.

    Examples
    --------
    >>> pairs = discover_variable_files(Path("data/raw"), "Uf", start=1, end=5)
    >>> pairs[0]
    (1, PosixPath('data/raw/Uf/Uf01.bin'))
    """
    result: List[Tuple[int, Path]] = []

    # Primary: directory named exactly as var (e.g. data/raw/QCLOUDf/)
    # Fallback: directory with trailing 'f' stripped (e.g. data/raw/PRECIP/ for var=PRECIPf)
    var_dir = data_dir / var
    if not var_dir.is_dir() and var.endswith("f"):
        var_dir = data_dir / var[:-1]

    for t in range(start, end + 1):
        path = var_dir / f"{var}{t:02d}.bin"
        if path.exists():
            result.append((t, path))
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_weather_matrix(
    path: Path,
    shape: Tuple[int, int, int] = _VOLUME_SHAPE,
    memmap: bool = False,
) -> torch.Tensor:
    """Load a binary float32 weather snapshot and reshape to matrix form.

    Parameters
    ----------
    path:
        Path to a ``{VAR}{T:02d}.bin`` file (little-endian float32).
    shape:
        Expected volume shape ``(z, y, x)``.  Default ``(100, 500, 500)``.
    memmap:
        If ``True``, use :func:`numpy.memmap` to avoid loading the full
        ~95 MB file into RAM at once.  Useful when memory is tight.

    Returns
    -------
    torch.Tensor
        Matrix of shape ``(z*y, x) = (250000, 100)`` in float32 on CPU.
        Rows are spatial ``(x, y)`` points; columns are z-levels.

    Raises
    ------
    RuntimeError
        If the file cannot be loaded or has an unexpected size.
    """
    try:
        if memmap:
            volume = np.memmap(path, dtype="<f4", mode="r", shape=shape)
        else:
            data = np.fromfile(path, dtype="<f4")
            expected = int(np.prod(shape))
            if data.size != expected:
                raise ValueError(
                    f"Expected {expected} float32 values, got {data.size} in {path}"
                )
            volume = data.reshape(shape)

        # Reshape (z, y, x) → (y*x, z) — rows are spatial points, cols are z-levels
        A = volume.reshape(shape[0], -1).T  # (500*500, 100) = (250000, 100)
        A = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32))
        return A
    except Exception as exc:
        raise RuntimeError(f"Failed to load {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Optimal rank-k baseline
# ---------------------------------------------------------------------------

def optimal_rank_k_rel_fro_error_from_gram(
    A: torch.Tensor,
    k: int,
) -> float:
    """Compute the optimal rank-*k* relative Frobenius error via the Gram matrix.

    Avoids a full SVD of the tall matrix ``A`` (250 000 × 100) by working with
    the square Gram matrix ``G = Aᵀ A`` (100 × 100) instead.

    The optimal relative error is:

    .. math::

        \\frac{\\|A - A_k\\|_F}{\\|A\\|_F}
        = \\sqrt{\\frac{\\sum_{i>k} \\sigma_i^2}{\\sum_i \\sigma_i^2}}

    where :math:`\\sigma_i` are the singular values of ``A``.

    Parameters
    ----------
    A:
        Matrix of shape ``(m, n)`` where ``m ≫ n`` (e.g. 250 000 × 100).
    k:
        Target rank.

    Returns
    -------
    float
        Relative Frobenius error of the best rank-*k* approximation.
        Returns ``0.0`` if ``k >= n`` or total energy is zero.
    """
    if k >= A.shape[1]:
        return 0.0

    with torch.no_grad():
        G = A.T @ A
        G = G.to(torch.float64)

        eigvals = torch.linalg.eigvalsh(G)   # ascending order
        eigvals = torch.clamp(eigvals, min=0.0)

        sigmas_sq = torch.flip(eigvals, dims=[0])  # descending order

        total_energy = torch.sum(sigmas_sq)
        if total_energy <= 0:
            return 0.0

        tail_energy = torch.sum(sigmas_sq[k:])
        opt_error = float(torch.sqrt(tail_energy / total_energy).item())

    return opt_error


__all__ = [
    "HURRICANE_VARIABLES",
    "MATRIX_SHAPE",
    "load_weather_matrix",
    "discover_variable_files",
    "optimal_rank_k_rel_fro_error_from_gram",
]
