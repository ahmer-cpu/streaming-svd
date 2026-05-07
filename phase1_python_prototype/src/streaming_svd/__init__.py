"""Streaming/Warm-started rSVD Research Package.

Public API
----------
Algorithms:
    rsvd        -- Cold-start randomized SVD (Halko et al. 2011)
    warm_rsvd   -- Warm-start randomized SVD using previous subspace

Metrics:
    rel_fro_error          -- Relative Frobenius norm reconstruction error
    rel_spec_error_est     -- Estimated relative spectral norm error
    subspace_sin_theta     -- Subspace distance (spectral norm of sin(Theta))
    subspace_sin_theta_fro -- Subspace distance (Frobenius norm of sin(Theta))
"""

__version__ = "0.1.0"

from streaming_svd.algos import (
    rsvd,
    warm_rsvd,
    rel_fro_error,
    rel_spec_error_est,
    subspace_sin_theta,
    subspace_sin_theta_fro,
)

__all__ = [
    "rsvd",
    "warm_rsvd",
    "rel_fro_error",
    "rel_spec_error_est",
    "subspace_sin_theta",
    "subspace_sin_theta_fro",
]
