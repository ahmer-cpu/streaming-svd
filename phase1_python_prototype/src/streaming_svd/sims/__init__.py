"""Simulation utilities for synthetic streaming experiments.

Three data-generation regimes are available:

perturbation  -- Additive noise: S_t = S_{t-1} + E_t  (stable subspace)
rotating      -- Rotating subspace: fixed spectrum, smoothly changing U/V
series        -- Independent random matrices (control / null case)
"""

from .perturbation import make_initial_matrix, perturb_step
from .rotating import make_initial_matrix_rotating, rotate_step
from .series import make_random_matrix, sample_independent_series

__all__ = [
    # Perturbation regime
    'make_initial_matrix',
    'perturb_step',
    # Rotating regime
    'make_initial_matrix_rotating',
    'rotate_step',
    # Independent series (control)
    'make_random_matrix',
    'sample_independent_series',
]
