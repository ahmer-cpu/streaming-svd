#pragma once

#include "matrix_types.hpp"

/// Relative Frobenius reconstruction error.
///
/// Uses the identity:
///   ||A - U diag(s) Vt||_F^2 = ||A||_F^2 - sum(s_i^2)
/// which avoids materialising the full m×n reconstruction.
///
/// @returns  sqrt( max(0, ||A||_F^2 - ||s||^2) ) / ||A||_F
float fro_error(const MatF& A, const VecF& s);

/// Optimal (truncated SVD) relative Frobenius error for rank-k approximation.
///
/// Avoids full SVD of the tall A by working with the (n×n) Gram matrix A.T A.
///   G = A.T @ A
///   eigenvalues lambda_i (ascending)
///   opt_err^2 = sum(lambda_0 .. lambda_{n-k-1}) / sum(all lambda)
///
/// @returns  sqrt( sum of k smallest eigenvalues / sum of all eigenvalues )
float optimal_fro_error(const MatF& A, int k);

/// Spectral norm error of the subspace approximation.
///
/// Estimates ||(I - U U.T) A||_2 / ||A||_2 via power iteration.
/// Uses n_iter=3 power steps (same as Python).
///
/// @param A      Input matrix (m × n).
/// @param U      Orthonormal basis (m × k).
/// @param n_iter Number of power iterations (default 3).
float spec_error(const MatF& A, const MatF& U, int n_iter = 3);

/// Principal angles between two subspaces.
///
/// Given orthonormal U1 (m × r1) and U2 (m × r2):
///   M = U1.T @ U2
///   sigma = singular values of M
///   sin_theta_i = sqrt(1 - sigma_i^2)
///
/// @param sin_theta_spec  Output: max(sin_theta_i)  [spectral norm]
/// @param sin_theta_fro   Output: ||sin_theta||_F   [Frobenius norm]
void subspace_sin_theta(const MatF& U1, const MatF& U2,
                        float& sin_theta_spec, float& sin_theta_fro);
