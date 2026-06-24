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

/// Compression-oriented reconstruction metrics.
///
/// All computed in a single column-by-column pass over A to avoid
/// materialising the full m×n reconstruction more than once.
/// Leverage-score fields require sigma_kp1_upper > 0 (pass the return value
/// of spectral_norm_residual); otherwise they are set to 0.
struct CompressionMetrics {
    float max_elem_error;   ///< max_{i,j} |A_{ij} - recon_{ij}|        (exact, Level 3)
    float psnr;             ///< 10·log10(peak² / MSE), dB  (peak = max|A|)
    float pctl_99;          ///< 99th percentile of |A - recon| element-wise
    float pctl_999;         ///< 99.9th percentile of |A - recon| element-wise

    // --- Error-bound hierarchy ---
    float min_leverage_U;   ///< min_i ||(U_k)_{i,:}||²  — worst-covered spatial point
    float min_leverage_V;   ///< min_j ||(V_k)_{j,:}||²  — worst-covered z-level
    float leverage_bound;   ///< sigma_kp1_upper·sqrt((1-min_lev_U)·(1-min_lev_V))
                            ///<   upper bound on max_elem_error  (Level 2)
};

/// Compute all compression-quality metrics in one pass.
///
/// @param A               Input matrix (m × n).
/// @param U               Left singular vectors (m × k).
/// @param s               Singular values (k).
/// @param Vt              Right singular vectors (k × n).
/// @param sigma_kp1_upper Upper bound on sigma_{k+1}(A); pass the return value
///                        of spectral_norm_residual(A, U).  Set to 0 to skip
///                        leverage_bound computation.
CompressionMetrics compression_metrics(const MatF& A, const MatF& U, const VecF& s,
                                       const MatF& Vt, float sigma_kp1_upper = 0.0f);

/// Absolute spectral norm of the residual subspace: ||(I - U U^T) A||_2.
///
/// This is an upper bound on sigma_{k+1}(A) when U spans an approximation
/// to the top-k left subspace.  Use it as sigma_kp1_upper in
/// compression_metrics() to obtain the leverage-score bound.
///
/// Cheaper than spec_error(): runs only the residual power iteration;
/// does NOT estimate ||A||_2, so no second power loop.
///
/// @param A      Input matrix (m × n).
/// @param U      Orthonormal basis (m × k).
/// @param n_iter Number of power iterations (default 3).
float spectral_norm_residual(const MatF& A, const MatF& U, int n_iter = 3);

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
