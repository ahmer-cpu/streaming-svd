#include "metrics.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Frobenius reconstruction error (cheap identity trick)
// ---------------------------------------------------------------------------
float fro_error(const MatF& A, const VecF& s) {
    // ||A - U diag(s) Vt||_F^2 = ||A||_F^2 - ||s||^2
    // This identity holds for rSVD output because U.T @ A = diag(s) @ Vt exactly
    // (see derivation in metrics.hpp).
    //
    // IMPORTANT: accumulate in double to avoid catastrophic cancellation.
    // ||A||_F^2 for a (250k x 100) float32 matrix can be ~1e10; the residual
    // may be <1% of that — tiny relative to the magnitude, fatal in float32.
    // Accumulate column-by-column to avoid a full (250k×100) double temporary
    double norm_A_sq = 0.0;
    for (Idx j = 0; j < A.cols(); ++j)
        norm_A_sq += A.col(j).cast<double>().squaredNorm();

    double norm_s_sq = 0.0;
    for (Idx i = 0; i < s.size(); ++i) {
        double v = static_cast<double>(s(i));
        norm_s_sq += v * v;
    }

    double err_sq = std::max(0.0, norm_A_sq - norm_s_sq);
    return static_cast<float>(std::sqrt(err_sq) / std::sqrt(norm_A_sq));
}

// ---------------------------------------------------------------------------
// Optimal rank-k Frobenius error via Gram matrix
// ---------------------------------------------------------------------------
float optimal_fro_error(const MatF& A, int k) {
    // G = A.T @ A  (n x n, symmetric PSD)
    MatF G = A.transpose() * A;

    // Eigenvalues in ascending order
    Eigen::SelfAdjointEigenSolver<MatF> eigs(G, Eigen::EigenvaluesOnly);
    if (eigs.info() != Eigen::Success)
        throw std::runtime_error("optimal_fro_error: eigendecomposition failed");

    VecF lam = eigs.eigenvalues();   // ascending, size n
    Idx  n   = lam.size();

    // sum of ALL eigenvalues = ||A||_F^2
    double total = lam.cast<double>().sum();
    if (total <= 0.0) return 0.0f;

    // sum of the k SMALLEST eigenvalues (tail energy after keeping k largest)
    // The k largest correspond to lam[n-k .. n-1] in ascending order.
    // Residual = sum of lam[0 .. n-k-1].
    double residual = 0.0;
    for (Idx i = 0; i < n - k; ++i)
        residual += static_cast<double>(lam(i));

    return static_cast<float>(std::sqrt(std::max(0.0, residual) / total));
}

// ---------------------------------------------------------------------------
// Spectral norm error via power iteration
// ---------------------------------------------------------------------------
float spec_error(const MatF& A, const MatF& U, int n_iter) {
    // Estimate ||(I - U U.T) A||_2 / ||A||_2
    // Power iteration on M = (I - U U.T) A A.T (I - U U.T)
    // v <- M v / ||M v||

    Idx m = A.rows();
    Idx n = A.cols();

    // Start with a random unit vector (use a fixed seed for reproducibility)
    VecF v = VecF::Ones(m);
    v /= v.norm();

    for (int i = 0; i < n_iter; ++i) {
        // w = A.T @ v  (n)
        VecF w = A.transpose() * v;
        // u = A @ w    (m)
        VecF u = A * w;
        // Project out the U-subspace: u = (I - U U.T) u
        u -= U * (U.transpose() * u);
        if (u.norm() < 1e-12f) break;
        v = u / u.norm();
    }

    // Rayleigh quotient for the residual operator.
    // After n_iter power iterations, v ≈ top eigenvector of
    //   M = (I - U Uᵀ) A Aᵀ (I - U Uᵀ)
    // with eigenvalue λ₁ = ||(I - U Uᵀ) A||₂².
    // One more application: u = M v ≈ λ₁ v,  so  vᵀu = λ₁.
    // We want ||(I - U Uᵀ) A||₂ = sqrt(λ₁), hence use vᵀu, not ||u||².
    VecF w = A.transpose() * v;
    VecF u = A * w;
    u -= U * (U.transpose() * u);
    // Rayleigh quotient: vᵀ (M v) = λ₁  (v is unit, M v = u)
    double lambda1 = v.cast<double>().dot(u.cast<double>());
    lambda1 = std::max(0.0, lambda1);   // guard against tiny negatives

    // Estimate ||A||₂ via a separate power iteration on A Aᵀ
    VecF v2 = VecF::Ones(m);
    v2 /= v2.norm();
    for (int i = 0; i < n_iter; ++i) {
        VecF w2 = A.transpose() * v2;
        VecF u2 = A * w2;
        if (u2.norm() < 1e-12f) break;
        v2 = u2 / u2.norm();
    }
    VecF w2 = A.transpose() * v2;
    // Rayleigh quotient of A Aᵀ at v2 ≈ σ₁² = ||A||₂²
    double norm_A_sq = (A * w2).cast<double>().squaredNorm()
                     / std::max(1e-24, w2.cast<double>().squaredNorm());
    double norm_A = std::sqrt(norm_A_sq);
    if (norm_A < 1e-12) return 0.0f;

    // spec_error = sqrt(λ₁) / ||A||₂ = ||(I - U Uᵀ) A||₂ / ||A||₂
    return static_cast<float>(std::sqrt(lambda1) / norm_A);
}

// ---------------------------------------------------------------------------
// Principal angles between subspaces
// ---------------------------------------------------------------------------
void subspace_sin_theta(const MatF& U1, const MatF& U2,
                        float& sin_theta_spec, float& sin_theta_fro) {
    // M = U1.T @ U2   (r1 x r2)
    MatF M = U1.transpose() * U2;

    // Singular values of M → cosines of principal angles
    Eigen::BDCSVD<MatF, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(M);
    VecF sigma = svd.singularValues();   // descending, size = min(r1, r2)

    // Clamp sigma to [-1, 1] to guard against numerical noise
    sigma = sigma.cwiseMax(-1.0f).cwiseMin(1.0f);

    // sin_theta_i = sqrt(1 - sigma_i^2)
    VecF sin_sq = VecF::Ones(sigma.size()) - sigma.cwiseAbs2();
    sin_sq = sin_sq.cwiseMax(0.0f);

    // Spectral: max sin_theta  = sqrt(1 - sigma_min^2)
    // (sigma is descending, so sigma_min = last element)
    sin_theta_spec = std::sqrt(sin_sq(sin_sq.size() - 1));

    // Frobenius: ||sin_theta||_F = sqrt(sum sin_theta_i^2)
    sin_theta_fro = std::sqrt(sin_sq.sum());
}
