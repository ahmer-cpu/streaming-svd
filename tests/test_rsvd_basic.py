"""Tests for rSVD algorithms, metrics, and simulation modules."""

import pytest
import torch

from streaming_svd.algos.rsvd import rsvd
from streaming_svd.algos.warm_rsvd import warm_rsvd
from streaming_svd.algos.metrics import (
    rel_fro_error,
    rel_spec_error_est,
    subspace_sin_theta,
    subspace_sin_theta_fro,
)
from streaming_svd.sims.perturbation import make_initial_matrix, perturb_step
from streaming_svd.sims.series import make_random_matrix, sample_independent_series
from streaming_svd.sims.rotating import make_initial_matrix_rotating, rotate_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rank10_matrix():
    """80×60 rank-10 test matrix with known factors."""
    S, U, s, Vt = make_initial_matrix(80, 60, rank=10, seed=0)
    return S, U, s, Vt


# ---------------------------------------------------------------------------
# Cold-start rSVD
# ---------------------------------------------------------------------------

class TestRsvd:
    def test_output_shapes(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U, s, Vt, _ = rsvd(S, k=5, p=5, seed=0)
        assert U.shape == (80, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 60)

    def test_U_orthonormal(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        err = torch.linalg.norm(U.T @ U - torch.eye(5)).item()
        assert err < 1e-5, f"U.T @ U != I, residual {err:.2e}"

    def test_singular_values_positive_and_sorted(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        _, s, _, _ = rsvd(S, k=8, p=5, seed=0)
        assert torch.all(s > 0), "Singular values must be positive"
        assert torch.all(s[:-1] >= s[1:]), "Singular values must be non-increasing"

    def test_reconstruction_error_at_true_rank(self, rank10_matrix):
        """Rank-10 rSVD on a rank-10 matrix should give near-zero error."""
        S, _, _, _ = rank10_matrix
        U, s, Vt, _ = rsvd(S, k=10, p=5, seed=0)
        err = rel_fro_error(S, U, s, Vt)
        assert err < 0.05, f"Error {err:.4f} too large for exact-rank approximation"

    def test_seed_reproducibility(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        _, s1, _, _ = rsvd(S, k=5, p=5, seed=42)
        _, s2, _, _ = rsvd(S, k=5, p=5, seed=42)
        assert torch.allclose(s1, s2)

    def test_power_iterations_do_not_worsen_error(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U0, s0, Vt0, _ = rsvd(S, k=5, p=2, q=0, seed=0)
        U2, s2, Vt2, _ = rsvd(S, k=5, p=2, q=2, seed=0)
        err0 = rel_fro_error(S, U0, s0, Vt0)
        err2 = rel_fro_error(S, U2, s2, Vt2)
        # More power iterations should not substantially increase error
        assert err2 <= err0 + 0.02, f"q=2 error {err2:.4f} worse than q=0 error {err0:.4f}"

    def test_return_stats_false(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        result = rsvd(S, k=5, return_stats=False)
        assert len(result) == 3

    def test_stats_structure(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        _, _, _, stats = rsvd(S, k=5, p=5, seed=0)
        assert 'timings' in stats
        assert 'matmul_counts' in stats
        assert 'params' in stats
        assert stats['timings']['total'] > 0
        assert stats['params']['k'] == 5

    def test_matmul_counts_q0(self, rank10_matrix):
        """q=0: 1 A@X (sketch) + 1 AT@X (projection B=Q.T@A)."""
        S, _, _, _ = rank10_matrix
        _, _, _, stats = rsvd(S, k=5, p=5, q=0, seed=0)
        assert stats['matmul_counts']['A@X'] == 1
        assert stats['matmul_counts']['AT@X'] == 1

    def test_matmul_counts_q2(self, rank10_matrix):
        """q=2: (1 + 2) A@X  +  (2 + 1) AT@X."""
        S, _, _, _ = rank10_matrix
        _, _, _, stats = rsvd(S, k=5, p=5, q=2, seed=0)
        assert stats['matmul_counts']['A@X'] == 3
        assert stats['matmul_counts']['AT@X'] == 3

    def test_invalid_k_raises(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        with pytest.raises(ValueError, match="k="):
            rsvd(S, k=100, p=5)  # k > min(80, 60) = 60

    def test_invalid_p_raises(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        with pytest.raises(ValueError, match="p="):
            rsvd(S, k=5, p=-1)


# ---------------------------------------------------------------------------
# Warm-start rSVD
# ---------------------------------------------------------------------------

class TestWarmRsvd:
    def test_fallback_to_cold_when_no_prev(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U, s, Vt, stats = warm_rsvd(S, None, k=5, p=5, seed=0)
        assert U.shape == (80, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 60)
        # Fallback: warm_start flag not set
        assert stats.get('params', {}).get('warm_start') is not True

    def test_output_shapes_with_prev(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        U, s, Vt, _ = warm_rsvd(S, U_prev, k=5, p=3, seed=1)
        assert U.shape == (80, 5)
        assert s.shape == (5,)
        assert Vt.shape == (5, 60)

    def test_warm_start_flag_in_stats(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        _, _, _, stats = warm_rsvd(S, U_prev, k=5, p=3, seed=1)
        assert stats['params']['warm_start'] is True
        assert stats['params']['r_prev'] == 5

    def test_U_orthonormal(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        U, _, _, _ = warm_rsvd(S, U_prev, k=5, p=3, seed=1)
        err = torch.linalg.norm(U.T @ U - torch.eye(5)).item()
        assert err < 1e-5, f"U.T @ U != I, residual {err:.2e}"

    def test_reconstruction_error_comparable_to_cold(self, rank10_matrix):
        """Warm rSVD error should be comparable to cold rSVD at the same rank."""
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        U_warm, s_warm, Vt_warm, _ = warm_rsvd(S, U_prev, k=5, p=3, seed=1)
        U_cold, s_cold, Vt_cold, _ = rsvd(S, k=5, p=5, seed=1)
        err_warm = rel_fro_error(S, U_warm, s_warm, Vt_warm)
        err_cold = rel_fro_error(S, U_cold, s_cold, Vt_cold)
        # Warm should not be dramatically worse than cold
        assert err_warm <= err_cold * 2.0, (
            f"Warm error {err_warm:.4f} far exceeds cold error {err_cold:.4f}"
        )

    def test_warm_with_perfect_prev_low_error(self):
        """Using the exact true subspace as U_prev should give near-optimal error."""
        S, U_true, s_true, Vt_true = make_initial_matrix(80, 60, rank=10, seed=0)
        U, s, Vt, _ = warm_rsvd(S, U_true, k=10, p=2, seed=0)
        err = rel_fro_error(S, U, s, Vt)
        assert err < 0.01, f"Error {err:.4f} too large with perfect prior"

    def test_matmul_counts_with_prev_q0(self, rank10_matrix):
        """U_prev given, q=0: A@X=2 (Y1,Y2) + AT@X=2 (G,projection)."""
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        _, _, _, stats = warm_rsvd(S, U_prev, k=5, p=3, q=0, seed=1)
        assert stats['matmul_counts']['A@X'] == 2
        assert stats['matmul_counts']['AT@X'] == 2

    def test_matmul_counts_with_prev_q1(self, rank10_matrix):
        """U_prev given, q=1: A@X=3 + AT@X=3."""
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        _, _, _, stats = warm_rsvd(S, U_prev, k=5, p=3, q=1, seed=1)
        assert stats['matmul_counts']['A@X'] == 3
        assert stats['matmul_counts']['AT@X'] == 3

    def test_k_exceeds_sketch_raises(self, rank10_matrix):
        """k > r_prev + p must raise, not silently return wrong shape."""
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=3, p=5, seed=0)  # r_prev = 3
        with pytest.raises(ValueError, match="sketch size"):
            warm_rsvd(S, U_prev, k=10, p=2)  # k=10 > r_prev+p=5

    def test_shape_mismatch_raises(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U_wrong = torch.randn(50, 5)  # row count doesn't match S
        with pytest.raises(ValueError):
            warm_rsvd(S, U_wrong, k=5, p=3)

    def test_invalid_k_raises(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U_prev, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        with pytest.raises(ValueError, match="k="):
            warm_rsvd(S, U_prev, k=200, p=5)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_rel_fro_error_exact_svd_is_zero(self):
        """Exact rank-k SVD of a rank-k matrix → error ≈ 0."""
        S, U, s, Vt = make_initial_matrix(50, 40, rank=8, seed=0)
        err = rel_fro_error(S, U, s, Vt)
        assert err < 1e-5, f"Exact SVD error should be ~0, got {err:.2e}"

    def test_rel_fro_error_in_unit_interval(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U, s, Vt, _ = rsvd(S, k=5, p=5, seed=0)
        err = rel_fro_error(S, U, s, Vt)
        assert 0.0 <= err <= 1.0

    def test_rel_fro_error_higher_rank_lower_error(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U5, s5, Vt5, _ = rsvd(S, k=5, p=5, seed=0)
        U10, s10, Vt10, _ = rsvd(S, k=10, p=5, seed=0)
        err5 = rel_fro_error(S, U5, s5, Vt5)
        err10 = rel_fro_error(S, U10, s10, Vt10)
        assert err10 <= err5, "Higher rank should not increase error"

    def test_subspace_sin_theta_identical_subspace(self):
        U = torch.linalg.qr(torch.randn(50, 5))[0]
        dist = subspace_sin_theta(U, U)
        # float32 round-trip through SVD of U.T @ U leaves ~1e-3 numerical noise
        assert dist < 1e-3, f"Identical subspaces should have distance ~0, got {dist:.2e}"

    def test_subspace_sin_theta_orthogonal_subspaces(self):
        """Orthogonal subspaces → sin-theta distance = 1."""
        Q = torch.linalg.qr(torch.randn(10, 10))[0]
        dist = subspace_sin_theta(Q[:, :5], Q[:, 5:])
        assert abs(dist - 1.0) < 1e-5, f"Orthogonal subspaces distance should be 1, got {dist:.4f}"

    def test_subspace_sin_theta_range(self, rank10_matrix):
        S, _, _, _ = rank10_matrix
        U1, _, _, _ = rsvd(S, k=5, p=5, seed=0)
        U2, _, _, _ = rsvd(S, k=5, p=5, seed=1)
        dist = subspace_sin_theta(U1, U2)
        assert 0.0 <= dist <= 1.0

    def test_subspace_sin_theta_fro_identical(self):
        U = torch.linalg.qr(torch.randn(50, 5))[0]
        dist = subspace_sin_theta_fro(U, U)
        assert dist < 1e-3, f"Identical subspaces should have distance ~0, got {dist:.2e}"

    def test_rel_spec_error_est_exact_subspace(self):
        """U spanning the exact column space → spec error ≈ 0."""
        S, _, _, _ = make_initial_matrix(50, 40, rank=5, seed=0)
        U_exact, _, _ = torch.linalg.svd(S, full_matrices=False)
        err = rel_spec_error_est(S, U_exact[:, :5], n_iter=5)
        assert err < 0.05, f"Spec error for exact subspace should be ~0, got {err:.4f}"


# ---------------------------------------------------------------------------
# Simulation modules
# ---------------------------------------------------------------------------

class TestSims:
    def test_make_initial_matrix_shapes(self):
        S, U, s, Vt = make_initial_matrix(60, 50, rank=8, seed=0)
        assert S.shape == (60, 50)
        assert U.shape == (60, 8)
        assert s.shape == (8,)
        assert Vt.shape == (8, 50)

    def test_make_initial_matrix_exact_reconstruction(self):
        S, U, s, Vt = make_initial_matrix(60, 50, rank=8, seed=0)
        S_approx = U @ torch.diag(s) @ Vt
        err = (torch.linalg.norm(S - S_approx) / torch.linalg.norm(S)).item()
        assert err < 1e-5, f"S should equal U @ diag(s) @ Vt, got err {err:.2e}"

    def test_make_initial_matrix_seed_reproducible(self):
        S1, _, _, _ = make_initial_matrix(30, 20, rank=5, seed=7)
        S2, _, _, _ = make_initial_matrix(30, 20, rank=5, seed=7)
        assert torch.allclose(S1, S2)

    def test_perturb_step_shapes(self):
        S, _, _, _ = make_initial_matrix(50, 40, rank=8, seed=0)
        S_new, E = perturb_step(S, eta=0.1, seed=1)
        assert S_new.shape == S.shape
        assert E.shape == S.shape

    def test_perturb_step_magnitude(self):
        """||E||_F should equal eta * ||S||_F exactly (by construction)."""
        S, _, _, _ = make_initial_matrix(50, 40, rank=8, seed=0)
        eta = 0.1
        _, E = perturb_step(S, eta=eta, seed=1)
        S_norm = torch.linalg.norm(S, 'fro').item()
        E_norm = torch.linalg.norm(E, 'fro').item()
        assert abs(E_norm - eta * S_norm) < 1e-4, (
            f"||E||_F={E_norm:.6f} should be eta*||S||_F={eta * S_norm:.6f}"
        )

    def test_perturb_step_lowrank_noise(self):
        S, _, _, _ = make_initial_matrix(50, 40, rank=8, seed=0)
        S_new, E = perturb_step(S, eta=0.1, noise_rank=3, seed=1)
        assert S_new.shape == S.shape
        assert E.shape == S.shape

    def test_sample_independent_series_count_and_shape(self):
        matrices = list(sample_independent_series(20, 15, T=5, rank=4, seed=0))
        assert len(matrices) == 5
        for S in matrices:
            assert S.shape == (20, 15)

    def test_sample_independent_series_independence(self):
        matrices = list(sample_independent_series(20, 15, T=3, rank=4, seed=0))
        assert not torch.allclose(matrices[0], matrices[1])
        assert not torch.allclose(matrices[1], matrices[2])

    def test_sample_independent_series_seed_reproducible(self):
        m1 = list(sample_independent_series(20, 15, T=3, rank=4, seed=42))
        m2 = list(sample_independent_series(20, 15, T=3, rank=4, seed=42))
        for a, b in zip(m1, m2):
            assert torch.allclose(a, b)

    def test_make_initial_matrix_rotating_shapes(self):
        S, U, s, Vt = make_initial_matrix_rotating(40, 30, rank=6, seed=0)
        assert S.shape == (40, 30)
        assert U.shape == (40, 6)
        assert s.shape == (6,)
        assert Vt.shape == (6, 30)

    def test_rotate_step_shapes(self):
        _, U, s, Vt = make_initial_matrix_rotating(40, 30, rank=6, seed=0)
        S_new, U_new, V_new = rotate_step(U, Vt.T, s, angle=0.05, seed=1)
        assert S_new.shape == (40, 30)
        assert U_new.shape == U.shape
        assert V_new.shape == Vt.T.shape

    def test_rotate_step_U_stays_orthonormal(self):
        _, U, s, Vt = make_initial_matrix_rotating(40, 30, rank=6, seed=0)
        _, U_new, _ = rotate_step(U, Vt.T, s, angle=0.1, seed=1, reorthonormalize=True)
        err = torch.linalg.norm(U_new.T @ U_new - torch.eye(6)).item()
        assert err < 1e-5, f"U_new not orthonormal after rotate_step, err {err:.2e}"
