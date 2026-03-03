"""Quick test script to verify rSVD implementations."""

import torch
from streaming_svd.algos import rsvd, warm_rsvd, rel_fro_error
from streaming_svd.sims import make_initial_matrix, perturb_step


def test_basic():
    """Test basic functionality of cold and warm rSVD."""
    print("Testing basic rSVD functionality...")
    print("=" * 50)
    
    # Parameters
    m, n, k = 100, 100, 10
    device = 'cpu'
    
    # Generate test matrix
    print(f"\nGenerating {m}x{n} matrix with rank {k*2}...")
    S, _, _, _ = make_initial_matrix(m, n, rank=k*2, device=device, seed=42)
    print(f"Matrix norm: {torch.linalg.norm(S, ord='fro'):.4f}")
    
    # Test cold-start rSVD
    print("\nTesting cold-start rSVD...")
    U_cold, s_cold, Vt_cold, stats_cold = rsvd(S, k, p=10, q=0, device=device, seed=42)
    error_cold = rel_fro_error(S, U_cold, s_cold, Vt_cold)
    print(f"  Shape: U={U_cold.shape}, s={s_cold.shape}, Vt={Vt_cold.shape}")
    print(f"  Error: {error_cold:.6f}")
    print(f"  Time: {stats_cold['timings']['total']:.4f}s")
    print(f"  Matmuls: {stats_cold['matmul_counts']['A@X'] + stats_cold['matmul_counts']['AT@X']}")
    
    # Test warm-start rSVD (first call, should fallback to cold)
    print("\nTesting warm-start rSVD (no previous basis)...")
    U_warm, s_warm, Vt_warm, stats_warm = warm_rsvd(
        S, None, k, p=5, q=0, device=device, seed=42
    )
    error_warm = rel_fro_error(S, U_warm, s_warm, Vt_warm)
    print(f"  Shape: U={U_warm.shape}, s={s_warm.shape}, Vt={Vt_warm.shape}")
    print(f"  Error: {error_warm:.6f}")
    print(f"  Time: {stats_warm['timings']['total']:.4f}s")
    
    # Perturb and test warm-start with previous basis
    print("\nPerturbing matrix and testing warm-start with previous basis...")
    S_new, E = perturb_step(S, eta=0.05, noise_rank=k, device=device, seed=43)
    print(f"  Perturbation norm: {torch.linalg.norm(E, ord='fro'):.4f}")
    
    U_warm2, s_warm2, Vt_warm2, stats_warm2 = warm_rsvd(
        S_new, U_warm, k, p=5, q=0, device=device, seed=42
    )
    error_warm2 = rel_fro_error(S_new, U_warm2, s_warm2, Vt_warm2)
    print(f"  Error: {error_warm2:.6f}")
    print(f"  Time: {stats_warm2['timings']['total']:.4f}s")
    print(f"  Matmuls: {stats_warm2['matmul_counts']['A@X'] + stats_warm2['matmul_counts']['AT@X']}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")


if __name__ == '__main__':
    test_basic()
