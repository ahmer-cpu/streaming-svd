"""
Synthetic streaming SVD experiment runner.

Compares cold-start rSVD vs warm-start rSVD on a synthetic streaming sequence.
"""

import argparse
import time
import torch
import numpy as np

from streaming_svd.algos.rsvd import rsvd
from streaming_svd.algos.warm_rsvd import warm_rsvd
from streaming_svd.algos.metrics import rel_fro_error, subspace_sin_theta
from streaming_svd.sims.perturbation import make_initial_matrix, perturb_step


def run_experiment(
    m=1000,
    n=1000,
    k=20,
    T=10,
    eta=0.05,
    p_cold=10,
    p_warm=5,
    q=0,
    device='cpu',
    seed=42,
    verbose=True,
):
    """
    Run synthetic streaming SVD experiment.
    
    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    k : int
        Target rank.
    T : int
        Number of time steps.
    eta : float
        Perturbation magnitude (relative to ||S||_F).
    p_cold : int
        Oversampling for cold-start rSVD.
    p_warm : int
        Oversampling for warm-start rSVD.
    q : int
        Number of power iterations.
    device : str
        Device for computation ('cpu' or 'cuda').
    seed : int
        Random seed.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    results : dict
        Dictionary containing errors, times, and matmul counts.
    """
    device = torch.device(device)
    device_str = str(device)
    
    if verbose:
        print("=" * 70)
        print("Synthetic Streaming SVD Experiment")
        print("=" * 70)
        print(f"Matrix size: {m} x {n}")
        print(f"Target rank k: {k}")
        print(f"Time steps T: {T}")
        print(f"Perturbation eta: {eta}")
        print(f"Cold-start: p={p_cold}, q={q}")
        print(f"Warm-start: p={p_warm}, q={q}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print("=" * 70)
    
    # Initialize results storage
    results = {
        'cold': {
            'errors': [],
            'times': [],
            'matmul_counts': [],
        },
        'warm': {
            'errors': [],
            'times': [],
            'matmul_counts': [],
        },
        'params': {
            'm': m, 'n': n, 'k': k, 'T': T, 'eta': eta,
            'p_cold': p_cold, 'p_warm': p_warm, 'q': q,
            'device': str(device), 'seed': seed,
        },
    }
    
    # Generate initial matrix
    if verbose:
        print("\nGenerating initial matrix S_1...")
    
    S, _, _, _ = make_initial_matrix(
        m, n, rank=k*2, decay=0.1,
        device=device_str, dtype=torch.float32, seed=seed
    )
    
    if verbose:
        print(f"Initial matrix norm: {torch.linalg.norm(S, ord='fro'):.4f}")
    
    # Initialize previous basis for warm-start
    U_warm_prev = None
    
    # Stream processing
    with torch.no_grad():
        for t in range(1, T + 1):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Time step {t}/{T}")
                print(f"{'='*70}")
            
            # Apply perturbation (except for first step)
            if t > 1:
                S, E = perturb_step(
                    S, eta, noise_rank=k,
                    device=device, seed=seed + t
                )
                if verbose:
                    E_norm = torch.linalg.norm(E, ord='fro')
                    S_norm = torch.linalg.norm(S, ord='fro')
                    print(f"Perturbation norm: {E_norm:.4f} ({E_norm/S_norm:.4f} relative)")
            
            S_norm = torch.linalg.norm(S, ord='fro').item()
            if verbose:
                print(f"Current matrix norm: {S_norm:.4f}")
            
            # Cold-start rSVD
            if verbose:
                print("\n  Running cold-start rSVD...")
            
            t0 = time.perf_counter()
            result_cold = rsvd(  # type: ignore[return-value]
                S, k, p=p_cold, q=q,
                device=device, seed=seed,
                return_stats=True
            )
            U_cold, s_cold, Vt_cold, stats_cold = result_cold  # type: ignore[misc]
            time_cold = time.perf_counter() - t0
            
            error_cold = rel_fro_error(S, U_cold, s_cold, Vt_cold)
            matmul_cold = stats_cold['matmul_counts']['A@X'] + stats_cold['matmul_counts']['AT@X']
            
            results['cold']['errors'].append(error_cold)
            results['cold']['times'].append(time_cold)
            results['cold']['matmul_counts'].append(matmul_cold)
            
            if verbose:
                print(f"    Time: {time_cold:.4f}s")
                print(f"    Error: {error_cold:.6f}")
                print(f"    Matmuls: {matmul_cold}")
            
            # Warm-start rSVD
            if verbose:
                print("\n  Running warm-start rSVD...")
            
            t0 = time.perf_counter()
            result_warm = warm_rsvd(  # type: ignore[return-value]
                S, U_warm_prev, k, p=p_warm, q=q,
                device=device, seed=seed,
                return_stats=True
            )
            U_warm, s_warm, Vt_warm, stats_warm = result_warm  # type: ignore[misc]
            time_warm = time.perf_counter() - t0
            
            error_warm = rel_fro_error(S, U_warm, s_warm, Vt_warm)
            matmul_warm = stats_warm['matmul_counts']['A@X'] + stats_warm['matmul_counts']['AT@X']
            
            results['warm']['errors'].append(error_warm)
            results['warm']['times'].append(time_warm)
            results['warm']['matmul_counts'].append(matmul_warm)
            
            if verbose:
                print(f"    Time: {time_warm:.4f}s")
                print(f"    Error: {error_warm:.6f}")
                print(f"    Matmuls: {matmul_warm}")
            
            # Update warm-start basis for next iteration
            U_warm_prev = U_warm
            
            # Compute subspace difference (if not first step)
            if t > 1:
                subspace_dist = subspace_sin_theta(U_cold, U_warm)
                if verbose:
                    print(f"\n  Subspace distance: {subspace_dist:.6f}")
    
    # Summary statistics
    if verbose:
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        
        avg_time_cold = np.mean(results['cold']['times'])
        avg_time_warm = np.mean(results['warm']['times'])
        avg_error_cold = np.mean(results['cold']['errors'])
        avg_error_warm = np.mean(results['warm']['errors'])
        avg_matmul_cold = np.mean(results['cold']['matmul_counts'])
        avg_matmul_warm = np.mean(results['warm']['matmul_counts'])
        
        speedup = avg_time_cold / avg_time_warm if avg_time_warm > 0 else 0
        matmul_reduction = (avg_matmul_cold - avg_matmul_warm) / avg_matmul_cold * 100
        
        print(f"\nCold-start rSVD:")
        print(f"  Average time:    {avg_time_cold:.4f}s")
        print(f"  Average error:   {avg_error_cold:.6f}")
        print(f"  Average matmuls: {avg_matmul_cold:.1f}")
        
        print(f"\nWarm-start rSVD:")
        print(f"  Average time:    {avg_time_warm:.4f}s")
        print(f"  Average error:   {avg_error_warm:.6f}")
        print(f"  Average matmuls: {avg_matmul_warm:.1f}")
        
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Matmul reduction: {matmul_reduction:.1f}%")
        print(f"Error ratio (warm/cold): {avg_error_warm/avg_error_cold:.3f}")
        print("=" * 70)
    
    return results


def main():
    """Command-line interface for synthetic experiment."""
    parser = argparse.ArgumentParser(
        description="Run synthetic streaming SVD experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--m', type=int, default=1000, help='Number of rows')
    parser.add_argument('--n', type=int, default=1000, help='Number of columns')
    parser.add_argument('--k', type=int, default=20, help='Target rank')
    parser.add_argument('--T', type=int, default=10, help='Number of time steps')
    parser.add_argument('--eta', type=float, default=0.05, help='Perturbation magnitude')
    parser.add_argument('--p-cold', type=int, default=10, help='Oversampling for cold-start')
    parser.add_argument('--p-warm', type=int, default=5, help='Oversampling for warm-start')
    parser.add_argument('--q', type=int, default=0, help='Power iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    parser.add_argument('--csv', type=str, default=None, help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        m=args.m,
        n=args.n,
        k=args.k,
        T=args.T,
        eta=args.eta,
        p_cold=args.p_cold,
        p_warm=args.p_warm,
        q=args.q,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    # Save to CSV if requested
    if args.csv:
        import pandas as pd  # type: ignore
        
        df_data = []
        for t in range(args.T):
            df_data.append({
                'timestep': t + 1,
                'cold_error': results['cold']['errors'][t],
                'cold_time': results['cold']['times'][t],
                'cold_matmuls': results['cold']['matmul_counts'][t],
                'warm_error': results['warm']['errors'][t],
                'warm_time': results['warm']['times'][t],
                'warm_matmuls': results['warm']['matmul_counts'][t],
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}")


if __name__ == '__main__':
    main()
