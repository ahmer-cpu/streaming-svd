"""Control experiment runner: independent random matrices.

Per-timestep seed fix: Each timestep uses fresh randomness for rSVD sketch (seed_t = seed + 10000*t)
while data matrices remain independently seeded via series.py (seed + t).
This ensures fair comparison: identical data independence, but randomized sketch variation.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from streaming_svd.algos.metrics import rel_fro_error, rel_spec_error_est, subspace_sin_theta
from streaming_svd.algos.rsvd import rsvd
from streaming_svd.algos.warm_rsvd import warm_rsvd
from streaming_svd.sims.series import sample_independent_series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def optimal_rank_k_rel_fro_error(A, k):
    """Compute optimal relative Frobenius error of rank-k truncation."""
    svals = torch.linalg.svdvals(A)
    if k >= svals.numel():
        return 0.0

    total_energy = torch.sum(svals**2)
    tail_energy = torch.sum(svals[k:]**2)
    if total_energy <= 0:
        return 0.0

    return float(torch.sqrt(tail_energy / total_energy).item())


def _maybe_compute_optimal(S, k, compute_optimal=True, optimal_max_dim=1200):
    """Compute optimal baseline if enabled and matrix is not too large."""
    if not compute_optimal:
        return np.nan

    m, n = S.shape
    if max(m, n) > optimal_max_dim:
        return np.nan

    return optimal_rank_k_rel_fro_error(S, k)


def _empty_results(params):
    return {
        'cold': {'errors': [], 'times': [], 'matmul_counts': [], 'spec_errors': []},
        'warm': {'errors': [], 'times': [], 'matmul_counts': [], 'spec_errors': []},
        'optimal': {'errors': []},
        'subspace_cold_vs_warm': [],
        'timesteps': [],
        'params': params,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_series_experiment(
    m=1000,
    n=1000,
    k=20,
    T=10,
    rank=None,
    decay=0.1,
    model='lowrank',
    p_cold=10,
    p_warm=5,
    q=0,
    compute_optimal=True,
    optimal_max_dim=1200,
    device='cpu',
    seed=42,
    verbose=True,
):
    """
    Run control experiment: independent random matrices.

    Each timestep S_t is generated independently from scratch, with no correlation
    between consecutive snapshots. This serves as a null/control case for comparison
    with correlated streaming scenarios.

    Parameters
    ----------
    m, n : int
        Matrix dimensions.
    k : int
        Target rank for SVD algorithms.
    T : int
        Number of timesteps.
    rank : int, optional
        Rank for lowrank model. If None, defaults to 2*k.
    decay : float, optional
        Singular value decay rate for lowrank model. Default is 0.1.
    model : str, optional
        Data generation model: 'lowrank' (default) or 'gaussian'.
    p_cold, p_warm : int
        Oversampling parameters for cold/warm rSVD.
    q : int
        Power iteration parameter.
    compute_optimal : bool
        Whether to compute optimal rank-k baseline.
    optimal_max_dim : int
        Maximum dimension to allow for optimal baseline computation.
    device : str
        Device for computation ('cpu' or 'cuda').
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    results : dict
        Dictionary with cold/warm/optimal errors, times, matmul counts, metrics.
    """
    device = torch.device(device)
    
    if rank is None:
        rank = 2 * k

    params = {
        'm': m,
        'n': n,
        'k': k,
        'T': T,
        'rank': rank,
        'decay': decay,
        'model': model,
        'p_cold': p_cold,
        'p_warm': p_warm,
        'q': q,
        'compute_optimal': compute_optimal,
        'optimal_max_dim': optimal_max_dim,
        'device': str(device),
        'seed': seed,
    }
    results = _empty_results(params)

    if verbose:
        print('=' * 70)
        print('Control Experiment: Independent Random Matrices')
        print('=' * 70)
        print(f'Model: {model}')
        if model == 'lowrank':
            print(f'Rank: {rank}, Decay: {decay}')
        print(f'Dimensions: {m} x {n}, Target rank k={k}, T={T} timesteps')
        print(f'Cold-start p={p_cold}, Warm-start p={p_warm}, q={q}')
        print('=' * 70)

    U_warm_prev = None

    with torch.no_grad():
        # Generator yielding independent matrices
        series_gen = sample_independent_series(
            m,
            n,
            T,
            rank=rank,
            decay=decay,
            model=model,
            device=str(device),
            dtype=torch.float32,
            seed=seed,
        )

        for t, S in enumerate(series_gen, start=1):
            results['timesteps'].append(t)

            # Fresh randomness for rSVD sketch each timestep (ensures fair Omega variation)
            seed_t = None if seed is None else seed + 10_000 * t

            t0 = time.perf_counter()
            result_cold = rsvd(
                S,
                k,
                p=p_cold,
                q=q,
                device=device,
                seed=seed_t,
                return_stats=True,
            )
            U_cold, s_cold, Vt_cold, stats_cold = result_cold  # type: ignore[misc]
            time_cold = time.perf_counter() - t0

            t0 = time.perf_counter()
            result_warm = warm_rsvd(
                S,
                U_warm_prev,
                k,
                p=p_warm,
                q=q,
                device=device,
                seed=seed_t,
                return_stats=True,
            )
            U_warm, s_warm, Vt_warm, stats_warm = result_warm  # type: ignore[misc]
            time_warm = time.perf_counter() - t0

            error_cold = rel_fro_error(S, U_cold, s_cold, Vt_cold)
            error_warm = rel_fro_error(S, U_warm, s_warm, Vt_warm)
            spec_cold = rel_spec_error_est(S, U_cold, n_iter=3)
            spec_warm = rel_spec_error_est(S, U_warm, n_iter=3)
            optimal_error = _maybe_compute_optimal(
                S,
                k,
                compute_optimal=compute_optimal,
                optimal_max_dim=optimal_max_dim,
            )

            matmul_cold = stats_cold['matmul_counts']['A@X'] + stats_cold['matmul_counts']['AT@X']
            matmul_warm = stats_warm['matmul_counts']['A@X'] + stats_warm['matmul_counts']['AT@X']

            results['cold']['errors'].append(error_cold)
            results['cold']['times'].append(time_cold)
            results['cold']['matmul_counts'].append(matmul_cold)
            results['cold']['spec_errors'].append(spec_cold)

            results['warm']['errors'].append(error_warm)
            results['warm']['times'].append(time_warm)
            results['warm']['matmul_counts'].append(matmul_warm)
            results['warm']['spec_errors'].append(spec_warm)

            results['optimal']['errors'].append(optimal_error)

            # Subspace distance between cold and warm (NaN for t=1)
            if t > 1:
                results['subspace_cold_vs_warm'].append(subspace_sin_theta(U_cold, U_warm))
            else:
                results['subspace_cold_vs_warm'].append(np.nan)

            U_warm_prev = U_warm

            if verbose:
                print(
                    f't={t:02d} | cold={error_cold:.4f} warm={error_warm:.4f} '
                    f'optimal={optimal_error:.4f}'
                )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_error_vs_optimal(results, output_file):
    """Plot cold, warm, and optimal errors over time."""
    timesteps = np.array(results['timesteps'])

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, results['cold']['errors'], 'o-', linewidth=2.3, markersize=7, label='Cold-start rSVD')
    plt.plot(timesteps, results['warm']['errors'], 's-', linewidth=2.3, markersize=7, label='Warm-start rSVD')

    optimal_errors = np.array(results['optimal']['errors'])
    if not np.all(np.isnan(optimal_errors)):
        plt.plot(
            timesteps,
            optimal_errors,
            '^-',
            linewidth=2.3,
            markersize=7,
            color='black',
            label='Optimal rank-k (truncated SVD)',
        )

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.title('Independent Random Matrices: Cold vs Warm rSVD Error', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_timing(results, output_file):
    """Plot cold vs warm timing."""
    timesteps = np.array(results['timesteps'])

    plt.figure(figsize=(10, 6))
    plt.plot(
        timesteps,
        np.array(results['cold']['times']) * 1000,
        'o-',
        linewidth=2.2,
        markersize=7,
        label='Cold-start rSVD',
    )
    plt.plot(
        timesteps,
        np.array(results['warm']['times']) * 1000,
        's-',
        linewidth=2.2,
        markersize=7,
        label='Warm-start rSVD',
    )

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Independent Random Matrices: Computational Time', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_plots(results, output_dir='results/figures'):
    """Generate and save plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    error_file = output_path / 'series_error_comparison.png'
    _plot_error_vs_optimal(results, error_file)
    print(f'✓ Saved error plot to: {error_file}')

    timing_file = output_path / 'series_timing_comparison.png'
    _plot_timing(results, timing_file)
    print(f'✓ Saved timing plot to: {timing_file}')

    print(f'All plots saved to: {output_path.absolute()}')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_csv(results, csv_path):
    """Save results to CSV."""
    try:
        import pandas as pd  # type: ignore

        df_data = []
        T = len(results['timesteps'])
        for i in range(T):
            df_data.append(
                {
                    'timestep': results['timesteps'][i],
                    'cold_error': results['cold']['errors'][i],
                    'warm_error': results['warm']['errors'][i],
                    'optimal_error': results['optimal']['errors'][i],
                    'cold_spec_error': results['cold']['spec_errors'][i],
                    'warm_spec_error': results['warm']['spec_errors'][i],
                    'cold_time': results['cold']['times'][i],
                    'warm_time': results['warm']['times'][i],
                    'cold_matmuls': results['cold']['matmul_counts'][i],
                    'warm_matmuls': results['warm']['matmul_counts'][i],
                    'subspace_cold_vs_warm': results['subspace_cold_vs_warm'][i],
                }
            )

        df = pd.DataFrame(df_data)
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f'Results saved to {csv_path}')
    except ImportError:
        import csv

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'timestep',
                    'cold_error',
                    'warm_error',
                    'optimal_error',
                    'cold_spec_error',
                    'warm_spec_error',
                    'cold_time',
                    'warm_time',
                    'cold_matmuls',
                    'warm_matmuls',
                    'subspace_cold_vs_warm',
                ],
            )
            writer.writeheader()
            T = len(results['timesteps'])
            for i in range(T):
                writer.writerow(
                    {
                        'timestep': results['timesteps'][i],
                        'cold_error': results['cold']['errors'][i],
                        'warm_error': results['warm']['errors'][i],
                        'optimal_error': results['optimal']['errors'][i],
                        'cold_spec_error': results['cold']['spec_errors'][i],
                        'warm_spec_error': results['warm']['spec_errors'][i],
                        'cold_time': results['cold']['times'][i],
                        'warm_time': results['warm']['times'][i],
                        'cold_matmuls': results['cold']['matmul_counts'][i],
                        'warm_matmuls': results['warm']['matmul_counts'][i],
                        'subspace_cold_vs_warm': results['subspace_cold_vs_warm'][i],
                    }
                )
        print(f'Results saved to {csv_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for control experiment."""
    parser = argparse.ArgumentParser(
        description='Run control experiment with independent random matrices',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--m', type=int, default=1000, help='Number of rows')
    parser.add_argument('--n', type=int, default=1000, help='Number of columns')
    parser.add_argument('--k', type=int, default=20, help='Target rank')
    parser.add_argument('--T', type=int, default=10, help='Number of time steps')
    parser.add_argument('--model', type=str, default='lowrank', choices=['lowrank', 'gaussian'],
                        help='Data generation model')
    parser.add_argument('--rank', type=int, default=None, help='Rank for lowrank model (default: 2*k)')
    parser.add_argument('--decay', type=float, default=0.1, help='Singular value decay for lowrank model')
    parser.add_argument('--p-cold', type=int, default=10, help='Oversampling for cold-start')
    parser.add_argument('--p-warm', type=int, default=5, help='Oversampling for warm-start')
    parser.add_argument('--q', type=int, default=0, help='Power iterations')
    parser.add_argument(
        '--compute-optimal',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Compute optimal rank-k baseline curve',
    )
    parser.add_argument(
        '--optimal-max-dim',
        type=int,
        default=1200,
        help='Skip optimal baseline if max(m,n) exceeds this value',
    )
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--csv', type=str, default=None, help='CSV output path')
    parser.add_argument('--output-dir', type=str, default='results/figures', help='Output directory for figures')

    args = parser.parse_args()
    verbose = not args.quiet

    # If rank not specified, use 2*k
    rank = args.rank if args.rank is not None else 2 * args.k

    results = run_series_experiment(
        m=args.m,
        n=args.n,
        k=args.k,
        T=args.T,
        rank=rank,
        decay=args.decay,
        model=args.model,
        p_cold=args.p_cold,
        p_warm=args.p_warm,
        q=args.q,
        compute_optimal=args.compute_optimal,
        optimal_max_dim=args.optimal_max_dim,
        device=args.device,
        seed=args.seed,
        verbose=verbose,
    )

    generate_plots(results, output_dir=args.output_dir)

    if args.csv:
        _save_csv(results, args.csv)


if __name__ == '__main__':
    main()
