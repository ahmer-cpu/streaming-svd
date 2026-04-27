"""Synthetic streaming SVD experiment runner for additive and rotating regimes."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from streaming_svd.algos.metrics import rel_fro_error, rel_spec_error_est, subspace_sin_theta
from streaming_svd.algos.rsvd import rsvd
from streaming_svd.algos.warm_rsvd import warm_rsvd
from streaming_svd.sims.perturbation import make_initial_matrix, perturb_step
from streaming_svd.sims.rotating import make_initial_matrix_rotating, rotate_step


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


def _empty_results(mode, params):
    return {
        'mode': mode,
        'cold': {'errors': [], 'times': [], 'matmul_counts': [], 'spec_errors': []},
        'warm': {'errors': [], 'times': [], 'matmul_counts': [], 'spec_errors': []},
        'optimal': {'errors': []},
        'subspace_cold_vs_warm': [],
        'subspace_warm_vs_true': [],
        'subspace_cold_vs_true': [],
        'timesteps': [],
        'params': params,
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_experiment_additive(
    m=1000,
    n=1000,
    k=20,
    T=10,
    eta=0.05,
    p_cold=10,
    p_warm=5,
    q=0,
    compute_optimal=True,
    optimal_max_dim=1200,
    device='cpu',
    seed=42,
    verbose=True,
):
    """Run additive-noise (rank-inflation) streaming experiment."""
    device = torch.device(device)
    params = {
        'm': m,
        'n': n,
        'k': k,
        'T': T,
        'eta': eta,
        'p_cold': p_cold,
        'p_warm': p_warm,
        'q': q,
        'compute_optimal': compute_optimal,
        'optimal_max_dim': optimal_max_dim,
        'device': str(device),
        'seed': seed,
    }
    results = _empty_results('additive', params)

    if verbose:
        print('=' * 70)
        print('Experiment: Additive noise (rank inflation) streaming')
        print('=' * 70)

    S, _, _, _ = make_initial_matrix(
        m=m,
        n=n,
        rank=min(k * 2, min(m, n)),
        decay=0.1,
        device=str(device),
        dtype=torch.float32,
        seed=seed,
    )

    U_warm_prev = None

    with torch.no_grad():
        for t in range(1, T + 1):
            results['timesteps'].append(t)

            if t > 1:
                S, _ = perturb_step(
                    S,
                    eta=eta,
                    noise_rank=k,
                    device=device,
                    dtype=torch.float32,
                    seed=seed + t,
                )

            t0 = time.perf_counter()
            result_cold = rsvd(
                S,
                k,
                p=p_cold,
                q=q,
                device=device,
                seed=seed,
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
                seed=seed,
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

            if t > 1:
                results['subspace_cold_vs_warm'].append(subspace_sin_theta(U_cold, U_warm))
            else:
                results['subspace_cold_vs_warm'].append(np.nan)

            results['subspace_warm_vs_true'].append(np.nan)
            results['subspace_cold_vs_true'].append(np.nan)
            U_warm_prev = U_warm

            if verbose:
                print(
                    f't={t:02d} | cold={error_cold:.4f} warm={error_warm:.4f} '
                    f'optimal={optimal_error:.4f}'
                )

    return results


def run_experiment_rotating(
    m=1000,
    n=1000,
    k=20,
    T=10,
    angle=0.03,
    p_cold=10,
    p_warm=5,
    q=0,
    compute_optimal=True,
    optimal_max_dim=1200,
    device='cpu',
    seed=42,
    verbose=True,
):
    """Run rotating-subspace (fixed spectrum) streaming experiment."""
    device = torch.device(device)
    true_rank = min(k * 2, min(m, n))
    params = {
        'm': m,
        'n': n,
        'k': k,
        'T': T,
        'angle': angle,
        'p_cold': p_cold,
        'p_warm': p_warm,
        'q': q,
        'compute_optimal': compute_optimal,
        'optimal_max_dim': optimal_max_dim,
        'device': str(device),
        'seed': seed,
    }
    results = _empty_results('rotating', params)

    if verbose:
        print('=' * 70)
        print('Experiment: Rotating subspace (fixed spectrum) streaming')
        print('=' * 70)

    S, U_true, s_true, Vt_true = make_initial_matrix_rotating(
        m=m,
        n=n,
        rank=true_rank,
        decay=0.1,
        seed=seed,
        device=str(device),
        dtype=torch.float32,
    )
    V_true = Vt_true.T

    U_warm_prev = None

    with torch.no_grad():
        for t in range(1, T + 1):
            results['timesteps'].append(t)

            if t > 1:
                S, U_true, V_true = rotate_step(
                    U_true,
                    V_true,
                    s_true,
                    angle=angle,
                    device=device,
                    dtype=torch.float32,
                    seed=seed + t,
                    rotate_both=True,
                    reorthonormalize=True,
                )

            t0 = time.perf_counter()
            result_cold = rsvd(
                S,
                k,
                p=p_cold,
                q=q,
                device=device,
                seed=seed,
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
                seed=seed,
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
            results['subspace_cold_vs_warm'].append(subspace_sin_theta(U_cold, U_warm) if t > 1 else np.nan)
            results['subspace_cold_vs_true'].append(subspace_sin_theta(U_cold, U_true[:, :k]))
            results['subspace_warm_vs_true'].append(subspace_sin_theta(U_warm, U_true[:, :k]))

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

def _plot_error_vs_optimal(results, title, output_file):
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
    plt.title(title, fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_rotating_subspace(results, output_file):
    timesteps = np.array(results['timesteps'])

    plt.figure(figsize=(10, 6))
    plt.plot(
        timesteps,
        results['subspace_cold_vs_true'],
        'o-',
        linewidth=2.2,
        markersize=6,
        label='Cold vs true subspace',
    )
    plt.plot(
        timesteps,
        results['subspace_warm_vs_true'],
        's-',
        linewidth=2.2,
        markersize=6,
        label='Warm vs true subspace',
    )

    cw = np.array(results['subspace_cold_vs_warm'])
    if np.any(~np.isnan(cw)):
        valid = ~np.isnan(cw)
        plt.plot(timesteps[valid], cw[valid], '^-', linewidth=2.0, markersize=6, label='Cold vs warm subspace')

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('sin(theta)', fontsize=12)
    plt.title('Rotating subspace (fixed spectrum) streaming: Subspace distance', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_plots(additive_results=None, rotating_results=None, output_dir='results/figures'):
    """Generate regime-specific figures and save them with distinct filenames."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if additive_results is not None:
        additive_file = output_path / 'error_additive_vs_optimal.png'
        _plot_error_vs_optimal(
            additive_results,
            'Additive noise (rank inflation) streaming',
            additive_file,
        )
        print(f'✓ Saved additive error figure to: {additive_file}')

    if rotating_results is not None:
        rotating_file = output_path / 'error_rotating_vs_optimal.png'
        _plot_error_vs_optimal(
            rotating_results,
            'Rotating subspace (fixed spectrum) streaming',
            rotating_file,
        )
        print(f'✓ Saved rotating error figure to: {rotating_file}')

        subspace_file = output_path / 'subspace_rotating.png'
        _plot_rotating_subspace(rotating_results, subspace_file)
        print(f'✓ Saved rotating subspace figure to: {subspace_file}')

    print(f'All plots saved to: {output_path.absolute()}')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_csv(results, csv_path):
    import pandas as pd  # type: ignore

    df_data = []
    T = len(results['timesteps'])
    for i in range(T):
        df_data.append(
            {
                'mode': results['mode'],
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
                'subspace_cold_vs_true': results['subspace_cold_vs_true'][i],
                'subspace_warm_vs_true': results['subspace_warm_vs_true'][i],
            }
        )

    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)
    print(f'Results saved to {csv_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for synthetic experiments."""
    parser = argparse.ArgumentParser(
        description='Run synthetic streaming SVD experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--mode', type=str, default='both', choices=['additive', 'rotating', 'both'])
    parser.add_argument('--m', type=int, default=1000, help='Number of rows')
    parser.add_argument('--n', type=int, default=1000, help='Number of columns')
    parser.add_argument('--k', type=int, default=20, help='Target rank')
    parser.add_argument('--T', type=int, default=10, help='Number of time steps')
    parser.add_argument('--eta', type=float, default=0.05, help='Additive perturbation magnitude')
    parser.add_argument('--angle', type=float, default=0.03, help='Rotation angle per step for rotating mode')
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
    parser.add_argument('--csv', type=str, default=None, help='Base CSV path (mode suffix is appended)')

    args = parser.parse_args()
    verbose = not args.quiet

    additive_results = None
    rotating_results = None

    if args.mode in ('additive', 'both'):
        additive_results = run_experiment_additive(
            m=args.m,
            n=args.n,
            k=args.k,
            T=args.T,
            eta=args.eta,
            p_cold=args.p_cold,
            p_warm=args.p_warm,
            q=args.q,
            compute_optimal=args.compute_optimal,
            optimal_max_dim=args.optimal_max_dim,
            device=args.device,
            seed=args.seed,
            verbose=verbose,
        )

    if args.mode in ('rotating', 'both'):
        rotating_results = run_experiment_rotating(
            m=args.m,
            n=args.n,
            k=args.k,
            T=args.T,
            angle=args.angle,
            p_cold=args.p_cold,
            p_warm=args.p_warm,
            q=args.q,
            compute_optimal=args.compute_optimal,
            optimal_max_dim=args.optimal_max_dim,
            device=args.device,
            seed=args.seed,
            verbose=verbose,
        )

    generate_plots(
        additive_results=additive_results,
        rotating_results=rotating_results,
        output_dir='results/figures',
    )

    if args.csv:
        csv_path = Path(args.csv)
        if additive_results is not None:
            _save_csv(additive_results, csv_path.with_name(f'{csv_path.stem}_additive{csv_path.suffix}'))
        if rotating_results is not None:
            _save_csv(rotating_results, csv_path.with_name(f'{csv_path.stem}_rotating{csv_path.suffix}'))


if __name__ == '__main__':
    main()
