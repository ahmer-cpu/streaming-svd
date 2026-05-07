"""Real-data streaming SVD experiment runner.

Loads volumetric weather snapshots (e.g. Uf01.bin ... Uf48.bin) and runs
cold-start vs warm-start rSVD across consecutive timesteps.

Data format: binary float32 files, shape (100, 500, 500) = (z, y, x).
Reshaped to matrix (250000, 100): rows are spatial (x,y) points, columns are z-levels.

Data loading utilities live in :mod:`streaming_svd.data` and are re-exported
here for backwards compatibility.

CLI entry point:
    python -m streaming_svd.experiments.run_weather --help
"""

import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from streaming_svd.algos.metrics import rel_fro_error, rel_spec_error_est, subspace_sin_theta
from streaming_svd.algos.rsvd import rsvd
from streaming_svd.algos.warm_rsvd import warm_rsvd
from streaming_svd.data import (
    load_weather_matrix,
    optimal_rank_k_rel_fro_error_from_gram,
)


def _empty_results():
    return {
        'timesteps': [],
        'cold': {
            'errors': [],
            'times': [],
            'matmuls': [],
            'spec_errors': [],
        },
        'warm': {
            'errors': [],
            'times': [],
            'matmuls': [],
            'spec_errors': [],
            'drift': [],
        },
        'optimal': {'errors': []},
        'params': {},
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_weather_experiment(
    data_dir: str = 'data/raw',
    var: str = 'Uf',
    start: int = 1,
    end: int = 10,
    k: int = 20,
    p_cold: int = 10,
    p_warm: int = 5,
    q: int = 0,
    dtype: str = 'float32',
    seed: int = 42,
    compute_optimal: bool = True,
    device: str = 'cpu',
    memmap: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run cold-start vs warm-start rSVD on real weather data.

    Parameters
    ----------
    data_dir : str
        Directory containing .bin files.
    var : str
        Variable prefix (e.g., 'Uf' for Uf01.bin, Uf02.bin, ...).
    start, end : int
        Timestep range (inclusive).
    k, p_cold, p_warm, q : int
        SVD parameters.
    dtype : str
        Data type (float32 or float64).
    seed : int
        Random seed.
    compute_optimal : bool
        Whether to compute optimal rank-k baseline.
    device : str
        Device ('cpu' or 'cuda').
    memmap : bool
        Whether to use numpy memmap for file loading.
    verbose : bool
        Verbosity.

    Returns
    -------
    results : dict
        Experiment results with errors, times, matmuls, optimal baseline.
    """
    device = torch.device(device)  # type: ignore[assignment]
    data_dir = Path(data_dir)  # type: ignore[assignment]

    results = _empty_results()
    results['params'] = {
        'data_dir': str(data_dir),
        'var': var,
        'start': start,
        'end': end,
        'k': k,
        'p_cold': p_cold,
        'p_warm': p_warm,
        'q': q,
        'dtype': dtype,
        'seed': seed,
        'compute_optimal': compute_optimal,
        'device': str(device),
        'memmap': memmap,
    }

    if verbose:
        print('=' * 70)
        print('Weather/Real Data SVD Experiment')
        print('=' * 70)
        print(f'Data directory: {data_dir}')
        print(f'Variable prefix: {var}')
        print(f'Timesteps: {start} to {end}')
        print(f'Target rank k: {k}')
        print(f'Cold-start p={p_cold}, q={q}')
        print(f'Warm-start p={p_warm}, q={q}')
        print(f'Device: {device}')
        
        # Normalize and display dtype
        normalized_dtype = 'float64' if dtype.lower() in ('float64', 'fp64', 'double') else 'float32'
        print(f'Data type: {normalized_dtype}')
        if normalized_dtype == 'float64':
            print('  WARNING: float64 computation may be slower and use more memory')
        print('=' * 70)

    U_warm_prev = None

    with torch.no_grad():
        for t in range(start, end + 1):
            file_path = data_dir / f'{var}{t:02d}.bin'  # type: ignore[operator]

            if not file_path.exists():
                if verbose:
                    print(f'WARNING: {file_path} not found, skipping')
                continue

            if verbose:
                print(f'\nProcessing timestep {t}: {file_path.name}')

            A = load_weather_matrix(file_path, memmap=memmap)
            
            # Cast A to specified dtype
            if dtype.lower() in ('float64', 'fp64', 'double'):
                A = A.to(device=device, dtype=torch.float64)
            else:
                A = A.to(device=device, dtype=torch.float32)

            optimal_error = np.nan
            if compute_optimal:
                optimal_error = optimal_rank_k_rel_fro_error_from_gram(A, k)

            t0 = time.perf_counter()
            result_cold = rsvd(
                A,
                k,
                p=p_cold,
                q=q,
                device=device,
                seed=seed,
                return_stats=True,
            )
            U_cold, s_cold, Vt_cold, stats_cold = result_cold  # type: ignore[misc]
            # Ensure SVD results match A's dtype
            U_cold = U_cold.to(dtype=A.dtype)
            s_cold = s_cold.to(dtype=A.dtype)
            Vt_cold = Vt_cold.to(dtype=A.dtype)
            time_cold = time.perf_counter() - t0

            t0 = time.perf_counter()
            result_warm = warm_rsvd(
                A,
                U_warm_prev,
                k,
                p=p_warm,
                q=q,
                device=device,
                seed=seed,
                return_stats=True,
            )
            U_warm, s_warm, Vt_warm, stats_warm = result_warm  # type: ignore[misc]
            # Ensure SVD results match A's dtype
            U_warm = U_warm.to(dtype=A.dtype)
            s_warm = s_warm.to(dtype=A.dtype)
            Vt_warm = Vt_warm.to(dtype=A.dtype)
            time_warm = time.perf_counter() - t0

            error_cold = rel_fro_error(A, U_cold, s_cold, Vt_cold)
            error_warm = rel_fro_error(A, U_warm, s_warm, Vt_warm)
            spec_cold = rel_spec_error_est(A, U_cold, n_iter=3)
            spec_warm = rel_spec_error_est(A, U_warm, n_iter=3)

            matmul_cold = stats_cold['matmul_counts']['A@X'] + stats_cold['matmul_counts']['AT@X']
            matmul_warm = stats_warm['matmul_counts']['A@X'] + stats_warm['matmul_counts']['AT@X']

            warm_drift = np.nan
            if U_warm_prev is not None:
                warm_drift = subspace_sin_theta(U_warm_prev, U_warm)

            results['timesteps'].append(t)
            results['cold']['errors'].append(error_cold)
            results['cold']['times'].append(time_cold)
            results['cold']['matmuls'].append(matmul_cold)
            results['cold']['spec_errors'].append(spec_cold)

            results['warm']['errors'].append(error_warm)
            results['warm']['times'].append(time_warm)
            results['warm']['matmuls'].append(matmul_warm)
            results['warm']['spec_errors'].append(spec_warm)
            results['warm']['drift'].append(warm_drift)

            results['optimal']['errors'].append(optimal_error)

            U_warm_prev = U_warm

            if verbose:
                print(f'  Cold error:    {error_cold:.6f}, time: {time_cold:.4f}s, matmuls: {matmul_cold}')
                print(f'  Warm error:    {error_warm:.6f}, time: {time_warm:.4f}s, matmuls: {matmul_warm}')
                print(f'  Optimal error: {optimal_error:.6f}')
                if not np.isnan(warm_drift):
                    print(f'  Warm drift:    {warm_drift:.6f}')

            del A, U_cold, s_cold, Vt_cold, U_warm, s_warm, Vt_warm
            del stats_cold, stats_warm
            gc.collect()

    if verbose:
        print('\n' + '=' * 70)
        print('Experiment complete')
        print('=' * 70)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_errors(results, output_file):
    """Plot cold, warm, and optimal errors over time."""
    timesteps = np.array(results['timesteps'], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, results['cold']['errors'], 'o-', linewidth=2.2, markersize=7, label='Cold-start rSVD')
    ax.plot(timesteps, results['warm']['errors'], 's-', linewidth=2.2, markersize=7, label='Warm-start rSVD')

    optimal_errors = np.array(results['optimal']['errors'])
    if np.any(~np.isnan(optimal_errors)):
        ax.plot(
            timesteps,
            optimal_errors,
            '^-',
            linewidth=2.2,
            markersize=7,
            color='black',
            label='Optimal rank-k',
        )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Relative Frobenius Error', fontsize=12)
    ax.set_title('Real Data (Uf): Cold vs Warm rSVD + Optimal Rank-k Baseline', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_timing(results, output_file):
    """Plot cold vs warm timing."""
    timesteps = np.array(results['timesteps'], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timesteps, np.array(results['cold']['times']) * 1000, 'o-', linewidth=2.2, markersize=7, label='Cold-start')
    ax.plot(
        timesteps,
        np.array(results['warm']['times']) * 1000,
        's-',
        linewidth=2.2,
        markersize=7,
        label='Warm-start',
    )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Real Data (Uf): Runtime Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_matmuls(results, output_file):
    """Plot matmul counts."""
    timesteps = np.array(results['timesteps'], dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        timesteps,
        results['cold']['matmuls'],
        'o-',
        linewidth=2.2,
        markersize=7,
        label='Cold-start',
    )
    ax.plot(
        timesteps,
        results['warm']['matmuls'],
        's-',
        linewidth=2.2,
        markersize=7,
        label='Warm-start',
    )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Matrix-Vector Multiplications', fontsize=12)
    ax.set_title('Real Data (Uf): Computational Work', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_drift(results, output_file):
    """Plot warm-vs-warm subspace drift."""
    timesteps = np.array(results['timesteps'], dtype=int)
    drift = np.array(results['warm']['drift'])

    fig, ax = plt.subplots(figsize=(10, 6))
    valid = ~np.isnan(drift)
    ax.plot(
        timesteps[valid],
        drift[valid],
        'D-',
        linewidth=2.2,
        markersize=7,
        color='purple',
        label='sin(theta)',
    )

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Subspace Distance', fontsize=12)
    ax.set_title('Real Data (Uf): Warm-Start Subspace Drift', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_plots(results, output_dir='results/figures'):
    """Generate all plots for weather experiment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _plot_errors(results, output_path / 'error_weather_vs_optimal.png')
    print(f'✓ Saved error plot to: {output_path / "error_weather_vs_optimal.png"}')

    _plot_timing(results, output_path / 'timing_weather.png')
    print(f'✓ Saved timing plot to: {output_path / "timing_weather.png"}')

    _plot_matmuls(results, output_path / 'matmuls_weather.png')
    print(f'✓ Saved matmuls plot to: {output_path / "matmuls_weather.png"}')

    _plot_drift(results, output_path / 'drift_weather.png')
    print(f'✓ Saved drift plot to: {output_path / "drift_weather.png"}')

    print(f'\nAll weather plots saved to: {output_path.absolute()}')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _save_csv(results, csv_path):
    """Save results to CSV."""
    try:
        import pandas as pd  # type: ignore

        df_data = []
        for i, t in enumerate(results['timesteps']):
            df_data.append(
                {
                    'timestep': t,
                    'cold_error': results['cold']['errors'][i],
                    'warm_error': results['warm']['errors'][i],
                    'optimal_error': results['optimal']['errors'][i],
                    'cold_time': results['cold']['times'][i],
                    'warm_time': results['warm']['times'][i],
                    'cold_matmuls': results['cold']['matmuls'][i],
                    'warm_matmuls': results['warm']['matmuls'][i],
                    'cold_spec_error': results['cold']['spec_errors'][i],
                    'warm_spec_error': results['warm']['spec_errors'][i],
                    'warm_drift': results['warm']['drift'][i],
                }
            )

        df = pd.DataFrame(df_data)
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f'Results saved to: {csv_path}')
    except ImportError:
        import csv

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            if results['timesteps']:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'timestep',
                        'cold_error',
                        'warm_error',
                        'optimal_error',
                        'cold_time',
                        'warm_time',
                        'cold_matmuls',
                        'warm_matmuls',
                        'cold_spec_error',
                        'warm_spec_error',
                        'warm_drift',
                    ],
                )
                writer.writeheader()
                for i, t in enumerate(results['timesteps']):
                    writer.writerow(
                        {
                            'timestep': t,
                            'cold_error': results['cold']['errors'][i],
                            'warm_error': results['warm']['errors'][i],
                            'optimal_error': results['optimal']['errors'][i],
                            'cold_time': results['cold']['times'][i],
                            'warm_time': results['warm']['times'][i],
                            'cold_matmuls': results['cold']['matmuls'][i],
                            'warm_matmuls': results['warm']['matmuls'][i],
                            'cold_spec_error': results['cold']['spec_errors'][i],
                            'warm_spec_error': results['warm']['spec_errors'][i],
                            'warm_drift': results['warm']['drift'][i],
                        }
                    )
        print(f'Results saved to: {csv_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for weather experiment."""
    parser = argparse.ArgumentParser(
        description='Run cold-start vs warm-start rSVD on real weather data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory containing .bin files')
    parser.add_argument('--var', type=str, default='Uf', help='Variable prefix (e.g., Uf for Uf01.bin)')
    parser.add_argument('--start', type=int, default=1, help='Starting timestep')
    parser.add_argument('--end', type=int, default=10, help='Ending timestep (inclusive)')
    parser.add_argument('--k', type=int, default=20, help='Target rank')
    parser.add_argument('--p-cold', type=int, default=10, help='Cold-start oversampling')
    parser.add_argument('--p-warm', type=int, default=5, help='Warm-start oversampling')
    parser.add_argument('--q', type=int, default=0, help='Power iterations')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--compute-optimal',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Compute optimal rank-k baseline',
    )
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--memmap', action='store_true', help='Use numpy memmap for file loading')
    parser.add_argument('--csv', type=str, default='results/weather_results.csv', help='CSV output path')
    parser.add_argument('--fig-dir', type=str, default='results/figures', help='Output directory for figures')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    results = run_weather_experiment(
        data_dir=args.data_dir,
        var=args.var,
        start=args.start,
        end=args.end,
        k=args.k,
        p_cold=args.p_cold,
        p_warm=args.p_warm,
        q=args.q,
        dtype=args.dtype,
        seed=args.seed,
        compute_optimal=args.compute_optimal,
        device=args.device,
        memmap=args.memmap,
        verbose=not args.quiet,
    )

    generate_plots(results, output_dir=args.fig_dir)
    _save_csv(results, args.csv)


if __name__ == '__main__':
    main()
