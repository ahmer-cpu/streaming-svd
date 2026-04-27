"""Parameterized sweep across experiment types comparing warm-start vs cold-start rSVD.

This module runs a large sweep over:
- Independent random matrices (series)
- Additive perturbation experiments (perturbation)
- Rotating subspace experiments (rotation)

Across parameter grids:
- Matrix dimensions: m, n
- Target rank: k
- Oversampling: p_cold, p_warm
- Power iterations: q

Only focuses on error metrics (ignores runtime, matmul counts, etc).
Aggregates statistics across seeds.
"""

import argparse
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Handle both relative and absolute imports
try:
    from .run_series import run_series_experiment
    from .run_synthetic import run_experiment_additive, run_experiment_rotating
except ImportError:
    from run_series import run_series_experiment
    from run_synthetic import run_experiment_additive, run_experiment_rotating


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_directories(output_dir, fig_dir):
    """Ensure output directories exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)


def compute_metrics(results, T):
    """
    Extract error metrics from experiment results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from an experiment runner.
    T : int
        Number of timesteps.
    
    Returns
    -------
    dict
        Dictionary with computed metrics.
    """
    cold_errors = np.array(results['cold']['errors'])
    warm_errors = np.array(results['warm']['errors'])
    
    if len(cold_errors) == 0 or len(warm_errors) == 0:
        return None
    
    mean_cold_error = np.mean(cold_errors)
    mean_warm_error = np.mean(warm_errors)
    
    # Element-wise differences: warm - cold
    gaps = warm_errors - cold_errors
    mean_gap = np.mean(gaps)
    
    # Element-wise ratios: warm / cold (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.divide(warm_errors, cold_errors, 
                          where=(cold_errors != 0),
                          out=np.ones_like(warm_errors))
    mean_ratio = np.mean(ratios)
    
    # Final gap: error at T
    if len(cold_errors) > 0 and len(warm_errors) > 0:
        final_gap = warm_errors[-1] - cold_errors[-1]
    else:
        final_gap = np.nan
    
    return {
        'mean_cold_error': mean_cold_error,
        'mean_warm_error': mean_warm_error,
        'mean_gap': mean_gap,
        'mean_ratio': mean_ratio,
        'final_gap': final_gap,
    }


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------

def run_single_config(experiment, m, n, k, p_cold, p_warm, q, T, seed, device='cpu', quiet=False):
    """
    Run a single experiment configuration.
    
    Returns
    -------
    dict or None
        Metrics dictionary, or None if run fails.
    """
    try:
        if experiment == "series":
            results = run_series_experiment(
                m=m, n=n, k=k, T=T,
                p_cold=p_cold, p_warm=p_warm, q=q,
                device=device, seed=seed, verbose=False
            )
        elif experiment == "perturbation":
            results = run_experiment_additive(
                m=m, n=n, k=k, T=T,
                p_cold=p_cold, p_warm=p_warm, q=q,
                device=device, seed=seed, verbose=False
            )
        elif experiment == "rotation":
            results = run_experiment_rotating(
                m=m, n=n, k=k, T=T,
                p_cold=p_cold, p_warm=p_warm, q=q,
                device=device, seed=seed, verbose=False
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
        
        metrics = compute_metrics(results, T)
        return metrics
    
    except Exception as e:
        if not quiet:
            print(f"Error in config: {experiment}, m={m}, n={n}, k={k}, "
                  f"p_cold={p_cold}, p_warm={p_warm}, q={q}, seed={seed}")
            print(f"  Exception: {e}")
        return None


def run_sweep(
    experiments,
    m_list, n_list, k_list,
    p_cold_list, p_warm_list, q_list,
    T, n_seeds, seed0,
    output_raw, output_summary, fig_dir,
    device='cpu', quiet=False
):
    """
    Execute the full parameter sweep.
    
    Returns
    -------
    pd.DataFrame
        Raw results table with one row per run.
    pd.DataFrame
        Aggregated summary table grouped by configuration.
    """
    raw_rows = []
    total_configs = (len(experiments) * len(m_list) * len(n_list) * len(k_list) * 
                     len(p_cold_list) * len(p_warm_list) * len(q_list) * n_seeds)
    
    run_count = 0
    
    for experiment in experiments:
        for m in m_list:
            for n in n_list:
                for k in k_list:
                    for p_cold in p_cold_list:
                        for p_warm in p_warm_list:
                            for q in q_list:
                                for seed_idx in range(n_seeds):
                                    seed = seed0 + seed_idx
                                    run_count += 1
                                    
                                    if not quiet:
                                        print(f"[{run_count}/{total_configs}] "
                                              f"{experiment}: m={m}, n={n}, k={k}, "
                                              f"p_cold={p_cold}, p_warm={p_warm}, q={q}, seed={seed}")
                                    
                                    metrics = run_single_config(
                                        experiment, m, n, k, p_cold, p_warm, q, T, seed,
                                        device=device, quiet=quiet
                                    )
                                    
                                    if metrics is not None:
                                        row = {
                                            'experiment': experiment,
                                            'seed': seed,
                                            'm': m,
                                            'n': n,
                                            'k': k,
                                            'p_cold': p_cold,
                                            'p_warm': p_warm,
                                            'q': q,
                                            'mean_cold_error': metrics['mean_cold_error'],
                                            'mean_warm_error': metrics['mean_warm_error'],
                                            'mean_gap': metrics['mean_gap'],
                                            'mean_ratio': metrics['mean_ratio'],
                                            'final_gap': metrics['final_gap'],
                                        }
                                        raw_rows.append(row)
    
    # Create raw results dataframe
    df_raw = pd.DataFrame(raw_rows)
    
    if not quiet:
        print(f"\nCompleted {len(raw_rows)} / {total_configs} configurations successfully.")
    
    # Aggregate across seeds
    groupby_cols = ['experiment', 'm', 'n', 'k', 'p_cold', 'p_warm', 'q']
    
    # Compute aggregated statistics
    agg_stats = df_raw.groupby(groupby_cols)[['mean_gap', 'mean_ratio']].agg(['mean', 'std']).reset_index()
    
    # Flatten column names for agg_stats
    agg_stats.columns = [col[0] if col[1] == '' else f'{col[0]}_{col[1]}' 
                         for col in agg_stats.columns.values]
    
    # Compute fraction_warm_better (compatible with pandas versions 1.3+)
    fraction_warm_better = (
        df_raw.groupby(groupby_cols)['mean_gap'].apply(
            lambda x: (x < 0).sum() / len(x)
        ).reset_index(name='fraction_warm_better')
    )
    
    # Merge results
    df_agg = agg_stats.merge(fraction_warm_better, on=groupby_cols)
    
    # Save raw results
    if not quiet:
        print(f"Saving raw results to {output_raw}")
    df_raw.to_csv(output_raw, index=False)
    
    # Save aggregated summary
    if not quiet:
        print(f"Saving aggregated summary to {output_summary}")
    df_agg.to_csv(output_summary, index=False)
    
    return df_raw, df_agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_gap_histogram(df_raw, fig_dir, quiet=False):
    """Plot histogram of mean_gap values for each experiment type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    experiments = df_raw['experiment'].unique()
    
    for ax, exp in zip(axes, sorted(experiments)):
        data = df_raw[df_raw['experiment'] == exp]['mean_gap'].dropna()
        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax.set_xlabel('Mean Gap (Warm - Cold Error)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{exp.capitalize()} Experiment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = fig_dir / 'sweep_error_gap_hist.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if not quiet:
        print(f"Saved histogram to {output_path}")
    plt.close()


def plot_fraction_warm_better(df_agg, fig_dir, quiet=False):
    """Plot fraction_warm_better vs p_warm for each experiment."""
    experiments = df_agg['experiment'].unique()
    
    fig, axes = plt.subplots(1, len(experiments), figsize=(5*len(experiments), 4))
    
    if len(experiments) == 1:
        axes = [axes]
    
    for ax, exp in zip(axes, sorted(experiments)):
        data = df_agg[df_agg['experiment'] == exp]
        
        # Group by p_warm and compute mean fraction_warm_better
        p_warm_values = data['p_warm'].unique()
        fractions = []
        for p_w in sorted(p_warm_values):
            mask = data['p_warm'] == p_w
            frac = data[mask]['fraction_warm_better'].mean()
            fractions.append(frac)
        
        ax.plot(sorted(p_warm_values), fractions, 'o-', linewidth=2, markersize=8)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% tie')
        ax.set_xlabel('p_warm (warm-start oversampling)')
        ax.set_ylabel('Fraction Runs: Warm Better')
        ax.set_title(f'{exp.capitalize()} Experiment')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    output_path = fig_dir / 'sweep_fraction_warm_better.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if not quiet:
        print(f"Saved fraction plot to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Parameterized sweep comparing warm-start vs cold-start rSVD'
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['series', 'perturbation', 'rotation'],
        choices=['series', 'perturbation', 'rotation'],
        help='Experiment types to run'
    )
    
    # Parameter grids
    parser.add_argument('--m-list', nargs='+', type=int, default=[500, 1000],
                       help='Row dimensions')
    parser.add_argument('--n-list', nargs='+', type=int, default=[500, 1000],
                       help='Column dimensions')
    parser.add_argument('--k-list', nargs='+', type=int, default=[10, 20, 40],
                       help='Target ranks')
    parser.add_argument('--p-cold-list', nargs='+', type=int, default=[5, 10, 20],
                       help='Cold-start oversampling parameters')
    parser.add_argument('--p-warm-list', nargs='+', type=int, default=[0, 5, 10, 20],
                       help='Warm-start oversampling parameters')
    parser.add_argument('--q-list', nargs='+', type=int, default=[0, 1],
                       help='Power iteration counts')
    
    # Experiment parameters
    parser.add_argument('--T', type=int, default=10,
                       help='Number of timesteps')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--seed0', type=int, default=42,
                       help='Base random seed')
    
    # Output paths (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent  # Navigate to project root
    parser.add_argument('--output-raw', type=Path, 
                       default=project_root / 'results' / 'sweep_raw.csv',
                       help='Path for raw results CSV')
    parser.add_argument('--output-summary', type=Path,
                       default=project_root / 'results' / 'sweep_summary.csv',
                       help='Path for aggregated summary CSV')
    parser.add_argument('--fig-dir', type=Path,
                       default=project_root / 'results' / 'figures',
                       help='Directory for output figures')
    
    # Misc options
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for computation')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Suppress torch warnings if not on GPU
    if args.device == 'cpu':
        warnings.filterwarnings('ignore', category=UserWarning)
    
    # Ensure output directories
    ensure_directories(args.output_raw.parent, args.fig_dir)
    
    if not args.quiet:
        print('=' * 70)
        print('Parameterized Sweep: Warm-start vs Cold-start rSVD')
        print('=' * 70)
        print(f'Experiments: {args.experiments}')
        print(f'Matrix dimensions: m={args.m_list}, n={args.n_list}')
        print(f'Ranks: k={args.k_list}')
        print(f'Oversampling: p_cold={args.p_cold_list}, p_warm={args.p_warm_list}')
        print(f'Power iterations: q={args.q_list}')
        print(f'Timesteps: T={args.T}')
        print(f'Seeds: {args.n_seeds} (starting from {args.seed0})')
        print('=' * 70)
        print()
    
    # Run the sweep
    df_raw, df_agg = run_sweep(
        experiments=args.experiments,
        m_list=args.m_list,
        n_list=args.n_list,
        k_list=args.k_list,
        p_cold_list=args.p_cold_list,
        p_warm_list=args.p_warm_list,
        q_list=args.q_list,
        T=args.T,
        n_seeds=args.n_seeds,
        seed0=args.seed0,
        output_raw=args.output_raw,
        output_summary=args.output_summary,
        fig_dir=args.fig_dir,
        device=args.device,
        quiet=args.quiet
    )
    
    if not args.quiet:
        print('\nGenerating plots...')
    
    # Generate plots
    plot_error_gap_histogram(df_raw, args.fig_dir, quiet=args.quiet)
    plot_fraction_warm_better(df_agg, args.fig_dir, quiet=args.quiet)
    
    if not args.quiet:
        print('\n' + '=' * 70)
        print('Sweep complete!')
        print(f'Raw results: {args.output_raw}')
        print(f'Summary: {args.output_summary}')
        print(f'Figures: {args.fig_dir}')
        print('=' * 70)


if __name__ == '__main__':
    main()
