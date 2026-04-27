"""Data characterization for Hurricane Isabel variables.

Quantifies sparsity, scale, and low-rank structure for each variable across all
48 timesteps.  Helps explain why some variables (e.g. QGRAUP, QRAIN) show little
warm-start benefit: they are dominated by near-zero values with no stable subspace.

Output
------
results/hurricane/data_characteristics.csv   — one row per variable
results/hurricane/figures/data_characteristics.png

Usage
-----
python -m streaming_svd.experiments.hurricane.characterize \
    --data-dir data/raw \
    --out results/hurricane/data_characteristics.csv \
    --fig results/hurricane/figures/data_characteristics.png \
    --k 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from streaming_svd.data import (
    HURRICANE_VARIABLES,
    discover_variable_files,
    load_weather_matrix,
    optimal_rank_k_rel_fro_error_from_gram,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-snapshot metrics
# ---------------------------------------------------------------------------

def _compute_snapshot_metrics(A: torch.Tensor, k: int, sparsity_thresh: float) -> Dict:
    """Compute characterization metrics for a single snapshot matrix.

    Parameters
    ----------
    A : shape (m, n) float32
    k : target rank for energy-concentration metrics
    sparsity_thresh : absolute threshold below which a value is considered "near-zero"

    Returns
    -------
    dict of scalar metrics
    """
    A_np = A.numpy()

    abs_vals = np.abs(A_np)
    total_elements = A_np.size

    max_abs = float(abs_vals.max())
    mean_abs = float(abs_vals.mean())
    std_val = float(A_np.std())

    # Sparsity: fraction of elements with |value| < threshold
    sparsity_abs = float((abs_vals < sparsity_thresh).sum() / total_elements)
    # Relative sparsity: fraction < 1% of max
    rel_thresh = max_abs * 0.01 if max_abs > 0 else 0.0
    sparsity_rel = float((abs_vals < rel_thresh).sum() / total_elements)

    # Singular values via Gram matrix (100×100 — cheap)
    with torch.no_grad():
        G = A.T @ A
        G64 = G.to(torch.float64)
        eigvals = torch.linalg.eigvalsh(G64)  # ascending
        eigvals = torch.clamp(eigvals, min=0.0)
        sigma_sq = torch.flip(eigvals, [0]).numpy()  # descending σ²

    sigma = np.sqrt(sigma_sq)
    total_energy = float(sigma_sq.sum())

    # Energy captured by top-k singular values
    top_k_energy = float(sigma_sq[:k].sum())
    energy_in_top_k = top_k_energy / total_energy if total_energy > 0 else 0.0

    # Optimal rank-k error (relative Frobenius)
    opt_err_k = float(np.sqrt(max(0.0, 1.0 - energy_in_top_k)))

    # Singular value decay: σ_{k+1}/σ_1
    sigma1 = sigma[0] if len(sigma) > 0 and sigma[0] > 0 else 1.0
    decay_ratio = float(sigma[k] / sigma1) if k < len(sigma) else 0.0

    # Stable rank = Σσ² / σ₁²  (Frobenius norm² / spectral norm²)
    stable_rank = (total_energy / float(sigma_sq[0])) if sigma_sq[0] > 0 else 0.0

    # Effective rank via entropy: exp(H(p)) where p_i = σ_i / Σσ_j
    sum_sigma = float(sigma.sum())
    if sum_sigma > 0:
        p = sigma / sum_sigma
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))
        effective_rank_entropy = float(np.exp(entropy))
    else:
        effective_rank_entropy = 0.0

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "std": std_val,
        "sparsity_abs": sparsity_abs,
        "sparsity_rel": sparsity_rel,
        "energy_in_top_k": energy_in_top_k,
        "opt_fro_error_k": opt_err_k,
        "decay_ratio": decay_ratio,
        "stable_rank": stable_rank,
        "effective_rank_entropy": effective_rank_entropy,
    }


# ---------------------------------------------------------------------------
# Per-variable aggregation
# ---------------------------------------------------------------------------

def characterize_variable(
    data_dir: Path,
    var: str,
    k: int,
    sparsity_thresh: float,
) -> Dict:
    """Aggregate characterization metrics across all timesteps for one variable."""
    files = discover_variable_files(data_dir, var, start=1, end=48)
    if not files:
        log.warning("No files found for variable %s in %s — skipping.", var, data_dir)
        return {}

    all_metrics: List[Dict] = []
    for t, path in files:
        A = load_weather_matrix(path)
        m = _compute_snapshot_metrics(A, k, sparsity_thresh)
        all_metrics.append(m)
        log.debug("  %s t=%02d: max_abs=%.3g sparsity_rel=%.3f energy_k=%.4f",
                  var, t, m["max_abs"], m["sparsity_rel"], m["energy_in_top_k"])

    keys = list(all_metrics[0].keys())
    agg: Dict = {"var": var, "n_timesteps": len(all_metrics)}
    for key in keys:
        vals = np.array([m[key] for m in all_metrics])
        agg[f"{key}_mean"] = float(vals.mean())
        agg[f"{key}_min"]  = float(vals.min())
        agg[f"{key}_max"]  = float(vals.max())

    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_characteristics(df: pd.DataFrame, fig_path: Path, k: int) -> None:
    """4-panel comparison figure across all 13 variables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vars_ordered = df["var"].tolist()
    x = np.arange(len(vars_ordered))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Hurricane Isabel — Data Characteristics (k={k})", fontsize=13)

    def bar(ax, col, title, ylabel, color):
        vals = df[col].values
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(vars_ordered, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        return bars

    bar(axes[0, 0],
        "sparsity_rel_mean",
        "Relative Sparsity (|v| < 1% of max)",
        "Fraction of elements",
        "#d62728")

    bar(axes[0, 1],
        "max_abs_mean",
        "Mean Max |value| across timesteps",
        "Max |value|",
        "#1f77b4")

    bar(axes[1, 0],
        "energy_in_top_k_mean",
        f"Energy captured by top-{k} singular values",
        "Fraction of total energy",
        "#2ca02c")

    bar(axes[1, 1],
        "stable_rank_mean",
        "Stable Rank Σσ²/σ₁²",
        "Stable rank",
        "#ff7f0e")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", fig_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Characterize hurricane data variables")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"),
                   help="Root data directory (default: data/raw)")
    p.add_argument("--out", type=Path,
                   default=Path("results/hurricane/data_characteristics.csv"),
                   help="Output CSV path")
    p.add_argument("--fig", type=Path,
                   default=Path("results/hurricane/figures/data_characteristics.png"),
                   help="Output figure path")
    p.add_argument("--k", type=int, default=20,
                   help="Rank for energy-concentration metrics (default: 20)")
    p.add_argument("--sparsity-thresh", type=float, default=1e-6,
                   help="Absolute threshold for near-zero count (default: 1e-6)")
    p.add_argument("--vars", nargs="+", default=None,
                   help="Subset of variables to process (default: all 13)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    variables = args.vars if args.vars else list(HURRICANE_VARIABLES)
    rows = []
    for var in variables:
        log.info("Processing %s ...", var)
        row = characterize_variable(args.data_dir, var, args.k, args.sparsity_thresh)
        if row:
            rows.append(row)

    if not rows:
        log.error("No data found. Check --data-dir.")
        return

    df = pd.DataFrame(rows)

    # Round for readability
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].round(6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Saved CSV: %s  (%d rows × %d cols)", args.out, len(df), len(df.columns))

    # Print concise summary table
    summary_cols = [
        "var",
        "sparsity_rel_mean",
        "max_abs_mean",
        "energy_in_top_k_mean",
        "opt_fro_error_k_mean",
        "stable_rank_mean",
        "effective_rank_entropy_mean",
    ]
    print("\n" + df[summary_cols].to_string(index=False))

    _plot_characteristics(df, args.fig, args.k)


if __name__ == "__main__":
    main()
