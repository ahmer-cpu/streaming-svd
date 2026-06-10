"""Plot adaptive experiment results.

Reads per-variable adaptive CSV files and generates figures:
  - k_star over time (adaptive rank tracking)
  - compression ratio over time
  - combined quality (PSNR, fro error, max error) over time
  - violation count and sparse cost over time

Usage:
    python analysis/hurricane/plot_adaptive.py \
        --raw-dir results/hurricane/adaptive \
        --fig-dir results/hurricane/figures_adaptive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_single_stage(df: pd.DataFrame) -> bool:
    """True for unified single-stage runs (L + S, no residual stage)."""
    if "stage2_skip_reason" in df.columns and \
            df["stage2_skip_reason"].astype(str).eq("single_stage").all():
        return True
    return "r_star" in df.columns and (df["r_star"] == 0).all() and \
           "stage2_rank_bytes" in df.columns and (df["stage2_rank_bytes"] == 0).all()


def plot_variable(df: pd.DataFrame, var: str, fig_dir: Path,
                  prefix: str | None = None) -> None:
    """Generate all plots for a single variable.  Figures are named
    <prefix>_*.png (default: <var>_*.png); pass the CSV stem as prefix to
    keep multiple runs of the same variable from overwriting each other."""
    if prefix is None:
        prefix = var
    t = df["timestep"].values
    single_stage = is_single_stage(df)
    combined_label = "Combined ($L + S$)" if single_stage else "Combined ($L_1 + L_2 + S$)"

    # --- 1. Adaptive rank (k* and r*) ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, df["k_star"], "o-", ms=4,
            label="$k^*$" if single_stage else "$k^*$ (stage 1)")
    if not single_stage and "r_star" in df.columns:
        ax.plot(t, df["r_star"], "s-", ms=4, label="$r^*$ (stage 2)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Adaptive rank")
    ax.set_title(f"{var}: Adaptive Rank Selection")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_adaptive_rank.png", dpi=150)
    plt.close(fig)

    # --- 2. Compression ratio ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, df["compression_ratio"], "o-", ms=4, color="tab:green")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Compression ratio")
    ax.set_title(f"{var}: Compression Ratio ($\\tau$-guaranteed)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_compression_ratio.png", dpi=150)
    plt.close(fig)

    # --- 3. Quality metrics: PSNR ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, df["warm_psnr"], "o-", ms=4, label="Rank-$k^*$ only", alpha=0.7)
    if "combined_psnr" in df.columns and df["combined_psnr"].notna().any():
        ax.plot(t, df["combined_psnr"], "s-", ms=4, label=combined_label, alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{var}: PSNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_psnr.png", dpi=150)
    plt.close(fig)

    # --- 4. Frobenius error ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, df["warm_fro_error"], "o-", ms=4, label="Rank-$k^*$ only", alpha=0.7)
    if "combined_fro_error" in df.columns:
        ax.plot(t, df["combined_fro_error"], "s-", ms=4, label=combined_label, alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title(f"{var}: Frobenius Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_fro_error.png", dpi=150)
    plt.close(fig)

    # --- 5. Violations and max error ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

    ax1.plot(t, df["s0_violations_at_kstar"] / 1e3, "o-", ms=4,
             label="Sparse entries" if single_stage else "After stage 1", alpha=0.7)
    if not single_stage and "r_star_violations" in df.columns:
        ax1.plot(t, df["r_star_violations"] / 1e3, "s-", ms=4, label="After stage 2", alpha=0.7)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Violations (thousands)")
    ax1.set_title(f"{var}: Violation Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    tau = df["tau"].iloc[0]
    ax2.plot(t, df["warm_max_elem_error"], "o-", ms=4, label="Rank-$k^*$ max error", alpha=0.7)
    ax2.axhline(tau, color="red", ls="--", lw=1.5, label=f"$\\tau = {tau}$")
    if "combined_max_elem_error" in df.columns:
        ax2.plot(t, df["combined_max_elem_error"], "s-", ms=4, label="Combined max error", alpha=0.7)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Max element-wise error")
    ax2.set_title(f"{var}: Max Error vs Tolerance")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_violations_maxerr.png", dpi=150)
    plt.close(fig)

    # --- 6. Storage breakdown ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    s1_mb = df["stage1_rank_bytes"] / 1e6
    sp_mb = df["sparse_bytes"] / 1e6
    if single_stage:
        ax.stackplot(t, s1_mb, sp_mb,
                     labels=["$L$ (rank $k^*$)", "Sparse $S$"], alpha=0.7)
    else:
        s2_mb = df["stage2_rank_bytes"] / 1e6
        ax.stackplot(t, s1_mb, s2_mb, sp_mb,
                     labels=["$L_1$ (rank $k^*$)", "$L_2$ (rank $r^*$)", "Sparse $S$"],
                     alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Storage (MB)")
    ax.set_title(f"{var}: Storage Breakdown")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_storage_breakdown.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot adaptive experiment results")
    parser.add_argument("--raw-dir", type=str, default="results/hurricane/adaptive")
    parser.add_argument("--fig-dir", type=str, default="results/hurricane/figures_adaptive")
    parser.add_argument("--vars", nargs="*", default=None,
                        help="Variables to plot (default: all found CSVs)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Find all adaptive CSVs: two-stage (<var>_adaptive.csv) and unified
    # single-stage (<var>_unified_tau<t>.csv / <var>_unified_eps<e>.csv).
    if args.vars:
        csv_files = []
        for v in args.vars:
            matches = sorted(raw_dir.glob(f"{v}_adaptive.csv")) + \
                      sorted(raw_dir.glob(f"{v}_unified*.csv"))
            if not matches:
                print(f"WARNING: no CSV for {v} in {raw_dir}, skipping")
            csv_files.extend(matches)
    else:
        csv_files = sorted(raw_dir.glob("*_adaptive.csv")) + \
                    sorted(raw_dir.glob("*_unified*.csv"))

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        # "<var>_adaptive" or "<var>_unified_tau1" -> "<var>"
        var = csv_path.stem.split("_unified")[0].replace("_adaptive", "")
        # Two-stage files keep the historical <var>_*.png figure names;
        # unified files use the full stem so runs don't overwrite each other.
        prefix = var if csv_path.stem == f"{var}_adaptive" else csv_path.stem
        print(f"Plotting {var} from {csv_path}")
        df = pd.read_csv(csv_path)
        plot_variable(df, var, fig_dir, prefix=prefix)
        print(f"  -> {fig_dir}/{prefix}_*.png")

    print("Done.")


if __name__ == "__main__":
    main()
