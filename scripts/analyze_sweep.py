"""Sweep analysis and figure generation for the Hurricane rSVD parameter study.

Reads the concatenated sweep_all_raw.csv produced by run_sweep.py and
generates summary tables and comparison plots showing how rank (k),
power iterations (q), and warm oversampling (p_warm) affect cold vs
warm rSVD accuracy and timing.

Usage:
    python scripts/analyze_sweep.py
    python scripts/analyze_sweep.py --input results/sweep/sweep_all_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "sweep" / "sweep_all_raw.csv"
DEFAULT_FIG_DIR = REPO_ROOT / "results" / "sweep" / "figures"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "sweep" / "sweep_summary.csv"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_COLD_COLOR = "#2171b5"
_WARM_COLOR = "#d94801"
_OPT_COLOR = "#252525"
_CMAP = plt.cm.viridis


def load_sweep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure sweep config columns exist (for baseline data that may lack them)
    for col, src in [("sweep_k", "k"), ("sweep_pwarm", "p_warm"), ("sweep_q", "q")]:
        if col not in df.columns and src in df.columns:
            df[col] = df[src]
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(k, p_warm, q) summary aggregated across all variables and timesteps."""
    groups = df.groupby(["sweep_k", "sweep_pwarm", "sweep_q"])

    rows = []
    for (k, pw, q), g in groups:
        # Only use timesteps where warm actually ran (t > 1)
        warm_rows = g.dropna(subset=["warm_fro_error"])
        warm_active = warm_rows[warm_rows["warm_stats_warm_start"] == True] if "warm_stats_warm_start" in warm_rows.columns else warm_rows[warm_rows["timestep"] > 1] if "timestep" in warm_rows.columns else warm_rows

        row = {
            "k": k, "p_warm": pw, "q": q,
            "n_rows": len(g),
            "cold_fro_mean": g["cold_fro_error"].mean(),
            "cold_fro_std": g["cold_fro_error"].std(),
            "warm_fro_mean": g["warm_fro_error"].mean(),
            "warm_fro_std": g["warm_fro_error"].std(),
            "optimal_fro_mean": g["optimal_fro_error"].mean() if "optimal_fro_error" in g.columns else np.nan,
            "fro_gap_mean": g["fro_error_gap"].mean(),
            "fro_gap_std": g["fro_error_gap"].std(),
            "frac_warm_better": (g["warm_fro_error"] < g["cold_fro_error"]).mean(),
        }

        # Timing (only for warm-active rows)
        if "cold_time_total" in warm_active.columns and "warm_time_total" in warm_active.columns:
            ct = warm_active["cold_time_total"].dropna() * 1000
            wt = warm_active["warm_time_total"].dropna() * 1000
            row["cold_time_ms"] = ct.mean()
            row["warm_time_ms"] = wt.mean()
            speedup = warm_active["time_speedup_ratio"].dropna()
            row["speedup_mean"] = speedup.mean() if len(speedup) > 0 else np.nan
            row["frac_warm_faster"] = (warm_active["warm_time_total"] < warm_active["cold_time_total"]).mean()
        else:
            row["cold_time_ms"] = row["warm_time_ms"] = row["speedup_mean"] = row["frac_warm_faster"] = np.nan

        # Spectral error
        if "cold_spec_error" in g.columns:
            row["cold_spec_mean"] = g["cold_spec_error"].mean()
            row["warm_spec_mean"] = g["warm_spec_error"].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def compute_per_var_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(var, k, p_warm, q) summary."""
    groups = df.groupby(["var", "sweep_k", "sweep_pwarm", "sweep_q"])

    rows = []
    for (var, k, pw, q), g in groups:
        row = {
            "var": var, "k": k, "p_warm": pw, "q": q,
            "cold_fro_mean": g["cold_fro_error"].mean(),
            "warm_fro_mean": g["warm_fro_error"].mean(),
            "fro_gap_mean": g["fro_error_gap"].mean(),
            "frac_warm_better": (g["warm_fro_error"] < g["cold_fro_error"]).mean(),
        }
        if "optimal_fro_error" in g.columns:
            row["optimal_fro_mean"] = g["optimal_fro_error"].mean()
        if "cold_time_total" in g.columns:
            warm_active = g.dropna(subset=["warm_time_total"])
            if len(warm_active) > 0:
                row["speedup_mean"] = warm_active["time_speedup_ratio"].dropna().mean()
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_vs_k(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Error vs k, one line per q value. Separate panels for cold, warm, gap."""
    for pw in summary["p_warm"].unique():
        sub = summary[summary["p_warm"] == pw]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for q_val in sorted(sub["q"].unique()):
            sq = sub[sub["q"] == q_val].sort_values("k")
            axes[0].plot(sq["k"], sq["cold_fro_mean"], "o-", label=f"q={q_val}")
            axes[1].plot(sq["k"], sq["warm_fro_mean"], "s-", label=f"q={q_val}")
            axes[2].plot(sq["k"], sq["fro_gap_mean"], "^-", label=f"q={q_val}")

        axes[0].set_title("Cold rSVD Error")
        axes[1].set_title("Warm rSVD Error")
        axes[2].set_title("Error Gap (warm − cold)")
        axes[2].axhline(0, color="gray", ls="--", lw=0.8)

        for ax in axes:
            ax.set_xlabel("Rank k")
            ax.set_ylabel("Mean Frobenius Error")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Error vs Rank (p_warm={pw})", fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_dir / f"error_vs_k_pwarm{pw}.png", dpi=150)
        fig.savefig(fig_dir / f"error_vs_k_pwarm{pw}.pdf")
        plt.close(fig)


def plot_error_vs_q(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Error vs q, one line per k value."""
    for pw in summary["p_warm"].unique():
        sub = summary[summary["p_warm"] == pw]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for k_val in sorted(sub["k"].unique()):
            sk = sub[sub["k"] == k_val].sort_values("q")
            axes[0].plot(sk["q"], sk["cold_fro_mean"], "o-", label=f"k={k_val}")
            axes[1].plot(sk["q"], sk["warm_fro_mean"], "s-", label=f"k={k_val}")
            axes[2].plot(sk["q"], sk["fro_gap_mean"], "^-", label=f"k={k_val}")

        axes[0].set_title("Cold rSVD Error")
        axes[1].set_title("Warm rSVD Error")
        axes[2].set_title("Error Gap (warm − cold)")
        axes[2].axhline(0, color="gray", ls="--", lw=0.8)

        for ax in axes:
            ax.set_xlabel("Power Iterations q")
            ax.set_ylabel("Mean Frobenius Error")
            ax.set_xticks(sorted(sub["q"].unique()))
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Error vs Power Iterations (p_warm={pw})", fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_dir / f"error_vs_q_pwarm{pw}.png", dpi=150)
        fig.savefig(fig_dir / f"error_vs_q_pwarm{pw}.pdf")
        plt.close(fig)


def plot_speedup_vs_k(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Speedup ratio vs k, one line per q."""
    if "speedup_mean" not in summary.columns:
        return

    for pw in summary["p_warm"].unique():
        sub = summary[summary["p_warm"] == pw]

        fig, ax = plt.subplots(figsize=(8, 5))
        for q_val in sorted(sub["q"].unique()):
            sq = sub[sub["q"] == q_val].sort_values("k")
            ax.plot(sq["k"], sq["speedup_mean"], "o-", label=f"q={q_val}")

        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="break-even")
        ax.set_xlabel("Rank k")
        ax.set_ylabel("Speedup (cold_time / warm_time)")
        ax.set_title(f"Timing Speedup vs Rank (p_warm={pw})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"speedup_vs_k_pwarm{pw}.png", dpi=150)
        fig.savefig(fig_dir / f"speedup_vs_k_pwarm{pw}.pdf")
        plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Heatmaps: (k, q) → gap and fraction_warm_better, for each p_warm."""
    for pw in summary["p_warm"].unique():
        sub = summary[summary["p_warm"] == pw]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Gap heatmap
        pivot_gap = sub.pivot_table(index="q", columns="k", values="fro_gap_mean")
        im0 = axes[0].imshow(pivot_gap.values, aspect="auto", cmap="RdBu_r",
                             vmin=-pivot_gap.abs().values.max(),
                             vmax=pivot_gap.abs().values.max())
        axes[0].set_xticks(range(len(pivot_gap.columns)))
        axes[0].set_xticklabels(pivot_gap.columns)
        axes[0].set_yticks(range(len(pivot_gap.index)))
        axes[0].set_yticklabels(pivot_gap.index)
        axes[0].set_xlabel("Rank k")
        axes[0].set_ylabel("Power Iterations q")
        axes[0].set_title("Mean Error Gap (warm − cold)")
        for i in range(len(pivot_gap.index)):
            for j in range(len(pivot_gap.columns)):
                v = pivot_gap.values[i, j]
                axes[0].text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im0, ax=axes[0])

        # Fraction warm better heatmap
        pivot_frac = sub.pivot_table(index="q", columns="k", values="frac_warm_better")
        im1 = axes[1].imshow(pivot_frac.values, aspect="auto", cmap="Greens",
                             vmin=0, vmax=1)
        axes[1].set_xticks(range(len(pivot_frac.columns)))
        axes[1].set_xticklabels(pivot_frac.columns)
        axes[1].set_yticks(range(len(pivot_frac.index)))
        axes[1].set_yticklabels(pivot_frac.index)
        axes[1].set_xlabel("Rank k")
        axes[1].set_ylabel("Power Iterations q")
        axes[1].set_title("Fraction Warm Better")
        for i in range(len(pivot_frac.index)):
            for j in range(len(pivot_frac.columns)):
                v = pivot_frac.values[i, j]
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im1, ax=axes[1])

        fig.suptitle(f"Parameter Sensitivity (p_warm={pw})", fontsize=14)
        fig.tight_layout()
        fig.savefig(fig_dir / f"heatmap_k_q_pwarm{pw}.png", dpi=150)
        fig.savefig(fig_dir / f"heatmap_k_q_pwarm{pw}.pdf")
        plt.close(fig)


def plot_pwarm_comparison(summary: pd.DataFrame, fig_dir: Path) -> None:
    """Compare p_warm=5 vs p_warm=10: error gap at each k."""
    if summary["p_warm"].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for q_val in sorted(summary["q"].unique()):
        for pw, marker, color in [(5, "o", _COLD_COLOR), (10, "s", _WARM_COLOR)]:
            sub = summary[(summary["p_warm"] == pw) & (summary["q"] == q_val)].sort_values("k")
            if sub.empty:
                continue
            axes[0].plot(sub["k"], sub["fro_gap_mean"], f"{marker}-", color=color,
                        label=f"p_warm={pw}, q={q_val}", alpha=0.8)
            axes[1].plot(sub["k"], sub["frac_warm_better"], f"{marker}-", color=color,
                        label=f"p_warm={pw}, q={q_val}", alpha=0.8)

    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Rank k")
    axes[0].set_ylabel("Mean Error Gap (warm − cold)")
    axes[0].set_title("Error Gap: p_warm=5 vs p_warm=10")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Rank k")
    axes[1].set_ylabel("Fraction Warm Better")
    axes[1].set_title("Win Rate: p_warm=5 vs p_warm=10")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "pwarm_comparison.png", dpi=150)
    fig.savefig(fig_dir / "pwarm_comparison.pdf")
    plt.close(fig)


def plot_per_variable_panel(per_var: pd.DataFrame, fig_dir: Path) -> None:
    """Per-variable error vs k, showing which variables benefit most."""
    # Fix p_warm=5, q=0 as the reference config
    sub = per_var[(per_var["p_warm"] == 5) & (per_var["q"] == 0)]
    if sub.empty:
        sub = per_var[per_var["q"] == 0]

    variables = sorted(sub["var"].unique())
    n_vars = len(variables)
    colors = plt.cm.tab20(np.linspace(0, 1, n_vars))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, var in enumerate(variables):
        sv = sub[sub["var"] == var].sort_values("k")
        axes[0].plot(sv["k"], sv["fro_gap_mean"], "o-", color=colors[i], label=var, markersize=4)
        axes[1].plot(sv["k"], sv["warm_fro_mean"], "s-", color=colors[i], label=var, markersize=4)

    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Rank k")
    axes[0].set_ylabel("Mean Error Gap (warm − cold)")
    axes[0].set_title("Per-Variable Error Gap vs Rank")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Rank k")
    axes[1].set_ylabel("Mean Warm Frobenius Error")
    axes[1].set_title("Per-Variable Warm Error vs Rank")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "per_variable_error_vs_k.png", dpi=150)
    fig.savefig(fig_dir / "per_variable_error_vs_k.pdf")
    plt.close(fig)


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 100)
    print("PARAMETER SWEEP SUMMARY (aggregated across all variables and timesteps)")
    print("=" * 100)

    header = (
        f"{'k':>3}  {'p_warm':>6}  {'q':>2}  "
        f"{'cold_fro':>9}  {'warm_fro':>9}  {'gap':>9}  "
        f"{'frac_better':>11}  {'speedup':>8}  {'cold_ms':>8}  {'warm_ms':>8}"
    )
    print(header)
    print("-" * 100)

    for _, r in summary.sort_values(["k", "p_warm", "q"]).iterrows():
        def _f(v, fmt=".5f"):
            return f"{v:{fmt}}" if not np.isnan(v) else "n/a"

        print(
            f"{int(r['k']):>3}  {int(r['p_warm']):>6}  {int(r['q']):>2}  "
            f"{_f(r['cold_fro_mean']):>9}  {_f(r['warm_fro_mean']):>9}  "
            f"{_f(r['fro_gap_mean']):>9}  "
            f"{_f(r['frac_warm_better'], '.2f'):>11}  "
            f"{_f(r.get('speedup_mean', np.nan), '.3f'):>8}  "
            f"{_f(r.get('cold_time_ms', np.nan), '.1f'):>8}  "
            f"{_f(r.get('warm_time_ms', np.nan), '.1f'):>8}"
        )

    print("-" * 100)
    print("  gap = mean(warm_fro - cold_fro), negative = warm better")
    print("  speedup = cold_time / warm_time, >1 = warm faster")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze hurricane rSVD parameter sweep")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                       help="Path to sweep_all_raw.csv")
    parser.add_argument("--fig-dir", type=str, default=str(DEFAULT_FIG_DIR),
                       help="Directory for output figures")
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY),
                       help="Path for summary CSV output")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    input_path = Path(args.input)
    fig_dir = Path(args.fig_dir)
    summary_path = Path(args.summary)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        print("Run scripts/run_sweep.py first.")
        return

    print(f"Loading sweep data from {input_path} ...")
    df = load_sweep(input_path)
    print(f"  {len(df)} rows, {df['var'].nunique()} variables")
    print(f"  k values: {sorted(df['sweep_k'].unique())}")
    print(f"  p_warm values: {sorted(df['sweep_pwarm'].unique())}")
    print(f"  q values: {sorted(df['sweep_q'].unique())}")

    # Compute summaries
    print("\nComputing summaries ...")
    summary = compute_summary(df)
    per_var = compute_per_var_summary(df)

    # Save summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"  Summary saved to {summary_path}")

    per_var_path = summary_path.parent / "sweep_per_var_summary.csv"
    per_var.to_csv(per_var_path, index=False)
    print(f"  Per-variable summary saved to {per_var_path}")

    # Print table
    print_summary_table(summary)

    # Generate plots
    if not args.no_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating plots to {fig_dir} ...")

        plot_error_vs_k(summary, fig_dir)
        print("  error_vs_k")

        plot_error_vs_q(summary, fig_dir)
        print("  error_vs_q")

        plot_speedup_vs_k(summary, fig_dir)
        print("  speedup_vs_k")

        plot_heatmaps(summary, fig_dir)
        print("  heatmaps")

        plot_pwarm_comparison(summary, fig_dir)
        print("  pwarm_comparison")

        plot_per_variable_panel(per_var, fig_dir)
        print("  per_variable_panel")

        print(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
