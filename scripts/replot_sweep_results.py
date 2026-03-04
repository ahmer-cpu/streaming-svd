"""Replot sweep results from CSV outputs.

This script regenerates sweep plots independently from:
- results/sweep_raw.csv
- results/sweep_summary.csv

Outputs:
- results/figures/sweep_error_gap_hist.(png|pdf)
- results/figures/sweep_fraction_warm_better.(png|pdf)
- results/figures/sweep_sampling_effect_mean_gap.(png|pdf)
- results/figures/sweep_sampling_effect_fraction.(png|pdf)
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_figure(fig, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_error_gap_histogram(df_raw: pd.DataFrame, fig_dir: Path) -> None:
    """Match original run_sweep style for gap histogram."""
    experiments = sorted(df_raw["experiment"].dropna().unique())
    if len(experiments) == 0:
        raise ValueError("No experiment labels found in raw CSV.")

    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 4))
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        data = df_raw[df_raw["experiment"] == exp]["mean_gap"].dropna()
        ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No improvement")
        ax.set_xlabel("Mean Gap (Warm - Cold Error)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{exp.capitalize()} Experiment")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, fig_dir / "sweep_error_gap_hist")

def plot_fraction_warm_better(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    """Match original run_sweep style for fraction plot."""
    experiments = sorted(df_summary["experiment"].dropna().unique())
    if len(experiments) == 0:
        raise ValueError("No experiment labels found in summary CSV.")

    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 4))
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        subset = df_summary[df_summary["experiment"] == exp]
        p_warm_values = subset["p_warm"].unique()
        fractions = []
        for p_w in sorted(p_warm_values):
            mask = subset["p_warm"] == p_w
            frac = subset[mask]["fraction_warm_better"].mean()
            fractions.append(frac)

        ax.plot(sorted(p_warm_values), fractions, "o-", linewidth=2, markersize=8)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% tie")
        ax.set_xlabel("p_warm (warm-start oversampling)")
        ax.set_ylabel("Fraction Runs: Warm Better")
        ax.set_title(f"{exp.capitalize()} Experiment")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_figure(fig, fig_dir / "sweep_fraction_warm_better")


def plot_sampling_effect_mean_gap(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    """Visualize p_cold and p_warm effects on mean gap."""
    experiments = sorted(df_summary["experiment"].dropna().unique())
    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 4))
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        subset = df_summary[df_summary["experiment"] == exp]
        grouped = (
            subset.groupby(["p_cold", "p_warm"], as_index=False)["mean_gap_mean"]
            .mean()
            .sort_values(["p_cold", "p_warm"])
        )
        for p_cold in sorted(grouped["p_cold"].unique()):
            g = grouped[grouped["p_cold"] == p_cold]
            ax.plot(g["p_warm"], g["mean_gap_mean"], "o-", linewidth=2, markersize=6, label=f"p_cold={int(p_cold)}")

        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="No improvement")
        ax.set_xlabel("p_warm")
        ax.set_ylabel("Mean of mean_gap")
        ax.set_title(f"{exp.capitalize()} Experiment")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, fig_dir / "sweep_sampling_effect_mean_gap")


def plot_sampling_effect_fraction(df_summary: pd.DataFrame, fig_dir: Path) -> None:
    """Visualize p_cold and p_warm effects on fraction warm better."""
    experiments = sorted(df_summary["experiment"].dropna().unique())
    fig, axes = plt.subplots(1, len(experiments), figsize=(5 * len(experiments), 4))
    if len(experiments) == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        subset = df_summary[df_summary["experiment"] == exp]
        grouped = (
            subset.groupby(["p_cold", "p_warm"], as_index=False)["fraction_warm_better"]
            .mean()
            .sort_values(["p_cold", "p_warm"])
        )
        for p_cold in sorted(grouped["p_cold"].unique()):
            g = grouped[grouped["p_cold"] == p_cold]
            ax.plot(g["p_warm"], g["fraction_warm_better"], "o-", linewidth=2, markersize=6, label=f"p_cold={int(p_cold)}")

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50% tie")
        ax.set_xlabel("p_warm")
        ax.set_ylabel("Mean Fraction Warm Better")
        ax.set_title(f"{exp.capitalize()} Experiment")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, fig_dir / "sweep_sampling_effect_fraction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot sweep outputs from CSV files")
    parser.add_argument("--raw", type=Path, default=Path("results/sweep_raw.csv"), help="Path to raw sweep CSV")
    parser.add_argument("--summary", type=Path, default=Path("results/sweep_summary.csv"), help="Path to summary sweep CSV")
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"), help="Directory for output figures")
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.raw.exists():
        raise FileNotFoundError(f"Raw CSV not found: {args.raw}")
    if not args.summary.exists():
        raise FileNotFoundError(f"Summary CSV not found: {args.summary}")

    df_raw = pd.read_csv(args.raw)
    df_summary = pd.read_csv(args.summary)

    if df_raw.empty:
        raise ValueError("Raw CSV is empty.")
    if df_summary.empty:
        raise ValueError("Summary CSV is empty.")

    plot_error_gap_histogram(df_raw, args.fig_dir)
    plot_fraction_warm_better(df_summary, args.fig_dir)
    plot_sampling_effect_mean_gap(df_summary, args.fig_dir)
    plot_sampling_effect_fraction(df_summary, args.fig_dir)

    print(f"Replotted figures to: {args.fig_dir}")


if __name__ == "__main__":
    main()
