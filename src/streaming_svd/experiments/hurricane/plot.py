"""Stage 3 — Hurricane experiment figure generation.

Reads the raw CSV files and summary CSV produced by Stages 1 & 2, and
generates a comprehensive set of publication-quality figures.  No SVD
computation is performed here — all input is tabular.

Output layout::

    results/hurricane/figures/
        per_variable/
            {VAR}_fro_error.png          — cold vs warm vs optimal (primary accuracy)
            {VAR}_spec_error.png         — spectral error cold vs warm
            {VAR}_runtime.png            — wall-clock time + speedup ratio
            {VAR}_subspace_drift.png     — Frobenius drift + fraction-aligned
            {VAR}_cold_vs_warm_subspace.png — output subspace agreement + prior quality
        timing_breakdown/               — generated separately (--timing-breakdown flag)
            {VAR}_timing_breakdown.png
        cross_variable/
            all_vars_fro_error_cold.png
            all_vars_fro_error_warm.png
            all_vars_fro_error_gap.png
            overhead_ratio_bar.png
            all_vars_subspace_stability.png
            variable_ranking_bar.png
            timing_speedup_bar.png
            scatter_stability_vs_benefit.png
            scatter_speedup_vs_accuracy.png

CLI usage::

    python -m streaming_svd.experiments.hurricane.plot --help

    python -m streaming_svd.experiments.hurricane.plot \\
        --raw-dir results/hurricane/raw \\
        --summary results/hurricane/hurricane_summary.csv \\
        --fig-dir results/hurricane/figures

    # Also generate timing breakdown stacked bars
    python -m streaming_svd.experiments.hurricane.plot \\
        --raw-dir results/hurricane/raw \\
        --summary results/hurricane/hurricane_summary.csv \\
        --fig-dir results/hurricane/figures \\
        --timing-breakdown
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from streaming_svd.experiments.hurricane.analyze import load_raw_results

# ---------------------------------------------------------------------------
# Colour palette and style constants
# ---------------------------------------------------------------------------

_COLD_COLOR = "#2171b5"
_WARM_COLOR = "#d94801"
_OPT_COLOR  = "#252525"
_GAP_ZERO_COLOR = "#cb181d"
_DRIFT_COLOR = "#6a3d9a"
_ALIGN_COLOR = "#2ca25f"
_PREV_QUAL_COLOR = "#238b45"
_CW_SUB_COLOR = "#cb181d"

_VAR_COLORS = [matplotlib.colormaps["tab20"](i / 20) for i in range(20)]

_LABEL_COLD = "Cold-start rSVD"
_LABEL_WARM = "Warm-start rSVD"
_LABEL_OPT  = "Optimal rank-k"

_FIG_W = 10
_FIG_H = 6


def _save_figure(fig: plt.Figure, path: Path, fmts: Sequence[str], dpi: int) -> None:
    for fmt in fmts:
        out = path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _apply_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)


def _k_val(df: pd.DataFrame) -> str:
    try:
        return str(int(df["k"].dropna().iloc[0]))
    except (IndexError, KeyError):
        return "?"


# ---------------------------------------------------------------------------
# Per-variable plots
# ---------------------------------------------------------------------------

def plot_fro_error_timeseries(
    df_var: pd.DataFrame, var: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Frobenius error vs timestep — cold, warm, optimal."""
    df = df_var.sort_values("timestep")
    ts = df["timestep"].values

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(ts, df["cold_fro_error"], "o-", color=_COLD_COLOR, lw=2.2, ms=6, label=_LABEL_COLD)
    ax.plot(ts, df["warm_fro_error"], "s-", color=_WARM_COLOR, lw=2.2, ms=6, label=_LABEL_WARM)

    if "optimal_fro_error" in df.columns and df["optimal_fro_error"].notna().any():
        ax.plot(ts, df["optimal_fro_error"], "^--", color=_OPT_COLOR, lw=1.8, ms=6,
                label=_LABEL_OPT, alpha=0.85)

    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Relative Frobenius Error", fontsize=12)
    ax.set_title(f"{var}: Frobenius Error — Cold vs Warm vs Optimal  (k={_k_val(df)})",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_spec_error_timeseries(
    df_var: pd.DataFrame, var: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Spectral error vs timestep — cold and warm."""
    df = df_var.sort_values("timestep")
    ts = df["timestep"].values

    if "cold_spec_error" not in df.columns or df["cold_spec_error"].isna().all():
        return

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(ts, df["cold_spec_error"], "o-", color=_COLD_COLOR, lw=2.2, ms=6, label=_LABEL_COLD)
    ax.plot(ts, df["warm_spec_error"], "s-", color=_WARM_COLOR, lw=2.2, ms=6, label=_LABEL_WARM)

    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Est. Relative Spectral Error", fontsize=12)
    ax.set_title(f"{var}: Spectral Error — Cold vs Warm", fontsize=13, fontweight="bold")
    _apply_grid(ax)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_runtime_timeseries(
    df_var: pd.DataFrame, var: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Wall-clock runtime per timestep — cold vs warm, with speedup ratio on secondary axis.

    Timing breakdown (stacked bar) is a separate figure; use --timing-breakdown to generate it.
    """
    df = df_var.sort_values("timestep")
    ts = df["timestep"].values

    if "cold_time_total" not in df.columns or df["cold_time_total"].isna().all():
        return

    cold_ms = df["cold_time_total"] * 1000
    warm_ms = df["warm_time_total"] * 1000

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(ts, cold_ms, "o-", color=_COLD_COLOR, lw=2.2, ms=6, label=_LABEL_COLD)
    ax.plot(ts, warm_ms, "s-", color=_WARM_COLOR, lw=2.2, ms=6, label=_LABEL_WARM)

    # Speedup ratio on secondary y-axis (only for warm-active timesteps)
    if "time_speedup_ratio" in df.columns:
        ax2 = ax.twinx()
        speedup = df["time_speedup_ratio"]
        mask = speedup.notna()
        ax2.plot(ts[mask], speedup[mask], "D--", color="grey", lw=1.4, ms=5,
                 alpha=0.7, label="Speedup (cold/warm)")
        ax2.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.5)
        ax2.set_ylabel("Speedup  (cold / warm)", fontsize=10, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")
        ax2.set_ylim(bottom=0)
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
    ax.set_title(f"{var}: Runtime — Cold vs Warm", fontsize=13, fontweight="bold")
    _apply_grid(ax)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_timing_breakdown(
    df_var: pd.DataFrame, var: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Stacked-bar timing breakdown (warm phases) — reference figure.

    Generated only when --timing-breakdown is passed to the CLI.
    """
    df = df_var.sort_values("timestep")
    ts = df["timestep"].values

    breakdown_cols = [
        ("warm_time_warm_proj",    "warm_proj (G=Aᵀ·U)",   "#c6dbef"),
        ("warm_time_warm_matmul",  "warm_matmul (A·G)",    "#6baed6"),
        ("warm_time_omega_gen",    "omega_gen",             "#9ecae1"),
        ("warm_time_random_matmul","random_matmul (A·Ω)",  "#2171b5"),
        ("warm_time_qr",           "QR",                   "#41ab5d"),
        ("warm_time_projection",   "projection (Qᵀ·A)",    "#a1d99b"),
        ("warm_time_small_svd",    "small SVD",            "#fdae6b"),
        ("warm_time_lift",         "lift (Q·Û)",           "#f16913"),
    ]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(_FIG_W, 10), sharex=True)

    cold_ms = df["cold_time_total"] * 1000 if "cold_time_total" in df.columns else None
    warm_ms = df["warm_time_total"] * 1000 if "warm_time_total" in df.columns else None

    if cold_ms is not None:
        ax_top.plot(ts, cold_ms, "o-", color=_COLD_COLOR, lw=2.0, ms=5, label=_LABEL_COLD)
    if warm_ms is not None:
        ax_top.plot(ts, warm_ms, "s-", color=_WARM_COLOR, lw=2.0, ms=5, label=_LABEL_WARM)
    ax_top.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax_top.set_title(f"{var}: Warm-Start Timing Breakdown", fontsize=13, fontweight="bold")
    _apply_grid(ax_top)
    ax_top.legend(fontsize=9)

    bottom = np.zeros(len(ts))
    for col, label, color in breakdown_cols:
        if col in df.columns:
            vals = df[col].fillna(0).values * 1000
            ax_bot.bar(ts, vals, bottom=bottom, label=label, color=color, width=0.8)
            bottom += vals

    ax_bot.set_xlabel("Timestep (hour)", fontsize=11)
    ax_bot.set_ylabel("Warm-start time (ms, stacked)", fontsize=11)
    _apply_grid(ax_bot)
    ax_bot.legend(fontsize=8, loc="upper right", ncol=2)
    fig.subplots_adjust(hspace=0.3)
    _save_figure(fig, out_stem, fmts, dpi)


def plot_subspace_drift_timeseries(
    df_var: pd.DataFrame, var: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Frobenius subspace drift + fraction-of-energy aligned vs timestep.

    Primary panel: warm_drift_fro (Frobenius ||sin Θ||_F, range [0, sqrt(k)]).
    Secondary axis: frac_aligned = 1 - drift_fro² / k  (0 = orthogonal, 1 = identical).

    The spectral (max-angle) metric is intentionally omitted — it saturates to 1
    for any two rank-k subspaces with k > 1 and is not informative.
    """
    df = df_var.sort_values("timestep")
    df_d = df.dropna(subset=["warm_drift_fro"])

    if df_d.empty:
        return

    ts = df_d["timestep"].values
    drift_fro = df_d["warm_drift_fro"].values

    k = int(df["k"].dropna().iloc[0]) if "k" in df.columns and df["k"].notna().any() else 20
    frac_aligned = np.clip(1.0 - drift_fro ** 2 / k, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.plot(ts, drift_fro, "v-", color=_DRIFT_COLOR, lw=2.2, ms=6,
            label=f"||sin Θ||_F  (Frobenius drift, max={np.sqrt(k):.2f})")
    ax.axhline(np.sqrt(k), color=_DRIFT_COLOR, lw=0.8, ls=":", alpha=0.4,
               label=f"max possible = √k = {np.sqrt(k):.2f}")
    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Frobenius Subspace Drift  ||sin Θ||_F", fontsize=12, color=_DRIFT_COLOR)
    ax.tick_params(axis="y", labelcolor=_DRIFT_COLOR)
    ax.set_ylim(bottom=0, top=np.sqrt(k) * 1.1)

    ax2 = ax.twinx()
    ax2.plot(ts, frac_aligned, "D--", color=_ALIGN_COLOR, lw=1.8, ms=5, alpha=0.85,
             label="Fraction of energy aligned  (1 − drift²/k)")
    ax2.set_ylabel("Fraction of energy aligned", fontsize=12, color=_ALIGN_COLOR)
    ax2.tick_params(axis="y", labelcolor=_ALIGN_COLOR)
    ax2.set_ylim(0, 1.05)

    ax.set_title(f"{var}: Warm-Start Subspace Drift  (k={k})", fontsize=13, fontweight="bold")
    _apply_grid(ax)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_cold_vs_warm_subspace(
    df_var: pd.DataFrame, var: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Subspace agreement between cold and warm — Frobenius metrics only.

    Two lines:
    * cold_vs_warm_subspace_fro — ||sin Θ||_F between cold and warm *output* at each step.
      Small = both methods found similar subspaces.
    * warm_prev_quality_fro — ||sin Θ||_F between warm *prior* (U from t-1) and cold output
      at t.  Small = prior was already a good initialisation for this step.

    Note: spectral (max-angle) variants are not plotted — they saturate to ≈1 for rank > 1
    and carry no useful information.
    """
    df = df_var.sort_values("timestep")
    ts = df["timestep"].values

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    plotted = False

    if "cold_vs_warm_subspace_fro" in df.columns and df["cold_vs_warm_subspace_fro"].notna().any():
        ax.plot(ts, df["cold_vs_warm_subspace_fro"], "o-", color=_CW_SUB_COLOR,
                lw=2.2, ms=6, label="cold vs warm output  ||sin Θ||_F")
        plotted = True

    if "warm_prev_quality_fro" in df.columns:
        df_q = df.dropna(subset=["warm_prev_quality_fro"])
        if len(df_q) > 0:
            ax.plot(df_q["timestep"], df_q["warm_prev_quality_fro"], "s--",
                    color=_PREV_QUAL_COLOR, lw=2.0, ms=6,
                    label="warm prior quality  ||sin Θ||_F  (U_{t-1} vs cold at t)")
            plotted = True

    if not plotted:
        plt.close(fig)
        warnings.warn(f"{var}: no Frobenius subspace columns found; skipping subspace plot.")
        return

    k = _k_val(df)
    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Subspace Distance  ||sin Θ||_F", fontsize=12)
    ax.set_title(f"{var}: Cold vs Warm Subspace Agreement  (k={k})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    _apply_grid(ax)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


# ---------------------------------------------------------------------------
# Cross-variable comparison plots
# ---------------------------------------------------------------------------

def _var_color(i: int):
    return _VAR_COLORS[i % len(_VAR_COLORS)]


def plot_all_vars_fro_error(
    df_raw: pd.DataFrame, method: str, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """All variables' Frobenius error over time — one line per variable."""
    assert method in ("cold", "warm")
    col = f"{method}_fro_error"
    title_method = "Cold-Start" if method == "cold" else "Warm-Start"

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, var in enumerate(sorted(df_raw["var"].unique())):
        df_v = df_raw[df_raw["var"] == var].sort_values("timestep")
        ax.plot(df_v["timestep"], df_v[col], "-", lw=1.8,
                color=_var_color(i), label=var, alpha=0.9)

    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Relative Frobenius Error", fontsize=12)
    ax.set_title(f"All Variables: {title_method} Frobenius Error over Time",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_fro_error_gap_timeseries(
    df_raw: pd.DataFrame, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Frobenius error gap (warm − cold) vs timestep for all variables."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, var in enumerate(sorted(df_raw["var"].unique())):
        df_v = df_raw[df_raw["var"] == var].sort_values("timestep")
        if "fro_error_gap" in df_v.columns:
            ax.plot(df_v["timestep"], df_v["fro_error_gap"], "-", lw=1.8,
                    color=_var_color(i), label=var, alpha=0.9)

    ax.axhline(0, color=_GAP_ZERO_COLOR, lw=2.0, ls="--", label="zero (break-even)")
    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Fro. Error Gap  (warm − cold)  ↓ better for warm", fontsize=11)
    ax.set_title("All Variables: Frobenius Error Gap — Warm minus Cold",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_overhead_ratio_bar(
    df_summary: pd.DataFrame, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Grouped bar: randomisation overhead above optimal baseline."""
    df = df_summary.dropna(subset=["cold_fro_overhead_mean"]).sort_values(
        "cold_fro_overhead_mean", ascending=False)
    if df.empty:
        return

    vars_ = df["var"].tolist()
    x = np.arange(len(vars_))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(vars_) * 0.8 + 2), 6))
    ax.bar(x - width / 2, df["cold_fro_overhead_mean"], width,
           color=_COLD_COLOR, alpha=0.85, label="Cold overhead")
    ax.bar(x + width / 2, df["warm_fro_overhead_mean"], width,
           color=_WARM_COLOR, alpha=0.85, label="Warm overhead")

    ax.set_xticks(x)
    ax.set_xticklabels(vars_, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean overhead above optimal  (relative Fro. error)", fontsize=11)
    ax.set_title("Randomisation Overhead above Optimal Rank-k Baseline",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_all_vars_subspace_stability(
    df_raw: pd.DataFrame, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Frobenius subspace drift over time for all variables.

    Uses warm_drift_fro (Frobenius sin-theta), not the spectral variant which
    saturates to ~1 and is uninformative for rank > 1.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    k = 20  # default; will be overridden per-df if available
    for i, var in enumerate(sorted(df_raw["var"].unique())):
        df_v = df_raw[df_raw["var"] == var].sort_values("timestep")
        if "warm_drift_fro" not in df_v.columns:
            continue
        df_d = df_v.dropna(subset=["warm_drift_fro"])
        if df_d.empty:
            continue
        if "k" in df_v.columns and df_v["k"].notna().any():
            k = int(df_v["k"].dropna().iloc[0])
        ax.plot(df_d["timestep"], df_d["warm_drift_fro"], "-", lw=1.8,
                color=_var_color(i), label=var, alpha=0.9)

    ax.axhline(np.sqrt(k), color="grey", lw=1.0, ls=":", alpha=0.5,
               label=f"max = √k = {np.sqrt(k):.2f}")
    ax.set_xlabel("Timestep (hour)", fontsize=12)
    ax.set_ylabel("Warm Subspace Drift  ||sin Θ||_F  (t-1 → t)", fontsize=11)
    ax.set_title("All Variables: Warm-Start Subspace Drift  (Frobenius)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(bottom=0)
    _apply_grid(ax)
    ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_variable_ranking_bar(
    df_summary: pd.DataFrame, out_stem: Path,
    metric: str = "fro_error_gap_mean",
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Horizontal bar: variables ranked by mean Frobenius error gap."""
    df = df_summary.dropna(subset=[metric]).sort_values(metric, ascending=True)
    if df.empty:
        return

    vars_ = df["var"].tolist()
    vals  = df[metric].values
    std_col = metric.replace("_mean", "_std")
    errs = df[std_col].values if std_col in df.columns else np.zeros_like(vals)
    colors = ["#41ab5d" if v < 0 else "#d7191c" for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(5, len(vars_) * 0.55 + 1.5)))
    y = np.arange(len(vars_))
    ax.barh(y, vals, xerr=errs, color=colors, alpha=0.85,
            error_kw={"elinewidth": 1.5, "capsize": 4})
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(vars_, fontsize=11)
    ax.set_xlabel("Mean Frobenius Error Gap  (warm − cold)  ←  warm better  |  cold better  →",
                  fontsize=10)
    ax.set_title("Variable Ranking by Warm-Start Accuracy Benefit\n"
                 "(green = warm better, red = cold better)",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_timing_speedup_bar(
    df_summary: pd.DataFrame, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Horizontal bar: mean timing speedup per variable."""
    df = df_summary.dropna(subset=["time_speedup_mean"]).sort_values(
        "time_speedup_mean", ascending=True)
    if df.empty:
        return

    vars_ = df["var"].tolist()
    vals  = df["time_speedup_mean"].values
    errs  = df["time_speedup_std"].fillna(0).values if "time_speedup_std" in df.columns else np.zeros_like(vals)
    frac  = df["fraction_warm_faster"].fillna(np.nan).values if "fraction_warm_faster" in df.columns else [np.nan] * len(vars_)
    colors = ["#41ab5d" if v >= 1 else "#d7191c" for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(5, len(vars_) * 0.55 + 1.5)))
    y = np.arange(len(vars_))
    ax.barh(y, vals, xerr=errs, color=colors, alpha=0.85,
            error_kw={"elinewidth": 1.5, "capsize": 4})
    ax.axvline(1.0, color="black", lw=1.5, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(vars_, fontsize=11)
    ax.set_xlabel("Timing Speedup  (cold / warm)  > 1 → warm faster", fontsize=11)
    ax.set_title("Warm-Start Timing Speedup per Variable", fontsize=13, fontweight="bold")

    for i, (f, v, e) in enumerate(zip(frac, vals, errs)):
        if not np.isnan(f):
            ax.text(max(v + e + 0.01, 0.05), i, f"{f*100:.0f}% faster",
                    va="center", fontsize=8)

    _apply_grid(ax)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_scatter_stability_vs_benefit(
    df_summary: pd.DataFrame, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Scatter: subspace stability (frac_aligned) vs warm-start accuracy benefit.

    X-axis: mean fraction of energy aligned between consecutive warm subspaces
            (1 = perfectly stable, 0 = completely rotates each step).
            Derived from warm_drift_fro_mean: frac_aligned = 1 - drift²/k.
    Y-axis: mean Frobenius error gap (warm − cold); negative = warm better.
    Point size: proportional to timing speedup.
    """
    needed = ["var", "frac_aligned_mean", "fro_error_gap_mean"]
    df = df_summary.dropna(subset=needed)
    if df.empty:
        warnings.warn("Not enough data for stability vs benefit scatter; skipping.")
        return

    # frac_aligned_mean is already computed correctly in analyze.py as
    # mean_t(1 - ||sinΘ||_F² / k), which differs from 1 - mean(||sinΘ||_F)² / k.
    frac_aligned = df["frac_aligned_mean"].values
    y     = df["fro_error_gap_mean"].values
    vars_ = df["var"].tolist()

    speedup = df["time_speedup_mean"].values if "time_speedup_mean" in df.columns else np.ones(len(df))
    speedup = np.nan_to_num(speedup, nan=1.0)
    sizes  = np.clip(speedup * 80, 40, 400)
    colors = ["#41ab5d" if v < 0 else "#d7191c" for v in y]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(frac_aligned, y, s=sizes, c=colors, alpha=0.85,
               edgecolors="black", linewidths=0.8)

    for xi, yi, var in zip(frac_aligned, y, vars_):
        ax.annotate(var, (xi, yi), textcoords="offset points", xytext=(6, 4),
                    fontsize=9, ha="left")

    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7, label="break-even")
    ax.axvline(np.median(frac_aligned), color="grey", lw=1.0, ls=":",
               label=f"median frac_aligned = {np.median(frac_aligned):.2f}")

    ax.set_xlabel("Mean Fraction of Energy Aligned  (1 − drift²/k)  →  more stable",
                  fontsize=11)
    ax.set_ylabel("Mean Fro. Error Gap  (warm − cold)  ↓ warm better", fontsize=11)
    ax.set_title("Subspace Stability vs Warm-Start Accuracy Benefit\n"
                 "(point size ∝ timing speedup; green = warm better)",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


def plot_scatter_speedup_vs_accuracy(
    df_summary: pd.DataFrame, out_stem: Path,
    fmts: Sequence[str] = ("png",), dpi: int = 150,
) -> None:
    """Scatter: timing speedup (x) vs accuracy improvement (y) per variable.

    Shows whether warm start achieves both benefits simultaneously.
    Upper-right quadrant = warm is both faster AND more accurate.
    """
    needed = ["var", "time_speedup_mean", "fro_error_gap_mean"]
    df = df_summary.dropna(subset=needed)
    if df.empty:
        return

    x     = df["time_speedup_mean"].values
    y     = df["fro_error_gap_mean"].values
    vars_ = df["var"].tolist()
    colors = ["#41ab5d" if (xi > 1 and yi < 0) else
              "#fdae6b" if (xi > 1 and yi >= 0) else "#d7191c"
              for xi, yi in zip(x, y)]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(x, y, s=120, c=colors, alpha=0.85, edgecolors="black", linewidths=0.8)

    for xi, yi, var in zip(x, y, vars_):
        ax.annotate(var, (xi, yi), textcoords="offset points", xytext=(6, 4),
                    fontsize=9, ha="left")

    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(1.0, color="black", lw=1.2, ls="--", alpha=0.7)

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(1.02, ylim[0] * 0.85, "Faster,\ncold more accurate",
            fontsize=8, color="#fdae6b", style="italic")
    ax.text(1.02, ylim[1] * 0.7, "Faster AND\nmore accurate",
            fontsize=8, color="#41ab5d", style="italic")

    ax.set_xlabel("Mean Timing Speedup  (cold / warm)  →  warm faster", fontsize=11)
    ax.set_ylabel("Mean Fro. Error Gap  (warm − cold)  ↓ warm better", fontsize=11)
    ax.set_title("Warm-Start: Timing Speedup vs Accuracy Benefit\n"
                 "(green = warm wins both; orange = faster but less accurate)",
                 fontsize=13, fontweight="bold")
    _apply_grid(ax)
    plt.tight_layout()
    _save_figure(fig, out_stem, fmts, dpi)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def plot_per_variable(
    raw_dir: Path,
    fig_dir: Path,
    variables: Optional[List[str]] = None,
    timing_breakdown: bool = False,
    fmts: Sequence[str] = ("png",),
    dpi: int = 150,
    verbose: bool = True,
) -> None:
    """Generate per-variable plots for each requested variable."""
    out_dir = Path(fig_dir) / "per_variable"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_raw_results(raw_dir, variables=variables)
    all_vars = sorted(df_all["var"].unique())

    for var in all_vars:
        if verbose:
            print(f"  {var}")
        df_v = df_all[df_all["var"] == var].copy()

        plot_fro_error_timeseries(df_v, var, out_dir / f"{var}_fro_error",
                                  fmts=fmts, dpi=dpi)
        plot_spec_error_timeseries(df_v, var, out_dir / f"{var}_spec_error",
                                   fmts=fmts, dpi=dpi)
        plot_runtime_timeseries(df_v, var, out_dir / f"{var}_runtime",
                                fmts=fmts, dpi=dpi)
        plot_subspace_drift_timeseries(df_v, var, out_dir / f"{var}_subspace_drift",
                                       fmts=fmts, dpi=dpi)
        plot_cold_vs_warm_subspace(df_v, var, out_dir / f"{var}_cold_vs_warm_subspace",
                                   fmts=fmts, dpi=dpi)

        if timing_breakdown:
            bd_dir = Path(fig_dir) / "timing_breakdown"
            bd_dir.mkdir(parents=True, exist_ok=True)
            plot_timing_breakdown(df_v, var, bd_dir / f"{var}_timing_breakdown",
                                  fmts=fmts, dpi=dpi)

    if verbose:
        print(f"  -> {out_dir.resolve()}")


def plot_cross_variable(
    raw_dir: Path,
    summary_path: Path,
    fig_dir: Path,
    variables: Optional[List[str]] = None,
    fmts: Sequence[str] = ("png",),
    dpi: int = 150,
    verbose: bool = True,
) -> None:
    """Generate all cross-variable comparison plots."""
    out_dir = Path(fig_dir) / "cross_variable"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_results(raw_dir, variables=variables)
    df_summary = pd.read_csv(summary_path)
    if variables is not None:
        df_summary = df_summary[df_summary["var"].isin(variables)]

    plots = [
        ("all_vars_fro_error_cold",
         lambda: plot_all_vars_fro_error(df_raw, "cold", out_dir / "all_vars_fro_error_cold",
                                         fmts=fmts, dpi=dpi)),
        ("all_vars_fro_error_warm",
         lambda: plot_all_vars_fro_error(df_raw, "warm", out_dir / "all_vars_fro_error_warm",
                                         fmts=fmts, dpi=dpi)),
        ("all_vars_fro_error_gap",
         lambda: plot_fro_error_gap_timeseries(df_raw, out_dir / "all_vars_fro_error_gap",
                                               fmts=fmts, dpi=dpi)),
        ("overhead_ratio_bar",
         lambda: plot_overhead_ratio_bar(df_summary, out_dir / "overhead_ratio_bar",
                                         fmts=fmts, dpi=dpi)),
        ("all_vars_subspace_stability",
         lambda: plot_all_vars_subspace_stability(df_raw, out_dir / "all_vars_subspace_stability",
                                                   fmts=fmts, dpi=dpi)),
        ("variable_ranking_bar",
         lambda: plot_variable_ranking_bar(df_summary, out_dir / "variable_ranking_bar",
                                           fmts=fmts, dpi=dpi)),
        ("timing_speedup_bar",
         lambda: plot_timing_speedup_bar(df_summary, out_dir / "timing_speedup_bar",
                                         fmts=fmts, dpi=dpi)),
        ("scatter_stability_vs_benefit",
         lambda: plot_scatter_stability_vs_benefit(df_summary,
                                                   out_dir / "scatter_stability_vs_benefit",
                                                   fmts=fmts, dpi=dpi)),
        ("scatter_speedup_vs_accuracy",
         lambda: plot_scatter_speedup_vs_accuracy(df_summary,
                                                  out_dir / "scatter_speedup_vs_accuracy",
                                                  fmts=fmts, dpi=dpi)),
    ]

    for name, fn in plots:
        if verbose:
            print(f"  {name}")
        try:
            fn()
        except Exception as exc:
            warnings.warn(f"  [WARN] {name} failed: {exc}")

    if verbose:
        print(f"  -> {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hurricane experiment — Stage 3: generate figures from CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=str, default="results/hurricane/raw",
                        help="Directory containing {VAR}_raw.csv files")
    parser.add_argument("--summary", type=str,
                        default="results/hurricane/hurricane_summary.csv",
                        help="Path to hurricane_summary.csv (output of Stage 2)")
    parser.add_argument("--fig-dir", type=str, default="results/hurricane/figures",
                        help="Root output directory for all figures")
    parser.add_argument("--vars", nargs="+", default=None, metavar="VAR",
                        help="Subset of variable names (default: all present in CSVs)")
    parser.add_argument("--per-variable-only", action="store_true",
                        help="Generate only per-variable plots")
    parser.add_argument("--cross-variable-only", action="store_true",
                        help="Generate only cross-variable comparison plots")
    parser.add_argument("--timing-breakdown", action="store_true",
                        help="Also generate timing breakdown stacked-bar figures "
                             "(saved to timing_breakdown/ subfolder)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fmt", nargs="+", default=["png"], metavar="FMT")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    verbose   = not args.quiet
    raw_dir   = Path(args.raw_dir)
    summary_path = Path(args.summary)
    fig_dir   = Path(args.fig_dir)
    do_per    = not args.cross_variable_only
    do_cross  = not args.per_variable_only

    if verbose:
        print(f"Figure output: {fig_dir.resolve()}  |  formats: {args.fmt}  |  dpi: {args.dpi}")

    if do_per:
        if verbose:
            print("\n--- Per-variable plots ---")
        plot_per_variable(
            raw_dir=raw_dir, fig_dir=fig_dir, variables=args.vars,
            timing_breakdown=args.timing_breakdown,
            fmts=args.fmt, dpi=args.dpi, verbose=verbose,
        )

    if do_cross:
        if not summary_path.exists():
            print(f"[WARN] Summary not found: {summary_path}\n"
                  "       Run analyze.py first.  Skipping cross-variable plots.")
        else:
            if verbose:
                print("\n--- Cross-variable plots ---")
            plot_cross_variable(
                raw_dir=raw_dir, summary_path=summary_path, fig_dir=fig_dir,
                variables=args.vars, fmts=args.fmt, dpi=args.dpi, verbose=verbose,
            )

    if verbose:
        print(f"\nDone. All figures in: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
