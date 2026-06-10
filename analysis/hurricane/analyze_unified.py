"""Analyze the unified Isabel adaptive sweep: warm vs cold, across eps.

Input is results/hurricane/unified/isabel_all.csv produced by
scripts/run_isabel_sweep.py, which concatenates the per-(var, eps, mode)
CSVs and adds `mode` ("warm" | "cold") and `eps` columns.

Outputs (under --fig-dir, default results/hurricane/figures_unified/):
  - unified_summary.csv          per (var, eps, mode) aggregates
  - warm_vs_cold_deltas.csv      per (var, eps) warm-vs-cold deltas (t >= 2)
  - wc_<var>_eps<eps>.png        2x3 time-series overlay, warm vs cold
  - metrics_vs_eps.png           2x3 aggregate metrics vs eps, all vars/modes
  - heatmap_summary.png          var x eps heatmaps per mode (k*, ratio, viol, time)
  - heatmap_warm_advantage.png   var x eps warm-vs-cold deltas
  - heatmap_timesteps_eps<e>.png timestep x var heatmaps (k*, log viol) per mode

Also asserts the tau guarantee (combined_max_elem_error <= tau) on every row.

Usage:
    python analysis/hurricane/analyze_unified.py
    python analysis/hurricane/analyze_unified.py --input <csv> --fig-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = REPO_ROOT / "results" / "hurricane" / "unified" / "isabel_all.csv"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "hurricane" / "figures_unified"

MODE_STYLE = {
    "warm": dict(color="tab:blue", marker="o", ls="-"),
    "cold": dict(color="tab:red",  marker="s", ls="--"),
}

# (column, axis label, log-scale)
TS_PANELS = [
    ("k_star",                 "Adaptive rank $k^*$",      False),
    ("s0_violations_at_kstar", "Violations (sparse $S$)",  True),
    ("stage1_time",            "Selection time (s)",       False),
    ("compression_ratio",      "Compression ratio (x)",    False),
    ("combined_psnr",          "Combined PSNR (dB)",       False),
    ("total_compressed_bytes", "Total storage (MB)",       False),
]


def load(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}\n"
                         "Run scripts/run_isabel_sweep.py first.")
    df = pd.read_csv(input_path)
    for col in ("mode", "eps"):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' missing — input must be the "
                             "concatenated isabel_all.csv (with mode/eps).")
    df = df.sort_values(["var", "eps", "mode", "timestep"]).reset_index(drop=True)
    df["total_mb"] = df["total_compressed_bytes"] / 1e6
    return df


def check_tau(df: pd.DataFrame) -> None:
    bad = df[df["combined_max_elem_error"] > df["tau"] * 1.0001]
    if len(bad):
        print(f"!! WARNING: {len(bad)} row(s) exceed the tau guarantee:")
        print(bad[["var", "eps", "mode", "timestep", "tau",
                   "combined_max_elem_error"]].to_string(index=False))
    else:
        print("[ok] tau guarantee holds for all rows "
              "(combined_max_elem_error <= tau).")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
AGG = {
    "k_star": "mean",
    "s0_violations_at_kstar": "mean",
    "stage1_time": "mean",
    "compression_ratio": "mean",
    "combined_psnr": "mean",
    "warm_psnr": "mean",
    "total_compressed_bytes": "sum",
    "k_expanded": "sum",
}


def summary_table(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    s = (df.groupby(["var", "eps", "mode"]).agg(AGG)
           .rename(columns={"k_star": "mean_k_star",
                            "s0_violations_at_kstar": "mean_violations",
                            "stage1_time": "mean_time_s",
                            "compression_ratio": "mean_ratio",
                            "combined_psnr": "mean_combined_psnr",
                            "warm_psnr": "mean_rank_only_psnr",
                            "total_compressed_bytes": "total_bytes",
                            "k_expanded": "n_expanded"})
           .reset_index())
    out = fig_dir / "unified_summary.csv"
    s.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 220,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print("\n=== Per (var, eps, mode) summary ===")
        print(s.to_string(index=False))
    print(f"\nSummary -> {out}")
    return s


def delta_table(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame | None:
    """Warm-vs-cold deltas per (var, eps), paired by timestep, t >= 2 only
    (t=1 is bootstrap in both arms and identical by construction)."""
    if df["mode"].nunique() < 2:
        print("\n(only one mode present — skipping warm-vs-cold delta table)")
        return None
    sub = df[df["timestep"] >= 2]
    keys = ["var", "eps", "timestep"]
    cols = ["k_star", "s0_violations_at_kstar", "stage1_time",
            "total_compressed_bytes", "combined_psnr"]
    w = sub[sub["mode"] == "warm"].set_index(keys)[cols]
    c = sub[sub["mode"] == "cold"].set_index(keys)[cols]
    j = w.join(c, lsuffix="_warm", rsuffix="_cold", how="inner")
    if j.empty:
        print("\n(no paired warm/cold rows — skipping delta table)")
        return None

    j = j.reset_index()
    j["psnr_delta"] = j["combined_psnr_warm"] - j["combined_psnr_cold"]
    j["same_k"] = (j["k_star_warm"] == j["k_star_cold"]).astype(float)
    sums = j.groupby(["var", "eps"]).sum(numeric_only=True)
    means = j.groupby(["var", "eps"]).mean(numeric_only=True)
    d = pd.DataFrame({
        "speedup_cold_over_warm":
            sums["stage1_time_cold"] / sums["stage1_time_warm"],
        "bytes_saved_pct":
            100 * (1 - sums["total_compressed_bytes_warm"]
                   / sums["total_compressed_bytes_cold"]),
        "violation_reduction_pct":
            100 * (1 - sums["s0_violations_at_kstar_warm"]
                   / sums["s0_violations_at_kstar_cold"].clip(lower=1)),
        "psnr_delta_db": means["psnr_delta"],
        "mean_k_warm": means["k_star_warm"],
        "mean_k_cold": means["k_star_cold"],
        "same_k_frac": means["same_k"],
    }).reset_index()
    out = fig_dir / "warm_vs_cold_deltas.csv"
    d.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 220,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print("\n=== Warm vs cold (paired, t >= 2) ===")
        print(d.to_string(index=False))
    print(f"\nDeltas -> {out}")
    return d


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_timeseries(df: pd.DataFrame, fig_dir: Path) -> None:
    for (var, eps), sub in df.groupby(["var", "eps"]):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for ax, (col, label, logy) in zip(axes.ravel(), TS_PANELS):
            for mode, ms in MODE_STYLE.items():
                mv = sub[sub["mode"] == mode]
                if mv.empty:
                    continue
                y = mv["total_mb"] if col == "total_compressed_bytes" else mv[col]
                ax.plot(mv["timestep"], y, ms=4, alpha=0.8, label=mode, **ms)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.suptitle(f"{var}  ($\\epsilon$ = {eps:g}, $\\tau$ per-timestep)",
                     fontsize=13)
        fig.tight_layout()
        out = fig_dir / f"wc_{var}_eps{eps:g}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {out}")


def plot_vs_eps(df: pd.DataFrame, fig_dir: Path) -> None:
    """Aggregate (mean over timesteps) metrics vs eps; one line per var x mode
    (warm solid, cold dashed, same colour per var)."""
    agg = (df.groupby(["var", "eps", "mode"])
             .agg({c: "mean" for c, _, _ in TS_PANELS})
             .reset_index())
    var_colors = {v: f"C{i}" for i, v in enumerate(sorted(agg["var"].unique()))}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (col, label, logy) in zip(axes.ravel(), TS_PANELS):
        for var in sorted(agg["var"].unique()):
            for mode in ("warm", "cold"):
                mv = agg[(agg["var"] == var) & (agg["mode"] == mode)]
                if mv.empty:
                    continue
                mv = mv.sort_values("eps")
                y = mv[col] / 1e6 if col == "total_compressed_bytes" else mv[col]
                ax.plot(mv["eps"], y, marker="o" if mode == "warm" else "s",
                        ls="-" if mode == "warm" else "--",
                        color=var_colors[var], ms=4, alpha=0.8,
                        label=f"{var} ({mode})")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.invert_xaxis()  # tighter tolerance to the right
        ax.set_xlabel(r"$\epsilon$")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Isabel featured variables: mean metrics vs tolerance "
                 "(solid = warm, dashed = cold)", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "metrics_vs_eps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def _imshow(ax, grid: pd.DataFrame, title: str, cmap: str, log: bool, fmt: str):
    M = grid.values.astype(float)
    norm = None
    if log and np.any(M > 0):
        norm = matplotlib.colors.LogNorm(vmin=max(np.nanmin(M[M > 0]), 1),
                                         vmax=np.nanmax(M))
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"{e:g}" for e in grid.columns], fontsize=7)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize=7)
    ax.set_xlabel(r"$\epsilon$", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if log else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    panels = [
        ("k_star", "mean $k^*$", "viridis", False, "{:.0f}"),
        ("compression_ratio", "mean ratio (x)", "YlGn", False, "{:.1f}"),
        ("s0_violations_at_kstar", "mean violations (log)", "magma", True, "{:.2g}"),
        ("stage1_time", "mean time (s)", "cividis", False, "{:.2f}"),
    ]
    modes = [m for m in ("warm", "cold") if m in df["mode"].unique()]
    eps_vals = sorted(df["eps"].unique(), reverse=True)
    variables = sorted(df["var"].unique())

    # --- per-mode summary heatmaps -------------------------------------
    fig, axes = plt.subplots(len(modes), len(panels),
                             figsize=(4.2 * len(panels), 2.6 * len(modes)),
                             squeeze=False)
    for r, mode in enumerate(modes):
        sub = df[df["mode"] == mode]
        for c, (col, title, cmap, log, fmt) in enumerate(panels):
            grid = (sub.pivot_table(index="var", columns="eps", values=col)
                       .reindex(index=variables, columns=eps_vals))
            _imshow(axes[r][c], grid, f"{mode}: {title}", cmap, log, fmt)
    fig.suptitle("Isabel: variable x $\\epsilon$ summary (mean over timesteps)",
                 fontsize=12)
    fig.tight_layout()
    out = fig_dir / "heatmap_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")

    # --- warm-advantage heatmaps (needs both modes, t >= 2) -------------
    if len(modes) == 2:
        sub = df[df["timestep"] >= 2]
        keys = ["var", "eps", "timestep"]
        w = sub[sub["mode"] == "warm"].set_index(keys)
        c = sub[sub["mode"] == "cold"].set_index(keys)
        j = w.join(c, lsuffix="_w", rsuffix="_c", how="inner").reset_index()
        if not j.empty:
            j["psnr_delta"] = j["combined_psnr_w"] - j["combined_psnr_c"]
            sums = j.groupby(["var", "eps"]).sum(numeric_only=True)
            means = j.groupby(["var", "eps"]).mean(numeric_only=True)
            grids = [
                ((sums["stage1_time_c"] / sums["stage1_time_w"]).unstack(),
                 "speedup (cold/warm time)", "YlGn", "{:.2f}"),
                ((100 * (1 - sums["total_compressed_bytes_w"]
                         / sums["total_compressed_bytes_c"])).unstack(),
                 "bytes saved by warm (%)", "RdYlGn", "{:.1f}"),
                (means["psnr_delta"].unstack(),
                 "PSNR delta (warm - cold, dB)", "RdYlGn", "{:.2f}"),
            ]
            fig, axes = plt.subplots(1, len(grids),
                                     figsize=(4.6 * len(grids), 2.8),
                                     squeeze=False)
            for ax, (grid, title, cmap, fmt) in zip(axes[0], grids):
                grid = grid.reindex(index=variables, columns=eps_vals)
                _imshow(ax, grid, title, cmap, False, fmt)
            fig.suptitle("Warm-start advantage (paired timesteps, t >= 2)",
                         fontsize=12)
            fig.tight_layout()
            out = fig_dir / "heatmap_warm_advantage.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  -> {out}")

    # --- timestep x variable heatmaps, one figure per eps ---------------
    ts_panels = [("k_star", "$k^*$", "viridis", False),
                 ("s0_violations_at_kstar", "violations (log)", "magma", True)]
    for eps in eps_vals:
        sub_e = df[df["eps"] == eps]
        fig, axes = plt.subplots(len(modes), len(ts_panels),
                                 figsize=(13, 1.1 * len(variables) * len(modes) + 2),
                                 squeeze=False)
        for r, mode in enumerate(modes):
            sub = sub_e[sub_e["mode"] == mode]
            for c, (col, title, cmap, log) in enumerate(ts_panels):
                grid = (sub.pivot_table(index="var", columns="timestep", values=col)
                           .reindex(index=variables))
                M = grid.values.astype(float)
                norm = None
                if log and np.any(M > 0):
                    norm = matplotlib.colors.LogNorm(
                        vmin=max(np.nanmin(M[M > 0]), 1), vmax=np.nanmax(M))
                ax = axes[r][c]
                im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
                ax.set_yticks(range(len(variables)))
                ax.set_yticklabels(variables, fontsize=7)
                ax.set_xlabel("Timestep", fontsize=8)
                ax.set_title(f"{mode}: {title}", fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.suptitle(f"Isabel: timestep x variable ($\\epsilon$ = {eps:g})",
                     fontsize=12)
        fig.tight_layout()
        out = fig_dir / f"heatmap_timesteps_eps{eps:g}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.input)
    check_tau(df)
    summary_table(df, args.fig_dir)
    delta_table(df, args.fig_dir)
    if not args.no_plots:
        print("\nGenerating figures ...")
        plot_timeseries(df, args.fig_dir)
        plot_vs_eps(df, args.fig_dir)
        plot_heatmaps(df, args.fig_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
