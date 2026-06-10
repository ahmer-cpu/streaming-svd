"""Analyze the static (single-snapshot) adaptive rSVD sweep on the SDRBench
datasets (NYX, Miranda).

Input is the concatenated results/static/static_all.csv produced by
scripts/run_static_sweep.py, which carries `dataset` and `eps` columns on top
of the standard adaptive schema.  Each row is one (dataset, variable, eps)
compression result.

Outputs:
  - A printed summary table (also written as static_summary.csv).
  - Figures under <fig-dir> (default results/static/figures/):
      * compression_ratio vs eps          (per variable, faceted by dataset)
      * combined_psnr vs eps              (per variable, faceted by dataset)
      * adaptive rank k* (and r*) vs eps  (per variable, faceted by dataset)
      * rate-distortion: ratio vs PSNR    (all variables, colored by dataset)
      * storage breakdown stacked bars    (per dataset, grouped by var x eps)

Usage:
    python analysis/static/analyze_static.py
    python analysis/static/analyze_static.py --input results/static/static_all.csv
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
DEFAULT_INPUT = REPO_ROOT / "results" / "static" / "static_all.csv"
DEFAULT_FIGDIR = REPO_ROOT / "results" / "static" / "figures"

SUMMARY_COLS = [
    "dataset", "var", "eps", "tau", "k_star", "r_star",
    "s0_violations_at_kstar", "r_star_violations",
    "compression_ratio", "combined_psnr", "combined_fro_error",
    "combined_max_elem_error", "stage2_skipped", "stage2_skip_reason",
]


def load(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}\n"
                         "Run scripts/run_static_sweep.py first.")
    df = pd.read_csv(input_path)
    df = df.sort_values(["dataset", "var", "eps"]).reset_index(drop=True)
    return df


def is_single_stage(df: pd.DataFrame) -> bool:
    """True for unified single-stage runs (L + S, no residual stage)."""
    if "stage2_skip_reason" in df.columns and \
            df["stage2_skip_reason"].astype(str).eq("single_stage").all():
        return True
    return "r_star" in df.columns and (df["r_star"] == 0).all() and \
           "stage2_rank_bytes" in df.columns and (df["stage2_rank_bytes"] == 0).all()


def print_summary(df: pd.DataFrame, fig_dir: Path, input_path: Path) -> None:
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    summary = df[cols].copy()

    # tau guarantee sanity check: combined max error must not exceed tau.
    if {"combined_max_elem_error", "tau"}.issubset(df.columns):
        violated = df[df["combined_max_elem_error"] > df["tau"] * 1.0001]
        if len(violated):
            print(f"!! WARNING: {len(violated)} row(s) exceed the tau guarantee "
                  "(combined_max_elem_error > tau):")
            print(violated[["dataset", "var", "eps", "tau",
                            "combined_max_elem_error"]].to_string(index=False))
        else:
            print("[ok] tau guarantee holds for all rows "
                  "(combined_max_elem_error <= tau).")

    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.4g}"):
        print("\n=== Static adaptive sweep summary ===")
        print(summary.to_string(index=False))

    # static_all.csv -> static_summary.csv; static_all_unified.csv -> static_summary_unified.csv
    out_csv = fig_dir.parent / input_path.name.replace("_all", "_summary")
    summary.to_csv(out_csv, index=False)
    print(f"\nSummary table -> {out_csv}")


def _facet_by_dataset(df, ycol, ylabel, title, out_path, logy=False, extra=None):
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = df[df["dataset"] == ds]
        for var in sorted(sub["var"].unique()):
            dv = sub[sub["var"] == var].sort_values("eps")
            ax.plot(dv["eps"], dv[ycol], "o-", ms=4, label=var)
            if extra and extra in sub.columns:
                ax.plot(dv["eps"], dv[extra], "s--", ms=3, alpha=0.5)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.invert_xaxis()  # tighter tolerance (smaller eps) to the right
        ax.set_xlabel(r"$\epsilon$ (value-range-relative tolerance)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ds}: {title}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_heatmaps(df: pd.DataFrame, fig_dir: Path) -> None:
    """Per-dataset variable x eps heatmaps (static analog of the supervisor's
    timestep x variable grids): adaptive rank k*, compression ratio, stage-1
    violation count (log), and stage-2 rank r*."""
    single_stage = is_single_stage(df)
    panels = [
        ("k_star", "Adaptive rank $k^*$", "viridis", False, "{:.0f}"),
        ("compression_ratio", "Compression ratio (x)", "YlGn", False, "{:.1f}"),
        ("s0_violations_at_kstar",
         "Sparse entries (log)" if single_stage else "Stage-1 violations (log)",
         "magma", True, "{:.2g}"),
        # Single-stage runs have no residual stage: show the search window
        # top (rank cap actually used) instead of the all-zero r*.
        ("k_search_hi", "Search window top $k_{hi}$", "cividis", False, "{:.0f}")
        if single_stage else
        ("r_star", "Stage-2 rank $r^*$", "cividis", False, "{:.0f}"),
    ]
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        eps_vals = sorted(sub["eps"].unique(), reverse=True)   # loose -> tight
        variables = sorted(sub["var"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(max(7, 1.6 * len(eps_vals) + 5),
                                                1.0 * len(variables) + 3))
        for ax, (col, title, cmap, logc, fmt) in zip(axes.ravel(), panels):
            if col not in sub.columns:
                ax.axis("off")
                continue
            grid = (sub.pivot_table(index="var", columns="eps", values=col)
                       .reindex(index=variables, columns=eps_vals))
            M = grid.values.astype(float)
            norm = matplotlib.colors.LogNorm(
                vmin=np.nanmax([np.nanmin(M[M > 0]) if np.any(M > 0) else 1, 1]),
                vmax=np.nanmax(M)) if logc and np.any(M > 0) else None
            im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm)
            ax.set_xticks(range(len(eps_vals)))
            ax.set_xticklabels([f"{e:g}" for e in eps_vals])
            ax.set_yticks(range(len(variables)))
            ax.set_yticklabels(variables, fontsize=7)
            ax.set_xlabel(r"$\epsilon$")
            ax.set_title(title, fontsize=10)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                                fontsize=6, color="white" if logc else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{ds}: adaptive metrics (variable x $\\epsilon$)", fontsize=12)
        fig.tight_layout()
        out = fig_dir / f"heatmap_{ds}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  -> {out}")


def plot_all(df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmaps(df, fig_dir)

    _facet_by_dataset(df, "compression_ratio", "Compression ratio (x)",
                      r"Compression vs tolerance",
                      fig_dir / "compression_ratio_vs_eps.png")

    if "combined_psnr" in df.columns:
        _facet_by_dataset(df, "combined_psnr", "PSNR (dB)",
                          r"Fidelity vs tolerance",
                          fig_dir / "psnr_vs_eps.png")

    if is_single_stage(df):
        _facet_by_dataset(df, "k_star", r"Adaptive rank $k^*$",
                          r"Adaptive rank vs tolerance",
                          fig_dir / "rank_vs_eps.png")
    else:
        _facet_by_dataset(df, "k_star", r"Adaptive rank $k^*$ (o) / $r^*$ (--)",
                          r"Adaptive rank vs tolerance",
                          fig_dir / "rank_vs_eps.png", extra="r_star")

    # Rate-distortion scatter: compression ratio vs PSNR, all variables.
    if "combined_psnr" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        markers = {"nyx": "o", "miranda": "s"}
        for ds in sorted(df["dataset"].unique()):
            sub = df[df["dataset"] == ds]
            ax.scatter(sub["combined_psnr"], sub["compression_ratio"],
                       marker=markers.get(ds, "o"), label=ds, alpha=0.7)
        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel("Compression ratio (x)")
        ax.set_title("Rate-distortion (each point = one variable at one $\\epsilon$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "rate_distortion.png", dpi=150)
        plt.close(fig)
        print(f"  -> {fig_dir / 'rate_distortion.png'}")

    # Storage breakdown stacked bars, one panel per dataset.
    need = {"stage1_rank_bytes", "stage2_rank_bytes", "sparse_bytes"}
    if need.issubset(df.columns):
        single_stage = is_single_stage(df)
        datasets = sorted(df["dataset"].unique())
        fig, axes = plt.subplots(len(datasets), 1,
                                 figsize=(13, 4.5 * len(datasets)), squeeze=False)
        for ax, ds in zip(axes[:, 0], datasets):
            sub = df[df["dataset"] == ds].sort_values(["var", "eps"])
            labels = [f"{r.var}\n{r.eps:g}" for r in sub.itertuples()]
            x = np.arange(len(sub))
            s1 = sub["stage1_rank_bytes"].values / 1e6
            s2 = sub["stage2_rank_bytes"].values / 1e6
            sp = sub["sparse_bytes"].values / 1e6
            if single_stage:
                ax.bar(x, s1, label="$L$ rank $k^*$")
                ax.bar(x, sp, bottom=s1, label="Sparse $S$")
            else:
                ax.bar(x, s1, label="$L_1$ rank $k^*$")
                ax.bar(x, s2, bottom=s1, label="$L_2$ rank $r^*$")
                ax.bar(x, sp, bottom=s1 + s2, label="Sparse $S$")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6, rotation=90)
            ax.set_ylabel("Storage (MB)")
            ax.set_title(f"{ds}: storage breakdown (var x $\\epsilon$)")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(fig_dir / "storage_breakdown.png", dpi=150)
        plt.close(fig)
        print(f"  -> {fig_dir / 'storage_breakdown.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    df = load(args.input)
    print_summary(df, args.fig_dir, args.input)
    if not args.no_plots:
        print("\nGenerating figures ...")
        plot_all(df, args.fig_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
