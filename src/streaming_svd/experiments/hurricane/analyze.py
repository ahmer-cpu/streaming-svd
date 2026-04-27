"""Stage 2 — Hurricane experiment analysis.

Loads per-variable raw CSVs produced by :mod:`collect` (Python) or the C++
``hurricane_bench`` binary, computes per-variable summary statistics, and
writes ``hurricane_summary.csv``.

The analysis step is intentionally decoupled from data collection: you can
re-run it at any time to change aggregation logic without re-running the SVD.

Summary CSV layout::

    results/hurricane/hurricane_summary.csv

CLI usage::

    python -m streaming_svd.experiments.hurricane.analyze --help

    python -m streaming_svd.experiments.hurricane.analyze \\
        --raw-dir results/hurricane/raw \\
        --out results/hurricane/hurricane_summary.csv \\
        --print-table

    # With full timing breakdown
    python -m streaming_svd.experiments.hurricane.analyze \\
        --raw-dir results/hurricane/raw \\
        --out results/hurricane/hurricane_summary.csv \\
        --print-table --print-timing
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from streaming_svd.data import HURRICANE_VARIABLES

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

# Cold-algorithm per-phase timing columns (without "cold_time_" prefix)
_COLD_PHASE_COLS = [
    "omega_gen", "initial_matmul", "power_iter",
    "qr", "projection", "small_svd", "lift",
]

# Warm-algorithm per-phase timing columns (without "warm_time_" prefix)
_WARM_PHASE_COLS = [
    "warm_proj", "warm_matmul",
    "omega_gen", "random_matmul", "concat",
    "power_iter", "qr", "projection", "small_svd", "lift",
]

_SUMMARY_COLUMNS: List[str] = [
    # Identity
    "var",
    "n_timesteps",
    "k",
    "p_cold",
    "p_warm",
    "q",
    # Accuracy
    "cold_fro_error_mean",
    "cold_fro_error_std",
    "cold_fro_error_min",
    "cold_fro_error_max",
    "warm_fro_error_mean",
    "warm_fro_error_std",
    "warm_fro_error_min",
    "warm_fro_error_max",
    "optimal_fro_error_mean",
    "optimal_fro_error_std",
    "fro_error_gap_mean",
    "fro_error_gap_std",
    "fro_error_ratio_mean",
    "cold_spec_error_mean",
    "warm_spec_error_mean",
    "fraction_warm_better_fro",
    "cold_fro_overhead_mean",
    "warm_fro_overhead_mean",
    # Subspace (Frobenius metrics only — spectral variants saturate to ~1 for rank > 1)
    "warm_drift_fro_mean",
    "warm_drift_fro_std",
    "frac_aligned_mean",          # 1 - drift_fro^2/k: fraction of energy aligned t-1 → t
    "cold_vs_warm_subspace_fro_mean",
    "cold_vs_warm_subspace_fro_std",
    "warm_prev_quality_fro_mean", # how close U_{t-1} is to current cold U (Frobenius)
    "warm_prev_quality_fro_std",
    # --- Timing: totals (ms) ---
    "cold_time_mean_ms",
    "cold_time_std_ms",
    "warm_time_mean_ms",
    "warm_time_std_ms",
    "time_speedup_mean",
    "time_speedup_std",
    "fraction_warm_faster",
    # --- Timing: cold phase breakdown (% of cold total) ---
    "cold_pct_omega_gen",
    "cold_pct_initial_matmul",
    "cold_pct_power_iter",
    "cold_pct_qr",
    "cold_pct_projection",
    "cold_pct_small_svd",
    "cold_pct_lift",
    # cold absolute mean (ms)
    "cold_ms_omega_gen",
    "cold_ms_initial_matmul",
    "cold_ms_power_iter",
    "cold_ms_qr",
    "cold_ms_projection",
    "cold_ms_small_svd",
    "cold_ms_lift",
    # --- Timing: warm phase breakdown (% of warm total) ---
    "warm_pct_warm_proj",
    "warm_pct_warm_matmul",
    "warm_pct_omega_gen",
    "warm_pct_random_matmul",
    "warm_pct_concat",
    "warm_pct_power_iter",
    "warm_pct_qr",
    "warm_pct_projection",
    "warm_pct_small_svd",
    "warm_pct_lift",
    # warm absolute mean (ms)
    "warm_ms_warm_proj",
    "warm_ms_warm_matmul",
    "warm_ms_omega_gen",
    "warm_ms_random_matmul",
    "warm_ms_concat",
    "warm_ms_power_iter",
    "warm_ms_qr",
    "warm_ms_projection",
    "warm_ms_small_svd",
    "warm_ms_lift",
    # --- Timing: phase delta (warm_ms - cold_ms for matching phases) ---
    "delta_ms_qr",             # warm QR cost vs cold QR cost (negative = warm cheaper)
    "delta_ms_projection",
    "delta_ms_small_svd",
    "delta_ms_lift",
    "delta_ms_matmuls_total",  # sum of all matmul-phase times: warm vs cold
    "warm_extra_ms_warm_proj_matmul",  # time unique to warm: warm_proj + warm_matmul
    # --- Legacy columns kept for backward compatibility ---
    "cold_time_breakdown_pct_matmul",
    "warm_time_breakdown_pct_warm_proj",
    "warm_time_breakdown_pct_matmul",
    # Matmuls
    "cold_matmuls_total_mean",
    "warm_matmuls_total_mean",
    "matmul_savings_mean",
]

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_results(
    raw_dir: Path,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate all ``{VAR}_raw.csv`` files from *raw_dir*.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw CSV files produced by :mod:`collect`.
    variables:
        Optional subset of variable names to load.  If ``None``, all CSVs
        present in *raw_dir* are loaded.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with one row per (variable, timestep).

    Raises
    ------
    FileNotFoundError
        If *raw_dir* does not exist.
    ValueError
        If no CSV files are found or expected columns are missing.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw results directory not found: {raw_dir}")

    if variables is None:
        csv_files = sorted(raw_dir.glob("*_raw.csv"))
    else:
        csv_files = [raw_dir / f"{v}_raw.csv" for v in variables]

    if not csv_files:
        raise ValueError(f"No *_raw.csv files found in {raw_dir}")

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        if not csv_path.exists():
            warnings.warn(f"Expected file not found: {csv_path}", stacklevel=2)
            continue
        try:
            df = pd.read_csv(csv_path)
            frames.append(df)
        except Exception as exc:
            warnings.warn(f"Could not read {csv_path}: {exc}", stacklevel=2)

    if not frames:
        raise ValueError("No valid CSV files could be read.")

    combined = pd.concat(frames, ignore_index=True)

    # Basic validation
    required = {"var", "timestep", "cold_fro_error", "warm_fro_error"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Combined DataFrame is missing expected columns: {missing}")

    return combined


# ---------------------------------------------------------------------------
# Summary computation helpers
# ---------------------------------------------------------------------------

def _safe_mean(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.mean()) if len(v) > 0 else _NAN


def _safe_std(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.std(ddof=1)) if len(v) > 1 else _NAN


def _safe_max(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.max()) if len(v) > 0 else _NAN


def _safe_min(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.min()) if len(v) > 0 else _NAN


def _pct_col(df: pd.DataFrame, num_col: str, denom_col: str) -> float:
    """Mean fraction of denom_col accounted for by num_col, across non-NaN rows."""
    if num_col not in df.columns or denom_col not in df.columns:
        return _NAN
    valid = df[[num_col, denom_col]].dropna()
    valid = valid[valid[denom_col] > 0]
    if valid.empty:
        return _NAN
    ratios = valid[num_col] / valid[denom_col]
    return float(ratios.mean())


def _col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column as Series, or empty float Series if absent."""
    return df[col] if col in df.columns else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_variable_summary(df_var: pd.DataFrame) -> Dict[str, object]:
    """Compute all summary statistics for a single variable's raw DataFrame.

    Parameters
    ----------
    df_var:
        Raw DataFrame containing rows for *one* variable only.

    Returns
    -------
    dict
        Flat row dict matching :data:`_SUMMARY_COLUMNS`.
    """
    var = str(df_var["var"].iloc[0]) if len(df_var) > 0 else "unknown"

    # Drop fully-NaN error rows (failed timesteps) for most metrics
    good = df_var.dropna(subset=["cold_fro_error", "warm_fro_error"])

    n_ts = len(good)

    def _param(col: str, default=_NAN):
        try:
            val = df_var[col].dropna().iloc[0]
            return int(val) if isinstance(val, (int, float)) and not np.isnan(float(val)) else default
        except (IndexError, KeyError):
            return default

    row: Dict[str, object] = {"var": var, "n_timesteps": n_ts}
    row["k"] = _param("k")
    row["p_cold"] = _param("p_cold")
    row["p_warm"] = _param("p_warm")
    row["q"] = _param("q")

    # --- Accuracy ---
    row["cold_fro_error_mean"] = _safe_mean(good["cold_fro_error"])
    row["cold_fro_error_std"]  = _safe_std(good["cold_fro_error"])
    row["cold_fro_error_min"]  = _safe_min(good["cold_fro_error"])
    row["cold_fro_error_max"]  = _safe_max(good["cold_fro_error"])

    row["warm_fro_error_mean"] = _safe_mean(good["warm_fro_error"])
    row["warm_fro_error_std"]  = _safe_std(good["warm_fro_error"])
    row["warm_fro_error_min"]  = _safe_min(good["warm_fro_error"])
    row["warm_fro_error_max"]  = _safe_max(good["warm_fro_error"])

    if "optimal_fro_error" in good.columns:
        row["optimal_fro_error_mean"] = _safe_mean(good["optimal_fro_error"])
        row["optimal_fro_error_std"]  = _safe_std(good["optimal_fro_error"])
    else:
        row["optimal_fro_error_mean"] = _NAN
        row["optimal_fro_error_std"]  = _NAN

    row["fro_error_gap_mean"]   = _safe_mean(good["fro_error_gap"])
    row["fro_error_gap_std"]    = _safe_std(good["fro_error_gap"])
    row["fro_error_ratio_mean"] = _safe_mean(good["fro_error_ratio"])

    row["cold_spec_error_mean"] = _safe_mean(_col(good, "cold_spec_error"))
    row["warm_spec_error_mean"] = _safe_mean(_col(good, "warm_spec_error"))

    if n_ts > 0:
        row["fraction_warm_better_fro"] = float(
            (good["warm_fro_error"] < good["cold_fro_error"]).mean()
        )
    else:
        row["fraction_warm_better_fro"] = _NAN

    row["cold_fro_overhead_mean"] = _safe_mean(_col(good, "cold_fro_overhead"))
    row["warm_fro_overhead_mean"] = _safe_mean(_col(good, "warm_fro_overhead"))

    # --- Subspace (Frobenius metrics; spectral variants omitted — they saturate for rank > 1) ---
    drift_fro = _col(good, "warm_drift_fro").dropna()
    row["warm_drift_fro_mean"] = _safe_mean(drift_fro)
    row["warm_drift_fro_std"]  = _safe_std(drift_fro)

    k_val = float(good["k"].dropna().iloc[0]) if "k" in good.columns and good["k"].notna().any() else 20.0
    if len(drift_fro) > 0 and k_val > 0:
        frac_al = (1.0 - drift_fro ** 2 / k_val).clip(0.0, 1.0)
        row["frac_aligned_mean"] = _safe_mean(frac_al)
    else:
        row["frac_aligned_mean"] = _NAN

    row["cold_vs_warm_subspace_fro_mean"] = _safe_mean(_col(good, "cold_vs_warm_subspace_fro").dropna())
    row["cold_vs_warm_subspace_fro_std"]  = _safe_std(_col(good, "cold_vs_warm_subspace_fro").dropna())

    row["warm_prev_quality_fro_mean"] = _safe_mean(_col(good, "warm_prev_quality_fro").dropna())
    row["warm_prev_quality_fro_std"]  = _safe_std(_col(good, "warm_prev_quality_fro").dropna())

    # --- Timing ---
    has_timing = "cold_time_total" in good.columns

    if has_timing:
        cold_ms = good["cold_time_total"] * 1000
        warm_ms = good["warm_time_total"].dropna() * 1000

        row["cold_time_mean_ms"] = _safe_mean(cold_ms)
        row["cold_time_std_ms"]  = _safe_std(cold_ms)
        row["warm_time_mean_ms"] = _safe_mean(warm_ms)
        row["warm_time_std_ms"]  = _safe_std(warm_ms)

        speedup = _col(good, "time_speedup_ratio").dropna()
        row["time_speedup_mean"] = _safe_mean(speedup)
        row["time_speedup_std"]  = _safe_std(speedup)

        valid_times = good[["cold_time_total", "warm_time_total"]].dropna()
        row["fraction_warm_faster"] = (
            float((valid_times["warm_time_total"] < valid_times["cold_time_total"]).mean())
            if not valid_times.empty else _NAN
        )

        # --- Cold phase breakdown ---
        warm_rows = good.dropna(subset=["warm_time_total"])  # rows where warm ran

        for phase in _COLD_PHASE_COLS:
            col = f"cold_time_{phase}"
            pct_key = f"cold_pct_{phase}"
            ms_key  = f"cold_ms_{phase}"
            if col in good.columns:
                ms_vals = good[col] * 1000
                row[ms_key]  = _safe_mean(ms_vals)
                row[pct_key] = _pct_col(good, col, "cold_time_total")
            else:
                row[ms_key]  = _NAN
                row[pct_key] = _NAN

        # --- Warm phase breakdown (only over warm-active timesteps t>=2) ---
        for phase in _WARM_PHASE_COLS:
            col = f"warm_time_{phase}"
            pct_key = f"warm_pct_{phase}"
            ms_key  = f"warm_ms_{phase}"
            if col in warm_rows.columns:
                ms_vals = warm_rows[col] * 1000
                row[ms_key]  = _safe_mean(ms_vals.dropna())
                row[pct_key] = _pct_col(warm_rows, col, "warm_time_total")
            else:
                row[ms_key]  = _NAN
                row[pct_key] = _NAN

        # --- Phase deltas: warm_ms - cold_ms for shared phases ---
        # Use only rows where both warm and cold ran (t >= 2)
        shared_rows = good.dropna(subset=["warm_time_total"])

        def _phase_delta(warm_col: str, cold_col: str) -> float:
            if warm_col not in shared_rows.columns or cold_col not in shared_rows.columns:
                return _NAN
            w = shared_rows[warm_col].dropna()
            c = shared_rows[cold_col].reindex(w.index).dropna()
            both = pd.concat([w.rename("w"), c.rename("c")], axis=1).dropna()
            if both.empty:
                return _NAN
            return float(((both["w"] - both["c"]) * 1000).mean())

        row["delta_ms_qr"]         = _phase_delta("warm_time_qr",         "cold_time_qr")
        row["delta_ms_projection"]  = _phase_delta("warm_time_projection",  "cold_time_projection")
        row["delta_ms_small_svd"]   = _phase_delta("warm_time_small_svd",   "cold_time_small_svd")
        row["delta_ms_lift"]        = _phase_delta("warm_time_lift",        "cold_time_lift")

        # Matmul phases: cold has initial_matmul; warm has warm_proj+warm_matmul+random_matmul
        cold_matmul_cols = [c for c in ["cold_time_initial_matmul", "cold_time_power_iter"] if c in shared_rows.columns]
        warm_matmul_cols = [c for c in ["warm_time_warm_proj", "warm_time_warm_matmul",
                                        "warm_time_random_matmul", "warm_time_power_iter"] if c in shared_rows.columns]
        if cold_matmul_cols and warm_matmul_cols:
            cm = shared_rows[cold_matmul_cols].sum(axis=1)
            wm = shared_rows[warm_matmul_cols].sum(axis=1)
            row["delta_ms_matmuls_total"] = float(((wm - cm) * 1000).mean())
        else:
            row["delta_ms_matmuls_total"] = _NAN

        # Cost unique to warm: warm_proj + warm_matmul (extra matmuls cold never does)
        unique_warm_cols = [c for c in ["warm_time_warm_proj", "warm_time_warm_matmul"] if c in shared_rows.columns]
        if unique_warm_cols:
            row["warm_extra_ms_warm_proj_matmul"] = float(
                shared_rows[unique_warm_cols].sum(axis=1).dropna().mean() * 1000
            )
        else:
            row["warm_extra_ms_warm_proj_matmul"] = _NAN

        # --- Legacy columns (backward compat) ---
        matmul_cols_cold = [c for c in ["cold_time_initial_matmul", "cold_time_power_iter"] if c in good.columns]
        if matmul_cols_cold:
            good_t = good[matmul_cols_cold + ["cold_time_total"]].dropna()
            if not good_t.empty:
                ms = good_t[matmul_cols_cold].sum(axis=1)
                row["cold_time_breakdown_pct_matmul"] = float(
                    (ms / good_t["cold_time_total"].replace(0, float("nan"))).mean()
                )
            else:
                row["cold_time_breakdown_pct_matmul"] = _NAN
        else:
            row["cold_time_breakdown_pct_matmul"] = _NAN

        row["warm_time_breakdown_pct_warm_proj"] = _pct_col(
            good, "warm_time_warm_proj", "warm_time_total"
        )
        row["warm_time_breakdown_pct_matmul"] = _pct_col(
            good, "warm_time_random_matmul", "warm_time_total"
        )
    else:
        # No timing columns — fill everything with NaN
        timing_keys = [
            "cold_time_mean_ms", "cold_time_std_ms", "warm_time_mean_ms", "warm_time_std_ms",
            "time_speedup_mean", "time_speedup_std", "fraction_warm_faster",
            "delta_ms_qr", "delta_ms_projection", "delta_ms_small_svd", "delta_ms_lift",
            "delta_ms_matmuls_total", "warm_extra_ms_warm_proj_matmul",
            "cold_time_breakdown_pct_matmul", "warm_time_breakdown_pct_warm_proj",
            "warm_time_breakdown_pct_matmul",
        ]
        for phase in _COLD_PHASE_COLS:
            timing_keys += [f"cold_pct_{phase}", f"cold_ms_{phase}"]
        for phase in _WARM_PHASE_COLS:
            timing_keys += [f"warm_pct_{phase}", f"warm_ms_{phase}"]
        for k in timing_keys:
            row[k] = _NAN

    # --- Matmuls ---
    row["cold_matmuls_total_mean"] = _safe_mean(_col(good, "cold_matmuls_total"))
    row["warm_matmuls_total_mean"] = _safe_mean(_col(good, "warm_matmuls_total"))
    row["matmul_savings_mean"]     = _safe_mean(_col(good, "matmul_savings"))

    return row


def analyze_hurricane_results(
    raw_dir: Path,
    out_path: Path,
    variables: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load raw CSVs, compute per-variable summaries, and save to *out_path*.

    Parameters
    ----------
    raw_dir:
        Directory of raw CSV files (output of :mod:`collect`).
    out_path:
        Destination path for the summary CSV.
    variables:
        Optional subset of variable names.  If ``None``, loads all present CSVs.
    verbose:
        Print progress and a brief summary table.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame (one row per variable).
    """
    if verbose:
        print("Loading raw results …")

    combined = load_raw_results(raw_dir, variables=variables)
    all_vars = sorted(combined["var"].unique())

    if verbose:
        print(f"  Loaded {len(combined)} rows across {len(all_vars)} variables: {all_vars}")

    rows = []
    for var in all_vars:
        df_var = combined[combined["var"] == var].copy()
        summary_row = compute_variable_summary(df_var)
        rows.append(summary_row)
        if verbose:
            gap     = summary_row.get("fro_error_gap_mean", _NAN)
            frac    = summary_row.get("fraction_warm_better_fro", _NAN)
            speedup = summary_row.get("time_speedup_mean", _NAN)
            print(
                f"  {var:<12}  gap={gap:+.5f}  frac_better={frac:.2f}  speedup={speedup:.3f}x"
            )

    df_summary = pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_path, index=False)

    if verbose:
        print(f"\nSummary saved to: {out_path.resolve()}")

    return df_summary


# ---------------------------------------------------------------------------
# Formatted tables
# ---------------------------------------------------------------------------

def print_summary_table(
    df_summary: pd.DataFrame,
    sort_by: str = "fro_error_gap_mean",
) -> None:
    """Print the primary accuracy + timing summary table."""
    if sort_by in df_summary.columns:
        df_sorted = df_summary.sort_values(sort_by, ascending=True)
    else:
        df_sorted = df_summary

    header = (
        f"{'Variable':<12} {'n_ts':>4} "
        f"{'cold_fro':>9} {'warm_fro':>9} {'optimal':>9} "
        f"{'gap':>9} {'frac_better':>11} "
        f"{'speedup':>9} {'frac_aligned':>13}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for _, r in df_sorted.iterrows():
        def _f(v, fmt=".5f"):
            try:
                if isinstance(v, float) and np.isnan(v):
                    return "   n/a   "
                return f"{float(v):{fmt}}"
            except (TypeError, ValueError):
                return "   n/a   "

        print(
            f"{str(r.get('var', '?')):<12} {int(r.get('n_timesteps', 0)):>4} "
            f"{_f(r.get('cold_fro_error_mean')):>9} "
            f"{_f(r.get('warm_fro_error_mean')):>9} "
            f"{_f(r.get('optimal_fro_error_mean')):>9} "
            f"{_f(r.get('fro_error_gap_mean')):>9} "
            f"{_f(r.get('fraction_warm_better_fro'), '.2f'):>11} "
            f"{_f(r.get('time_speedup_mean'), '.3f'):>9} "
            f"{_f(r.get('frac_aligned_mean'), '.3f'):>13}"
        )

    print(sep + "\n")
    print(f"  Sorted by: {sort_by}")
    print(f"  gap          = mean(warm_fro - cold_fro)  [negative = warm is better]")
    print(f"  speedup      = mean(cold_time / warm_time) [>1 = warm is faster]")
    print(f"  frac_aligned = mean(1 - drift_fro²/k)  [1=subspace stable, 0=rotates completely]\n")


def print_timing_breakdown(df_summary: pd.DataFrame) -> None:
    """Print a detailed per-phase timing breakdown table.

    Shows, for each variable:
      - Mean cold and warm total time (ms)
      - Per-phase percentage share of total time for both algorithms
      - Delta columns: how much more/less time warm spends vs cold per phase
      - Extra cost unique to warm (warm_proj + warm_matmul)

    This is the primary diagnostic table for understanding *where* the
    warm-vs-cold timing discrepancy arises.
    """
    df = df_summary.copy()

    def _f(v, fmt="7.1f"):
        """Format a number; return 'n/a' for NaN/None."""
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                width = int(fmt.split(".")[0]) if fmt[0].isdigit() else 7
                return f"{'n/a':>{width}}"
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "n/a"

    # ---- Phase breakdown table ----
    print("\n" + "=" * 120)
    print("COLD-START PHASE BREAKDOWN  (mean % of total cold time)")
    print("=" * 120)
    hdr_phases = f"{'Var':<12}  {'total_ms':>8}  " + "  ".join(
        f"{p:>10}" for p in _COLD_PHASE_COLS
    )
    print(hdr_phases)
    print("-" * 120)
    for _, r in df.iterrows():
        line = f"{str(r['var']):<12}  {_f(r.get('cold_time_mean_ms'), '7.1f'):>8}  "
        for p in _COLD_PHASE_COLS:
            pct = r.get(f"cold_pct_{p}", _NAN)
            if isinstance(pct, float) and not np.isnan(pct):
                line += f"  {pct*100:>9.1f}%"
            else:
                line += f"  {'n/a':>9} "
        print(line)
    print()

    print("=" * 120)
    print("WARM-START PHASE BREAKDOWN  (mean % of total warm time, t>=2 only)")
    print("=" * 120)
    hdr_warm = f"{'Var':<12}  {'total_ms':>8}  " + "  ".join(
        f"{p:>12}" for p in _WARM_PHASE_COLS
    )
    print(hdr_warm)
    print("-" * 120)
    for _, r in df.iterrows():
        line = f"{str(r['var']):<12}  {_f(r.get('warm_time_mean_ms'), '7.1f'):>8}  "
        for p in _WARM_PHASE_COLS:
            pct = r.get(f"warm_pct_{p}", _NAN)
            if isinstance(pct, float) and not np.isnan(pct):
                line += f"  {pct*100:>11.1f}%"
            else:
                line += f"  {'n/a':>11} "
        print(line)
    print()

    print("=" * 100)
    print("PHASE DELTA TABLE  (warm_ms - cold_ms, averaged over t>=2 timesteps)")
    print("Positive = warm spends MORE time on this phase than cold")
    print("=" * 100)
    delta_cols = [
        ("qr",          "delta_ms_qr"),
        ("projection",  "delta_ms_projection"),
        ("small_svd",   "delta_ms_small_svd"),
        ("lift",        "delta_ms_lift"),
        ("all_matmuls", "delta_ms_matmuls_total"),
        ("extra(w_p+wm)","warm_extra_ms_warm_proj_matmul"),
    ]
    hdr_delta = f"{'Var':<12}  {'speedup':>8}  " + "  ".join(
        f"{label:>14}" for label, _ in delta_cols
    )
    print(hdr_delta)
    print("-" * 100)
    for _, r in df.iterrows():
        spd = r.get("time_speedup_mean", _NAN)
        line = f"{str(r['var']):<12}  {_f(spd, '7.3f'):>8}  "
        for label, key in delta_cols:
            v = r.get(key, _NAN)
            if isinstance(v, float) and not np.isnan(v):
                line += f"  {v:>+14.2f}"
            else:
                line += f"  {'n/a':>14}"
        print(line)
    print()

    print("  delta_ms_qr          : warm QR time minus cold QR time (both operate on Y,")
    print("                         but warm Y has r_prev+p cols vs cold's k+p cols)")
    print("  delta_ms_all_matmuls : warm total matmul time minus cold total matmul time")
    print("                         (warm does 2 AX + 2 ATX vs cold's 1 AX + 1 ATX for q=0)")
    print("  extra(w_p+wm)        : absolute cost of warm_proj + warm_matmul (unique to warm)")
    print()

    # ---- Absolute ms breakdown side-by-side ----
    shared_phases = ["qr", "projection", "small_svd", "lift"]
    print("=" * 90)
    print("ABSOLUTE PHASE TIMES (ms) — cold vs warm for shared phases")
    print("=" * 90)
    hdr_abs = f"{'Var':<12}  " + "  ".join(
        f"{'cold_'+p:>12}  {'warm_'+p:>12}" for p in shared_phases
    )
    print(hdr_abs)
    print("-" * 90)
    for _, r in df.iterrows():
        line = f"{str(r['var']):<12}  "
        for p in shared_phases:
            cv = r.get(f"cold_ms_{p}", _NAN)
            wv = r.get(f"warm_ms_{p}", _NAN)
            line += f"  {_f(cv, '7.2f'):>12}  {_f(wv, '7.2f'):>12}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for Stage 2 (analysis)."""
    parser = argparse.ArgumentParser(
        description="Hurricane experiment — Stage 2: analyze raw CSV results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", type=str, default="results/hurricane/raw",
        help="Directory containing {VAR}_raw.csv files",
    )
    parser.add_argument(
        "--out", type=str, default="results/hurricane/hurricane_summary.csv",
        help="Path to write the summary CSV",
    )
    parser.add_argument(
        "--vars", nargs="+", default=None, metavar="VAR",
        help="Subset of variables to include (default: all present CSVs)",
    )
    parser.add_argument(
        "--print-table", action="store_true",
        help="Print the accuracy + speedup summary table",
    )
    parser.add_argument(
        "--print-timing", action="store_true",
        help="Print the detailed per-phase timing breakdown tables",
    )
    parser.add_argument(
        "--sort-by", type=str, default="fro_error_gap_mean",
        help="Column to sort the printed table by",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    df_summary = analyze_hurricane_results(
        raw_dir=Path(args.raw_dir),
        out_path=Path(args.out),
        variables=args.vars,
        verbose=not args.quiet,
    )

    if args.print_table:
        print_summary_table(df_summary, sort_by=args.sort_by)

    if args.print_timing:
        print_timing_breakdown(df_summary)


if __name__ == "__main__":
    main()
