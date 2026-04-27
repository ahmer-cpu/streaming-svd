"""Stage 1 — Hurricane experiment data collection.

Runs cold-start and warm-start rSVD on every (variable, timestep) pair in the
Hurricane Isabel dataset and writes per-variable raw CSV files.  Each CSV row
captures ~53 metrics: reconstruction quality, subspace distances, full timing
breakdowns, and matmul counts.

The collection loop is **resumable**: if a variable's CSV already contains all
expected timestep rows it is skipped.  Partial CSVs are continued from the last
complete row.

Results directory layout::

    results/hurricane/raw/
        CLOUDf_raw.csv
        Pf_raw.csv
        ...
        Wf_raw.csv

CLI usage::

    python -m streaming_svd.experiments.hurricane.collect --help

    # Quick smoke test (one variable, 5 timesteps)
    python -m streaming_svd.experiments.hurricane.collect \\
        --data-dir data/raw --vars Uf --start 1 --end 5 \\
        --k 20 --p-cold 10 --p-warm 5

    # Full run (all 13 variables × 48 timesteps)
    python -m streaming_svd.experiments.hurricane.collect \\
        --data-dir data/raw --start 1 --end 48 \\
        --k 20 --p-cold 10 --p-warm 5
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from streaming_svd.algos.metrics import (
    rel_fro_error,
    rel_spec_error_est,
    subspace_sin_theta,
    subspace_sin_theta_fro,
)
from streaming_svd.algos.rsvd import rsvd
from streaming_svd.algos.warm_rsvd import warm_rsvd
from streaming_svd.data import (
    HURRICANE_VARIABLES,
    discover_variable_files,
    load_weather_matrix,
    optimal_rank_k_rel_fro_error_from_gram,
)

# ---------------------------------------------------------------------------
# CSV column order
# ---------------------------------------------------------------------------

_RAW_COLUMNS: List[str] = [
    # --- Identity / experiment metadata ---
    "var",
    "timestep",
    "k",
    "p_cold",
    "p_warm",
    "q",
    "seed",
    "dtype",
    "device",
    "data_file",
    "run_timestamp",
    # --- Approximation quality ---
    "cold_fro_error",
    "warm_fro_error",
    "optimal_fro_error",
    "cold_spec_error",
    "warm_spec_error",
    "fro_error_gap",
    "fro_error_ratio",
    "cold_fro_overhead",
    "warm_fro_overhead",
    "cold_spec_gap",
    # --- Subspace quality ---
    "warm_drift_spec",
    "warm_drift_fro",
    "cold_vs_warm_subspace_spec",
    "cold_vs_warm_subspace_fro",
    "warm_prev_quality_spec",
    # --- Timing (wall-clock, seconds) ---
    "cold_time_total",
    "warm_time_total",
    "time_load_matrix",
    "time_speedup_ratio",
    # Cold timing breakdown
    "cold_time_omega_gen",
    "cold_time_initial_matmul",
    "cold_time_power_iter",
    "cold_time_qr",
    "cold_time_projection",
    "cold_time_small_svd",
    "cold_time_lift",
    "cold_time_stats_total",
    # Warm timing breakdown
    "warm_time_warm_proj",
    "warm_time_warm_matmul",
    "warm_time_omega_gen",
    "warm_time_random_matmul",
    "warm_time_concat",
    "warm_time_power_iter",
    "warm_time_qr",
    "warm_time_projection",
    "warm_time_small_svd",
    "warm_time_lift",
    "warm_time_stats_total",
    # --- Matmul counts ---
    "cold_matmuls_AX",
    "cold_matmuls_ATX",
    "cold_matmuls_total",
    "warm_matmuls_AX",
    "warm_matmuls_ATX",
    "warm_matmuls_total",
    "matmul_savings",
    # --- Stats params (sanity checks) ---
    "cold_stats_k",
    "cold_stats_p",
    "warm_stats_r_prev",
    "warm_stats_warm_start",
]

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def build_raw_row(
    *,
    var: str,
    timestep: int,
    data_file: str,
    params: Dict[str, Any],
    A: torch.Tensor,
    U_cold: torch.Tensor,
    s_cold: torch.Tensor,
    Vt_cold: torch.Tensor,
    stats_cold: Dict[str, Any],
    time_cold: float,
    U_warm: torch.Tensor,
    s_warm: torch.Tensor,
    Vt_warm: torch.Tensor,
    stats_warm: Dict[str, Any],
    time_warm: float,
    U_warm_prev: Optional[torch.Tensor],
    optimal_error: float,
    time_load: float,
) -> Dict[str, Any]:
    """Assemble a single raw-CSV row dict from live tensors.

    This is a **pure function** with no I/O or side effects.  It must be called
    *before* any tensor is deleted, since it computes subspace metrics in-place.

    Parameters
    ----------
    var:            Variable name (e.g. ``"Uf"``).
    timestep:       1-indexed timestep.
    data_file:      Basename of the binary file (e.g. ``"Uf03.bin"``).
    params:         Experiment hyper-parameters dict (k, p_cold, p_warm, q, seed, dtype, device).
    A:              Data matrix (250000, 100).
    U_cold, s_cold, Vt_cold: Cold-start rSVD output.
    stats_cold:     Stats dict returned by :func:`rsvd`.
    time_cold:      Wall-clock seconds for the cold rSVD call.
    U_warm, s_warm, Vt_warm: Warm-start rSVD output.
    stats_warm:     Stats dict returned by :func:`warm_rsvd`.
    time_warm:      Wall-clock seconds for the warm rSVD call.
    U_warm_prev:    Warm-start basis from the previous timestep; ``None`` at t=1.
    optimal_error:  Optimal rank-k relative Frobenius error (NaN if disabled).
    time_load:      Wall-clock seconds to load the binary file.

    Returns
    -------
    dict
        Flat row matching :data:`_RAW_COLUMNS` (53 keys).
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Approximation quality ---
    cold_fro = rel_fro_error(A, U_cold, s_cold, Vt_cold)
    warm_fro = rel_fro_error(A, U_warm, s_warm, Vt_warm)
    cold_spec = rel_spec_error_est(A, U_cold, n_iter=3)
    warm_spec = rel_spec_error_est(A, U_warm, n_iter=3)

    fro_gap = warm_fro - cold_fro
    fro_ratio = (warm_fro / cold_fro) if cold_fro > 0 else _NAN
    cold_overhead = (cold_fro - optimal_error) if not math.isnan(optimal_error) else _NAN
    warm_overhead = (warm_fro - optimal_error) if not math.isnan(optimal_error) else _NAN
    spec_gap = warm_spec - cold_spec

    # --- Subspace quality ---
    if U_warm_prev is not None:
        warm_drift_spec = subspace_sin_theta(U_warm_prev, U_warm)
        warm_drift_fro_ = subspace_sin_theta_fro(U_warm_prev, U_warm)
        warm_prev_quality = subspace_sin_theta(U_warm_prev, U_cold)
    else:
        warm_drift_spec = _NAN
        warm_drift_fro_ = _NAN
        warm_prev_quality = _NAN

    cv_w_spec = subspace_sin_theta(U_cold, U_warm)
    cv_w_fro = subspace_sin_theta_fro(U_cold, U_warm)

    # --- Timing ---
    speedup = (time_cold / time_warm) if time_warm > 0 else _NAN

    def _t(d: Dict, key: str) -> float:
        return float(d.get("timings", {}).get(key, _NAN))

    # --- Matmul counts ---
    c_mc = stats_cold.get("matmul_counts", {})
    w_mc = stats_warm.get("matmul_counts", {})
    cold_AX = int(c_mc.get("A@X", 0))
    cold_ATX = int(c_mc.get("AT@X", 0))
    warm_AX = int(w_mc.get("A@X", 0))
    warm_ATX = int(w_mc.get("AT@X", 0))

    # --- Stats params ---
    c_p = stats_cold.get("params", {})
    w_p = stats_warm.get("params", {})

    return {
        # Identity
        "var": var,
        "timestep": timestep,
        "k": params["k"],
        "p_cold": params["p_cold"],
        "p_warm": params["p_warm"],
        "q": params["q"],
        "seed": params["seed"],
        "dtype": params["dtype"],
        "device": params["device"],
        "data_file": data_file,
        "run_timestamp": ts,
        # Approximation quality
        "cold_fro_error": cold_fro,
        "warm_fro_error": warm_fro,
        "optimal_fro_error": optimal_error,
        "cold_spec_error": cold_spec,
        "warm_spec_error": warm_spec,
        "fro_error_gap": fro_gap,
        "fro_error_ratio": fro_ratio,
        "cold_fro_overhead": cold_overhead,
        "warm_fro_overhead": warm_overhead,
        "cold_spec_gap": spec_gap,
        # Subspace quality
        "warm_drift_spec": warm_drift_spec,
        "warm_drift_fro": warm_drift_fro_,
        "cold_vs_warm_subspace_spec": cv_w_spec,
        "cold_vs_warm_subspace_fro": cv_w_fro,
        "warm_prev_quality_spec": warm_prev_quality,
        # Timing
        "cold_time_total": time_cold,
        "warm_time_total": time_warm,
        "time_load_matrix": time_load,
        "time_speedup_ratio": speedup,
        # Cold timing breakdown
        "cold_time_omega_gen": _t(stats_cold, "omega_gen"),
        "cold_time_initial_matmul": _t(stats_cold, "initial_matmul"),
        "cold_time_power_iter": _t(stats_cold, "power_iterations"),
        "cold_time_qr": _t(stats_cold, "qr"),
        "cold_time_projection": _t(stats_cold, "projection"),
        "cold_time_small_svd": _t(stats_cold, "small_svd"),
        "cold_time_lift": _t(stats_cold, "lift"),
        "cold_time_stats_total": _t(stats_cold, "total"),
        # Warm timing breakdown
        "warm_time_warm_proj": _t(stats_warm, "warm_proj"),
        "warm_time_warm_matmul": _t(stats_warm, "warm_matmul"),
        "warm_time_omega_gen": _t(stats_warm, "omega_gen"),
        "warm_time_random_matmul": _t(stats_warm, "random_matmul"),
        "warm_time_concat": _t(stats_warm, "concat"),
        "warm_time_power_iter": _t(stats_warm, "power_iterations"),
        "warm_time_qr": _t(stats_warm, "qr"),
        "warm_time_projection": _t(stats_warm, "projection"),
        "warm_time_small_svd": _t(stats_warm, "small_svd"),
        "warm_time_lift": _t(stats_warm, "lift"),
        "warm_time_stats_total": _t(stats_warm, "total"),
        # Matmul counts
        "cold_matmuls_AX": cold_AX,
        "cold_matmuls_ATX": cold_ATX,
        "cold_matmuls_total": cold_AX + cold_ATX,
        "warm_matmuls_AX": warm_AX,
        "warm_matmuls_ATX": warm_ATX,
        "warm_matmuls_total": warm_AX + warm_ATX,
        "matmul_savings": (cold_AX + cold_ATX) - (warm_AX + warm_ATX),
        # Stats params
        "cold_stats_k": c_p.get("k", _NAN),
        "cold_stats_p": c_p.get("p", _NAN),
        "warm_stats_r_prev": w_p.get("r_prev", _NAN),
        "warm_stats_warm_start": w_p.get("warm_start", False),
    }


# ---------------------------------------------------------------------------
# Per-variable runner
# ---------------------------------------------------------------------------

def _open_csv_writer(
    out_path: Path,
    append: bool,
) -> Tuple[Any, Any]:
    """Open a CSV file for writing or appending.  Returns (file_handle, writer)."""
    mode = "a" if append else "w"
    fh = open(out_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=_RAW_COLUMNS, extrasaction="ignore")
    if not append:
        writer.writeheader()
    return fh, writer


def _load_present_timesteps(csv_path: Path) -> List[int]:
    """Read a partial CSV and return sorted list of timestep values already written."""
    if not csv_path.exists():
        return []
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path, usecols=["timestep"])
        return sorted(df["timestep"].dropna().astype(int).tolist())
    except Exception:
        return []


def run_single_variable(
    var: str,
    data_dir: Path,
    start: int,
    end: int,
    k: int,
    p_cold: int,
    p_warm: int,
    q: int,
    seed: int,
    dtype: str,
    device: str,
    compute_optimal: bool,
    memmap: bool,
    out_path: Path,
    resume_from: int = 0,
    verbose: bool = True,
) -> int:
    """Run cold + warm rSVD for one variable over ``[start, end]``.

    Rows are appended to ``out_path`` immediately after each timestep (per-row
    flush), so partial results are preserved if the process is interrupted.

    Parameters
    ----------
    var:            Variable name (e.g. ``"Uf"``).
    data_dir:       Root data directory.
    start, end:     Inclusive timestep range (1-indexed).
    k, p_cold, p_warm, q: SVD hyper-parameters.
    seed:           Random seed for sketch matrices.
    dtype:          ``"float32"`` or ``"float64"``.
    device:         ``"cpu"`` or ``"cuda"``.
    compute_optimal: Whether to compute the Gram-matrix optimal baseline.
    memmap:         Whether to use numpy memmap for loading.
    out_path:       Destination CSV path.
    resume_from:    Skip timesteps ≤ this value (used for partial-resume).
    verbose:        Print per-timestep progress.

    Returns
    -------
    int
        Number of rows successfully written.
    """
    torch_device = torch.device(device)
    torch_dtype = torch.float64 if dtype.lower() in ("float64", "fp64", "double") else torch.float32

    params = {
        "k": k,
        "p_cold": p_cold,
        "p_warm": p_warm,
        "q": q,
        "seed": seed,
        "dtype": dtype,
        "device": device,
    }

    file_pairs = discover_variable_files(data_dir, var, start, end)
    if not file_pairs:
        if verbose:
            print(f"  [WARN] No files found for {var} in {data_dir / var}")
        return 0

    # Decide whether to append to an existing partial CSV
    append_mode = resume_from > 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh, writer = _open_csv_writer(out_path, append=append_mode)

    n_written = 0
    U_warm_prev: Optional[torch.Tensor] = None

    # If we are resuming mid-variable we need to reconstruct U_warm_prev by
    # running the warm algorithm (without writing rows) up to resume_from.
    if resume_from > 0:
        if verbose:
            print(f"  [RESUME] Replaying {var} t=1..{resume_from} to rebuild warm state …")
        replay_pairs = [(t, p) for t, p in file_pairs if t <= resume_from]
        for t_r, path_r in replay_pairs:
            try:
                A_r = load_weather_matrix(path_r, memmap=memmap).to(device=torch_device, dtype=torch_dtype)
                with torch.no_grad():
                    res_w = warm_rsvd(A_r, U_warm_prev, k, p=p_warm, q=q,
                                      device=torch_device, seed=seed, return_stats=True)
                U_warm_prev = res_w[0].to(dtype=torch_dtype)
                del A_r, res_w
                gc.collect()
            except Exception:
                U_warm_prev = None

    try:
        for t, file_path in file_pairs:
            if t <= resume_from:
                continue

            if verbose:
                print(f"  t={t:02d}  {file_path.name}", end=" ", flush=True)

            try:
                # --- Load ---
                t0_load = time.perf_counter()
                A = load_weather_matrix(file_path, memmap=memmap)
                time_load = time.perf_counter() - t0_load
                A = A.to(device=torch_device, dtype=torch_dtype)

                # --- Optimal baseline ---
                if compute_optimal:
                    try:
                        optimal_error = optimal_rank_k_rel_fro_error_from_gram(A, k)
                    except Exception:
                        optimal_error = _NAN
                else:
                    optimal_error = _NAN

                # --- Cold rSVD ---
                with torch.no_grad():
                    t0 = time.perf_counter()
                    result_cold = rsvd(A, k, p=p_cold, q=q,
                                       device=torch_device, seed=seed, return_stats=True)
                    time_cold = time.perf_counter() - t0
                U_cold, s_cold, Vt_cold, stats_cold = result_cold  # type: ignore[misc]
                U_cold = U_cold.to(dtype=torch_dtype)
                s_cold = s_cold.to(dtype=torch_dtype)
                Vt_cold = Vt_cold.to(dtype=torch_dtype)

                # --- Warm rSVD ---
                with torch.no_grad():
                    t0 = time.perf_counter()
                    result_warm = warm_rsvd(A, U_warm_prev, k, p=p_warm, q=q,
                                            device=torch_device, seed=seed, return_stats=True)
                    time_warm = time.perf_counter() - t0
                U_warm, s_warm, Vt_warm, stats_warm = result_warm  # type: ignore[misc]
                U_warm = U_warm.to(dtype=torch_dtype)
                s_warm = s_warm.to(dtype=torch_dtype)
                Vt_warm = Vt_warm.to(dtype=torch_dtype)

                # --- Build row (while all tensors are still alive) ---
                row = build_raw_row(
                    var=var,
                    timestep=t,
                    data_file=file_path.name,
                    params=params,
                    A=A,
                    U_cold=U_cold,
                    s_cold=s_cold,
                    Vt_cold=Vt_cold,
                    stats_cold=stats_cold,
                    time_cold=time_cold,
                    U_warm=U_warm,
                    s_warm=s_warm,
                    Vt_warm=Vt_warm,
                    stats_warm=stats_warm,
                    time_warm=time_warm,
                    U_warm_prev=U_warm_prev,
                    optimal_error=optimal_error,
                    time_load=time_load,
                )

                if verbose:
                    print(
                        f"cold_fro={row['cold_fro_error']:.5f}  "
                        f"warm_fro={row['warm_fro_error']:.5f}  "
                        f"gap={row['fro_error_gap']:+.5f}  "
                        f"t_cold={time_cold*1000:.1f}ms  "
                        f"t_warm={time_warm*1000:.1f}ms"
                    )

                # --- Update warm state ---
                U_warm_prev = U_warm.detach().clone()

                # --- Write row immediately (per-row flush) ---
                writer.writerow(row)
                fh.flush()
                n_written += 1

                # --- Cleanup ---
                del A, U_cold, s_cold, Vt_cold, U_warm, s_warm, Vt_warm
                del stats_cold, stats_warm, result_cold, result_warm
                gc.collect()

            except Exception as exc:
                if verbose:
                    print(f"\n  [ERROR] {var} t={t}: {exc}")
                    traceback.print_exc()
                # Write a NaN row so the timestep slot is recorded (makes gap
                # detection easier in the resume logic).
                nan_row: Dict[str, Any] = {col: _NAN for col in _RAW_COLUMNS}
                nan_row.update({
                    "var": var,
                    "timestep": t,
                    "data_file": file_path.name,
                    "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **{c: params[c] for c in ("k", "p_cold", "p_warm", "q", "seed", "dtype", "device")},
                })
                writer.writerow(nan_row)
                fh.flush()
                # Do NOT update U_warm_prev; keep it from the last successful step
                gc.collect()
    finally:
        fh.close()

    return n_written


# ---------------------------------------------------------------------------
# Multi-variable orchestrator
# ---------------------------------------------------------------------------

def collect_hurricane_experiment(
    data_dir: Path,
    variables: List[str],
    start: int = 1,
    end: int = 48,
    k: int = 20,
    p_cold: int = 10,
    p_warm: int = 5,
    q: int = 0,
    seed: int = 42,
    dtype: str = "float32",
    device: str = "cpu",
    compute_optimal: bool = True,
    memmap: bool = False,
    out_dir: Path = Path("results/hurricane/raw"),
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Orchestrate data collection for all specified variables.

    Parameters
    ----------
    data_dir:         Root data directory (contains one sub-folder per variable).
    variables:        List of variable names to process.
    start, end:       Inclusive timestep range (1-indexed).
    k, p_cold, p_warm, q: SVD hyper-parameters.
    seed:             Random seed.
    dtype:            ``"float32"`` or ``"float64"``.
    device:           ``"cpu"`` or ``"cuda"``.
    compute_optimal:  Whether to compute the Gram-matrix optimal baseline.
    memmap:           Use numpy memmap for file loading.
    out_dir:          Directory to write ``{VAR}_raw.csv`` files.
    resume:           If ``True``, skip fully-complete variables and continue
                      partial ones.  If ``False``, overwrite all output files.
    verbose:          Print progress.

    Returns
    -------
    dict
        Mapping ``{var: csv_path}`` for each variable that was processed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed: Dict[str, Path] = {}

    for i_var, var in enumerate(variables):
        csv_path = out_dir / f"{var}_raw.csv"

        if verbose:
            print(f"\n{'='*70}")
            print(f"[{i_var+1}/{len(variables)}] Variable: {var}  ->  {csv_path}")
            print(f"{'='*70}")

        # --- Discover expected timesteps ---
        expected_pairs = discover_variable_files(data_dir, var, start, end)
        expected_ts = {t for t, _ in expected_pairs}

        if not expected_ts:
            if verbose:
                print(f"  [SKIP] No data files found for {var} under {data_dir / var}")
            continue

        resume_from = 0  # 0 = start from scratch

        if resume and csv_path.exists():
            present_ts = set(_load_present_timesteps(csv_path))
            missing_ts = expected_ts - present_ts

            if not missing_ts:
                if verbose:
                    n = len(expected_ts)
                    print(f"  [SKIP] {var}: all {n} timesteps already present — skipping")
                completed[var] = csv_path
                continue
            else:
                # Partial resume: continue from the last present timestep
                if present_ts:
                    resume_from = max(present_ts)
                    if verbose:
                        print(
                            f"  [RESUME] {var}: {len(present_ts)}/{len(expected_ts)} rows found; "
                            f"resuming from t={resume_from+1}"
                        )

        n_written = run_single_variable(
            var=var,
            data_dir=data_dir,
            start=start,
            end=end,
            k=k,
            p_cold=p_cold,
            p_warm=p_warm,
            q=q,
            seed=seed,
            dtype=dtype,
            device=device,
            compute_optimal=compute_optimal,
            memmap=memmap,
            out_path=csv_path,
            resume_from=resume_from,
            verbose=verbose,
        )

        if verbose:
            print(f"\n  [DONE] {var}: {n_written} rows written to {csv_path}")

        completed[var] = csv_path

    if verbose:
        print(f"\n{'='*70}")
        print(f"Collection complete.  {len(completed)}/{len(variables)} variables written.")
        print(f"Output directory: {out_dir.resolve()}")
        print(f"{'='*70}")

    return completed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for Stage 1 (data collection)."""
    parser = argparse.ArgumentParser(
        description="Hurricane experiment — Stage 1: collect raw SVD metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Root data directory containing one sub-folder per variable",
    )
    parser.add_argument(
        "--vars", nargs="+", default=list(HURRICANE_VARIABLES),
        metavar="VAR",
        help="Variable names to process (default: all 13)",
    )
    parser.add_argument("--start", type=int, default=1, help="First timestep (inclusive)")
    parser.add_argument("--end", type=int, default=48, help="Last timestep (inclusive)")
    parser.add_argument("--k", type=int, default=20, help="Target rank")
    parser.add_argument("--p-cold", type=int, default=10, help="Cold-start oversampling")
    parser.add_argument("--p-warm", type=int, default=5, help="Warm-start oversampling")
    parser.add_argument("--q", type=int, default=0, help="Power iteration count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"],
        help="Computation dtype",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument(
        "--out-dir", type=str, default="results/hurricane/raw",
        help="Output directory for raw CSV files",
    )
    parser.add_argument(
        "--no-compute-optimal", action="store_true",
        help="Skip the Gram-matrix optimal baseline (saves time and memory)",
    )
    parser.add_argument("--memmap", action="store_true", help="Use numpy memmap for file loading")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Overwrite existing CSVs rather than resuming partial runs",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    collect_hurricane_experiment(
        data_dir=Path(args.data_dir),
        variables=args.vars,
        start=args.start,
        end=args.end,
        k=args.k,
        p_cold=args.p_cold,
        p_warm=args.p_warm,
        q=args.q,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        compute_optimal=not args.no_compute_optimal,
        memmap=args.memmap,
        out_dir=Path(args.out_dir),
        resume=not args.no_resume,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
