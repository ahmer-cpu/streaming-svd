"""Parameter sweep orchestrator for the Hurricane rSVD benchmark.

Invokes cpp/build/Release/hurricane_bench.exe for each parameter
configuration in the grid, writing results to separate subdirectories
under results/sweep/.  After all runs complete, concatenates raw CSVs
into a single sweep_all_raw.csv for downstream analysis.

Baseline config (k=20, p_cold=10, p_warm=5, q=0) is assumed already
computed in results/hurricane/raw_cpp/ and is copied into the sweep
output rather than re-run.

Usage:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --dry-run        # print commands only
    python scripts/run_sweep.py --vars Uf TCf     # subset of variables
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

K_VALUES = [5, 10, 20]
P_COLD = 10           # fixed
P_WARM_VALUES = [5, 10]
Q_VALUES = [0, 1, 2]
SEED = 42

BASELINE_K = 20      # baseline rank (for reference)
BASELINE_Q = 0       # baseline power iterations (for reference)

SWEEP_VARS = [
    "CLOUDf", "Pf", "PRECIPf", "QCLOUDf", "QICEf",
    "QSNOWf", "QVAPORf", "TCf", "Uf", "Vf", "Wf",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXE = REPO_ROOT / "cpp" / "build" / "Release" / "hurricane_bench.exe"
DATA_DIR = REPO_ROOT / "data" / "ISABEL_raw"
SWEEP_DIR = REPO_ROOT / "results" / "sweep"


def config_dir(k: int, p_warm: int, q: int) -> Path:
    return SWEEP_DIR / f"k{k}_pcold{P_COLD}_pwarm{p_warm}_q{q}"


def build_grid() -> List[Tuple[int, int]]:
    """Return list of (k, q) configs.

    Each config runs ALL p_warm values in a single invocation, so the
    grid is over (k, q) only.  All configs are re-run with the new
    per-timestep seed strategy.
    """
    return list(itertools.product(K_VALUES, Q_VALUES))


def run_config(
    k: int, q: int,
    p_warm_values: List[int],
    variables: List[str],
    dry_run: bool = False,
    skip_optimal: bool = False,
) -> bool:
    """Run hurricane_bench.exe for one (k, q) config with all p_warm values.

    Data is loaded once, cold rSVD runs once, optimal error computed once,
    and warm rSVD runs for each p_warm value — all within a single process
    invocation.
    """
    # Build list of output dirs (one per p_warm)
    out_dirs = []
    for pw in p_warm_values:
        d = config_dir(k, pw, q)
        d.mkdir(parents=True, exist_ok=True)
        out_dirs.append(str(d))

    cmd = [
        str(EXE),
        "--data-dir", str(DATA_DIR),
        "--out-dir", *out_dirs,
        "--vars", *variables,
        "--start", "1",
        "--end", "48",
        "--k", str(k),
        "--p-cold", str(P_COLD),
        "--p-warm", *[str(pw) for pw in p_warm_values],
        "--q", str(q),
        "--seed", str(SEED),
    ]
    if skip_optimal:
        cmd.append("--skip-optimal")

    label = f"k={k} q={q} p_warm={p_warm_values}"

    if dry_run:
        print(f"  [DRY RUN] {label}")
        print(f"    {' '.join(cmd)}")
        return True

    print(f"  Running {label} ...", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    FAILED ({elapsed:.1f}s)")
        print(f"    stderr: {result.stderr[:500]}")
        return False

    print(f"    done ({elapsed:.1f}s)")
    return True


def concatenate_results(variables: List[str]) -> Path:
    """Concatenate all per-config raw CSVs into sweep_all_raw.csv."""
    import pandas as pd

    all_frames = []
    for subdir in sorted(SWEEP_DIR.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("k"):
            continue
        # Parse config from directory name
        parts = subdir.name.split("_")
        config = {}
        for part in parts:
            for prefix in ("k", "pcold", "pwarm", "q"):
                if part.startswith(prefix) and part[len(prefix):].isdigit():
                    config[prefix] = int(part[len(prefix):])

        for csv_path in sorted(subdir.glob("*_raw.csv")):
            var = csv_path.stem.replace("_raw", "")
            if variables and var not in variables:
                continue
            try:
                df = pd.read_csv(csv_path)
                # Tag with sweep config for easy grouping
                df["sweep_k"] = config.get("k")
                df["sweep_pcold"] = config.get("pcold")
                df["sweep_pwarm"] = config.get("pwarm")
                df["sweep_q"] = config.get("q")
                all_frames.append(df)
            except Exception as e:
                print(f"  Warning: could not read {csv_path}: {e}")

    if not all_frames:
        print("  No CSV files found to concatenate.")
        return SWEEP_DIR / "sweep_all_raw.csv"

    combined = pd.concat(all_frames, ignore_index=True)
    out_path = SWEEP_DIR / "sweep_all_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Concatenated {len(combined)} rows from {len(all_frames)} files")
    print(f"  -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hurricane rSVD parameter sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--vars", nargs="+", default=SWEEP_VARS, help="Variables to sweep over")
    parser.add_argument("--skip-concat", action="store_true", help="Don't concatenate at the end")
    parser.add_argument("--skip-optimal", action="store_true", help="Skip optimal Frobenius error computation")
    args = parser.parse_args()

    if not EXE.exists():
        print(f"Error: executable not found at {EXE}")
        print("Build first: cmake --build cpp/build --config Release")
        sys.exit(1)

    grid = build_grid()
    total = len(grid)

    # Each (k, q) config runs all p_warm values in one invocation
    total_output_configs = total * len(P_WARM_VALUES)

    print(f"Hurricane rSVD Parameter Sweep (optimized)")
    print(f"  Variables: {' '.join(args.vars)}")
    print(f"  Grid: k={K_VALUES}, p_cold={P_COLD}, p_warm={P_WARM_VALUES}, q={Q_VALUES}")
    print(f"  Baseline: k={BASELINE_K}, q={BASELINE_Q}")
    print(f"  Invocations: {total} (each runs {len(P_WARM_VALUES)} p_warm values)")
    print(f"  Total output configs: {total_output_configs}")
    print(f"  SVD pairs per invocation: {len(args.vars)} vars x 48 timesteps x {len(P_WARM_VALUES)} p_warm = {len(args.vars) * 48 * len(P_WARM_VALUES)}")
    if args.skip_optimal:
        print(f"  Skipping optimal Frobenius error computation")
    print()

    # Run sweep
    print("Running sweep ...")
    succeeded = 0
    failed = 0
    t_sweep_start = time.time()

    for i, (k, q) in enumerate(grid, 1):
        print(f"[{i}/{total}]", end="")
        ok = run_config(k, q, P_WARM_VALUES, args.vars,
                       dry_run=args.dry_run, skip_optimal=args.skip_optimal)
        if ok:
            succeeded += 1
        else:
            failed += 1

        if not args.dry_run and i < total:
            elapsed = time.time() - t_sweep_start
            avg = elapsed / i
            remaining = avg * (total - i)
            print(f"    ETA: {remaining / 60:.1f} min remaining")

    t_total = time.time() - t_sweep_start
    print(f"\nSweep complete: {succeeded} succeeded, {failed} failed ({t_total / 60:.1f} min total)")

    # Concatenate
    if not args.skip_concat and not args.dry_run:
        print("\nConcatenating results ...")
        concatenate_results(args.vars)


if __name__ == "__main__":
    main()
