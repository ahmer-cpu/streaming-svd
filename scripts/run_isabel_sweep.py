"""Isabel adaptive warm-vs-cold sweep (unified single-stage compressor).

Runs unified_adaptive_bench for the 4 featured Hurricane Isabel variables at
value-range-relative tolerances, in BOTH modes:

  warm : U_prev / k* carried across timesteps (streaming algorithm)
  cold : every timestep bootstrapped standalone (control arm; each timestep
         is treated exactly like one static NYX/Miranda variable)

tau is per-timestep (tau = eps * range(A_t)), so each Isabel snapshot is
methodologically comparable to a static-dataset variable.  This choice is
flagged for the presentation (PROGRESS.md, Phase-4 backlog).

Each (var, eps, mode) run writes results/hurricane/unified/
<var>_unified[_cold]_eps<eps>.csv.  Afterwards everything is concatenated
into results/hurricane/unified/isabel_all.csv with added `eps` and `mode`
columns for downstream warm-vs-cold analysis.

Usage:
    python scripts/run_isabel_sweep.py
    python scripts/run_isabel_sweep.py --dry-run
    python scripts/run_isabel_sweep.py --vars Uf --eps 1e-3 --modes warm
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXE = REPO_ROOT / "phase2_cpp" / "build" / "Release" / "unified_adaptive_bench.exe"
OUT_DIR = REPO_ROOT / "results" / "hurricane" / "unified"

# Featured variables (Phase 3 selection): dense wind, temperature, vapour,
# plus the sparse/unstable-rank control QRAINf.
VARS = ["Uf", "TCf", "QVAPORf", "QRAINf"]
EPS_VALUES = ["1e-2", "1e-3", "1e-4"]
MODES = ["warm", "cold"]

START_T, END_T = 1, 48

# Adaptive parameters (driver defaults, pinned here for the record).
K_MAX_INIT = 50
K_DELTA = 8
K_EXPAND = 16
FINE_RADIUS = 3
P_COLD = 10
P_WARM = 5
Q = 0
C_ENTRY = 12
SEED = 42


def run_one(var: str, eps: str, mode: str, dry_run: bool) -> Path:
    cmd = [
        str(EXE),
        "--dataset", "isabel",
        "--out-dir", str(OUT_DIR),
        "--vars", var,
        "--start", str(START_T),
        "--end", str(END_T),
        "--mode", mode,
        "--tau-mode", "vrel",
        "--eps", eps,
        "--k-max-init", str(K_MAX_INIT),
        "--k-delta", str(K_DELTA),
        "--k-expand", str(K_EXPAND),
        "--fine-radius", str(FINE_RADIUS),
        "--p-cold", str(P_COLD),
        "--p-warm", str(P_WARM),
        "--q", str(Q),
        "--c-entry", str(C_ENTRY),
        "--seed", str(SEED),
    ]
    mode_tag = "_cold" if mode == "cold" else ""
    csv_path = OUT_DIR / f"{var}_unified{mode_tag}_eps{eps}.csv"

    print(f"\n=== {var}  eps={eps}  mode={mode} ===")
    print("  " + " ".join(cmd))
    if dry_run:
        return csv_path

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"unified_adaptive_bench failed: {var} eps={eps} mode={mode}")
    print(f"  done in {time.time() - t0:.1f}s -> {csv_path}")
    return csv_path


def concatenate(csv_paths: list[Path]) -> Path:
    import pandas as pd

    frames = []
    for p in csv_paths:
        if not p.exists():
            print(f"  (skip missing {p})")
            continue
        df = pd.read_csv(p)
        m = re.match(r"(?P<var>.+?)_unified(?P<cold>_cold)?_eps(?P<eps>.+)\.csv", p.name)
        df.insert(0, "mode", "cold" if (m and m.group("cold")) else "warm")
        df.insert(1, "eps", float(m.group("eps")) if m else float("nan"))
        frames.append(df)

    if not frames:
        raise SystemExit("No CSVs to concatenate.")
    combined = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR / "isabel_all.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nConcatenated {len(frames)} CSV(s), {len(combined)} rows -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vars", nargs="+", default=VARS)
    ap.add_argument("--eps", nargs="+", default=EPS_VALUES, dest="eps_values")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-concat", action="store_true")
    args = ap.parse_args()

    if not EXE.exists():
        raise SystemExit(f"Executable not found: {EXE}\n"
                         "Build it: cmake --build phase2_cpp/build --config Release "
                         "--target unified_adaptive_bench")

    csv_paths = []
    for var in args.vars:
        for eps in args.eps_values:
            for mode in args.modes:
                csv_paths.append(run_one(var, eps, mode, args.dry_run))

    if args.dry_run or args.no_concat:
        return
    concatenate(csv_paths)


if __name__ == "__main__":
    main()
