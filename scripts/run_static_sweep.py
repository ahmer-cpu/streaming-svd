"""Static (single-snapshot) adaptive rSVD sweep for the SDRBench datasets.

Runs the adaptive compressor for each (dataset, eps) pair, where eps is the
value-range-relative error bound (tau = eps * (max - min) per variable).
Two benchmarks are supported via --bench:

  unified    (default)  cpp .. unified_adaptive_bench.exe
                        single-stage L + S compressor
  two-stage             cpp .. static_adaptive_bench.exe
                        legacy L1 + L2 + S compressor (kept as a baseline)

Each run writes one CSV under results/<dataset>/<unified|adaptive>/.  After
all runs complete, the per-run CSVs are concatenated into
results/static/static_all.csv (or static_all_unified.csv) with added
`dataset` and `eps` columns for downstream analysis.

Usage:
    python scripts/run_static_sweep.py
    python scripts/run_static_sweep.py --bench two-stage
    python scripts/run_static_sweep.py --dry-run
    python scripts/run_static_sweep.py --datasets miranda --eps 1e-3
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = REPO_ROOT / "cpp" / "build" / "Release"
EXES = {
    "unified":   BUILD_DIR / "unified_adaptive_bench.exe",
    "two-stage": BUILD_DIR / "static_adaptive_bench.exe",
}
OUT_ROOT = REPO_ROOT / "results"
STATIC_DIR = OUT_ROOT / "static"

DATASETS = ["nyx", "miranda"]
EPS_VALUES = ["1e-2", "1e-3", "1e-4"]

# Initial rank cap for the search.  Both drivers' boundary expansion grows
# this automatically when a tighter eps needs a higher rank, so one modest
# value works for every eps (the exact prune keeps low-rank cases cheap).
K_MAX_INIT = 32
K_EXPAND = 32

# Fixed adaptive-stage parameters (match the streaming driver defaults).
P_COLD = 10
Q = 0
R_MAX = 4        # two-stage only
R_EXPAND = 4     # two-stage only
P_STAGE2 = 5     # two-stage only
C_ENTRY = 12
SEED = 42


def run_one(bench: str, dataset: str, eps: str, dry_run: bool) -> Path:
    common = [
        "--dataset", dataset,
        "--tau-mode", "vrel",
        "--eps", eps,
        "--k-max-init", str(K_MAX_INIT),
        "--k-expand", str(K_EXPAND),
        "--p-cold", str(P_COLD),
        "--q", str(Q),
        "--c-entry", str(C_ENTRY),
        "--seed", str(SEED),
    ]
    if bench == "unified":
        out_dir = OUT_ROOT / dataset / "unified"
        cmd = [str(EXES[bench]), "--out-dir", str(out_dir)] + common
        csv_path = out_dir / f"{dataset}_unified_eps{eps}.csv"
    else:
        out_dir = OUT_ROOT / dataset / "adaptive"
        cmd = [str(EXES[bench]), "--out-dir", str(out_dir)] + common + [
            "--r-max", str(R_MAX),
            "--r-expand", str(R_EXPAND),
            "--p-stage2", str(P_STAGE2),
        ]
        csv_path = out_dir / f"{dataset}_static_adaptive_eps{eps}.csv"

    print(f"\n=== [{bench}] {dataset}  eps={eps}  "
          f"(k_max_init={K_MAX_INIT}, k_expand={K_EXPAND}) ===")
    print("  " + " ".join(cmd))
    if dry_run:
        return csv_path

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"static_adaptive_bench failed for {dataset} eps={eps}")

    n_warn = len(re.findall(r"\[WARN k\* hit", proc.stdout))
    if n_warn:
        print(f"  !! {n_warn} variable(s) rank-capped at eps={eps}; "
              f"consider raising K_MAX_INIT['{eps}'].")
    print(f"  done in {time.time() - t0:.1f}s -> {csv_path}")
    return csv_path


def concatenate(csv_paths: list[Path], bench: str) -> Path:
    import pandas as pd

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in csv_paths:
        if not p.exists():
            print(f"  (skip missing {p})")
            continue
        df = pd.read_csv(p)
        # Recover dataset + eps from either filename pattern:
        #   <dataset>_static_adaptive_eps<eps>.csv  (two-stage)
        #   <dataset>_unified_eps<eps>.csv          (unified)
        m = re.match(r"(?P<ds>.+?)_(?:static_adaptive|unified)_eps(?P<eps>.+)\.csv",
                     p.name)
        df.insert(0, "dataset", m.group("ds") if m else p.parent.parent.name)
        df.insert(1, "eps", float(m.group("eps")) if m else float("nan"))
        frames.append(df)

    if not frames:
        raise SystemExit("No CSVs to concatenate.")
    combined = pd.concat(frames, ignore_index=True)
    name = "static_all_unified.csv" if bench == "unified" else "static_all.csv"
    out_path = STATIC_DIR / name
    combined.to_csv(out_path, index=False)
    print(f"\nConcatenated {len(frames)} CSV(s), {len(combined)} rows -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default="unified", choices=list(EXES),
                    help="Which compressor to run (default: unified)")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--eps", nargs="+", default=EPS_VALUES, dest="eps_values")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-concat", action="store_true",
                    help="Skip the concatenation step")
    args = ap.parse_args()

    exe = EXES[args.bench]
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe}\n"
                         "Build it: cmake --build cpp/build --config Release "
                         f"--target {exe.stem}")

    csv_paths = []
    for dataset in args.datasets:
        for eps in args.eps_values:
            csv_paths.append(run_one(args.bench, dataset, eps, args.dry_run))

    if args.dry_run or args.no_concat:
        return
    concatenate(csv_paths, args.bench)


if __name__ == "__main__":
    main()
