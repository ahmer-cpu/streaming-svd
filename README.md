# Streaming SVD

Research package for Streaming and Warm-started Randomized Singular Value Decomposition (rSVD).
This repository implements and evaluates adaptive, error-bounded lossy compression of scientific
simulation data: each snapshot is stored as a cost-optimal low-rank layer plus exact sparse
corrections, guaranteeing `||A - A_hat||_max <= tau` by construction. Temporal datasets
(Hurricane Isabel) additionally warm-start each timestep's factorization from the previous one.

The current compressor is a **unified single-stage** design (`unified_adaptive_bench`), evaluated
on Hurricane Isabel (warm vs cold), NYX, and Miranda.

> **Note.** Early exploratory experiments — applying the warm-started algorithm to a single
> Hurricane Isabel variable to verify feasibility and initialize the method — were prototyped in
> Python. That prototyping is not included in this repository; everything here is the C++
> implementation and the analysis pipeline used to produce the reported results.

## Quick Start

```powershell
# Build (from repo root; requires CMake + MSVC + Eigen3 + OpenBLAS)
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release

# Isabel (temporal, warm-started), absolute tolerance
./cpp/build/Release/unified_adaptive_bench.exe --dataset isabel --vars Uf --tau 1.0

# NYX / Miranda (static, cold), value-range-relative tolerance
./cpp/build/Release/unified_adaptive_bench.exe --dataset miranda --tau-mode vrel --eps 1e-3

# Full sweeps + analysis
python scripts/run_isabel_sweep.py          # 4 featured vars x 3 eps x warm/cold
python scripts/run_static_sweep.py          # NYX + Miranda, all vars x 3 eps
python analysis/hurricane/analyze_unified.py
python analysis/static/analyze_static.py --input results/static/static_all_unified.csv
```

## Requirements

- **C++ build** — CMake ≥ 3.18, MSVC, Eigen3, OpenBLAS (managed via vcpkg; see `cpp/README.md`).
- **Analysis** — Python 3.8+ with `pandas` and `matplotlib` for the scripts under `analysis/`
  and `scripts/`.

## Project Structure

```
streaming-svd/
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore rules
│
├── cpp/                               # C++ CPU implementation (Eigen + OpenBLAS)
│   ├── include/                       #   Headers: rsvd.hpp, warm_rsvd.hpp, metrics.hpp, ...
│   ├── src/                           #   Sources: rsvd.cpp, warm_rsvd.cpp, *_experiment.cpp, ...
│   ├── CMakeLists.txt                 #   Build system
│   ├── vcpkg.json                     #   Dependency manifest
│   └── README.md                      #   Build + executable reference
│
├── analysis/                         # Python analysis/plotting on C++ output CSVs
│   ├── hurricane/
│   │   ├── analyze.py                #   Aggregate fixed-rank raw CSVs into summary statistics
│   │   ├── plot.py                   #   Fixed-rank figures
│   │   ├── plot_adaptive.py          #   Per-variable adaptive figures
│   │   └── analyze_unified.py        #   Warm-vs-cold tables, overlays, heatmaps (isabel_all.csv)
│   └── static/
│       └── analyze_static.py         #   NYX/Miranda sweep summary + figures
│
├── scripts/                          # Experiment runners
│   ├── run_isabel_sweep.py           #   Isabel: 4 featured vars × eps × warm/cold → isabel_all.csv
│   ├── run_static_sweep.py           #   NYX/Miranda eps sweep (--bench unified|two-stage)
│   └── run_sweep.py, analyze_sweep.py#   Fixed-rank (k, p, q) parameter sweep
│
├── data/                             # Datasets (raw binaries, not tracked in git)
│   ├── ISABEL_raw/                   #   Hurricane Isabel: 13 vars × 48 timesteps, {VAR}/{VAR}{T:02d}.bin
│   ├── MIRANDA_raw/                  #   Miranda: 7 vars, 384×384×256 float64 ({var}.d64)
│   └── NYX_raw/                      #   NYX: 6 vars, 512³ float32 ({var}.f32)
│
└── results/                          # Output CSVs (not in git) and figures (PNGs tracked)
    ├── hurricane/                    #   raw_cpp/ (fixed-rank), adaptive/ (two-stage legacy),
    │                                 #   unified/ (warm/cold), figures_* dirs
    ├── nyx/, miranda/                #   unified/ static-run CSVs per dataset
    ├── static/                       #   Concatenated static results + figures
    └── sweep/                        #   Parameter-sweep outputs
```

### Layout Notes

- **`cpp/`** — the C++ codebase (CMake + Eigen + OpenBLAS). Executables:
  - `unified_adaptive_bench` — **active**: single-stage adaptive compressor (L + S), one driver
    for Isabel (`--mode warm|cold`) and NYX/Miranda (cold static);
  - `adaptive_bench` / `static_adaptive_bench` — legacy two-stage (L1+L2+S) baselines;
  - `hurricane_bench` / `hurricane_dumb_bench` — fixed-rank cold-vs-warm benchmarks.
- **`analysis/`** — Python scripts that aggregate and plot the CSVs emitted by the C++ benchmarks.
- **`data/`** — three raw datasets (`ISABEL_raw`, `MIRANDA_raw`, `NYX_raw`); binaries are not
  tracked in git.

## License

MIT
