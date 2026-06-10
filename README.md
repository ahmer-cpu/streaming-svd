# Streaming SVD

Research package for Streaming and Warm-started Randomized Singular Value Decomposition (rSVD).
This repository implements and evaluates adaptive, error-bounded lossy compression of scientific
simulation data: each snapshot is stored as a cost-optimal low-rank layer plus exact sparse
corrections, guaranteeing `||A - A_hat||_max <= tau` by construction. Temporal datasets
(Hurricane Isabel) additionally warm-start each timestep's factorization from the previous one.

**Current phase (Phase 4):** unified single-stage adaptive compressor (`unified_adaptive_bench`)
evaluated on Hurricane Isabel (warm vs cold), NYX, and Miranda. See `PROGRESS.md` for the full
project history, results, and backlog.

## Quick Start (current phase)

```powershell
# Build (from repo root; requires CMake + MSVC + Eigen3 + OpenBLAS)
cmake -B phase2_cpp/build -S phase2_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build phase2_cpp/build --config Release

# Isabel (temporal, warm-started), absolute tolerance
./phase2_cpp/build/Release/unified_adaptive_bench.exe --dataset isabel --vars Uf --tau 1.0

# NYX / Miranda (static, cold), value-range-relative tolerance
./phase2_cpp/build/Release/unified_adaptive_bench.exe --dataset miranda --tau-mode vrel --eps 1e-3

# Full sweeps + analysis
python scripts/run_isabel_sweep.py          # 4 featured vars x 3 eps x warm/cold
python scripts/run_static_sweep.py          # NYX + Miranda, all vars x 3 eps
python analysis/hurricane/analyze_unified.py
python analysis/static/analyze_static.py --input results/static/static_all_unified.csv
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

4. **(Optional) Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

```
streaming-svd/
├── README.md                          # Project documentation
├── CLAUDE.md                          # Guidance for Claude Code
├── .gitignore                         # Git ignore rules
│
├── phase2_cpp/                        # ** ACTIVE ** C++ CPU implementation (Eigen + OpenBLAS)
│   ├── include/                       #   Headers: rsvd.hpp, warm_rsvd.hpp, metrics.hpp, ...
│   ├── src/                           #   Sources: rsvd.cpp, warm_rsvd.cpp, hurricane_experiment.cpp, ...
│   ├── CMakeLists.txt                 #   Build system
│   └── vcpkg.json                     #   Dependency manifest
│
├── analysis/                         # Python analysis/plotting on C++ output CSVs
│   ├── hurricane/
│   │   ├── analyze.py                #   Aggregate fixed-rank raw CSVs into summary statistics
│   │   ├── plot.py                   #   Fixed-rank figures
│   │   ├── plot_adaptive.py          #   Per-variable adaptive figures (two-stage + unified CSVs)
│   │   └── analyze_unified.py        #   Warm-vs-cold tables, overlays, heatmaps (isabel_all.csv)
│   └── static/
│       └── analyze_static.py         #   NYX/Miranda sweep summary + figures
│
├── phase1_python_prototype/          # Archived Python/PyTorch prototype (Phase 1 — complete)
│   ├── src/streaming_svd/            #   Original package: algos/, sims/, experiments/, data/, utils/
│   ├── tests/                        #   pytest tests
│   ├── notebooks/                    #   Exploratory Jupyter notebooks
│   ├── scripts/                      #   Plotting scripts for preliminary sweeps
│   └── pyproject.toml                #   Python package metadata
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
├── results/                          # Output CSVs (not in git) and figures (PNGs tracked)
│   ├── hurricane/                    #   raw_cpp/ (fixed-rank), adaptive/ (two-stage legacy),
│   │                                 #   unified/ (Phase-4 warm/cold), figures_* dirs
│   ├── nyx/, miranda/                #   unified/ static-run CSVs per dataset
│   ├── static/                       #   Concatenated static results + figures
│   └── sweep/                        #   Parameter-sweep outputs
│
└── docs/                             # Documentation + LaTeX slides/report (not tracked in git)
```

### Layout Notes

- **`phase2_cpp/`** — the active codebase (CMake + Eigen + OpenBLAS). Executables:
  - `unified_adaptive_bench` — **active**: single-stage adaptive compressor (L + S), one driver
    for Isabel (`--mode warm|cold`) and NYX/Miranda (cold static);
  - `adaptive_bench` / `static_adaptive_bench` — legacy two-stage (L1+L2+S) baselines;
  - `hurricane_bench` / `hurricane_dumb_bench` — fixed-rank cold-vs-warm benchmarks (Phase 2).
- **`analysis/`** — Python scripts that aggregate and plot the CSVs emitted by the C++ benchmarks.
- **`phase1_python_prototype/`** — the original PyTorch prototype, archived for reference; not under active development.
- **`data/`** — three raw datasets (`ISABEL_raw`, `MIRANDA_raw`, `NYX_raw`); binaries are not tracked in git.

## Archived Python Prototype (Phase 1)

The `streaming_svd` Python package, its tests (`pytest`), and dev tooling live under
`phase1_python_prototype/` and are kept for reference only — see its `pyproject.toml`.
The Installation section above applies to that archive; the active C++ phase only needs
Python (with pandas/matplotlib) for the analysis scripts.

## License

MIT

