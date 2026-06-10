# Streaming SVD

Research package for Streaming and Warm-started Randomized Singular Value Decomposition (rSVD).
This repository provides a framework for implementing and evaluating streaming SVD algorithms with support for warm-start initialization.

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
│   └── hurricane/
│       ├── analyze.py                #   Aggregate raw CSVs into summary statistics
│       ├── plot.py                   #   Generate figures from CSVs
│       └── plot_adaptive.py          #   Figures for the adaptive-rank experiment
│
├── phase1_python_prototype/          # Archived Python/PyTorch prototype (Phase 1 — complete)
│   ├── src/streaming_svd/            #   Original package: algos/, sims/, experiments/, data/, utils/
│   ├── tests/                        #   pytest tests
│   ├── notebooks/                    #   Exploratory Jupyter notebooks
│   ├── scripts/                      #   Plotting scripts for preliminary sweeps
│   └── pyproject.toml                #   Python package metadata
│
├── scripts/                          # Parameter-sweep driver + analysis (run_sweep.py, analyze_sweep.py)
│
├── data/                             # Datasets (raw float32 binaries, not tracked in git)
│   ├── ISABEL_raw/                   #   Hurricane Isabel: 13 vars × 48 timesteps, {VAR}/{VAR}{T:02d}.bin
│   ├── MIRANDA_raw/                  #   Miranda dataset
│   └── NYX_raw/                      #   Nyx dataset
│
├── results/                          # Output CSVs and figures (not tracked in git)
│   ├── hurricane/                    #   C++ benchmark outputs (raw_cpp/, figures_cpp/, ...)
│   ├── sweep/                        #   Parameter-sweep outputs
│   └── preliminary/                  #   Phase 1 Python sweep results
│
└── docs/                             # Documentation
```

### Layout Notes

- **`phase2_cpp/`** — the active codebase: cold-start (Halko) vs. warm-start (Brand) rSVD in C++, built with CMake + Eigen + OpenBLAS.
- **`analysis/hurricane/`** — Python scripts that aggregate and plot the per-timestep CSVs emitted by the C++ benchmark.
- **`phase1_python_prototype/`** — the original PyTorch prototype, archived for reference; not under active development.
- **`data/`** — three raw datasets (`ISABEL_raw`, `MIRANDA_raw`, `NYX_raw`); binaries are not tracked in git.

## Usage

```python
from streaming_svd import algos, data, experiments, utils

# Import specific modules as needed
# from streaming_svd.algos import your_algorithm
# from streaming_svd.data import your_loader
```

## Tests

Run the test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov=streaming_svd
```

## Development

Format code with black:
```bash
black src tests
```

Sort imports with isort:
```bash
isort src tests
```

Type check with mypy:
```bash
mypy src
```

Lint with flake8:
```bash
flake8 src tests
```

## License

MIT

