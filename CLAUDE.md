# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Streaming SVD** is a research package for implementing and comparing cold-start vs. warm-start randomized SVD (rSVD) algorithms in streaming scenarios. The central research question is whether re-using the previous snapshot's left singular vectors (`U_prev`) as a warm start reduces approximation error compared to the standard Halko et al. cold-start approach.

Three streaming regimes are studied:
- **Series**: Independent random matrices (control — warm start should provide no benefit)
- **Perturbation**: Additive noise (`S_t = S_{t-1} + E_t`) — subspace is stable across timesteps
- **Rotation**: Rotating subspace — smooth changes in principal directions

## Commands

### Setup

```bash
# Install in editable mode (from repo root)
pip install -e .

# Install with dev tools
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run the basic standalone test directly
python tests/test_rsvd_basic.py
```

### Code Quality

```bash
black src tests          # Format (line-length=100)
isort src tests          # Sort imports (black profile)
flake8 src tests         # Lint
mypy src                 # Type check
```

> See `docs/QUICK_START.md` for detailed sweep parameter reference and metric interpretation.

### Running Experiments

```bash
# Series (control) experiment — fast
python -m streaming_svd.experiments.run_series \
    --m 100 --n 100 --k 5 --T 5 --p-cold 10 --p-warm 5

# Synthetic dual-regime (additive + rotating)
python -m streaming_svd.experiments.run_synthetic \
    --m 1000 --n 1000 --k 20 --T 10 --eta 0.05 \
    --p-cold 10 --p-warm 5 --q 0 --device cpu

# Real weather data experiment
python -m streaming_svd.experiments.run_weather \
    --data-dir data/raw --var Uf --start 1 --end 10 \
    --k 20 --p-cold 10 --p-warm 5

# Full parameter sweep (can take 1-2 hours on CPU)
python -m streaming_svd.experiments.run_sweep \
    --experiments series perturbation rotation \
    --m-list 500 1000 --n-list 500 1000 \
    --k-list 10 20 40 --p-cold-list 5 10 20 \
    --p-warm-list 0 5 10 20 --q-list 0 1 \
    --T 10 --n-seeds 5 --device cpu

# Regenerate figures from saved CSV results (no re-computation)
python scripts/replot_sweep_results.py \
    --raw results/sweep_raw.csv \
    --summary results/sweep_summary.csv \
    --fig-dir results/figures
```

## Architecture

### Package layout: `src/streaming_svd/`

```
algos/          Core algorithm implementations
sims/           Synthetic data generators for streaming scenarios
experiments/    Experiment runners that tie algos + sims together
utils/          Shared utilities (logging, timing)
data/           Data loading/preprocessing
```

### Core Algorithms (`algos/`)

**`rsvd.py`** — Cold-start baseline (Halko et al. 2011):
- Draws `k+p` random Gaussian vectors, applies `q` power iterations, QR decompose, project, small SVD
- Signature: `rsvd(A, k, p=10, q=0, ...) -> (U, s, Vt, stats)`

**`warm_rsvd.py`** — Warm-start novel approach (Brand 2006):
- Computes `G = A.T @ U_prev` (warm component from previous basis) and concatenates with `p` fresh random vectors (exploration)
- Key advantage: total sketch vectors = `rank(U_prev) + p` instead of `k + p`; reuses structure across timesteps
- Signature: `warm_rsvd(A, U_prev, k, p=5, q=0, ...) -> (U, s, Vt, stats)`

**`metrics.py`** — Evaluation metrics:
- `rel_fro_error()`, `rel_spec_error_est()` — reconstruction quality
- `subspace_sin_theta()`, `subspace_sin_theta_fro()` — subspace alignment (canonical angles)

### Simulation Modules (`sims/`)

Each module generates a streaming sequence of matrices for a specific regime:
- **`perturbation.py`**: `make_initial_matrix()` + `perturb_step()` — additive low-rank noise each step
- **`rotating.py`**: `make_initial_matrix_rotating()` + `rotate_step()` — latent factors undergo orthogonal rotation each step
- **`series.py`**: `sample_independent_series()` — i.i.d. random matrices (no temporal correlation)

### Experiment Runners (`experiments/`)

Each runner imports from both `algos/` and `sims/`, runs both cold and warm rSVD at each timestep, collects metrics, saves CSVs and figures to `results/`:
- **`run_series.py`** — control experiment
- **`run_synthetic.py`** — additive + rotating dual-regime
- **`run_sweep.py`** — grid search over `[m, n, k, p_cold, p_warm, q]` × 3 regime types × multiple seeds; key output metrics are `mean_gap` (warm error minus cold error) and `fraction_warm_better`

### Results Layout

```
results/
  sweep_raw.csv       # One row per (config × seed)
  sweep_summary.csv   # Aggregated statistics per config
  series_results.csv
  weather_results.csv
  figures/            # PNG + PDF plots
```

### Data

`data/raw/` contains binary `.bin` files (`Uf01.bin`–`Uf17.bin`) representing 24-hour volumetric weather (temperature) data. See `docs/data_readme.txt` for format details.

## Key Dependencies

| Package | Role |
|---------|------|
| `numpy` | Matrix operations, random draws |
| `scipy` | QR decomposition, linear algebra |
| `torch` | Optional GPU acceleration (`--device cuda`) |
| `pandas` | Results aggregation and CSV I/O |
| `matplotlib` | Figure generation |

## References

- Halko, Martinsson, Tropp (2011) — "Finding structure with randomness"
- Brand (2006) — Warm-start / incremental SVD update strategy
