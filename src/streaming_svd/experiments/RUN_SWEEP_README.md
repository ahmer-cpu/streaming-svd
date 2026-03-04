# run_sweep.py - Parameterized Sweep Runner

## Overview

`run_sweep.py` is a comprehensive parameter sweep runner that studies the behavior of warm-start vs cold-start randomized SVD across three different experiment types:

1. **Series**: Independent random matrices (control experiment)
2. **Perturbation**: Additive perturbation experiment
3. **Rotation**: Rotating subspace experiment

The script collects error metrics, aggregates results across seeds, and generates plots comparing warm-start and cold-start performance.

## Quick Start

Run a minimal default sweep:
```bash
cd src/streaming_svd/experiments
python run_sweep.py
```

## Usage Examples

### Run only series experiments with custom parameters:
```bash
python run_sweep.py \
  --experiments series \
  --m-list 500 1000 \
  --n-list 500 1000 \
  --k-list 10 20 \
  --p-cold-list 5 10 \
  --p-warm-list 0 5 10 \
  --q-list 0 1 \
  --T 20 \
  --n-seeds 10
```

### Run all three experiment types with reduced parameters:
```bash
python run_sweep.py \
  --experiments series perturbation rotation \
  --m-list 200 \
  --n-list 200 \
  --k-list 10 \
  --p-cold-list 5 \
  --p-warm-list 0 5 \
  --T 10 \
  --n-seeds 3
```

### Run with GPU acceleration:
```bash
python run_sweep.py \
  --device cuda \
  --n-seeds 5
```

### Run quietly (suppress progress output):
```bash
python run_sweep.py --quiet
```

## Command-Line Options

### Experiment Selection
- `--experiments` {series,perturbation,rotation}: Which experiment types to run (default: all three)

### Parameter Grids
- `--m-list`: Row dimensions (default: 500 1000)
- `--n-list`: Column dimensions (default: 500 1000)
- `--k-list`: Target ranks (default: 10 20 40)
- `--p-cold-list`: Cold-start oversampling parameters (default: 5 10 20)
- `--p-warm-list`: Warm-start oversampling parameters (default: 0 5 10 20)
- `--q-list`: Power iteration counts (default: 0 1)

### Experiment Parameters
- `--T`: Number of timesteps (default: 10)
- `--n-seeds`: Number of random seeds per configuration (default: 5)
- `--seed0`: Base random seed (default: 42)

### Output Options
- `--output-raw`: Output path for raw results (default: results/sweep_raw.csv)
- `--output-summary`: Output path for aggregated summary (default: results/sweep_summary.csv)
- `--fig-dir`: Directory for output figures (default: results/figures)

### Miscellaneous
- `--device`: {cpu, cuda} - Computation device (default: cpu)
- `--quiet`: Suppress progress output

## Output Files

### 1. `sweep_raw.csv`
Raw results with one row per configuration run.

Columns:
- `experiment`: Experiment type (series, perturbation, rotation)
- `seed`: Random seed
- `m`, `n`, `k`: Matrix dimensions and target rank
- `p_cold`, `p_warm`, `q`: Algorithm parameters
- `mean_cold_error`: Mean error across timesteps for cold-start
- `mean_warm_error`: Mean error across timesteps for warm-start
- `mean_gap`: Average difference (warm - cold)
- `mean_ratio`: Average ratio (warm / cold)
- `final_gap`: Difference at final timestep (warm_T - cold_T)

### 2. `sweep_summary.csv`
Aggregated statistics grouped by configuration (excluding seed).

Columns:
- Grouping columns: `experiment`, `m`, `n`, `k`, `p_cold`, `p_warm`, `q`
- `mean_gap_mean`: Mean of mean_gap across seeds
- `mean_gap_std`: Standard deviation across seeds
- `mean_ratio_mean`: Mean of mean_ratio across seeds
- `mean_ratio_std`: Standard deviation across seeds
- `fraction_warm_better`: Proportion of runs where warm-start has lower error

### 3. Plots
- `sweep_error_gap_hist.png`: Histograms of mean_gap for each experiment type
- `sweep_fraction_warm_better.png`: Fraction of runs where warm-start is better vs p_warm

## Key Metrics

The sweep focuses on **error metrics only** and ignores runtime, matmul counts, and other complexity metrics:

- **mean_gap**: How much warm-start differs from cold-start on average
  - Negative = warm-start is better
  - Positive = cold-start is better

- **mean_ratio**: Multiplicative factor of warm-start vs cold-start error
  - < 1 = warm-start is better
  - > 1 = cold-start is better

- **fraction_warm_better**: Percentage of individual runs where warm-start outperforms cold-start
  - Useful for understanding consistency of improvements

## Error Handling

The script includes robust error handling:
- Failed individual runs are logged and execution continues
- Output directories are created automatically
- Use `--quiet` flag to suppress error messages during production runs

## Performance Considerations

- Default parameters run ~625 configurations (5 experiments × 4m × 4n × 3k × 3p_cold × 4p_warm × 2q × 5 seeds)
- Each configuration takes 1-5 seconds depending on matrix dimensions
- Full default sweep takes ~1-2 hours
- Use smaller parameter grids for quick testing

## Example Results Interpretation

Looking at `sweep_summary.csv`:

```
experiment,m,n,k,p_cold,p_warm,q,mean_gap_mean,mean_gap_std,fraction_warm_better
perturbation,500,500,20,10,5,0,-0.05,0.01,0.95
```

This shows:
- For perturbation experiments with 500×500 matrices and k=20
- Using p_cold=10 and p_warm=5
- Warm-start is on average 0.05 units better than cold-start
- This improvement holds in 95% of runs
- Variation across seeds is low (std=0.01)

## Modifying Existing Experiments

The script imports and uses existing experiment runners without modification:
- `run_series_experiment()` from `run_series.py`
- `run_experiment_additive()` from `run_synthetic.py`
- `run_experiment_rotating()` from `run_synthetic.py`

To change experiment behavior, modify the respective `run_*.py` files and re-run the sweep.
