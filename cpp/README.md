# Streaming SVD C++ Benchmarks

C++ implementation of the cold- vs warm-start rSVD experiments and the adaptive
error-bounded compressor. Timing measurements wrap only the Eigen/BLAS calls.

## Executables

| Target | Status | Purpose |
|--------|--------|---------|
| `unified_adaptive_bench` | **active** | Single-stage adaptive compressor `A_hat = L(rank k*) + S(sparse)` with hard guarantee `\|\|A - A_hat\|\|_max <= tau`. One driver: `--dataset isabel` (temporal; `--mode warm\|cold`) and `--dataset nyx\|miranda` (static, cold). |
| `adaptive_bench` | legacy baseline | Two-stage (L1+L2+S) adaptive driver for Isabel (superseded June 2026). |
| `static_adaptive_bench` | legacy baseline | Two-stage adaptive driver for NYX/Miranda. |
| `hurricane_bench` | baseline | Fixed-rank cold-vs-warm rSVD benchmark (all 13 Isabel variables). |
| `hurricane_dumb_bench` | control | Fixed-rank cold vs naive warm-start variant. |

## Prerequisites

- **CMake** ≥ 3.18
- **Visual Studio 2019/2022** (MSVC)
- **vcpkg** (for Eigen3 + OpenBLAS)

## Setup

### 1. Install vcpkg (if not already installed)

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
```

### 2. Install dependencies via vcpkg

```powershell
cd E:\PhD Year 2\SVD Project\streaming-svd\cpp
C:\vcpkg\vcpkg install --triplet x64-windows
```

This reads `vcpkg.json` and installs Eigen3 and OpenBLAS automatically.

### 3. Configure and build

```powershell
cd E:\PhD Year 2\SVD Project\streaming-svd

cmake -B cpp/build -S cpp `
      -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
      -DVCPKG_TARGET_TRIPLET=x64-windows `
      -DCMAKE_BUILD_TYPE=Release

cmake --build cpp/build --config Release
```

Executables land in `cpp/build/Release/`.

## Usage

### Adaptive compressor (unified, active)

```powershell
# Isabel, warm-started streaming, absolute tolerance
./cpp/build/Release/unified_adaptive_bench.exe `
    --dataset isabel --vars Uf TCf --start 1 --end 48 --tau 1.0

# Isabel cold control arm, per-snapshot relative tolerance
./cpp/build/Release/unified_adaptive_bench.exe `
    --dataset isabel --mode cold --tau-mode vrel --eps 1e-3

# NYX / Miranda (static, cold)
./cpp/build/Release/unified_adaptive_bench.exe `
    --dataset miranda --tau-mode vrel --eps 1e-3
```

Key options: `--k-max-init` (bootstrap rank cap), `--k-delta` (warm window
half-width), `--k-expand` (window growth on boundary hit), `--fine-radius`
(fine sweep half-width, default 3 = coarse grid step − 1), `--c-entry`
(sparse entry cost model, bytes). Run with `-h` for the full list.

### Fixed-rank benchmark

```powershell
# Single variable smoke test (3 timesteps)
./cpp/build/Release/hurricane_bench.exe `
    --data-dir data/ISABEL_raw `
    --out-dir results/hurricane/raw_cpp `
    --vars Uf --start 1 --end 3

# Full experiment (all 13 variables × 48 timesteps)
./cpp/build/Release/hurricane_bench.exe `
    --data-dir data/ISABEL_raw `
    --out-dir results/hurricane/raw_cpp `
    --start 1 --end 48 `
    --k 20 --p-cold 10 --p-warm 5 --q 0 --seed 42
```

## Downstream analysis

```bash
# Fixed-rank benchmark CSVs
python analysis/hurricane/analyze.py \
    --raw-dir results/hurricane/raw_cpp \
    --out results/hurricane/hurricane_summary_cpp.csv --print-table
python analysis/hurricane/plot.py \
    --raw-dir results/hurricane/raw_cpp \
    --summary results/hurricane/hurricane_summary_cpp.csv \
    --fig-dir results/hurricane/figures_cpp

# Unified adaptive sweeps (runners write the concatenated inputs)
python scripts/run_isabel_sweep.py
python scripts/run_static_sweep.py
python analysis/hurricane/analyze_unified.py
python analysis/static/analyze_static.py --input results/static/static_all_unified.csv
```

## Algorithm notes

| Algorithm | Sketch width | Matmuls (q=0) | Bottleneck |
|-----------|-------------|---------------|------------|
| Cold rSVD | k + p_cold = 30 | AX:1, ATX:1 | QR of (250k × 30) |
| Warm rSVD | r_prev + p_warm = 25 | AX:2, ATX:2 | QR of (250k × 25) + 2 extra matmuls |

Key timing phases recorded per timestep:

- **Cold**: `omega_gen`, `initial_matmul`, `power_iter`, `qr`, `projection`, `small_svd`, `lift`
- **Warm**: `warm_proj`, `warm_matmul`, `omega_gen`, `random_matmul`, `concat`, `power_iter`, `qr`, `projection`, `small_svd`, `lift`

All timings use `std::chrono::steady_clock` wrapping only the Eigen/BLAS calls.
