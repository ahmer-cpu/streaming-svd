# Hurricane SVD C++ Benchmark

C++ reimplementation of the hurricane cold- vs warm-start rSVD experiment.
Removes all Python/PyTorch overhead from timing measurements.

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

The executable will be at `cpp/build/Release/hurricane_bench.exe`.

## Usage

```powershell
# Single variable smoke test (3 timesteps)
./cpp/build/Release/hurricane_bench.exe `
    --data-dir data/raw `
    --out-dir results/hurricane/raw_cpp `
    --vars Uf --start 1 --end 3

# Full experiment (all 13 variables × 48 timesteps)
./cpp/build/Release/hurricane_bench.exe `
    --data-dir data/raw `
    --out-dir results/hurricane/raw_cpp `
    --start 1 --end 48 `
    --k 20 --p-cold 10 --p-warm 5 --q 0 --seed 42
```

## Downstream analysis

The output CSVs in `results/hurricane/raw_cpp/` have the same column schema as
the Python pipeline.  You can run the existing analysis and plot stages on them:

```bash
python -m streaming_svd.experiments.hurricane.analyze \
    --raw-dir results/hurricane/raw_cpp \
    --out results/hurricane/hurricane_summary_cpp.csv \
    --print-table

python -m streaming_svd.experiments.hurricane.plot \
    --raw-dir results/hurricane/raw_cpp \
    --summary results/hurricane/hurricane_summary_cpp.csv \
    --fig-dir results/hurricane/figures_cpp
```

## Algorithm notes

| Algorithm | Sketch width | Matmuls (q=0) | Bottleneck |
|-----------|-------------|---------------|------------|
| Cold rSVD | k + p_cold = 30 | AX:1, ATX:1 | QR of (250k × 30) |
| Warm rSVD | r_prev + p_warm = 25 | AX:2, ATX:2 | QR of (250k × 25) + 2 extra matmuls |

Key timing phases recorded per timestep:

- **Cold**: `omega_gen`, `initial_matmul`, `power_iter`, `qr`, `projection`, `small_svd`, `lift`
- **Warm**: `warm_proj`, `warm_matmul`, `omega_gen`, `random_matmul`, `concat`, `power_iter`, `qr`, `projection`, `small_svd`, `lift`

All timings use `std::chrono::steady_clock` wrapping only the Eigen/BLAS calls,
with no Python interpreter overhead.
