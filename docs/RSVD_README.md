# Streaming rSVD Implementation

Implementation of cold-start and warm-start randomized SVD algorithms for streaming matrices.

## Installation

```bash
# Install in editable mode with dependencies
pip install -e .
```

This will install PyTorch, NumPy, and SciPy as dependencies.

## Quick Start

### Running the Synthetic Experiment

The main experiment runner compares cold-start vs warm-start rSVD on synthetic streaming data:

```bash
# Basic run with default parameters
python -m streaming_svd.experiments.run_synthetic

# Custom parameters
python -m streaming_svd.experiments.run_synthetic \
    --m 2000 \
    --n 2000 \
    --k 50 \
    --T 20 \
    --eta 0.1 \
    --p-cold 15 \
    --p-warm 8 \
    --q 1 \
    --device cpu \
    --seed 42
```

**Parameters:**
- `--m`: Number of rows (default: 1000)
- `--n`: Number of columns (default: 1000)
- `--k`: Target rank (default: 20)
- `--T`: Number of time steps (default: 10)
- `--eta`: Perturbation magnitude (default: 0.05)
- `--p-cold`: Oversampling for cold-start (default: 10)
- `--p-warm`: Oversampling for warm-start (default: 5)
- `--q`: Power iterations (default: 0)
- `--device`: 'cpu' or 'cuda' (default: 'cpu')
- `--seed`: Random seed (default: 42)
- `--csv`: Save results to CSV file (optional)

### Using the algorithms directly

```python
import torch
from streaming_svd.algos import rsvd, warm_rsvd, rel_fro_error
from streaming_svd.sims import make_initial_matrix, perturb_step

# Generate synthetic matrix
m, n, k = 1000, 1000, 20
S, _, _, _ = make_initial_matrix(m, n, rank=k*2, device='cpu', seed=42)

# Cold-start rSVD
U_cold, s_cold, Vt_cold, stats = rsvd(S, k, p=10, q=0, device='cpu')
error = rel_fro_error(S, U_cold, s_cold, Vt_cold)
print(f"Cold-start error: {error:.6f}")
print(f"Time: {stats['timings']['total']:.4f}s")

# Perturb the matrix
S_new, E = perturb_step(S, eta=0.05, noise_rank=k, device='cpu')

# Warm-start rSVD using previous basis
U_warm, s_warm, Vt_warm, stats = warm_rsvd(
    S_new, U_cold, k, p=5, q=0, device='cpu'
)
error = rel_fro_error(S_new, U_warm, s_warm, Vt_warm)
print(f"Warm-start error: {error:.6f}")
print(f"Time: {stats['timings']['total']:.4f}s")
```

## Module Structure

### Core Algorithms (`streaming_svd.algos`)

- **`rsvd(A, k, p, q, ...)`**: Cold-start randomized SVD
  - Implements the Halko et al. algorithm
  - Returns U, s, Vt factorization and statistics
  
- **`warm_rsvd(A, U_prev, k, p, q, ...)`**: Warm-start randomized SVD
  - Uses previous subspace `U_prev` to reduce computation
  - Falls back to cold-start if `U_prev` is None
  
- **`rel_fro_error(A, U, s, Vt)`**: Relative Frobenius norm error
- **`rel_spec_error_est(A, U)`**: Estimated relative spectral error
- **`subspace_sin_theta(U1, U2)`**: Subspace distance (principal angles)

### Synthetic Data (`streaming_svd.sims`)

- **`make_initial_matrix(m, n, rank, decay, ...)`**: Generate low-rank matrix
  - Returns S = U @ diag(s) @ Vt with exponentially decaying singular values
  
- **`perturb_step(S_prev, eta, ...)`**: Add perturbation to matrix
  - Returns S_new = S_prev + E with controlled perturbation magnitude

### Experiments (`streaming_svd.experiments`)

- **`run_synthetic.py`**: Main experiment script
  - Generates streaming sequence S_t = S_{t-1} + E_t
  - Compares cold vs warm rSVD across time steps
  - Reports errors, times, and matmul counts

## Device Support

All functions accept a `device` parameter:

```python
# CPU
U, s, Vt, stats = rsvd(A, k, device='cpu')

# GPU (if CUDA available)
U, s, Vt, stats = rsvd(A, k, device='cuda')
```

## Testing

Run the basic test:

```bash
python tests/test_rsvd_basic.py
```

## Implementation Details

### Cold-start rSVD Algorithm

1. Draw random Gaussian matrix Ω ~ N(0,1) of shape (n, k+p)
2. Compute Y = A @ Ω
3. Power iterations: Y = A @ (A^T @ Y) repeated q times
4. Orthonormalize: Q = orth(Y) via QR decomposition
5. Project: B = Q^T @ A
6. Small SVD: B = Û @ diag(s) @ Vt
7. Lift: U = Q @ Û
8. Truncate to rank k

### Warm-start rSVD Algorithm

1. Compute G = A^T @ U_prev (warm component)
2. Y1 = A @ G (equals (A @ A^T) @ U_prev)
3. Draw Ω ~ N(0,1) of shape (n, p)
4. Y2 = A @ Ω (exploration component)
5. Y = [Y1, Y2] (concatenate)
6. Power iterations: Y = A @ (A^T @ Y) repeated q times
7. Orthonormalize: Q = orth(Y) via QR
8. Project: B = Q^T @ A
9. Small SVD: B = Û @ diag(s) @ Vt
10. Lift: U = Q @ Û
11. Truncate to rank k

**Key advantage**: Warm-start requires fewer random vectors (p << k+p typically), reducing matrix-vector products with A.

## References

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.

- Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear algebra and its applications, 415(1), 20-30.
