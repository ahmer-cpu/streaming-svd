# Streaming SVD — CLI Handoff Plan
*Generated 2026-05-26. Pick up any task below directly in CLI Claude Code.*

> **STATUS (2026-06-10): SUPERSEDED.** Tasks 1–5 below were completed (or made moot)
> by the Phase-3 two-stage adaptive compressor and the Phase-4 unified single-stage
> redesign. Current state, results, and backlog live in `PROGRESS.md` (Phase 4 section).
> This file is kept for the theoretical notes (error-bound derivations) and the
> dataset survey in the backlog section.

---

## What Has Been Implemented (Phase 2 C++)

### Error-Bound Hierarchy (completed this session)

Three-level hierarchy for bounding `||A - A_k||_max` (max elementwise reconstruction error):

| Level | Bound | Formula | Notes |
|-------|-------|---------|-------|
| 1 | Spectral | `||(I - UU^T)A||_2` | Upper bound via power iteration; `spectral_norm_residual()` |
| 2 | Leverage-score | `σ_{k+1} · sqrt((1-min_τ_U)(1-min_τ_V))` | Tighter when subspace is incoherent |
| 3 | Exact | `max|R_{ij}|` = `cold_max_elem_error` | Direct materialization of residual |

**Files modified:**
- `phase2_cpp/include/metrics.hpp` — `CompressionMetrics` struct extended (3 new fields), `spectral_norm_residual()` declared, `compression_metrics()` signature updated
- `phase2_cpp/src/metrics.cpp` — Both new functions implemented
- `phase2_cpp/include/csv_writer.hpp` — 8 new `RowData` fields; `NUM_COLS` = 77 (was 69)
- `phase2_cpp/src/csv_writer.cpp` — 8 new header columns + `write_row()` entries
- `phase2_cpp/src/hurricane_experiment.cpp` — `spectral_norm_residual()` called for cold/warm; new row fields populated

**CSV schema (77 columns):** The 8 new columns appended at the end (cols 70–77):
```
cold_spectral_bound, warm_spectral_bound,
cold_leverage_bound, warm_leverage_bound,
cold_min_leverage_U, warm_min_leverage_U,
cold_min_leverage_V, warm_min_leverage_V
```

---

## Empirical Findings (from results already run)

Variables tested: **Uf, TCf, QVAPORf** (plus Vf, QSNOWf from earlier slides).

### Key results
- PSNR: 50–80 dB depending on variable (warm consistently 1.5–3 dB better than cold)
- Speedup: warm is ~1.2× faster than cold (consistent across variables)
- **Max error is 10× larger than the 99th-percentile error** — spatially concentrated near the hurricane eyewall

### Error bounds are extremely loose
- `cold_spectral_bound / cold_max_elem_error` ≈ **63–118× loose**
- Leverage bound provides no improvement over spectral bound
- Root cause: `cold_min_leverage_U` ≈ **0.000004** (some spatial rows are nearly orthogonal to the entire top-20 subspace)
- Interpretation: hurricane data is **coherent** at eyewall points — the data violates the incoherence assumption needed for tight leverage bounds
- **Conclusion:** bounds are theoretically correct but practically useless for this dataset; the exact residual is the tool of interest

### Residual regime identification
The residual `R = A - A_k` likely falls in **Regime 1 (low-rank) or Regime 2 (row-sparse)**:
- **Regime 1 (low-rank):** `σ_{k+1}(R) / ||R||_F ≈ 1` — residual is essentially rank-1 or rank-2; second-stage rSVD directly applicable
- **Regime 2 (row-sparse):** `min_leverage_U ≈ 0` identifies specific spatial rows (eyewall points) that carry most of the error; storing those rows explicitly gives near-exact reconstruction for those points
- **Regime 3 (entry-sparse):** NOT the case here — eyewall errors are spatially structured, not isolated entries

**Diagnostic:** `ρ = spectral_bound / (||R||_F)` — if ρ ≈ 1, residual is approximately rank-1 → Regime 1.

---

## Next Implementation Tasks (in priority order)

### Task 1 — Residual Regime Diagnostic
**Goal:** Automatically classify the residual into Regime 1/2/3 and output the regime class per timestep.

**What to add to `hurricane_experiment.cpp`:**
```cpp
// After computing cold_cm and warm_cm:
// Regime diagnostic: ρ = spectral_bound / ||R||_F
// ||R||_F ≈ sqrt(||A||_F^2 - ||s||^2)  [already have fro_error*||A||_F]
// For now just output spectral_bound / (cold_fro_error * ||A||_F) as a ratio
```

Add to `RowData` / CSV: `cold_residual_regime_rho`, `warm_residual_regime_rho`

### Task 2 — Second-Stage rSVD on Residual
**Goal:** After computing `A_k = U_k Σ_k V_k^T`, compute a low-rank approximation to `R = A - A_k` with rank `k2`.

**Approach:**
```
R = A - U_k * (s.asDiagonal() * Vt_k)  // materialize residual (250000×100, ~95MB)
SVDResult r2_cold = cold_rsvd(R, k2, p2, q2, seed+1);
// Warm-started: use cold r2_cold.U from t-1 as warm start for r2_warm
SVDResult r2_warm = warm_rsvd(R, &r2_prev_U, k2, p2, q2, seed+1);

// Combined approximation:
// A ≈ U_k Σ_k V_k^T + U2 Σ2 V2^T
// max_elem_error of combined vs. single-stage
```

**Parameters to sweep:** `k2 = 2, 3, 5` (expect k2=2–3 sufficient for eyewall anomaly)

**Metrics to add:**
- `r2_cold_fro_error`, `r2_warm_fro_error` — does second stage close the gap?
- `r2_cold_max_elem_error`, `r2_warm_max_elem_error` — combined max error
- `r2_cold_time`, `r2_warm_time` — overhead of second stage
- `r2_warm_speedup` — warm speedup on second stage (should be larger, residual more stable)

**Implementation notes:**
- Materialize R explicitly: `MatF R = A - U * s.asDiagonal() * Vt;`
- Second-stage warm start: save `r2_U` at end of each timestep, pass as `U_prev` to next
- Second-stage is independently warm-startable (residual subspace also temporally coherent)

### Task 3 — Row-Sparse Correction
**Goal:** For rows with high leverage deficit `(1 - τ_i)`, store the full residual row explicitly to guarantee hard L∞ error control.

**Algorithm:**
```cpp
// Identify "bad" rows: leverage deficit above threshold
float threshold_tau = 0.01f;  // rows with tau_i < threshold_tau
VecF row_leverage = U.rowwise().squaredNorm();  // (m,) = tau_i for each spatial point
// bad_rows = {i : row_leverage(i) < threshold_tau}
// store R(bad_rows, :) = A(bad_rows, :) - A_k(bad_rows, :)
// At decode time: A_hat(bad_rows,:) = stored residual + A_k(bad_rows,:)  → exact
```

**Metrics to add:**
- `num_corrected_rows`, `corrected_row_fraction` — how many rows corrected
- `corrected_max_elem_error` — max error after row correction
- `corrected_storage_overhead` — bits for stored rows / bits for rank-k factors

**Expected result:** Very few rows need correction (eyewall ≈ <1% of 250000 spatial points), so storage overhead is small.

### Task 4 — New Variables to Test
**Recommended variables** (in order of interest):
- **Wf** — vertical velocity, most turbulent, expected worst bounds + most coherent eyewall
- **QRAINf or PRECIPf** — most spatially localized, good test for row-sparse regime
- **Pf** — pressure, smoothest field, expect tightest bounds and best compression

Run with: `--vars Wf QRAINf Pf --start 1 --end 48 --k 20 --p-cold 10 --p-warm 5 --q 0 --seed 42`

### Task 5 — Downstream Python Analysis for Error Hierarchy
**Add to `analysis/hurricane/plot.py`:**
1. Bound hierarchy over time: plot `cold_spectral_bound`, `cold_leverage_bound`, `cold_max_elem_error` on same axis per variable
2. Leverage score map: plot `cold_min_leverage_U` over timesteps (does it vary?)
3. Warm vs cold leverage comparison: `warm_min_leverage_U` vs `cold_min_leverage_U`
4. Regime diagnostic: `cold_residual_regime_rho` over time per variable

---

## Build & Run Commands

```powershell
# From repo root (Windows)
# Configure
cmake -B phase2_cpp/build -S phase2_cpp -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build phase2_cpp/build --config Release

# Smoke test (Uf, 3 timesteps)
.\phase2_cpp\build\Release\hurricane_bench.exe `
    --data-dir data/ISABEL_raw --out-dir results/hurricane/raw_cpp `
    --vars Uf --start 1 --end 3

# Full run (all variables)
.\phase2_cpp\build\Release\hurricane_bench.exe `
    --data-dir data/ISABEL_raw --out-dir results/hurricane/raw_cpp `
    --start 1 --end 48 --k 20 --p-cold 10 --p-warm 5 --q 0 --seed 42

# New variables only
.\phase2_cpp\build\Release\hurricane_bench.exe `
    --data-dir data/ISABEL_raw --out-dir results/hurricane/raw_cpp `
    --vars Wf QRAINf Pf --start 1 --end 48 --k 20 --p-cold 10 --p-warm 5 --q 0 --seed 42

# Downstream analysis
python analysis/hurricane/analyze.py `
    --raw-dir results/hurricane/raw_cpp `
    --out results/hurricane/hurricane_summary_cpp.csv --print-table

python analysis/hurricane/plot.py `
    --raw-dir results/hurricane/raw_cpp `
    --summary results/hurricane/hurricane_summary_cpp.csv `
    --fig-dir results/hurricane/figures_cpp
```

---

## Key Theoretical Notes (for context)

### Why σ_{k+1} from the small SVD is NOT an upper bound
The rSVD computes `B = Q^T A` (small, `(k+p) × n`). The SVD of `B` gives `σ_ℓ(B)`.
Since `Q` is a projection (orthonormal, tall): `σ_ℓ(Q^T A) ≤ σ_ℓ(A)` for all ℓ.
So `svd(B).singularValues()(k)` is a **lower bound** on `σ_{k+1}(A)`, NOT an upper bound.
**Correct upper bound:** `spectral_norm_residual(A, U)` = `||(I - UU^T)A||_2` via power iteration.

### Why leverage bounds are loose for hurricane data
The bound `||R||_max ≤ σ_{k+1} · sqrt((1-min_τ_U)(1-min_τ_V))` is tight only when the matrix is *incoherent* (leverage scores approximately uniform). Hurricane eyewall has `min_τ_U ≈ 4×10^{-6}`, making `(1-min_τ_U) ≈ 1`, so the bound collapses to `σ_{k+1}` (spectral bound), with no tightening. This is mathematically correct but diagnostically useless.

### Cauchy-Schwarz derivation of Level 2 bound
```
R_{ij} = Σ_{ℓ>k} σ_ℓ u_{iℓ} v_{jℓ}
        = a · b   where a = (σ_{k+1} u_{i,k+1}, ...), b = (v_{j,k+1}, ...)
|R_{ij}| ≤ ||a||_2 · ||b||_2
         = sqrt(Σ_{ℓ>k} σ_ℓ² u_{iℓ}²) · sqrt(Σ_{ℓ>k} v_{jℓ}²)
         ≤ σ_{k+1} · sqrt(Σ_{ℓ>k} u_{iℓ}²) · ||v_{j,>k}||
         = σ_{k+1} · sqrt(1 - τ_i^U) · sqrt(1 - τ_j^V)
||R||_max ≤ σ_{k+1} · sqrt((1-min_τ_U)(1-min_τ_V))
```

### Second-stage rSVD error bound
If `R = A - A_k` and `R_2 = R - R_k2` (second-stage residual), then:
```
||A - (A_k + R_k2)||_max = ||R - R_k2||_max ≤ σ_{k2+1}(R) · (leverage terms for R)
```
Since `R` is (likely) low-rank (Regime 1), `σ_{k2+1}(R)` drops sharply → tight bound.

---

## File Map (active files only)

```
phase2_cpp/
  include/
    metrics.hpp          ← CompressionMetrics{+min_lev_U, +min_lev_V, +lev_bound}
                            spectral_norm_residual() declared
                            compression_metrics(A, U, s, Vt, sigma_kp1_upper) declared
    csv_writer.hpp       ← RowData{+8 error-bound fields}, NUM_COLS=77
    rsvd.hpp             ← cold_rsvd() unchanged
    warm_rsvd.hpp        ← warm_rsvd() unchanged
  src/
    metrics.cpp          ← spectral_norm_residual() implemented
                            compression_metrics() now computes leverage bound
    csv_writer.cpp       ← 8 new header cols + write_row entries
    hurricane_experiment.cpp ← calls spectral_norm_residual, populates 8 new fields
    rsvd.cpp             ← unchanged
    warm_rsvd.cpp        ← unchanged

analysis/hurricane/
  analyze.py             ← aggregate CSVs (reads first 69 cols, safe w/ 77-col files)
  plot.py                ← generate figures (same, append-safe)

results/hurricane/raw_cpp/
  {VAR}_raw.csv          ← 77-col output (Uf, Vf, TCf, QVAPORf, QSNOWf done)
```

---

## Future Ideas / Backlog

### Second temporal dataset for warm-start (beyond Hurricane Isabel)
*Logged 2026-06-09.* The new SDRBench datasets (NYX, Miranda) are **single-snapshot** — one 3D volume per variable, so they only exercise the cold adaptive-rank path, not warm-start. Neither has a time-stepped version on SDRBench (the larger Miranda 3072³ is just higher spatial resolution, still one time step). If we want a *second* temporal sequence (different physics from Hurricane) to test warm-start `U_prev` reuse:

- **XGC** (SDRBench, fusion) — 20694×512 unstructured mesh, **9 timesteps**, 1.2 GB. Cleanest small multi-timestep option, but **ADIOS/BP format** → needs `adios2` reader or one-time conversion to raw `.f32` before it fits the existing loader.
- **NSTX GPI** (SDRBench, fusion) — 80×64 frames, **369k-step "movie"**, 4.1 GB, also ADIOS/BP.
- **JHTDB** (Johns Hopkins Turbulence DB) — time-resolved isotropic/channel turbulence; download cutouts at arbitrary timesteps as raw binary/HDF5. Lowest-friction (no exotic format), genuinely time-evolving.
- **NYX-with-time** only exists via running the Nyx code (AMReX-Astro) and reading its plotfiles (AMReX format → yt/VisIt → convert). Not a clean download.

Next step if pursued: pick XGC (on-site, ADIOS conversion) or JHTDB (off-site, clean format); build an ingest path → raw `.f32` → reuse `load_static_matrix` / warm `adaptive_bench`.

---

## Notes on Plugin Hook Error (harmless)
Every `Edit` call triggers: `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/check-sql-files.py"` with unexpanded env var → error. All edits succeed; this is a misconfigured hook in an unrelated plugin. Ignore it.
