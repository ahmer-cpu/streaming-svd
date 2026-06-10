# Streaming SVD — Progress Tracker

**Project:** Warm-started higher-order randomized SVD (rHOSVD) on GPUs for streaming tensor data
**PI / Supervisor:** Dr. Kai Zhao (Department of Computer Science, Florida State University)
**Collaborator:** Argonne National Laboratory
**Researcher:** Ahmer Nadeem Khan (Research Assistant)
**Repository:** [ahmer-cpu/streaming-svd](https://github.com/ahmer-cpu/streaming-svd)

---

## Project Goal

Develop warm-started randomized HOSVD algorithms optimized for GPUs to efficiently conduct lossy compression of streaming tensor data from massive scientific simulations (e.g. weather/climate). The central research question: does reusing the previous snapshot's left singular vectors (`U_prev`) as a warm start reduce approximation error and/or wall-clock time compared to the standard Halko et al. (2011) cold-start rSVD?

---

## Dataset

**Hurricane Isabel** (IEEE Visualization 2004 Contest, WRF/NCAR simulation)

| Property | Value |
|----------|-------|
| Variables | 13: CLOUDf, Pf, PRECIPf, QCLOUDf, QGRAUPf, QICEf, QRAINf, QSNOWf, QVAPORf, TCf, Uf, Vf, Wf |
| Timesteps | 48 hourly snapshots per variable |
| Grid per timestep | 100 x 500 x 500 (25M voxels, ~95 MB float32) |
| Matrix form | Reshaped to (250,000 rows x 100 cols): rows = spatial points, cols = z-levels |
| Total volume | ~15.6 billion voxels across all variables and timesteps |
| Storage | `data/ISABEL_raw/{VAR}/{VAR}{T:02d}.bin` (raw float32, not tracked in git) |

**Two data regimes observed:**
- **Dense fields** (all grid points carry signal): Uf, Vf, Wf, TCf, Pf, QVAPORf — large dynamic ranges, 0% sparsity.
- **Sparse fields** (most grid points zero/near-zero): QCLOUDf (96%), QGRAUPf (83%), QRAINf (80%), QICEf (76%), CLOUDf (76%), QSNOWf (31%), PRECIPf (30%) — localised around the hurricane eyewall with very small magnitudes.

---

## Timeline & Milestones

### Phase 1 — Python Prototype (Feb–Mar 2026) [COMPLETE, ARCHIVED]

| Date | Milestone | Presentation |
|------|-----------|--------------|
| Feb 2026 | Ideation & literature review. Conceptualized warm-started rHOSVD for streaming tensor decomposition. | SVD 01 — Ideation |
| Mar 2026 | Built Python/PyTorch prototype. Validated on synthetic data with dual-regime experiments: **additive perturbation** (stable subspace with noise drift) and **rotating subspace** (smoothly changing U/V with fixed spectrum). Initial Hurricane Isabel runs. | SVD 02 — Preliminary Experiments |
| Mar 2026 | Comprehensive parameter sweep infrastructure over (k, p, q) grids with automated analysis and figure generation. | — |

**Deliverables:** Working Python prototype with cold-start and warm-start rSVD, synthetic experiments (additive perturbation + rotating subspace regimes), preliminary Hurricane Isabel results, and parameter sweep tooling.

---

### Phase 2 — C++ Fixed-Rank Benchmark (Apr–May 2026) [COMPLETE]

| Date | Milestone | Presentation |
|------|-----------|--------------|
| Apr 2026 | Rewrote core algorithms in C++ (Eigen + OpenBLAS) for production-grade benchmarking without Python overhead. | SVD 03 — Hurricane Benchmark |
| Apr 2026 | Full Hurricane Isabel benchmark: cold vs. warm rSVD across all 13 variables x 48 timesteps at fixed rank k=20. | SVD 03 |
| May 2026 | Compression fidelity deep-dive: PSNR, max elementwise error, tail percentiles, and reconstruction quality analysis across all 13 variables. | SVD 04 — Compression Fidelity |
| May 2026 | Error-bound hierarchy: 3-level bounding for max elementwise error. Extended CSV schema to 77 columns. | SVD 04 |

**Algorithms implemented:**
- **Cold-start rSVD** (Halko et al. 2011): Standard randomised SVD with `k+p` Gaussian sketch vectors.
- **Warm-start rSVD** (Brand 2006): Sketch = `[A(A^T U_prev), A Omega]` — reuses prior subspace + random exploration.
- **Naive warm-start rSVD** (control): Uses `U_prev` directly as sketch vectors without subspace iteration.

**Experiment design:** k=20, p_cold=10, p_warm=5, q=0, seed=42. All 13 variables x 48 timesteps. t=1 is cold-start only; t>=2 runs both cold and warm on the same data.

#### Phase 2 Key Results

**Compression fidelity summary (all 13 variables, rank k=20, 5x compression):**

| Variable | Cold PSNR | Warm PSNR | PSNR Gain | Max Elem Err (Cold) | Max Elem Err (Warm) | Speedup |
|----------|-----------|-----------|-----------|---------------------|---------------------|---------|
| Uf       | 50.6 dB   | 52.8 dB   | +2.3 dB   | 6.95                | 5.93                | 1.21x   |
| Vf       | 51.0 dB   | 53.2 dB   | +2.2 dB   | 6.22                | 5.24                | 1.21x   |
| TCf      | 55.7 dB   | 57.4 dB   | +1.8 dB   | 8.02                | 7.05                | 1.22x   |
| CLOUDf   | 59.1 dB   | 60.3 dB   | +1.2 dB   | 2.8e-4              | 2.8e-4              | 1.15x   |
| QVAPORf  | 60.4 dB   | 62.3 dB   | +2.0 dB   | 3.0e-3              | 2.2e-3              | 1.20x   |
| QCLOUDf  | 66.1 dB   | 67.1 dB   | +1.0 dB   | 2.8e-4              | 2.8e-4              | 1.18x   |
| Pf       | 72.7 dB   | 74.1 dB   | +1.4 dB   | 284                 | 277                 | 1.15x   |
| QICEf    | 74.4 dB   | 75.9 dB   | +1.5 dB   | 2.5e-5              | 2.5e-5              | 1.21x   |
| Wf       | 76.2 dB   | 78.7 dB   | +2.5 dB   | 0.311               | 0.246               | 1.18x   |
| QSNOWf   | 79.1 dB   | 79.9 dB   | +0.8 dB   | 4.2e-5              | 4.8e-5              | 1.12x   |
| PRECIPf  | 84.9 dB   | 85.6 dB   | +0.7 dB   | 1.2e-4              | 1.3e-4              | 1.10x   |
| QGRAUPf  | 98.5 dB   | 98.7 dB   | +0.2 dB   | 3.6e-5              | 4.6e-5              | 1.08x   |
| QRAINf   | **100.2 dB** | 99.4 dB | **-0.9 dB** | 3.1e-5           | 3.1e-5              | 1.05x   |

**Fixed-rank headline findings:**
1. **Warm-start improves fidelity for 12/13 variables** — PSNR gain +0.2 to +2.5 dB. Only QRAINf is worse (-0.9 dB) due to unstable subspace at ~100 dB.
2. **All 13 variables achieve >50 dB PSNR** at rank k=20 (5x compression). 99% of grid points have error below 1 m/s even for the hardest wind fields.
3. **Warm-start is faster, not slower** — 5–22% speedup across all 13 variables (peak 1.22x). Better fidelity *and* lower cost.
4. **Benefit scales with subspace stability** — strongest for wind/temperature, weakest for sporadic fields (QGRAUPf, QRAINf).
5. **Tail errors are spatially concentrated** — 99.9th percentile ~2-3x the 99th; max errors are 10x+ larger, concentrated at the hurricane eyewall.

**Error-bound hierarchy findings:**
- Spectral bound is **63–118x loose** relative to actual max error.
- Leverage-score bound provides **no tightening** for this dataset.
- Root cause: `min_leverage_U ~ 4e-6` at eyewall — hurricane data is highly coherent, violating incoherence assumptions.
- **Conclusion:** Bounds are theoretically correct but practically uninformative. Direct residual materialisation is the only reliable metric.

---

### Phase 3 — Adaptive Error-Bounded Compression (May 2026) [COMPLETE]

| Date | Milestone | Presentation |
|------|-----------|--------------|
| May 2026 | Designed and implemented the L1+L2+S three-layer decomposition with hard error guarantee `\|\|A - A_hat\|\|_max <= tau`. | SVD 05 — Adaptive Compression |
| May 2026 | Two-stage cost-driven rank optimisation with warm-state management across both stages. | SVD 05 |
| May 2026 | Ran adaptive experiments on 4 featured variables (Uf, TCf, QVAPORf, QRAINf) with scientifically meaningful tolerances. | SVD 05 |

**Algorithm: Three-Layer Decomposition (L1 + L2 + S)**

Given a user tolerance tau > 0, each snapshot is represented as:
```
A_hat = L1 (rank k*) + L2 (rank r*) + S (sparse corrections)
```

- **L1 — Main low-rank:** Warm-started rSVD at adaptive rank k*. Captures dominant smooth structure.
- **L2 — Residual low-rank:** Optional rSVD on R = A - L1 at adaptive rank r*. Captures coherent residual structure (e.g. eyewall anomaly). Skipped when residual is flat or sparse-only is cheaper.
- **S — Sparse corrections:** Stores entries where |error| > tau. At decode: corrected entries have error = 0, uncorrected have |error| <= tau.
- **Result:** `||A - A_hat||_max <= tau` **by construction**.

**Rank optimisation (greedy two-stage):**
- **Stage 1:** Compute one oversized rSVD (cold bootstrap at t=1, warm-tracked for t>=2), sweep candidate ranks k by truncation, select k* = argmin [C_rank(k) + violations(k) * c_entry].
- **Stage 2 skip logic:** (a) no violations → done; (b) sparse-only cheaper than 1 residual rank → skip; (c) residual spectral concentration < 20% → flat spectrum, skip.
- **Stage 2:** If not skipped, compute warm rSVD on residual, sweep r candidates, select r* by same cost criterion.
- **Escape mechanisms:** Expand search window if optimal rank hits boundary with high sparse cost.

**Warm-state management:** Stage 1 carries U_prev (k* columns); Stage 2 carries U2_prev (r* columns, or null). Both stages warm-started from previous timestep.

#### Phase 3 Key Results

**Variables and tolerances tested:**

| Variable | Tolerance tau | tau/Range |
|----------|--------------|-----------|
| Uf (wind) | 1.0 m/s | 0.9% |
| TCf (temperature) | 1.0 C | 0.9% |
| QVAPORf (vapour) | 1e-4 kg/kg | 0.5% |
| QRAINf (rain) | 1e-5 kg/kg | 0.4% |

**Per-variable adaptive results:**

| Variable | k* Range | r* Behaviour | Compression | Notes |
|----------|----------|--------------|-------------|-------|
| Uf | 12 → 16 (stable) | r*=1 at 2/48 steps | 5.3–7.4x | Sparse layer dominates; max error clamped to tau=1.0 m/s |
| TCf | 12 (constant) | r*=1 at 7/48 steps | 6.9–7.8x | Most stable subspace; stage 2 reduces violations ~45% when active |
| QVAPORf | 16 → 20 (rises at t=13) | r*=1 at 7/48 steps | 4.5–6.0x | Rank jump at t=13 drops violations from 344k to 34k |
| QRAINf | 1–6 (highly dynamic) | r*=0 (never activates) | 10.7–23.7x | Sparse-only always cheapest; handles unstable subspace naturally |

**Adaptive headline findings:**
1. **Hard error guarantee achieved** — `||A - A_hat||_max <= tau` holds by construction for all timesteps and all variables tested.
2. **Adaptive rank selection dramatically improves storage efficiency** by choosing variable- and timestep-dependent ranks instead of a fixed k=20.
3. **The L1+L2+S decomposition naturally handles both data regimes** — dense fields (Uf, TCf, QVAPORf) use moderate ranks + sparse corrections; sparse fields (QRAINf) use very low ranks + sparse-only, achieving 10-24x compression.
4. **QRAINf is no longer problematic** — the adaptive algorithm sidesteps the warm-start instability that hurt fixed-rank by choosing very low ranks and relying on sparse storage.
5. **Stage 2 is activated sparingly** (0–15% of timesteps) — the cost-driven skip logic avoids unnecessary residual SVDs.
6. **Combined PSNR is 1–3 dB above stage-1 alone** — sparse corrections improve average quality, not just worst-case.

---

### Phase 4 — Unified Single-Stage Compressor (June 2026) [IN PROGRESS]

| Date | Milestone |
|------|-----------|
| Jun 2026 | Redundancy analysis of the two-stage (L1+L2+S) design against the Phase-3 result data. |
| Jun 2026 | Implemented `unified_adaptive_bench`: one single-stage driver (L + S) for Isabel (warm) and NYX/Miranda (cold). Back-ported the incremental-reconstruction sweep + exact prune into the legacy two-stage driver (bit-identical selections, 3–9x faster stage 1). |
| Jun 2026 | Added `--mode cold` control arm + `scripts/run_isabel_sweep.py`. Experiment design locked (see below). |

**Phase-4 experiment design:**
- **Isabel**: 4 featured variables only (Uf, TCf, QVAPORf, QRAINf; expand later), t=1..48, eps in {1e-2, 1e-3, 1e-4} with **per-timestep tau = eps * range(A_t)**, in BOTH modes: `warm` (streaming, U_prev carried) and `cold` (standalone bootstrap per timestep) — the adaptive warm-vs-cold comparison. t=1 rows are identical in both arms (same seed) — built-in sanity check. Because the arms may pick different k*, compare at the system level (total bytes, total time, fidelity at chosen storage), not same-k quality. Runner: `scripts/run_isabel_sweep.py` -> `results/hurricane/unified/isabel_all.csv` (+ `mode`, `eps` columns).
- **NYX + Miranda**: all variables, same eps grid, cold/static path. Runner: `scripts/run_static_sweep.py` (unified default) -> `results/static/static_all_unified.csv`.
- Metric battery per row (59-col schema): k*, search window, violations, sparse/rank/total bytes, compression ratio, selection time, warm flag, fro error + optimal-fro overhead, PSNR (rank-only and combined), tail percentiles, max elementwise error vs tau.
- Smoke evidence (Uf, eps=1e-3, t=1..3): warm matches cold's k*=28 with ~10% fewer violations, ~1.5% smaller totals, ~15% less time.

**Why stage 2 was removed (evidence from Phase-3 CSVs):**
- Stage 2 changed the outcome in 4/39 static runs and ~8% of hurricane timesteps, always with r* in {1, 2}.
- Net byte benefit when it did fire: <0.3% of total storage. Wall-time cost: 8–32% per timestep.
- The spectrum probe gating stage 2 cost an rSVD of the same matrix at the same rank as the stage it gated, and passed in 13/39 static runs where the sweep then chose r*=0 anyway.
- The skip rules `no_violations` and `sparse_cheap` are theorems about the cost sweep's r=0 baseline (early exits, not logic).
- Structural flaw fixed: stage 2 could only ADD rank above the coarse grid (step 4); optima sitting 1–3 ranks BELOW a coarse point (~1 MB/rank on Isabel, up to 5–15% of storage) were invisible. The unified two-sided fine sweep finds them. Validated: Uf t=3 two-stage k*=16 / 17.1 MB -> unified k*=14 / 16.0 MB (−6%), faster.

**Unified algorithm (one sentence):** cost-optimal truncation rank of a single warm/cold rSVD — coarse grid, then a two-sided ±3 fine sweep and walk-down, expanding the search window whenever the argmin touches its top — plus exact sparse storage of all remaining tau-violations (which alone carries the guarantee).

**Optimality status:** exact over truncations of the computed factorization *under unimodality of cost(k)*; the coarse step (4) and fine radius (3) are coupled so the bracket argument is airtight. Not guaranteed under non-unimodal cost (violation "cliffs"), over all rank-k approximations (rSVD ≠ exact SVD; p/q shift the curve), or beyond the c_entry=12 byte cost model.

#### Backlog / flags for the next presentation

1. **q=0 → q=1 power-iteration sweep** — the designated replacement for stage 2's fresh-sketch error correction; cheap, not yet run under the adaptive cost objective.
2. **Per-snapshot `vrel` semantics** — `tau = eps * range(A_t)` is recomputed per snapshot, so tau drifts over a temporal sequence for the same variable (e.g. Uf: 0.106 -> 0.125 over t=1..3 as the storm intensifies). **Deliberate experiment-design choice (June 2026):** it makes every Isabel timestep methodologically identical to one static NYX/Miranda variable, so warm-vs-cold and cross-dataset comparisons share one knob (eps). The guarantee is per-snapshot relative, not absolute over the sequence — say this explicitly in the presentation.
3. **Gap-bound optimality certificate** — monotonicity of viol(k) gives a certified lower bound on every unevaluated rank (`cost(k) >= (k_a+1)*rank_bytes + viol(k_b)*c_entry` for k in gap (k_a, k_b)); a cheap post-search scan would upgrade "heuristic under unimodality" to "provably cost-optimal over truncations". Backlogged.
4. **fine_radius / grid-step coupling** — the ±3 fine sweep brackets exactly because the coarse step is 4; if the grid is ever coarsened the radius must track it (or item 3 makes the bracket moot).
5. **c_entry = 12 B/entry cost model** — naive COO assumption; eyewall violations are spatially clustered, so a real sparse encoder pays less, which would shift all optima toward lower rank. Revisit with the Phase-4 compression-backend work.
6. **Legacy drivers** (`adaptive_bench`, `static_adaptive_bench`) kept until the head-to-head benchmark is run, then archive.
7. **Warm-state policy (`U_prev`) alternatives** — today we carry exactly the k* stored columns from t-1 (principled: encoder state = decoder-reconstructible from the compressed stream). Ideas to test later:
   - **Carry all k_hi computed columns (or k* + fine_radius)** — free at t-1, shrinks the random padding `p_needed = k_hi - r_prev + p_warm`, directly attacks warm-sketch dilution at `k_expanded` timesteps. Caveat: the columns above k* are the least accurate (sketch-error tail) and widen the warm matmuls. One-line change; hypothesis = better quality/cost on expansion steps.
   - **V_prev warm start** — `Y = A @ V_prev` is ONE matmul vs two for `A @ (A^T @ U_prev)`, and Isabel's V (z-mixing, 100 x k) is likely very stable. But the current scheme implicitly applies one power iteration on the NEW data (`(A A^T) U_prev`), which self-corrects subspace drift; `A @ V_prev` is a plain projection without that. Cheaper vs more robust.
   - **Multi-step state** (orthogonalize [U_{t-1} | U_{t-2}] or Grassmannian extrapolation) — only worth it if some variable shows oscillating subspaces; backlog.
   - **Known no-op:** passing `U_prev @ diag(s)` changes nothing — column scaling does not change the span, and the QR re-normalizes. Only the span matters.
   - **Diagnostic to watch for:** if k* dips for one timestep and the rank re-climbs with poor warm quality right after, the state was discarded too aggressively — that is the signature that a k*+buffer policy would help.

---

## Presentations

1. **SVD 01 — Ideation** (Feb 2026): Initial conceptualization and literature review.
2. **SVD 02 — Preliminary Experiments** (Mar 2026): Synthetic validation (additive perturbation + rotating subspace regimes) and initial benchmarks.
3. **SVD 03 — Hurricane Benchmark** (Apr 2026): Full Hurricane Isabel performance evaluation across all 13 variables.
4. **SVD 04 — Compression Fidelity** (May 2026): PSNR and reconstruction quality across all 13 variables, error-bound hierarchy analysis.
5. **SVD 05 — Adaptive Compression** (May 2026): Adaptive error-bounded three-layer decomposition (L1+L2+S) with tolerance-guaranteed compression.

---

## Output Artifacts

| Artifact | Location |
|----------|----------|
| Raw per-timestep CSVs (C++, 77 cols) | `results/hurricane/raw_cpp/` |
| Raw per-timestep CSVs (Python, archived) | `results/hurricane/raw/` |
| Naive warm-start CSVs | `results/hurricane/raw_dumb/` |
| Adaptive experiment CSVs (59 cols) | `results/hurricane/adaptive/` |
| Summary statistics | `results/hurricane/hurricane_summary_cpp.csv` |
| Parameter sweep results | `results/sweep/` |
| Fixed-rank figures (PNG + PDF) | `results/hurricane/figures_cpp/` |
| Adaptive figures | `results/hurricane/figures_adaptive/` |

---

## Goals & Next Steps (June 2026, two-week sprint)

### 1. New Datasets

Extend the benchmark beyond Hurricane Isabel to establish generality:

- **NYX Cosmology** (SDR benchmark dataset) — cosmological hydrodynamics simulation. Tests the algorithm on a fundamentally different physical domain with different spectral characteristics (dark matter density, baryon density, temperature, velocity fields).
- **MIRANDA** — hydrodynamics simulation dataset. Turbulent mixing flows with sharp interfaces and multi-scale structure; tests robustness to non-smooth fields.

### 2. Rank Unfolding

Move beyond the current mode-3 matrix unfolding (spatial x z-levels) to explore alternative unfoldings of the 3D tensor. Different unfolding choices change the row/column structure and may yield better compression ratios or more stable subspaces for certain variables.

### 3. Adaptive Rank Visualisation

Create 2D heatmap visualisations of the adaptive rank results to reveal patterns across variables and timesteps simultaneously:
- Heatmap: timestep x variable with colour = adaptive rank k*
- Heatmap: timestep x variable with colour = compression ratio
- Heatmap: timestep x variable with colour = violation count (log scale)
- Heatmap: timestep x variable with colour = stage-2 activation / r*

### 4. Compression Backend Research

Investigate established scientific data compression methods and entropy coding to improve the sparse storage layer:

- **SZ2 and SZ3** — error-bounded lossy compressors for scientific floating-point data. Understand their prediction + quantisation + entropy coding pipeline and how our L1+L2+S decomposition could feed into or replace stages of their pipeline.
- **Sparse storage implementations** — survey efficient sparse formats (COO, CSR, CSC, compressed sparse blocks) and evaluate which best suits the sparse correction layer S, given the typical sparsity patterns observed (spatially clustered at the eyewall).
- **Huffman encoding** — study Huffman coding as an entropy coder for the quantised residual or sparse index streams. Evaluate potential bitrate savings on top of the current 12-byte-per-entry sparse format.

### 5. Research POD (Proper Orthogonal Decomposition)

Study the connection between POD and the streaming SVD framework. POD is the standard tool in computational fluid dynamics for extracting coherent structures from time-resolved simulation data. Understand:
- How our warm-started rSVD relates to incremental/streaming POD methods in the literature.
- Whether POD-specific diagnostics (mode energy, temporal coefficients) provide additional insight into rank selection or warm-start benefit.
- Positioning of this work relative to the POD community for eventual publication.

### Ideas to Explore

1. **Multi-variable joint decomposition** — Instead of compressing each variable independently, stack correlated variables (e.g. Uf + Vf + Wf wind triplet) into a single matrix and decompose jointly. Correlated variables may share a subspace, reducing total storage and enabling cross-variable warm-starting.

2. **Adaptive tolerance scheduling** — Rather than a fixed tau per variable, let tau vary over timesteps based on a total storage budget. During calm periods (low rank, few violations) tighten tau for higher fidelity; during turbulent periods relax tau to avoid sparse storage blow-up. This converts the per-snapshot guarantee into a budget-constrained stream-level optimisation.

3. **Incremental SVD update (rank-1 / rank-p corrections)** — Instead of recomputing the full rSVD at each timestep, investigate whether the warm-start can be reformulated as a rank-p update to the previous factorisation (Brand 2006 incremental SVD, Bunch–Nielsen–Sorensen). This could reduce per-timestep cost from O(mnk) to O(mn·p) when the subspace drift is small.

4. **Spatial error localisation and adaptive refinement** — Use leverage scores or residual magnitude maps to identify spatial regions (e.g. eyewall) where error concentrates. Apply a second-pass local refinement (higher rank or denser sparse storage) only in those regions, leaving the rest at the baseline rank. This is a spatial analogue of the current L1+L2+S temporal adaptivity.

5. **Streaming error accumulation analysis** — Quantify how compression error propagates when the compressed output is used as input for downstream analysis (e.g. vorticity computation, derivative fields). A small pointwise error in velocity may amplify in gradient-based quantities. Understanding this would inform tolerance selection.

6. **Randomised power iteration tuning** — The current experiments use q=0 power iterations. Investigate whether q=1 or q=2 disproportionately benefits the warm-start (since the warm sketch already has a better starting subspace, power iteration may converge faster). The parameter sweep data in `results/sweep/` partially addresses this but hasn't been analysed in the context of the adaptive algorithm.

7. **Temporal prediction for rank and sparse cost** — Fit a lightweight model (e.g. exponential smoothing or AR(1)) to the time series of k*, r*, and violation counts. Use predictions to pre-allocate the search window and skip unnecessary rank candidates, reducing the sweep cost at each timestep.

8. **Comparison with Tucker decomposition** — The current approach decomposes mode-3 unfoldings independently. Compare against a full Tucker decomposition (rHOSVD) that simultaneously decompresses along all three spatial modes. This is the natural next step toward the original project goal of streaming tensor decomposition.

### Long-Term

1. **GPU implementation and testing** — Port the core rSVD kernels (sketch generation, matmul, QR, small SVD, lift) to CUDA. Benchmark GPU vs. CPU performance across all three phases (fixed-rank, adaptive, L1+L2+S) and evaluate whether warm-start speedup amplifies on GPU where memory bandwidth is the bottleneck.
2. **Complete compression pipeline with quantisation and downstream lossless encoding** — Build an end-to-end pipeline: L1+L2+S decomposition → quantisation of the low-rank factors and sparse entries → lossless entropy coding (e.g. Huffman, arithmetic, or ANS) → bitstream output. Measure final compressed bitrate (bits per element) against SZ3 and other state-of-the-art scientific lossy compressors.

---

*Last updated: 2026-06-01*
