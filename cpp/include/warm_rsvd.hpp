#pragma once

#include "rsvd.hpp"   // SVDResult

/// Warm-start randomized SVD (Brand 2006).
///
/// Reuses U_prev (left singular vectors from the previous timestep) to build a
/// richer sketch with fewer random columns.  Falls back to cold_rsvd() when
/// U_prev is nullptr (first timestep).
///
/// Sketch structure:
///   Y1 = A @ (A.T @ U_prev)   [m × r_prev]  — warm component
///   Y2 = A @ Omega             [m × p]        — random exploration
///   Y  = [Y1, Y2]              [m × (r_prev + p)]
///
/// @param A       Input matrix, shape (m, n).
/// @param U_prev  Previous left singular vectors, shape (m, r_prev), or nullptr.
/// @param k       Target rank.
/// @param p       Warm oversampling (default 5, smaller than cold's 10).
/// @param q       Power iterations (default 0).
/// @param seed    RNG seed for the random exploration sketch.
SVDResult warm_rsvd(const MatF& A, const MatF* U_prev,
                    int k, int p = 5, int q = 0, uint64_t seed = 42);
