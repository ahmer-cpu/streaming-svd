#pragma once

#include "rsvd.hpp"   // SVDResult

/// Dumb warm-start randomized SVD.
///
/// Like warm_rsvd(), reuses U_prev from the previous timestep, but skips the
/// embedded power iteration on that prior basis.  The sketch is simply:
///
///   Y = [U_prev,  A @ Omega]   [m × (r_prev + p)]
///
/// U_prev is copied directly into Y without any matrix multiplication, saving
/// the two matmuls that warm_rsvd() spends on  G = A.T @ U_prev  and
/// Y1 = A @ G.  This makes the matmul cost equal to cold_rsvd() (one AX +
/// one ATX at q=0), while still injecting prior subspace information.
///
/// Expected trade-off vs warm_rsvd():  faster, but less accurate because the
/// warm columns are not amplified through the data matrix.
///
/// Falls back to cold_rsvd() when U_prev is nullptr (first timestep).
///
/// @param A       Input matrix, shape (m, n).
/// @param U_prev  Previous left singular vectors, shape (m, r_prev), or nullptr.
/// @param k       Target rank.
/// @param p       Oversampling for the random exploration columns (default 5).
/// @param q       Power iterations applied to the full sketch Y (default 0).
/// @param seed    RNG seed for the random exploration sketch.
SVDResult dumb_warm_rsvd(const MatF& A, const MatF* U_prev,
                          int k, int p = 5, int q = 0, uint64_t seed = 42);
