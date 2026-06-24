#pragma once

#include "matrix_types.hpp"
#include <unordered_map>
#include <string>
#include <cstdint>

/// Result returned by both cold_rsvd() and warm_rsvd().
struct SVDResult {
    MatF U;    ///< Left singular vectors,  shape (m, k).
    VecF s;    ///< Singular values,         shape (k,),  descending.
    MatF Vt;   ///< Right singular vectors,  shape (k, n).

    /// Wall-clock time (seconds) spent in each named phase.
    /// Keys match the Python CSV column names (without the "cold_time_" prefix):
    ///   omega_gen, initial_matmul, power_iter, qr, projection, small_svd, lift
    /// Plus "total" = end-to-end wall time including all phases.
    std::unordered_map<std::string, double> timings;

    /// Number of matrix–vector products performed.
    /// Keys: "AX"  (forward, A @ X)
    ///       "ATX" (transpose, A.T @ X)
    std::unordered_map<std::string, int> matmuls;

    /// Whether this result came from the warm path (false = cold / fallback).
    bool warm_start = false;

    /// r_prev used (0 for cold).
    int r_prev = 0;
};

/// Cold-start randomized SVD (Halko, Martinsson, Tropp 2011).
///
/// Computes a rank-k approximation A ≈ U diag(s) Vt from scratch.
///
/// @param A     Input matrix, shape (m, n).
/// @param k     Target rank.
/// @param p     Oversampling parameter (default 10).
/// @param q     Power iterations (default 0).
/// @param seed  RNG seed for the random sketch.
SVDResult cold_rsvd(const MatF& A, int k, int p = 10, int q = 0, uint64_t seed = 42);
