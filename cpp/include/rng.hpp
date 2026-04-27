#pragma once

#include "matrix_types.hpp"
#include <random>
#include <cstdint>

/// Fill a MatF with i.i.d. N(0, 1) draws using a seeded Mersenne Twister.
///
/// @param rows  Number of rows.
/// @param cols  Number of columns.
/// @param seed  RNG seed (use the same seed as Python's --seed argument).
/// @returns     New MatF of shape (rows, cols).
///
/// Note: C++ std::mt19937 with std::normal_distribution produces different
/// values than PyTorch's internal RNG even for the same seed.  This is
/// acceptable — the goal is accurate timing, not bit-identical results.
inline MatF randn_matrix(Idx rows, Idx cols, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    MatF out(rows, cols);
    for (Idx j = 0; j < cols; ++j)
        for (Idx i = 0; i < rows; ++i)
            out(i, j) = dist(rng);
    return out;
}
