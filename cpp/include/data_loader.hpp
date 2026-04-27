#pragma once

#include "matrix_types.hpp"
#include <string>

/// Load one Hurricane Isabel binary file and return a float32 matrix.
///
/// File format:
///   - Raw little-endian float32 values, no header.
///   - Logical shape: (z=100, y=500, x=500) = 25,000,000 values ≈ 95 MB.
///
/// Returned matrix shape: (250000, 100)
///   - Rows   = spatial grid points (y*x = 500*500 = 250,000)
///   - Cols   = z-levels (altitude, 100)
///
/// This matches the Python reshape:
///   volume.reshape(100, -1).T   →  (250000, 100)
///
/// @param path  Absolute or relative path to the .bin file.
/// @throws std::runtime_error if the file cannot be opened or has wrong size.
MatF load_bin_matrix(const std::string& path);
