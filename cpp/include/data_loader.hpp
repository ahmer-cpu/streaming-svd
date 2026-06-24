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

/// Load a raw little-endian binary cube and reshape to (n_rows, n_cols).
///
/// The matrix column axis is the outermost (slowest-varying) axis of the cube:
/// column j is the contiguous block of n_rows values at element offset
/// j * n_rows.  This matches the Hurricane convention (outer axis -> columns)
/// and, for the SDRBench datasets, places the chosen axis on the columns
/// (Miranda: the z / mixing direction; NYX: an arbitrary axis of the isotropic
/// cube).
///
/// @param path       Path to the raw binary file (no header).
/// @param n_rows     Number of matrix rows = product of the two inner axes.
/// @param n_cols     Number of matrix cols = size of the outermost axis.
/// @param is_double  If true the file holds float64 values, downcast to float32.
/// @throws std::runtime_error on open failure or short read.
MatF load_static_matrix(const std::string& path, Idx n_rows, int n_cols, bool is_double);
