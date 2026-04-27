#pragma once

// EIGEN_USE_BLAS is defined by CMakeLists.txt via add_compile_definitions.
// No need to redefine here.
#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// Core matrix / vector aliases (float32, column-major)
// ---------------------------------------------------------------------------

/// Dense float32 matrix, column-major (Eigen default).
using MatF = Eigen::MatrixXf;

/// Dense float32 column vector.
using VecF = Eigen::VectorXf;

/// Dense float32 matrix, row-major (used when reading raw binary data).
using MatFR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Convenience: Eigen index type.
using Idx = Eigen::Index;
