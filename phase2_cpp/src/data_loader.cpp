#include "data_loader.hpp"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <vector>

// Hurricane Isabel binary format:
//   - Raw float32, little-endian, no header.
//   - Logical shape: (z=100, y=500, x=500).
//   - Total values: 100 * 500 * 500 = 25,000,000.
//
// Python reshape logic:
//   volume = data.reshape(100, 500, 500)   # (z, y, x)
//   A = volume.reshape(100, -1).T          # (250000, 100)
//
// Interpretation:
//   - volume.reshape(100, -1) collapses (y,x) into a single axis → (100, 250000)
//   - .T transposes to (250000, 100)
// So column j of A = all spatial values at altitude level j.
// Row i of A = all altitude values at spatial point i.
//
// Memory layout in file: z-major → for each z in 0..99, y in 0..499, x in 0..499.
// Flat index of (z, y, x) = z*500*500 + y*500 + x.
// After transpose: A[i, j] = file[j*250000 + i]  (where i = y*500+x, j = z).
//
// Eigen is column-major by default, so A(i, j) maps column j contiguously.
// We load column-by-column: column j = z-level j = 250,000 consecutive floats
// starting at file offset j * 250000 * sizeof(float).

MatF load_bin_matrix(const std::string& path) {
    static constexpr int NZ  = 100;
    static constexpr int NYX = 500 * 500;   // 250,000 spatial points
    static constexpr std::size_t TOTAL = static_cast<std::size_t>(NZ) * NYX;
    static constexpr std::size_t BYTES = TOTAL * sizeof(float);

    // Open in binary mode
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("load_bin_matrix: cannot open '" + path + "'");

    // Read all bytes at once
    std::vector<float> buf(TOTAL);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(BYTES));
    if (!f)
        throw std::runtime_error("load_bin_matrix: short read on '" + path +
                                 "' (expected " + std::to_string(BYTES) + " bytes)");

    // The file is little-endian float32.  On Windows (little-endian) no byte-swap needed.
    // If porting to big-endian systems, swap bytes here.

    // Build Eigen matrix (NYX, NZ) = (250000, 100), column-major.
    // Column j = z-level j = buf[j*NYX .. (j+1)*NYX - 1].
    MatF A(NYX, NZ);
    for (int j = 0; j < NZ; ++j) {
        // Map raw buffer slice as an Eigen column vector and copy.
        Eigen::Map<const Eigen::VectorXf> col(buf.data() + j * NYX, NYX);
        A.col(j) = col;
    }

    return A;   // (250000, 100)
}

// ---------------------------------------------------------------------------
// Generic single-snapshot loader for the SDRBench cubes (NYX, Miranda).
//
// The cube is flattened C-order with the outermost (slowest-varying) axis as
// the matrix columns, so column j occupies the contiguous element range
// [j*n_rows, (j+1)*n_rows).  Eigen is column-major, so each column copies as a
// single contiguous block.  When is_double, the file holds little-endian
// float64 that we downcast to float32 to feed the existing MatF pipeline.
// ---------------------------------------------------------------------------
MatF load_static_matrix(const std::string& path, Idx n_rows, int n_cols, bool is_double) {
    const std::size_t total =
        static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_cols);

    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("load_static_matrix: cannot open '" + path + "'");

    MatF A(n_rows, n_cols);

    if (is_double) {
        std::vector<double> buf(total);
        const std::size_t bytes = total * sizeof(double);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes));
        if (!f)
            throw std::runtime_error("load_static_matrix: short read on '" + path +
                                     "' (expected " + std::to_string(bytes) + " bytes)");
        for (int j = 0; j < n_cols; ++j) {
            const double* src = buf.data() + static_cast<std::size_t>(j) * n_rows;
            for (Idx i = 0; i < n_rows; ++i)
                A(i, j) = static_cast<float>(src[i]);
        }
    } else {
        std::vector<float> buf(total);
        const std::size_t bytes = total * sizeof(float);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(bytes));
        if (!f)
            throw std::runtime_error("load_static_matrix: short read on '" + path +
                                     "' (expected " + std::to_string(bytes) + " bytes)");
        for (int j = 0; j < n_cols; ++j) {
            Eigen::Map<const Eigen::VectorXf> col(
                buf.data() + static_cast<std::size_t>(j) * n_rows, n_rows);
            A.col(j) = col;
        }
    }

    return A;
}
