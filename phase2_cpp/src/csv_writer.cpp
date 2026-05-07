#include "csv_writer.hpp"

#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Header definition (60 columns, matching Python collect.py exactly)
// ---------------------------------------------------------------------------
const std::vector<std::string>& RawRowWriter::header() {
    static const std::vector<std::string> H = {
        // Identity (11)
        "var", "timestep", "k", "p_cold", "p_warm", "q", "seed",
        "dtype", "device", "data_file", "run_timestamp",
        // Approximation quality (10)
        "cold_fro_error", "warm_fro_error", "optimal_fro_error",
        "cold_spec_error", "warm_spec_error",
        "fro_error_gap", "fro_error_ratio",
        "cold_fro_overhead", "warm_fro_overhead", "cold_spec_gap",
        // Subspace quality (5)
        "warm_drift_spec", "warm_drift_fro",
        "cold_vs_warm_subspace_spec", "cold_vs_warm_subspace_fro",
        "warm_prev_quality_spec", "warm_prev_quality_fro",
        // Timing — overview (4)
        "cold_time_total", "warm_time_total",
        "time_load_matrix", "time_speedup_ratio",
        // Timing — cold breakdown (8)
        "cold_time_omega_gen", "cold_time_initial_matmul", "cold_time_power_iter",
        "cold_time_qr", "cold_time_projection", "cold_time_small_svd",
        "cold_time_lift", "cold_time_stats_total",
        // Timing — warm breakdown (11)
        "warm_time_warm_proj", "warm_time_warm_matmul",
        "warm_time_omega_gen", "warm_time_random_matmul", "warm_time_concat",
        "warm_time_power_iter", "warm_time_qr", "warm_time_projection",
        "warm_time_small_svd", "warm_time_lift", "warm_time_stats_total",
        // Matmul counts (7)
        "cold_matmuls_AX", "cold_matmuls_ATX", "cold_matmuls_total",
        "warm_matmuls_AX", "warm_matmuls_ATX", "warm_matmuls_total",
        "matmul_savings",
        // Config params (4)
        "cold_stats_k", "cold_stats_p", "warm_stats_r_prev", "warm_stats_warm_start"
    };
    return H;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
RawRowWriter::RawRowWriter(const std::string& path) {
    out_.open(path);
    if (!out_)
        throw std::runtime_error("RawRowWriter: cannot open '" + path + "'");

    // Write header
    const auto& h = header();
    for (std::size_t i = 0; i < h.size(); ++i) {
        out_ << h[i];
        if (i + 1 < h.size()) out_ << ',';
    }
    out_ << '\n';
    out_.flush();
}

RawRowWriter::~RawRowWriter() {
    if (out_.is_open()) out_.close();
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
std::string RawRowWriter::fmt(double v, int precision) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream ss;
    ss << std::setprecision(precision) << v;
    return ss.str();
}

std::string RawRowWriter::fmt(float v, int precision) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream ss;
    ss << std::setprecision(precision) << v;
    return ss.str();
}

std::string RawRowWriter::fmt_bool(bool v) {
    return v ? "True" : "False";
}

// ---------------------------------------------------------------------------
// write_row
// ---------------------------------------------------------------------------
void RawRowWriter::write_row(const RowData& r) {
    const char sep = ',';
    out_
        // Identity
        << r.var                              << sep
        << r.timestep                         << sep
        << r.k                                << sep
        << r.p_cold                           << sep
        << r.p_warm                           << sep
        << r.q                                << sep
        << r.seed                             << sep
        << r.dtype                            << sep
        << r.device                           << sep
        << r.data_file                        << sep
        << r.run_timestamp                    << sep
        // Approximation quality
        << fmt(r.cold_fro_error)              << sep
        << fmt(r.warm_fro_error)              << sep
        << fmt(r.optimal_fro_error)           << sep
        << fmt(r.cold_spec_error)             << sep
        << fmt(r.warm_spec_error)             << sep
        << fmt(r.fro_error_gap)               << sep
        << fmt(r.fro_error_ratio)             << sep
        << fmt(r.cold_fro_overhead)           << sep
        << fmt(r.warm_fro_overhead)           << sep
        << fmt(r.cold_spec_gap)               << sep
        // Subspace quality
        << fmt(r.warm_drift_spec)             << sep
        << fmt(r.warm_drift_fro)              << sep
        << fmt(r.cold_vs_warm_subspace_spec)  << sep
        << fmt(r.cold_vs_warm_subspace_fro)   << sep
        << fmt(r.warm_prev_quality_spec)      << sep
        << fmt(r.warm_prev_quality_fro)       << sep
        // Timing overview
        << fmt(r.cold_time_total, 9)          << sep
        << fmt(r.warm_time_total, 9)          << sep
        << fmt(r.time_load_matrix, 9)         << sep
        << fmt(r.time_speedup_ratio, 6)       << sep
        // Cold breakdown
        << fmt(r.cold_time_omega_gen, 9)      << sep
        << fmt(r.cold_time_initial_matmul, 9) << sep
        << fmt(r.cold_time_power_iter, 9)     << sep
        << fmt(r.cold_time_qr, 9)             << sep
        << fmt(r.cold_time_projection, 9)     << sep
        << fmt(r.cold_time_small_svd, 9)      << sep
        << fmt(r.cold_time_lift, 9)           << sep
        << fmt(r.cold_time_stats_total, 9)    << sep
        // Warm breakdown
        << fmt(r.warm_time_warm_proj, 9)      << sep
        << fmt(r.warm_time_warm_matmul, 9)    << sep
        << fmt(r.warm_time_omega_gen, 9)      << sep
        << fmt(r.warm_time_random_matmul, 9)  << sep
        << fmt(r.warm_time_concat, 9)         << sep
        << fmt(r.warm_time_power_iter, 9)     << sep
        << fmt(r.warm_time_qr, 9)             << sep
        << fmt(r.warm_time_projection, 9)     << sep
        << fmt(r.warm_time_small_svd, 9)      << sep
        << fmt(r.warm_time_lift, 9)           << sep
        << fmt(r.warm_time_stats_total, 9)    << sep
        // Matmul counts
        << r.cold_matmuls_AX                  << sep
        << r.cold_matmuls_ATX                 << sep
        << r.cold_matmuls_total               << sep
        << r.warm_matmuls_AX                  << sep
        << r.warm_matmuls_ATX                 << sep
        << r.warm_matmuls_total               << sep
        << r.matmul_savings                   << sep
        // Config
        << r.cold_stats_k                     << sep
        << r.cold_stats_p                     << sep
        << r.warm_stats_r_prev                << sep
        << fmt_bool(r.warm_stats_warm_start)
        << '\n';

    out_.flush();
}
