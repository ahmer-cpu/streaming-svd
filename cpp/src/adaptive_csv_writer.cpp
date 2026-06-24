#include "adaptive_csv_writer.hpp"

#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Header definition (59 columns)
// ---------------------------------------------------------------------------
const std::vector<std::string>& AdaptiveRowWriter::header() {
    static const std::vector<std::string> H = {
        // Identity (15)
        "var", "timestep", "k_max_init", "k_delta",
        "p_cold", "p_warm", "q", "seed",
        "dtype", "device", "data_file", "run_timestamp",
        "tau", "r_max", "c_entry_bytes",
        // Stage 1 adaptive rank (8)
        "k_star", "k_search_lo", "k_search_hi", "k_expanded",
        "s0_violations_at_kstar", "stage1_rank_bytes",
        "stage1_time", "stage1_warm_start",
        // Stage 1 quality (8)
        "warm_fro_error", "warm_max_elem_error", "warm_psnr",
        "warm_pctl_99", "warm_pctl_999",
        "cold_fro_error_at_kstar", "optimal_fro_error_at_kstar",
        "warm_fro_overhead",
        // Stage 1 sweep (3)
        "stage1_sweep_ranks", "stage1_sweep_violations", "stage1_sweep_costs",
        // Residual diagnostics (6)
        "residual_fro_norm", "residual_spectral_concentration",
        "residual_sv_1", "residual_sv_2", "residual_sv_5", "residual_sv_10",
        // Stage 2 decision (8)
        "r_star", "r_star_violations", "stage2_rank_bytes", "sparse_bytes",
        "stage2_time", "stage2_warm_start", "stage2_skipped", "stage2_skip_reason",
        // Stage 2 sweep (3)
        "stage2_sweep_ranks", "stage2_sweep_violations", "stage2_sweep_costs",
        // Total compression (4)
        "total_compressed_bytes", "original_bytes",
        "compression_ratio", "total_sparse_entries",
        // Combined quality (3)
        "combined_max_elem_error", "combined_psnr", "combined_fro_error",
        // Residual spectrum (1)
        "residual_sv_spectrum"
    };
    return H;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
AdaptiveRowWriter::AdaptiveRowWriter(const std::string& path) {
    out_.open(path);
    if (!out_)
        throw std::runtime_error("AdaptiveRowWriter: cannot open '" + path + "'");

    const auto& h = header();
    if (static_cast<int>(h.size()) != NUM_COLS)
        throw std::runtime_error("AdaptiveRowWriter: header size " +
            std::to_string(h.size()) + " != NUM_COLS " + std::to_string(NUM_COLS));

    for (std::size_t i = 0; i < h.size(); ++i) {
        out_ << h[i];
        if (i + 1 < h.size()) out_ << ',';
    }
    out_ << '\n';
    out_.flush();
}

AdaptiveRowWriter::~AdaptiveRowWriter() {
    if (out_.is_open()) out_.close();
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------
std::string AdaptiveRowWriter::fmt(double v, int precision) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream ss;
    ss << std::setprecision(precision) << v;
    return ss.str();
}

std::string AdaptiveRowWriter::fmt(float v, int precision) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    std::ostringstream ss;
    ss << std::setprecision(precision) << v;
    return ss.str();
}

std::string AdaptiveRowWriter::fmt_bool(bool v) {
    return v ? "True" : "False";
}

std::string AdaptiveRowWriter::fmt_i64(int64_t v) {
    return std::to_string(v);
}

// ---------------------------------------------------------------------------
// write_row
// ---------------------------------------------------------------------------
void AdaptiveRowWriter::write_row(const RowData& r) {
    const char s = ',';
    out_
        // Identity (15)
        << r.var                          << s
        << r.timestep                     << s
        << r.k_max_init                   << s
        << r.k_delta                      << s
        << r.p_cold                       << s
        << r.p_warm                       << s
        << r.q                            << s
        << r.seed                         << s
        << r.dtype                        << s
        << r.device                       << s
        << r.data_file                    << s
        << r.run_timestamp                << s
        << fmt(r.tau)                     << s
        << r.r_max                        << s
        << r.c_entry_bytes                << s
        // Stage 1 adaptive rank (8)
        << r.k_star                       << s
        << r.k_search_lo                  << s
        << r.k_search_hi                  << s
        << fmt_bool(r.k_expanded)         << s
        << fmt_i64(r.s0_violations_at_kstar) << s
        << fmt_i64(r.stage1_rank_bytes)   << s
        << fmt(r.stage1_time, 9)          << s
        << fmt_bool(r.stage1_warm_start)  << s
        // Stage 1 quality (8)
        << fmt(r.warm_fro_error)          << s
        << fmt(r.warm_max_elem_error)     << s
        << fmt(r.warm_psnr)              << s
        << fmt(r.warm_pctl_99)            << s
        << fmt(r.warm_pctl_999)           << s
        << fmt(r.cold_fro_error_at_kstar) << s
        << fmt(r.optimal_fro_error_at_kstar) << s
        << fmt(r.warm_fro_overhead)       << s
        // Stage 1 sweep (3)
        << r.stage1_sweep_ranks           << s
        << r.stage1_sweep_violations      << s
        << r.stage1_sweep_costs           << s
        // Residual diagnostics (6)
        << fmt(r.residual_fro_norm)       << s
        << fmt(r.residual_spectral_concentration) << s
        << fmt(r.residual_sv_1)           << s
        << fmt(r.residual_sv_2)           << s
        << fmt(r.residual_sv_5)           << s
        << fmt(r.residual_sv_10)          << s
        // Stage 2 decision (8)
        << r.r_star                       << s
        << fmt_i64(r.r_star_violations)   << s
        << fmt_i64(r.stage2_rank_bytes)   << s
        << fmt_i64(r.sparse_bytes)        << s
        << fmt(r.stage2_time, 9)          << s
        << fmt_bool(r.stage2_warm_start)  << s
        << fmt_bool(r.stage2_skipped)     << s
        << r.stage2_skip_reason           << s
        // Stage 2 sweep (3)
        << r.stage2_sweep_ranks           << s
        << r.stage2_sweep_violations      << s
        << r.stage2_sweep_costs           << s
        // Total compression (4)
        << fmt_i64(r.total_compressed_bytes) << s
        << fmt_i64(r.original_bytes)      << s
        << fmt(r.compression_ratio)       << s
        << fmt_i64(r.total_sparse_entries) << s
        // Combined quality (3)
        << fmt(r.combined_max_elem_error) << s
        << fmt(r.combined_psnr)           << s
        << fmt(r.combined_fro_error)      << s
        // Residual spectrum (1)
        << r.residual_sv_spectrum
        << '\n';

    out_.flush();
}

// ---------------------------------------------------------------------------
// Join helpers (free functions)
// ---------------------------------------------------------------------------
std::string join_ints(const std::vector<int>& v, const char* sep) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) ss << sep;
        ss << v[i];
    }
    return ss.str();
}

std::string join_int64s(const std::vector<int64_t>& v, const char* sep) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) ss << sep;
        ss << v[i];
    }
    return ss.str();
}

std::string join_floats(const std::vector<float>& v, const char* sep, int precision) {
    std::ostringstream ss;
    ss << std::setprecision(precision);
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) ss << sep;
        ss << v[i];
    }
    return ss.str();
}
