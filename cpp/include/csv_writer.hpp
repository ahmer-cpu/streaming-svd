#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cmath>

/// Writes per-timestep rows to a raw CSV file with columns identical to the
/// Python hurricane collect.py pipeline (60 columns).
///
/// Column order and names match exactly so that analyze.py / plot.py can
/// consume results/hurricane/raw_cpp/{VAR}_raw.csv without modification.
class RawRowWriter {
public:
    /// Open (or create) the CSV at the given path and write the header row.
    /// @throws std::runtime_error if the file cannot be opened.
    explicit RawRowWriter(const std::string& path);

    ~RawRowWriter();

    /// All data needed to build one row.
    struct RowData {
        // --- Identity ---
        std::string var;
        int         timestep;
        int         k;
        int         p_cold;
        int         p_warm;
        int         q;
        int         seed;
        std::string dtype        = "float32";
        std::string device       = "cpu_cpp";
        std::string data_file;
        std::string run_timestamp;

        // --- Approximation quality ---
        float cold_fro_error     = std::numeric_limits<float>::quiet_NaN();
        float warm_fro_error     = std::numeric_limits<float>::quiet_NaN();
        float optimal_fro_error  = std::numeric_limits<float>::quiet_NaN();
        float cold_spec_error    = std::numeric_limits<float>::quiet_NaN();
        float warm_spec_error    = std::numeric_limits<float>::quiet_NaN();
        float fro_error_gap      = std::numeric_limits<float>::quiet_NaN();
        float fro_error_ratio    = std::numeric_limits<float>::quiet_NaN();
        float cold_fro_overhead  = std::numeric_limits<float>::quiet_NaN();
        float warm_fro_overhead  = std::numeric_limits<float>::quiet_NaN();
        float cold_spec_gap      = std::numeric_limits<float>::quiet_NaN();

        // --- Subspace quality ---
        float warm_drift_spec              = std::numeric_limits<float>::quiet_NaN();
        float warm_drift_fro               = std::numeric_limits<float>::quiet_NaN();
        float cold_vs_warm_subspace_spec   = std::numeric_limits<float>::quiet_NaN();
        float cold_vs_warm_subspace_fro    = std::numeric_limits<float>::quiet_NaN();
        float warm_prev_quality_spec       = std::numeric_limits<float>::quiet_NaN();
        float warm_prev_quality_fro        = std::numeric_limits<float>::quiet_NaN();

        // --- Timing (seconds) ---
        double cold_time_total        = std::numeric_limits<double>::quiet_NaN();
        double warm_time_total        = std::numeric_limits<double>::quiet_NaN();
        double time_load_matrix       = std::numeric_limits<double>::quiet_NaN();
        double time_speedup_ratio     = std::numeric_limits<double>::quiet_NaN();

        // Cold breakdown
        double cold_time_omega_gen      = std::numeric_limits<double>::quiet_NaN();
        double cold_time_initial_matmul = std::numeric_limits<double>::quiet_NaN();
        double cold_time_power_iter     = std::numeric_limits<double>::quiet_NaN();
        double cold_time_qr             = std::numeric_limits<double>::quiet_NaN();
        double cold_time_projection     = std::numeric_limits<double>::quiet_NaN();
        double cold_time_small_svd      = std::numeric_limits<double>::quiet_NaN();
        double cold_time_lift           = std::numeric_limits<double>::quiet_NaN();
        double cold_time_stats_total    = std::numeric_limits<double>::quiet_NaN();

        // Warm breakdown
        double warm_time_warm_proj      = std::numeric_limits<double>::quiet_NaN();
        double warm_time_warm_matmul    = std::numeric_limits<double>::quiet_NaN();
        double warm_time_omega_gen      = std::numeric_limits<double>::quiet_NaN();
        double warm_time_random_matmul  = std::numeric_limits<double>::quiet_NaN();
        double warm_time_concat         = std::numeric_limits<double>::quiet_NaN();
        double warm_time_power_iter     = std::numeric_limits<double>::quiet_NaN();
        double warm_time_qr             = std::numeric_limits<double>::quiet_NaN();
        double warm_time_projection     = std::numeric_limits<double>::quiet_NaN();
        double warm_time_small_svd      = std::numeric_limits<double>::quiet_NaN();
        double warm_time_lift           = std::numeric_limits<double>::quiet_NaN();
        double warm_time_stats_total    = std::numeric_limits<double>::quiet_NaN();

        // --- Matmul counts ---
        int cold_matmuls_AX    = 0;
        int cold_matmuls_ATX   = 0;
        int cold_matmuls_total = 0;
        int warm_matmuls_AX    = 0;
        int warm_matmuls_ATX   = 0;
        int warm_matmuls_total = 0;
        int matmul_savings     = 0;

        // --- Config params ---
        int  cold_stats_k         = 0;
        int  cold_stats_p         = 0;
        int  warm_stats_r_prev    = 0;
        bool warm_stats_warm_start = false;
    };

    /// Write one row and flush to disk immediately.
    void write_row(const RowData& row);

    /// Column count (for validation).
    static constexpr int NUM_COLS = 61;

private:
    std::ofstream out_;

    static const std::vector<std::string>& header();

    // Helper: format a float/double as string (NaN → "nan")
    static std::string fmt(double v, int precision = 9);
    static std::string fmt(float v,  int precision = 9);
    static std::string fmt_bool(bool v);
};
