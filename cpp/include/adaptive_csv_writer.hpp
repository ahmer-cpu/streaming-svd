#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <limits>

/// Writes per-timestep rows for the adaptive warm residual-corrected rSVD
/// experiment.  Each row captures:
///   - stage-1 adaptive rank selection (k*)
///   - residual diagnostics
///   - stage-2 residual rank selection (r*)
///   - sparse correction stats
///   - combined quality metrics
class AdaptiveRowWriter {
public:
    explicit AdaptiveRowWriter(const std::string& path);
    ~AdaptiveRowWriter();

    struct RowData {
        // ---- Identity (15) ----
        std::string var;
        int         timestep        = 0;
        int         k_max_init      = 0;
        int         k_delta         = 0;
        int         p_cold          = 0;
        int         p_warm          = 0;
        int         q               = 0;
        int         seed            = 0;
        std::string dtype           = "float32";
        std::string device          = "cpu_cpp";
        std::string data_file;
        std::string run_timestamp;
        float       tau             = 0.0f;
        int         r_max           = 0;
        int         c_entry_bytes   = 12;

        // ---- Stage 1 adaptive rank (8) ----
        int   k_star            = 0;
        int   k_search_lo       = 0;
        int   k_search_hi       = 0;
        bool  k_expanded        = false;
        int64_t s0_violations_at_kstar = 0;
        int64_t stage1_rank_bytes = 0;
        double  stage1_time     = std::numeric_limits<double>::quiet_NaN();
        bool    stage1_warm_start = false;

        // ---- Stage 1 quality (8) ----
        float warm_fro_error          = std::numeric_limits<float>::quiet_NaN();
        float warm_max_elem_error     = std::numeric_limits<float>::quiet_NaN();
        float warm_psnr               = std::numeric_limits<float>::quiet_NaN();
        float warm_pctl_99            = std::numeric_limits<float>::quiet_NaN();
        float warm_pctl_999           = std::numeric_limits<float>::quiet_NaN();
        float cold_fro_error_at_kstar = std::numeric_limits<float>::quiet_NaN();
        float optimal_fro_error_at_kstar = std::numeric_limits<float>::quiet_NaN();
        float warm_fro_overhead       = std::numeric_limits<float>::quiet_NaN();

        // ---- Stage 1 sweep (3, semicolon-delimited) ----
        std::string stage1_sweep_ranks;
        std::string stage1_sweep_violations;
        std::string stage1_sweep_costs;

        // ---- Residual diagnostics (6) ----
        float residual_fro_norm               = std::numeric_limits<float>::quiet_NaN();
        float residual_spectral_concentration = std::numeric_limits<float>::quiet_NaN();
        float residual_sv_1  = std::numeric_limits<float>::quiet_NaN();
        float residual_sv_2  = std::numeric_limits<float>::quiet_NaN();
        float residual_sv_5  = std::numeric_limits<float>::quiet_NaN();
        float residual_sv_10 = std::numeric_limits<float>::quiet_NaN();

        // ---- Stage 2 decision (8) ----
        int     r_star              = 0;
        int64_t r_star_violations   = 0;
        int64_t stage2_rank_bytes   = 0;
        int64_t sparse_bytes        = 0;
        double  stage2_time         = std::numeric_limits<double>::quiet_NaN();
        bool    stage2_warm_start   = false;
        bool    stage2_skipped      = false;
        std::string stage2_skip_reason;   // "none", "no_violations", "sparse_cheap", "flat_spectrum"

        // ---- Stage 2 sweep (3, semicolon-delimited) ----
        std::string stage2_sweep_ranks;
        std::string stage2_sweep_violations;
        std::string stage2_sweep_costs;

        // ---- Total compression (4) ----
        int64_t total_compressed_bytes = 0;
        int64_t original_bytes         = 0;
        float   compression_ratio      = std::numeric_limits<float>::quiet_NaN();
        int64_t total_sparse_entries    = 0;

        // ---- Combined quality (3) ----
        float combined_max_elem_error = std::numeric_limits<float>::quiet_NaN();
        float combined_psnr           = std::numeric_limits<float>::quiet_NaN();
        float combined_fro_error      = std::numeric_limits<float>::quiet_NaN();

        // ---- Residual spectrum (1, semicolon-delimited) ----
        std::string residual_sv_spectrum;
    };

    void write_row(const RowData& row);

    static constexpr int NUM_COLS = 59;

private:
    std::ofstream out_;

    static const std::vector<std::string>& header();
    static std::string fmt(double v, int precision = 9);
    static std::string fmt(float v,  int precision = 9);
    static std::string fmt_bool(bool v);
    static std::string fmt_i64(int64_t v);
};

// Helpers for semicolon-delimited array columns
std::string join_ints(const std::vector<int>& v, const char* sep = ";");
std::string join_int64s(const std::vector<int64_t>& v, const char* sep = ";");
std::string join_floats(const std::vector<float>& v, const char* sep = ";", int precision = 6);
