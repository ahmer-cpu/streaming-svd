#include "rsvd.hpp"
#include "warm_rsvd.hpp"
#include "metrics.hpp"
#include "data_loader.hpp"
#include "csv_writer.hpp"
#include "timer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// All 13 Hurricane Isabel variable names
// ---------------------------------------------------------------------------
static const std::vector<std::string> ALL_VARS = {
    "CLOUDf", "Pf", "PRECIPf", "QCLOUDf", "QGRAUPf", "QICEf",
    "QRAINf",  "QSNOWf", "QVAPORf", "TCf", "Uf", "Vf", "Wf"
};

// ---------------------------------------------------------------------------
// Simple CLI argument parser
// ---------------------------------------------------------------------------
struct Config {
    std::string              data_dir  = "data/raw";
    std::string              out_dir   = "results/hurricane/raw";
    std::vector<std::string> vars      = ALL_VARS;
    int                      start_t   = 1;
    int                      end_t     = 48;
    int                      k         = 20;
    int                      p_cold    = 10;
    int                      p_warm    = 5;
    int                      q         = 0;
    uint64_t                 seed      = 42;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --data-dir  <path>        Root data directory          (default: data/raw)\n"
        << "  --out-dir   <path>        Output CSV directory         (default: results/hurricane/raw)\n"
        << "  --vars      v1 v2 ...     Variable names (space-sep)   (default: all 13)\n"
        << "  --start     <int>         First timestep (1-indexed)   (default: 1)\n"
        << "  --end       <int>         Last  timestep (inclusive)   (default: 48)\n"
        << "  --k         <int>         Target rank                  (default: 20)\n"
        << "  --p-cold    <int>         Cold oversampling            (default: 10)\n"
        << "  --p-warm    <int>         Warm oversampling            (default: 5)\n"
        << "  --q         <int>         Power iterations             (default: 0)\n"
        << "  --seed      <uint64>      RNG seed                     (default: 42)\n"
        << "  --help                    Show this message\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (arg == "--data-dir"  && i+1 < argc) { cfg.data_dir  = argv[++i]; }
        else if (arg == "--out-dir"   && i+1 < argc) { cfg.out_dir   = argv[++i]; }
        else if (arg == "--start"     && i+1 < argc) { cfg.start_t   = std::stoi(argv[++i]); }
        else if (arg == "--end"       && i+1 < argc) { cfg.end_t     = std::stoi(argv[++i]); }
        else if (arg == "--k"         && i+1 < argc) { cfg.k         = std::stoi(argv[++i]); }
        else if (arg == "--p-cold"    && i+1 < argc) { cfg.p_cold    = std::stoi(argv[++i]); }
        else if (arg == "--p-warm"    && i+1 < argc) { cfg.p_warm    = std::stoi(argv[++i]); }
        else if (arg == "--q"         && i+1 < argc) { cfg.q         = std::stoi(argv[++i]); }
        else if (arg == "--seed"      && i+1 < argc) { cfg.seed      = static_cast<uint64_t>(std::stoull(argv[++i])); }
        else if (arg == "--vars") {
            cfg.vars.clear();
            while (i+1 < argc && argv[i+1][0] != '-') {
                cfg.vars.push_back(argv[++i]);
            }
            if (cfg.vars.empty()) {
                std::cerr << "Error: --vars requires at least one variable name\n";
                std::exit(1);
            }
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// ISO-8601 timestamp string
// ---------------------------------------------------------------------------
static std::string iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

// ---------------------------------------------------------------------------
// Build binary file path: data_dir/VAR/VARtt.bin  (tt = zero-padded 2 digits)
// ---------------------------------------------------------------------------
static std::string bin_path(const std::string& data_dir,
                             const std::string& var, int t) {
    std::ostringstream ss;
    ss << data_dir << "/" << var << "/" << var
       << std::setw(2) << std::setfill('0') << t << ".bin";
    return ss.str();
}

// ---------------------------------------------------------------------------
// Compute NaN-safe speedup ratio
// ---------------------------------------------------------------------------
static double speedup_ratio(double cold_s, double warm_s) {
    if (std::isnan(warm_s) || warm_s <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return cold_s / warm_s;
}

// ---------------------------------------------------------------------------
// Main experiment loop
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    // Create output directory
    fs::create_directories(cfg.out_dir);

    std::cout << "Hurricane C++ SVD Benchmark\n"
              << "  data_dir : " << cfg.data_dir  << "\n"
              << "  out_dir  : " << cfg.out_dir   << "\n"
              << "  vars     : ";
    for (auto& v : cfg.vars) std::cout << v << " ";
    std::cout << "\n"
              << "  timesteps: " << cfg.start_t << " to " << cfg.end_t << "\n"
              << "  k=" << cfg.k << "  p_cold=" << cfg.p_cold
              << "  p_warm=" << cfg.p_warm << "  q=" << cfg.q
              << "  seed=" << cfg.seed << "\n\n";

    for (const auto& var : cfg.vars) {
        std::string csv_path = cfg.out_dir + "/" + var + "_raw.csv";
        std::cout << "[" << var << "] writing -> " << csv_path << "\n";

        RawRowWriter writer(csv_path);

        const MatF* U_warm_prev_ptr = nullptr;
        MatF        U_warm_prev;   // storage for previous warm U

        for (int t = cfg.start_t; t <= cfg.end_t; ++t) {
            std::string fpath = bin_path(cfg.data_dir, var, t);
            std::cout << "  t=" << std::setw(2) << t << "  " << fpath << " ... " << std::flush;

            // ------------------------------------------------------------------
            // Load data
            // ------------------------------------------------------------------
            double t_load = 0.0;
            MatF A;
            {
                auto t0 = std::chrono::steady_clock::now();
                try {
                    A = load_bin_matrix(fpath);
                } catch (const std::exception& e) {
                    std::cerr << "\n  ERROR loading " << fpath << ": " << e.what() << "\n";
                    continue;
                }
                t_load = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
            }

            // ------------------------------------------------------------------
            // Cold rSVD
            // ------------------------------------------------------------------
            SVDResult cold = cold_rsvd(A, cfg.k, cfg.p_cold, cfg.q, cfg.seed);

            // ------------------------------------------------------------------
            // Warm rSVD  (U_warm_prev_ptr == nullptr at t=start_t)
            // ------------------------------------------------------------------
            SVDResult warm = warm_rsvd(A, U_warm_prev_ptr,
                                       cfg.k, cfg.p_warm, cfg.q, cfg.seed);

            // ------------------------------------------------------------------
            // Accuracy metrics
            // ------------------------------------------------------------------
            float cold_fro = fro_error(A, cold.s);
            float warm_fro = fro_error(A, warm.s);
            float opt_fro  = optimal_fro_error(A, cfg.k);

            float cold_spec = spec_error(A, cold.U);
            float warm_spec = spec_error(A, warm.U);

            // Subspace metrics (require U_warm_prev from PREVIOUS step)
            float warm_drift_spec    = std::numeric_limits<float>::quiet_NaN();
            float warm_drift_fro     = std::numeric_limits<float>::quiet_NaN();
            float cw_sub_spec        = std::numeric_limits<float>::quiet_NaN();
            float cw_sub_fro         = std::numeric_limits<float>::quiet_NaN();
            float warm_prev_qual     = std::numeric_limits<float>::quiet_NaN();
            float warm_prev_qual_fro = std::numeric_limits<float>::quiet_NaN();

            if (U_warm_prev_ptr != nullptr) {
                // warm_drift: distance from previous warm U to current warm U
                subspace_sin_theta(*U_warm_prev_ptr, warm.U, warm_drift_spec, warm_drift_fro);
                // cold vs warm subspace agreement (current step)
                subspace_sin_theta(cold.U, warm.U, cw_sub_spec, cw_sub_fro);
                // warm_prev_quality: how close prev warm U is to current cold U
                subspace_sin_theta(*U_warm_prev_ptr, cold.U, warm_prev_qual, warm_prev_qual_fro);
            } else {
                // At t=start, still compute cold vs warm agreement
                subspace_sin_theta(cold.U, warm.U, cw_sub_spec, cw_sub_fro);
            }

            // ------------------------------------------------------------------
            // Assemble CSV row
            // ------------------------------------------------------------------
            RawRowWriter::RowData row;
            row.var           = var;
            row.timestep      = t;
            row.k             = cfg.k;
            row.p_cold        = cfg.p_cold;
            row.p_warm        = cfg.p_warm;
            row.q             = cfg.q;
            row.seed          = static_cast<int>(cfg.seed);
            row.data_file     = fpath;
            row.run_timestamp = iso_timestamp();

            // Accuracy
            row.cold_fro_error    = cold_fro;
            row.warm_fro_error    = warm_fro;
            row.optimal_fro_error = opt_fro;
            row.cold_spec_error   = cold_spec;
            row.warm_spec_error   = warm_spec;
            row.fro_error_gap     = warm_fro - cold_fro;       // positive = warm worse
            row.fro_error_ratio   = (cold_fro > 0) ? warm_fro / cold_fro
                                                   : std::numeric_limits<float>::quiet_NaN();
            row.cold_fro_overhead = (opt_fro > 0) ? cold_fro / opt_fro - 1.0f : std::numeric_limits<float>::quiet_NaN();
            row.warm_fro_overhead = (opt_fro > 0) ? warm_fro / opt_fro - 1.0f : std::numeric_limits<float>::quiet_NaN();
            row.cold_spec_gap     = warm_spec - cold_spec;

            // Subspace
            row.warm_drift_spec             = warm_drift_spec;
            row.warm_drift_fro              = warm_drift_fro;
            row.cold_vs_warm_subspace_spec  = cw_sub_spec;
            row.cold_vs_warm_subspace_fro   = cw_sub_fro;
            row.warm_prev_quality_spec      = warm_prev_qual;
            row.warm_prev_quality_fro       = warm_prev_qual_fro;

            // Timing
            row.cold_time_total   = cold.timings.count("total")  ? cold.timings.at("total")  : std::numeric_limits<double>::quiet_NaN();
            row.time_load_matrix  = t_load;
            row.time_speedup_ratio = std::numeric_limits<double>::quiet_NaN();

            row.cold_time_omega_gen      = cold.timings.count("omega_gen")      ? cold.timings.at("omega_gen")      : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_initial_matmul = cold.timings.count("initial_matmul") ? cold.timings.at("initial_matmul") : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_power_iter     = cold.timings.count("power_iter")     ? cold.timings.at("power_iter")     : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_qr             = cold.timings.count("qr")             ? cold.timings.at("qr")             : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_projection     = cold.timings.count("projection")     ? cold.timings.at("projection")     : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_small_svd      = cold.timings.count("small_svd")      ? cold.timings.at("small_svd")      : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_lift           = cold.timings.count("lift")           ? cold.timings.at("lift")           : std::numeric_limits<double>::quiet_NaN();
            row.cold_time_stats_total    = row.cold_time_total;   // same as total (no Python overhead)

            if (warm.warm_start) {
                row.warm_time_total       = warm.timings.count("total") ? warm.timings.at("total") : std::numeric_limits<double>::quiet_NaN();
                row.time_speedup_ratio    = speedup_ratio(row.cold_time_total, row.warm_time_total);

                row.warm_time_warm_proj     = warm.timings.count("warm_proj")     ? warm.timings.at("warm_proj")     : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_warm_matmul   = warm.timings.count("warm_matmul")   ? warm.timings.at("warm_matmul")   : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_omega_gen     = warm.timings.count("omega_gen")     ? warm.timings.at("omega_gen")     : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_random_matmul = warm.timings.count("random_matmul") ? warm.timings.at("random_matmul") : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_concat        = warm.timings.count("concat")        ? warm.timings.at("concat")        : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_power_iter    = warm.timings.count("power_iter")    ? warm.timings.at("power_iter")    : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_qr            = warm.timings.count("qr")            ? warm.timings.at("qr")            : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_projection    = warm.timings.count("projection")    ? warm.timings.at("projection")    : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_small_svd     = warm.timings.count("small_svd")     ? warm.timings.at("small_svd")     : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_lift          = warm.timings.count("lift")          ? warm.timings.at("lift")          : std::numeric_limits<double>::quiet_NaN();
                row.warm_time_stats_total   = row.warm_time_total;
            }
            // else: warm timings remain NaN (t=start, no prior)

            // Matmul counts
            row.cold_matmuls_AX    = cold.matmuls.count("AX")  ? cold.matmuls.at("AX")  : 0;
            row.cold_matmuls_ATX   = cold.matmuls.count("ATX") ? cold.matmuls.at("ATX") : 0;
            row.cold_matmuls_total = row.cold_matmuls_AX + row.cold_matmuls_ATX;
            row.warm_matmuls_AX    = warm.matmuls.count("AX")  ? warm.matmuls.at("AX")  : 0;
            row.warm_matmuls_ATX   = warm.matmuls.count("ATX") ? warm.matmuls.at("ATX") : 0;
            row.warm_matmuls_total = row.warm_matmuls_AX + row.warm_matmuls_ATX;
            row.matmul_savings     = row.cold_matmuls_total - row.warm_matmuls_total;

            // Config
            row.cold_stats_k          = cfg.k;
            row.cold_stats_p          = cfg.p_cold;
            row.warm_stats_r_prev     = warm.r_prev;
            row.warm_stats_warm_start = warm.warm_start;

            writer.write_row(row);

            // ------------------------------------------------------------------
            // Progress output
            // ------------------------------------------------------------------
            std::cout << std::fixed << std::setprecision(3)
                      << "cold=" << row.cold_time_total * 1000.0 << "ms";
            if (warm.warm_start)
                std::cout << "  warm=" << row.warm_time_total * 1000.0 << "ms"
                          << "  speedup=" << std::setprecision(3) << row.time_speedup_ratio;
            std::cout << "  cold_err=" << std::setprecision(5) << cold_fro
                      << "  warm_err=" << warm_fro
                      << "\n";

            // ------------------------------------------------------------------
            // Update warm state for next iteration.
            // At t=start_t, warm falls back to cold_rsvd with p=p_warm (< p_cold),
            // so seed the chain with the higher-quality cold result instead.
            // For t > start_t, use the warm result (correct streaming semantics).
            // ------------------------------------------------------------------
            U_warm_prev     = warm.warm_start ? warm.U : cold.U;
            U_warm_prev_ptr = &U_warm_prev;
        }

        std::cout << "  -> done: " << csv_path << "\n\n";
    }

    std::cout << "All done.\n";
    return 0;
}
