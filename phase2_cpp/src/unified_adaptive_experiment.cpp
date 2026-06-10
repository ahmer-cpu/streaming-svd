// ===========================================================================
// Unified single-stage adaptive error-bounded rSVD compressor.
//
// Supersedes the two-stage (L1+L2+S) drivers adaptive_experiment.cpp and
// static_adaptive_experiment.cpp with one driver and one stage:
//
//   A_hat = L (rank k*)  +  S (sparse: entries with |A - L|_ij > tau)
//
//   k* = argmin_k  cost(k) = k*(m+n+1)*c_float + violations(k)*c_entry
//
// Rank search per snapshot:
//   1. one rSVD at the top of the search window — warm-started from U_prev
//      for temporal datasets (Isabel), cold otherwise / at bootstrap;
//   2. coarse ascending sweep over a candidate grid, scored by truncating
//      that single factorization via an incrementally accumulated rank-k
//      reconstruction (O(m*n) per candidate), with an exact upward prune:
//      once k*rank_bytes alone >= best cost, no larger k can win;
//   3. two-sided fine sweep over every rank within +/-fine_radius of the
//      coarse argmin, then a walk-down one rank at a time while the lowest
//      rank keeps winning (stops at the first non-improving step);
//   4. if the argmin sits on the window top, widen the window, recompute the
//      rSVD, and repeat (uniform boundary rule; downward never needs a new
//      rSVD since smaller ranks are truncations of the same factorization).
//
// The guarantee ||A - A_hat||_max <= tau comes from S alone, by construction.
// Stage 2 of the old drivers (residual rSVD + spectrum probe + skip rules) is
// removed: across results/static + results/hurricane/adaptive it changed the
// outcome in <10% of runs for <0.3% of bytes while costing 8-32% of wall
// time, and its systematic contribution (testing k*+1, k*+2 above a coarse
// grid point) is subsumed by the fine sweep, which also searches *below* the
// coarse argmin — something the two-stage design could not do.
// ===========================================================================
#include "rsvd.hpp"
#include "warm_rsvd.hpp"
#include "metrics.hpp"
#include "data_loader.hpp"
#include "adaptive_csv_writer.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <algorithm>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Dataset presets
// ---------------------------------------------------------------------------
struct DatasetPreset {
    std::string              name;
    bool                     temporal;     // true: vars x timesteps, warm-startable
    Idx                      n_rows;       // static datasets only
    int                      n_cols;       //   "
    bool                     is_double;    //   "
    std::string              ext;          //   "
    std::vector<std::string> vars;
    std::string              default_data_dir;
    int                      default_k_max_init;
};

static DatasetPreset make_dataset(const std::string& name) {
    if (name == "isabel") {
        return {"isabel", true, 0, 0, false, "",
                {"CLOUDf", "Pf", "PRECIPf", "QCLOUDf", "QGRAUPf", "QICEf",
                 "QRAINf", "QSNOWf", "QVAPORf", "TCf", "Uf", "Vf", "Wf"},
                "data/ISABEL_raw", 50};
    }
    if (name == "nyx") {
        // 512 x 512 x 512 float32 -> (262144, 512)
        return {"nyx", false, 512 * 512, 512, false, ".f32",
                {"baryon_density", "dark_matter_density", "temperature",
                 "velocity_x", "velocity_y", "velocity_z"},
                "data/NYX_raw", 32};
    }
    if (name == "miranda") {
        // x:384 (fastest), y:384, z:256 (slowest) float64 -> (147456, 256)
        return {"miranda", false, 384 * 384, 256, true, ".d64",
                {"density", "diffusivity", "pressure",
                 "velocityx", "velocityy", "velocityz", "viscocity"},
                "data/MIRANDA_raw", 32};
    }
    throw std::runtime_error("unknown dataset '" + name + "' (expected isabel|nyx|miranda)");
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    std::string              dataset;            // required
    std::string              data_dir;           // default: preset
    std::string              out_dir;            // default: results/<dataset>/unified
    std::vector<std::string> vars;               // default: preset
    int                      start_t     = 1;    // temporal datasets only
    int                      end_t       = 48;   //   "

    std::string              tau_mode    = "abs";   // "abs" | "vrel"
    float                    tau         = -1.0f;   // required when abs
    float                    eps         = -1.0f;   // required when vrel
    std::string              eps_str;               // raw --eps text, for filenames

    // Temporal datasets only: "warm" carries U_prev/k* across timesteps;
    // "cold" treats every timestep as a standalone bootstrap (the control
    // arm for adaptive warm-vs-cold comparisons).
    std::string              mode        = "warm";

    int                      k_max_init  = -1;   // default: preset
    int                      k_delta     = 8;    // warm window half-width
    int                      k_expand    = 16;   // window growth on boundary hit
    int                      fine_radius = 3;    // fine sweep half-width
    int                      p_cold      = 10;
    int                      p_warm      = 5;
    int                      q           = 0;
    int                      c_entry     = 12;
    uint64_t                 seed        = 42;

    int c_float = 4;  // sizeof(float) — baseline representation
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --dataset <isabel|nyx|miranda> [options]\n"
        << "  --dataset     <name>   Dataset preset                  (REQUIRED)\n"
        << "  --data-dir    <path>   Root data directory             (default: preset)\n"
        << "  --out-dir     <path>   Output CSV directory            (default: results/<dataset>/unified)\n"
        << "  --vars        v1 v2    Variable names (space-sep)      (default: all in dataset)\n"
        << "  --start       <int>    First timestep, isabel only     (default: 1)\n"
        << "  --end         <int>    Last timestep, isabel only      (default: 48)\n"
        << "  --mode        <mode>   warm | cold (temporal only)     (default: warm)\n"
        << "  --tau-mode    <mode>   Tolerance mode: abs | vrel      (default: abs)\n"
        << "  --tau         <float>  Absolute elementwise tolerance  (REQUIRED if abs)\n"
        << "  --eps         <float>  Relative tol; tau=eps*range     (REQUIRED if vrel)\n"
        << "  --k-max-init  <int>    Bootstrap rank cap              (default: preset 50/32)\n"
        << "  --k-delta     <int>    Warm window half-width          (default: 8)\n"
        << "  --k-expand    <int>    Window growth on boundary hit   (default: 16)\n"
        << "  --fine-radius <int>    Fine sweep half-width           (default: 3)\n"
        << "  --p-cold      <int>    Cold rSVD oversampling          (default: 10)\n"
        << "  --p-warm      <int>    Warm rSVD oversampling          (default: 5)\n"
        << "  --q           <int>    Power iterations                (default: 0)\n"
        << "  --c-entry     <int>    Sparse entry cost (bytes)       (default: 12)\n"
        << "  --seed        <uint>   RNG seed                        (default: 42)\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--dataset"     && i+1 < argc) { cfg.dataset     = argv[++i]; }
        else if (arg == "--data-dir"    && i+1 < argc) { cfg.data_dir    = argv[++i]; }
        else if (arg == "--out-dir"     && i+1 < argc) { cfg.out_dir     = argv[++i]; }
        else if (arg == "--start"       && i+1 < argc) { cfg.start_t     = std::stoi(argv[++i]); }
        else if (arg == "--end"         && i+1 < argc) { cfg.end_t       = std::stoi(argv[++i]); }
        else if (arg == "--mode"        && i+1 < argc) { cfg.mode        = argv[++i]; }
        else if (arg == "--tau-mode"    && i+1 < argc) { cfg.tau_mode    = argv[++i]; }
        else if (arg == "--tau"         && i+1 < argc) { cfg.tau         = std::stof(argv[++i]); }
        else if (arg == "--eps"         && i+1 < argc) { cfg.eps_str     = argv[++i]; cfg.eps = std::stof(cfg.eps_str); }
        else if (arg == "--k-max-init"  && i+1 < argc) { cfg.k_max_init  = std::stoi(argv[++i]); }
        else if (arg == "--k-delta"     && i+1 < argc) { cfg.k_delta     = std::stoi(argv[++i]); }
        else if (arg == "--k-expand"    && i+1 < argc) { cfg.k_expand    = std::stoi(argv[++i]); }
        else if (arg == "--fine-radius" && i+1 < argc) { cfg.fine_radius = std::stoi(argv[++i]); }
        else if (arg == "--p-cold"      && i+1 < argc) { cfg.p_cold      = std::stoi(argv[++i]); }
        else if (arg == "--p-warm"      && i+1 < argc) { cfg.p_warm      = std::stoi(argv[++i]); }
        else if (arg == "--q"           && i+1 < argc) { cfg.q           = std::stoi(argv[++i]); }
        else if (arg == "--c-entry"     && i+1 < argc) { cfg.c_entry     = std::stoi(argv[++i]); }
        else if (arg == "--seed"        && i+1 < argc) { cfg.seed        = static_cast<uint64_t>(std::stoull(argv[++i])); }
        else if (arg == "--vars") {
            cfg.vars.clear();
            while (i+1 < argc && argv[i+1][0] != '-')
                cfg.vars.push_back(argv[++i]);
            if (cfg.vars.empty()) { std::cerr << "Error: --vars requires at least one name\n"; std::exit(1); }
        }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); std::exit(1); }
    }
    if (cfg.dataset.empty()) { std::cerr << "Error: --dataset is required\n"; print_usage(argv[0]); std::exit(1); }
    if (cfg.tau_mode == "vrel") {
        if (cfg.eps <= 0) { std::cerr << "Error: --eps (>0) is required for --tau-mode vrel\n"; print_usage(argv[0]); std::exit(1); }
    } else if (cfg.tau_mode == "abs") {
        if (cfg.tau < 0)  { std::cerr << "Error: --tau is required for --tau-mode abs\n"; print_usage(argv[0]); std::exit(1); }
    } else {
        std::cerr << "Error: --tau-mode must be 'abs' or 'vrel'\n"; std::exit(1);
    }
    if (cfg.mode != "warm" && cfg.mode != "cold") {
        std::cerr << "Error: --mode must be 'warm' or 'cold'\n"; std::exit(1);
    }
    if (cfg.fine_radius < 0) cfg.fine_radius = 0;
    return cfg;
}

// ---------------------------------------------------------------------------
// Utility
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
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

static std::string isabel_bin_path(const std::string& data_dir,
                                   const std::string& var, int t) {
    std::ostringstream ss;
    ss << data_dir << "/" << var << "/" << var
       << std::setfill('0') << std::setw(2) << t << ".bin";
    return ss.str();
}

// "0.001000" -> "0.001"
static std::string trim_float(float v) {
    std::string s = std::to_string(v);
    s.erase(s.find_last_not_of('0') + 1);
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------
// Violations of a precomputed rank-k reconstruction against A:
//   #{ (i,j) : |A_ij - recon_ij| > tau }.   O(m*n) per call.
// ---------------------------------------------------------------------------
static int64_t count_violations_recon(const MatF& A, const MatF& recon, float tau) {
    int64_t cnt = 0;
    for (Idx j = 0; j < A.cols(); ++j)
        for (Idx i = 0; i < A.rows(); ++i)
            if (std::abs(A(i, j) - recon(i, j)) > tau) ++cnt;
    return cnt;
}

// ---------------------------------------------------------------------------
// Coarse candidate grid within [k_lo, k_hi]: every rank up to 8, then
// multiples of 4, plus both window endpoints.
// ---------------------------------------------------------------------------
static std::vector<int> coarse_candidates(int k_lo, int k_hi) {
    std::vector<int> cands;
    for (int k = std::max(1, k_lo); k <= std::min(k_hi, 8); ++k)
        cands.push_back(k);
    int first4 = ((std::max(k_lo, 9) + 3) / 4) * 4;   // first multiple of 4 >= max(k_lo, 9)
    for (int k = std::max(12, first4); k <= k_hi; k += 4)
        cands.push_back(k);
    cands.push_back(std::max(1, k_lo));
    cands.push_back(k_hi);
    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
    return cands;
}

// ---------------------------------------------------------------------------
// Rank selection: one rSVD per window, coarse + fine sweep, boundary expansion.
// On return `recon` holds the rank-k* reconstruction (for the metrics pass).
// ---------------------------------------------------------------------------
struct SelectionResult {
    SVDResult svd;                       // factorization at the final window top
    int       k_star       = 1;
    int64_t   k_star_viol  = 0;
    int64_t   best_cost    = 0;
    int       k_search_lo  = 1;
    int       k_search_hi  = 0;
    bool      expanded     = false;
    bool      warm         = false;
    std::vector<int>     sweep_ranks;    // all evaluated, ascending
    std::vector<int64_t> sweep_viols;
    std::vector<int64_t> sweep_costs;
};

static SelectionResult select_rank(const MatF& A, float tau, const Config& cfg,
                                   int k_max_init,
                                   const MatF* U_prev, int k_prev,
                                   MatF& recon) {
    const Idx m = A.rows();
    const Idx n = A.cols();
    const int64_t per_rank_bytes = static_cast<int64_t>(m + n + 1) * cfg.c_float;
    const bool bootstrap = (U_prev == nullptr);

    SelectionResult res;
    int k_lo, k_hi;
    if (bootstrap) {
        k_lo = 1;
        k_hi = std::min(k_max_init, static_cast<int>(n));
    } else {
        k_lo = std::max(1, k_prev - cfg.k_delta);
        k_hi = std::min(static_cast<int>(n), k_prev + cfg.k_delta);
    }
    res.k_search_lo = k_lo;

    recon.resize(m, n);

    std::map<int, std::pair<int64_t, int64_t>> evals;  // rank -> (viol, cost)
    int recon_rank = 0;

    // recon = sum of the first k rank-1 terms of the current factorization.
    // Ascending adds reproduce the float accumulation order of a fresh build;
    // descending subtracts differ only by float rounding (~eps*|A| << tau).
    auto set_rank = [&](int k) {
        while (recon_rank < k) {
            recon.noalias() += res.svd.U.col(recon_rank)
                * (res.svd.s(recon_rank) * res.svd.Vt.row(recon_rank));
            ++recon_rank;
        }
        while (recon_rank > k) {
            --recon_rank;
            recon.noalias() -= res.svd.U.col(recon_rank)
                * (res.svd.s(recon_rank) * res.svd.Vt.row(recon_rank));
        }
    };
    auto eval = [&](int k) {
        set_rank(k);
        int64_t viol = count_violations_recon(A, recon, tau);
        int64_t cost = static_cast<int64_t>(k) * per_rank_bytes + viol * cfg.c_entry;
        evals[k] = {viol, cost};
        if (cost < res.best_cost ||
            (cost == res.best_cost && k < res.k_star)) {
            res.best_cost   = cost;
            res.k_star      = k;
            res.k_star_viol = viol;
        }
    };

    bool searching = true;
    while (searching) {
        searching = false;

        // rSVD at the window top
        if (bootstrap) {
            res.svd  = cold_rsvd(A, k_hi, cfg.p_cold, cfg.q, cfg.seed);
            res.warm = false;
        } else {
            int r_prev_cols = static_cast<int>(U_prev->cols());
            int p_needed = std::max(cfg.p_warm, k_hi - r_prev_cols + cfg.p_warm);
            res.svd  = warm_rsvd(A, U_prev, k_hi, p_needed, cfg.q, cfg.seed);
            res.warm = res.svd.warm_start;
        }

        evals.clear();
        res.best_cost = std::numeric_limits<int64_t>::max();
        res.k_star = k_lo;
        recon.setZero();
        recon_rank = 0;

        // -- coarse ascending sweep with exact upward prune ----------------
        for (int k : coarse_candidates(k_lo, k_hi)) {
            if (k > static_cast<int>(res.svd.s.size())) break;
            // cost(k') >= k'*per_rank_bytes >= k*per_rank_bytes for k' >= k
            if (static_cast<int64_t>(k) * per_rank_bytes >= res.best_cost) break;
            eval(k);
        }

        // -- two-sided fine sweep around the coarse argmin -----------------
        int fhi = std::min(k_hi, res.k_star + cfg.fine_radius);
        int flo = std::max(1,    res.k_star - cfg.fine_radius);
        for (int k = fhi; k >= flo; --k)            // descending: cheap subtracts
            if (!evals.count(k)) eval(k);

        // -- walk down while the lowest rank keeps winning ------------------
        // Marginal cost of dropping one rank (added violations * c_entry -
        // per_rank_bytes) grows as k shrinks, so stop at the first
        // non-improving step.  Each step is one O(m*n) subtract + count.
        while (res.k_star == flo && flo > 1) {
            --flo;
            eval(flo);
        }

        // -- uniform boundary rule: argmin on window top => widen -----------
        if (res.k_star == k_hi && k_hi < static_cast<int>(n)) {
            k_hi = std::min(static_cast<int>(n), k_hi + cfg.k_expand);
            res.expanded = true;
            searching = true;
        }
    }
    res.k_search_hi = k_hi;

    // Leave recon at k* for the caller's single metrics pass.
    set_rank(res.k_star);

    for (auto& [k, vc] : evals) {
        res.sweep_ranks.push_back(k);
        res.sweep_viols.push_back(vc.first);
        res.sweep_costs.push_back(vc.second);
    }
    return res;
}

// ---------------------------------------------------------------------------
// Single pass over A - recon: pre-sparse residual stats + post-sparse
// (combined) quality metrics.  Entries with |e| > tau are stored exactly in
// S, so their post-sparse error is 0.
// ---------------------------------------------------------------------------
struct FinalMetrics {
    float  residual_fro_norm;   // ||A - L||_F (pre-sparse)
    float  combined_max_err;    // <= tau by construction
    float  combined_psnr;
    float  combined_fro;        // relative, post-sparse
};

static FinalMetrics final_metrics(const MatF& A, const MatF& recon, float tau,
                                  float peak_A, double norm_A_sq) {
    const Idx m = A.rows(), n = A.cols();
    double resid_sq = 0.0, comb_sq = 0.0;
    float  comb_max = 0.0f;
    for (Idx j = 0; j < n; ++j) {
        for (Idx i = 0; i < m; ++i) {
            float ae = std::abs(A(i, j) - recon(i, j));
            resid_sq += static_cast<double>(ae) * ae;
            if (ae <= tau) {
                if (ae > comb_max) comb_max = ae;
                comb_sq += static_cast<double>(ae) * ae;
            }
        }
    }
    FinalMetrics fm;
    fm.residual_fro_norm = static_cast<float>(std::sqrt(resid_sq));
    fm.combined_max_err  = comb_max;
    fm.combined_fro = (norm_A_sq > 0)
        ? static_cast<float>(std::sqrt(comb_sq / norm_A_sq)) : 0.0f;
    double mse = comb_sq / (static_cast<double>(m) * static_cast<double>(n));
    if (mse < 1e-30 || peak_A < 1e-30f)
        fm.combined_psnr = std::numeric_limits<float>::infinity();
    else
        fm.combined_psnr = static_cast<float>(10.0 * std::log10(
            static_cast<double>(peak_A) * static_cast<double>(peak_A) / mse));
    return fm;
}

// ---------------------------------------------------------------------------
// Process one snapshot end-to-end; returns k* (and the factor for warm reuse
// via out-params).
// ---------------------------------------------------------------------------
static void run_snapshot(const MatF& A, const std::string& var, int timestep,
                         const std::string& fpath, const std::string& dtype,
                         float tau, const Config& cfg, int k_max_init,
                         MatF* U_prev_inout, int* k_prev_inout,
                         MatF& recon, AdaptiveRowWriter& writer) {
    const Idx m = A.rows();
    const Idx n = A.cols();
    const int64_t original_bytes = static_cast<int64_t>(m) * n * cfg.c_float;
    const float peak_A = A.cwiseAbs().maxCoeff();
    double norm_A_sq = 0.0;
    for (Idx j = 0; j < n; ++j)
        norm_A_sq += A.col(j).cast<double>().squaredNorm();

    const MatF* U_prev = (U_prev_inout && U_prev_inout->size() > 0)
                       ? U_prev_inout : nullptr;

    auto t_start = std::chrono::steady_clock::now();
    SelectionResult sel = select_rank(A, tau, cfg, k_max_init,
                                      U_prev, k_prev_inout ? *k_prev_inout : 0,
                                      recon);
    double t_select = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    const int     k_star = sel.k_star;
    const int64_t s0     = sel.k_star_viol;
    const int64_t rank_bytes   = static_cast<int64_t>(k_star) * (m + n + 1) * cfg.c_float;
    const int64_t sparse_bytes = s0 * cfg.c_entry;
    const int64_t total_bytes  = rank_bytes + sparse_bytes;
    const float compression_ratio = (total_bytes > 0)
        ? static_cast<float>(original_bytes) / static_cast<float>(total_bytes)
        : std::numeric_limits<float>::infinity();

    // Quality at k*
    float fro     = fro_error(A, sel.svd.s.head(k_star));
    float opt_fro = optimal_fro_error(A, k_star);
    float fro_overhead = (opt_fro > 0) ? fro / opt_fro - 1.0f
                       : std::numeric_limits<float>::quiet_NaN();
    CompressionMetrics cm = compression_metrics(
        A, sel.svd.U.leftCols(k_star), sel.svd.s.head(k_star),
        sel.svd.Vt.topRows(k_star));
    FinalMetrics fm = final_metrics(A, recon, tau, peak_A, norm_A_sq);

    // ---- CSV row (streaming schema; stage-2 fields are structurally N/A) ----
    AdaptiveRowWriter::RowData row;
    row.var           = var;
    row.timestep      = timestep;
    row.k_max_init    = k_max_init;
    row.k_delta       = cfg.k_delta;
    row.p_cold        = cfg.p_cold;
    row.p_warm        = cfg.p_warm;
    row.q             = cfg.q;
    row.seed          = static_cast<int>(cfg.seed);
    row.dtype         = dtype;
    row.device        = "cpu_cpp";
    row.data_file     = fpath;
    row.run_timestamp = iso_timestamp();
    row.tau           = tau;
    row.r_max         = 0;
    row.c_entry_bytes = cfg.c_entry;

    row.k_star                 = k_star;
    row.k_search_lo            = sel.k_search_lo;
    row.k_search_hi            = sel.k_search_hi;
    row.k_expanded             = sel.expanded;
    row.s0_violations_at_kstar = s0;
    row.stage1_rank_bytes      = rank_bytes;
    row.stage1_time            = t_select;
    row.stage1_warm_start      = sel.warm;

    row.warm_fro_error             = fro;
    row.warm_max_elem_error        = cm.max_elem_error;
    row.warm_psnr                  = cm.psnr;
    row.warm_pctl_99               = cm.pctl_99;
    row.warm_pctl_999              = cm.pctl_999;
    row.cold_fro_error_at_kstar    = sel.warm ? std::numeric_limits<float>::quiet_NaN() : fro;
    row.optimal_fro_error_at_kstar = opt_fro;
    row.warm_fro_overhead          = fro_overhead;

    row.stage1_sweep_ranks      = join_ints(sel.sweep_ranks);
    row.stage1_sweep_violations = join_int64s(sel.sweep_viols);
    row.stage1_sweep_costs      = join_int64s(sel.sweep_costs);

    row.residual_fro_norm = fm.residual_fro_norm;

    // Single-stage: no residual stage exists.
    row.r_star             = 0;
    row.r_star_violations  = s0;
    row.stage2_rank_bytes  = 0;
    row.sparse_bytes       = sparse_bytes;
    row.stage2_time        = 0.0;
    row.stage2_warm_start  = false;
    row.stage2_skipped     = true;
    row.stage2_skip_reason = "single_stage";

    row.total_compressed_bytes = total_bytes;
    row.original_bytes         = original_bytes;
    row.compression_ratio      = compression_ratio;
    row.total_sparse_entries   = s0;

    row.combined_max_elem_error = fm.combined_max_err;
    row.combined_psnr           = fm.combined_psnr;
    row.combined_fro_error      = fm.combined_fro;

    writer.write_row(row);

    std::cout << std::fixed
              << "tau=" << std::setprecision(4) << tau
              << "  k*=" << std::setw(3) << k_star
              << " [" << sel.k_search_lo << "," << sel.k_search_hi << "]"
              << (sel.expanded ? "+" : " ")
              << (sel.warm ? " warm" : " cold")
              << "  s0=" << s0
              << "  maxe=" << std::setprecision(3) << fm.combined_max_err
              << "  ratio=" << std::setprecision(1) << compression_ratio << "x"
              << "  total=" << total_bytes / 1e6 << "MB"
              << "  t=" << std::setprecision(0) << t_select * 1000 << "ms\n";

    // Warm state update
    if (U_prev_inout && k_prev_inout) {
        *U_prev_inout = sel.svd.U.leftCols(k_star);
        *k_prev_inout = k_star;
    }
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    DatasetPreset ds = make_dataset(cfg.dataset);
    if (cfg.data_dir.empty())   cfg.data_dir   = ds.default_data_dir;
    if (cfg.out_dir.empty())    cfg.out_dir    = "results/" + cfg.dataset + "/unified";
    if (cfg.vars.empty())       cfg.vars       = ds.vars;
    const int k_max_init = (cfg.k_max_init > 0) ? cfg.k_max_init
                                                : ds.default_k_max_init;

    fs::create_directories(cfg.out_dir);

    std::cout << "Unified Single-Stage Adaptive rSVD (rank + sparse)\n"
              << "  dataset    : " << ds.name
              << (ds.temporal ? "  (temporal, warm-startable)\n"
                              : "  (static, cold-only)\n")
              << "  data_dir   : " << cfg.data_dir << "\n"
              << "  out_dir    : " << cfg.out_dir  << "\n"
              << "  vars       : ";
    for (auto& v : cfg.vars) std::cout << v << " ";
    std::cout << "\n";
    if (ds.temporal)
        std::cout << "  timesteps  : " << cfg.start_t << " to " << cfg.end_t
                  << "    mode = " << cfg.mode
                  << (cfg.mode == "cold" ? "  (standalone bootstrap per timestep)" : "")
                  << "\n";
    if (cfg.tau_mode == "vrel")
        std::cout << "  tau_mode   = vrel  eps = " << cfg.eps
                  << "  (tau = eps * value_range, per snapshot)\n";
    else
        std::cout << "  tau_mode   = abs   tau = " << cfg.tau << "\n";
    std::cout << "  k_max_init = " << k_max_init
              << "  k_delta = "     << cfg.k_delta
              << "  k_expand = "    << cfg.k_expand
              << "  fine_radius = " << cfg.fine_radius << "\n"
              << "  p_cold = " << cfg.p_cold << "  p_warm = " << cfg.p_warm
              << "  q = " << cfg.q << "  c_entry = " << cfg.c_entry
              << " bytes  seed = " << cfg.seed << "\n\n";

    const std::string tag = (cfg.tau_mode == "vrel")
        ? "_eps" + cfg.eps_str
        : "_tau" + trim_float(cfg.tau);

    MatF recon;  // reusable m x n reconstruction buffer

    if (ds.temporal) {
        // ---------------- Isabel: vars x timesteps ------------------------
        // mode=warm: U_prev/k* carried across timesteps (streaming).
        // mode=cold: every timestep is a standalone bootstrap (control arm,
        //            methodologically identical to one static variable).
        const bool cold_mode = (cfg.mode == "cold");
        const std::string mode_tag = cold_mode ? "_cold" : "";
        for (const auto& var : cfg.vars) {
            std::string csv_path = cfg.out_dir + "/" + var + "_unified"
                                 + mode_tag + tag + ".csv";
            std::cout << "[" << var << "] writing -> " << csv_path << "\n";
            AdaptiveRowWriter writer(csv_path);

            MatF U_prev;   // empty => bootstrap
            int  k_prev = 0;

            for (int t = cfg.start_t; t <= cfg.end_t; ++t) {
                std::string fpath = isabel_bin_path(cfg.data_dir, var, t);
                std::cout << "  t=" << std::setw(2) << t << "  " << std::flush;
                MatF A;
                try {
                    A = load_bin_matrix(fpath);
                } catch (const std::exception& e) {
                    std::cerr << "ERROR loading " << fpath << ": " << e.what() << "\n";
                    continue;
                }
                const float tau = (cfg.tau_mode == "vrel")
                    ? cfg.eps * (A.maxCoeff() - A.minCoeff())
                    : cfg.tau;
                run_snapshot(A, var, t, fpath, "float32", tau, cfg, k_max_init,
                             cold_mode ? nullptr : &U_prev,
                             cold_mode ? nullptr : &k_prev,
                             recon, writer);
            }
            std::cout << "  -> done: " << csv_path << "\n\n";
        }
    } else {
        // ---------------- NYX / Miranda: one cold snapshot per var ---------
        std::string csv_path = cfg.out_dir + "/" + cfg.dataset + "_unified" + tag + ".csv";
        AdaptiveRowWriter writer(csv_path);
        std::cout << "Writing -> " << csv_path << "\n\n";
        const std::string dtype = ds.is_double ? "float64->float32" : "float32";

        for (const auto& var : cfg.vars) {
            std::string fpath = cfg.data_dir + "/" + var + ds.ext;
            std::cout << "[" << std::setw(20) << std::left << var << "] " << std::flush;
            MatF A;
            try {
                A = load_static_matrix(fpath, ds.n_rows, ds.n_cols, ds.is_double);
            } catch (const std::exception& e) {
                std::cerr << "ERROR loading " << fpath << ": " << e.what() << "\n";
                continue;
            }
            const float tau = (cfg.tau_mode == "vrel")
                ? cfg.eps * (A.maxCoeff() - A.minCoeff())
                : cfg.tau;
            run_snapshot(A, var, 0, fpath, dtype, tau, cfg, k_max_init,
                         nullptr, nullptr, recon, writer);
        }
        std::cout << "\nAll done -> " << csv_path << "\n";
        return 0;
    }

    std::cout << "All done.\n";
    return 0;
}
