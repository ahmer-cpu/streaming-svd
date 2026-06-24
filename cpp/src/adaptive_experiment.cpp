#include "rsvd.hpp"
#include "warm_rsvd.hpp"
#include "metrics.hpp"
#include "data_loader.hpp"
#include "adaptive_csv_writer.hpp"
#include "timer.hpp"

#include <iostream>
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
#include <numeric>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// All 13 Hurricane Isabel variable names
// ---------------------------------------------------------------------------
static const std::vector<std::string> ALL_VARS = {
    "CLOUDf", "Pf", "PRECIPf", "QCLOUDf", "QGRAUPf", "QICEf",
    "QRAINf",  "QSNOWf", "QVAPORf", "TCf", "Uf", "Vf", "Wf"
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    std::string              data_dir    = "data/ISABEL_raw";
    std::string              out_dir     = "results/hurricane/adaptive";
    std::vector<std::string> vars        = ALL_VARS;
    int                      start_t     = 1;
    int                      end_t       = 48;

    float                    tau         = -1.0f;   // required
    int                      k_max_init  = 50;
    int                      k_delta     = 8;
    int                      k_expand    = 16;
    int                      p_cold      = 10;
    int                      p_warm      = 5;
    int                      q           = 0;
    int                      r_max       = 4;
    int                      r_expand    = 4;
    int                      p_stage2    = 5;
    int                      q_probe     = -1;  // default = r_max
    int                      p_probe     = 5;
    int                      c_entry     = 12;
    uint64_t                 seed        = 42;

    // derived
    int c_float = 4;  // sizeof(float)
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --data-dir    <path>   Root data directory               (default: data/ISABEL_raw)\n"
        << "  --out-dir     <path>   Output CSV directory              (default: results/hurricane/adaptive)\n"
        << "  --vars        v1 v2    Variable names (space-sep)        (default: all 13)\n"
        << "  --start       <int>    First timestep (1-indexed)        (default: 1)\n"
        << "  --end         <int>    Last timestep                     (default: 48)\n"
        << "  --tau         <float>  Max elementwise error tolerance   (REQUIRED)\n"
        << "  --k-max-init  <int>    Oversized rank for cold bootstrap (default: 50)\n"
        << "  --k-delta     <int>    Rank search half-width            (default: 8)\n"
        << "  --k-expand    <int>    Rank expansion on boundary hit    (default: 16)\n"
        << "  --p-cold      <int>    Cold rSVD oversampling            (default: 10)\n"
        << "  --p-warm      <int>    Warm rSVD oversampling            (default: 5)\n"
        << "  --q           <int>    Power iterations                  (default: 0)\n"
        << "  --r-max       <int>    Max residual rank for stage 2     (default: 4)\n"
        << "  --r-expand    <int>    Residual rank expansion           (default: 4)\n"
        << "  --p-stage2    <int>    Oversampling for stage 2          (default: 5)\n"
        << "  --q-probe     <int>    Probe rank for residual spectrum  (default: r_max)\n"
        << "  --p-probe     <int>    Oversampling for spectrum probe   (default: 5)\n"
        << "  --c-entry     <int>    Sparse entry cost (bytes)         (default: 12)\n"
        << "  --seed        <uint>   RNG seed                         (default: 42)\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-dir"   && i+1 < argc) { cfg.data_dir   = argv[++i]; }
        else if (arg == "--out-dir"    && i+1 < argc) { cfg.out_dir    = argv[++i]; }
        else if (arg == "--start"      && i+1 < argc) { cfg.start_t    = std::stoi(argv[++i]); }
        else if (arg == "--end"        && i+1 < argc) { cfg.end_t      = std::stoi(argv[++i]); }
        else if (arg == "--tau"        && i+1 < argc) { cfg.tau        = std::stof(argv[++i]); }
        else if (arg == "--k-max-init" && i+1 < argc) { cfg.k_max_init = std::stoi(argv[++i]); }
        else if (arg == "--k-delta"    && i+1 < argc) { cfg.k_delta    = std::stoi(argv[++i]); }
        else if (arg == "--k-expand"   && i+1 < argc) { cfg.k_expand   = std::stoi(argv[++i]); }
        else if (arg == "--p-cold"     && i+1 < argc) { cfg.p_cold     = std::stoi(argv[++i]); }
        else if (arg == "--p-warm"     && i+1 < argc) { cfg.p_warm     = std::stoi(argv[++i]); }
        else if (arg == "--q"          && i+1 < argc) { cfg.q          = std::stoi(argv[++i]); }
        else if (arg == "--r-max"      && i+1 < argc) { cfg.r_max      = std::stoi(argv[++i]); }
        else if (arg == "--r-expand"   && i+1 < argc) { cfg.r_expand   = std::stoi(argv[++i]); }
        else if (arg == "--p-stage2"   && i+1 < argc) { cfg.p_stage2   = std::stoi(argv[++i]); }
        else if (arg == "--q-probe"    && i+1 < argc) { cfg.q_probe    = std::stoi(argv[++i]); }
        else if (arg == "--p-probe"    && i+1 < argc) { cfg.p_probe    = std::stoi(argv[++i]); }
        else if (arg == "--c-entry"    && i+1 < argc) { cfg.c_entry    = std::stoi(argv[++i]); }
        else if (arg == "--seed"       && i+1 < argc) { cfg.seed       = static_cast<uint64_t>(std::stoull(argv[++i])); }
        else if (arg == "--vars") {
            cfg.vars.clear();
            while (i+1 < argc && argv[i+1][0] != '-')
                cfg.vars.push_back(argv[++i]);
            if (cfg.vars.empty()) { std::cerr << "Error: --vars requires at least one name\n"; std::exit(1); }
        }
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown argument: " << arg << "\n"; print_usage(argv[0]); std::exit(1); }
    }
    if (cfg.tau < 0) { std::cerr << "Error: --tau is required\n"; print_usage(argv[0]); std::exit(1); }
    if (cfg.q_probe < 0) cfg.q_probe = cfg.r_max;
    return cfg;
}

// ---------------------------------------------------------------------------
// Utility: build file path
// ---------------------------------------------------------------------------
static std::string bin_path(const std::string& data_dir,
                             const std::string& var, int t) {
    std::ostringstream ss;
    ss << data_dir << "/" << var << "/" << var
       << std::setfill('0') << std::setw(2) << t << ".bin";
    return ss.str();
}

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

// ---------------------------------------------------------------------------
// Count violations: |R_ij| > tau, column-by-column
// Also computes ||R||_F in double precision as a side output.
// ---------------------------------------------------------------------------
struct ViolationResult {
    int64_t count;
    double  fro_norm_sq;   // ||R||_F^2
};

static ViolationResult count_violations(const MatF& R, float tau) {
    int64_t cnt = 0;
    double  fro_sq = 0.0;
    for (Idx j = 0; j < R.cols(); ++j) {
        for (Idx i = 0; i < R.rows(); ++i) {
            float ae = std::abs(R(i, j));
            fro_sq += static_cast<double>(ae) * static_cast<double>(ae);
            if (ae > tau) ++cnt;
        }
    }
    return {cnt, fro_sq};
}

// ---------------------------------------------------------------------------
// Violations of a precomputed rank-k reconstruction against A:
//   #{ (i,j) : |A_ij - recon_ij| > tau }.
// O(m*n).  The stage-1 sweep maintains `recon` incrementally (one rank-1 term
// per rank, ascending), replacing the old O(m*n*k)-per-candidate truncation
// loop with identical float accumulation order.
// ---------------------------------------------------------------------------
static int64_t count_violations_recon(const MatF& A, const MatF& recon, float tau) {
    int64_t cnt = 0;
    for (Idx j = 0; j < A.cols(); ++j)
        for (Idx i = 0; i < A.rows(); ++i)
            if (std::abs(A(i, j) - recon(i, j)) > tau) ++cnt;
    return cnt;
}

// ---------------------------------------------------------------------------
// Count violations for stage-2 correction on residual R1.
// Counts #{|R1_ij - (U2[:,:r] * diag(s2[:r]) * Vt2[:r,:])_ij| > tau}
// Streaming, no materialization.
// ---------------------------------------------------------------------------
static int64_t count_violations_stage2(
    const MatF& R1,
    const MatF& U2, const VecF& s2, const MatF& Vt2,
    int r, float tau)
{
    int64_t cnt = 0;
    const Idx m = R1.rows();
    const Idx n = R1.cols();
    VecF scaled(r);
    for (Idx j = 0; j < n; ++j) {
        for (int ell = 0; ell < r; ++ell)
            scaled(ell) = s2(ell) * Vt2(ell, j);
        for (Idx i = 0; i < m; ++i) {
            float corr = 0.0f;
            for (int ell = 0; ell < r; ++ell)
                corr += U2(i, ell) * scaled(ell);
            if (std::abs(R1(i, j) - corr) > tau) ++cnt;
        }
    }
    return cnt;
}

// ---------------------------------------------------------------------------
// Compute compression cost in bytes
// ---------------------------------------------------------------------------
static int64_t compute_cost(int rank, Idx m, Idx n, int64_t violations,
                             int c_float, int c_entry) {
    return static_cast<int64_t>(rank) * (m + n + 1) * c_float
         + violations * c_entry;
}

// ---------------------------------------------------------------------------
// Build candidate rank set: geometric grid around center
// ---------------------------------------------------------------------------
static std::vector<int> build_stage1_candidates(int k_lo, int k_hi) {
    std::vector<int> cands;
    // Dense at small ranks, then every 4
    for (int k = k_lo; k <= std::min(k_hi, 8); ++k)
        cands.push_back(k);
    for (int k = 12; k <= k_hi; k += 4) {
        if (k > 8) cands.push_back(k);
    }
    // Ensure k_hi is included
    if (cands.empty() || cands.back() != k_hi)
        cands.push_back(k_hi);
    // Deduplicate
    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
    return cands;
}

// Build stage-2 candidate ranks: {0, 1, 2, 4, 8, ...}
static std::vector<int> build_stage2_candidates(int r_max_t) {
    std::vector<int> cands = {0};
    for (int r = 1; r <= r_max_t; r *= 2)
        cands.push_back(r);
    if (cands.back() != r_max_t)
        cands.push_back(r_max_t);
    return cands;
}

// ---------------------------------------------------------------------------
// Materialize residual: R = A - U[:,:k] * diag(s[:k]) * Vt[:k,:]
// ---------------------------------------------------------------------------
static void materialize_residual(MatF& R, const MatF& A,
                                  const MatF& U, const VecF& s, const MatF& Vt,
                                  int k) {
    R.resize(A.rows(), A.cols());
    // Compute U[:,:k] * diag(s[:k])
    MatF US = U.leftCols(k) * s.head(k).asDiagonal();
    // R = A - US * Vt[:k,:]
    R.noalias() = A - US * Vt.topRows(k);
}

// ---------------------------------------------------------------------------
// Compute combined quality metrics after L1 + L2 + sparse correction
// ---------------------------------------------------------------------------
struct CombinedMetrics {
    float max_elem_error;
    float psnr;
    float fro_error;  // relative
};

static CombinedMetrics compute_combined_metrics(
    const MatF& R1,                // residual after stage 1
    const MatF* U2, const VecF* s2, const MatF* Vt2, int r_star,
    float tau, float peak_A, double norm_A_sq, Idx m, Idx n)
{
    float max_err = 0.0f;
    double sum_sq = 0.0;

    for (Idx j = 0; j < n; ++j) {
        for (Idx i = 0; i < m; ++i) {
            float e = R1(i, j);
            if (r_star > 0 && U2 && s2 && Vt2) {
                float corr = 0.0f;
                for (int ell = 0; ell < r_star; ++ell)
                    corr += (*U2)(i, ell) * (*s2)(ell) * (*Vt2)(ell, j);
                e -= corr;
            }
            // After sparse correction: entries with |e| > tau are corrected to 0
            float final_err = (std::abs(e) > tau) ? 0.0f : std::abs(e);
            if (final_err > max_err) max_err = final_err;
            sum_sq += static_cast<double>(final_err) * static_cast<double>(final_err);
        }
    }

    // Relative Frobenius error: ||E_sparse||_F / ||A||_F
    float fro_err = (norm_A_sq > 0)
        ? static_cast<float>(std::sqrt(sum_sq / norm_A_sq))
        : 0.0f;

    // PSNR = 10 log10(peak^2 / MSE)
    double mse = sum_sq / (static_cast<double>(m) * static_cast<double>(n));
    float psnr;
    if (mse < 1e-30 || peak_A < 1e-30f)
        psnr = std::numeric_limits<float>::infinity();
    else
        psnr = static_cast<float>(10.0 * std::log10(
            static_cast<double>(peak_A) * static_cast<double>(peak_A) / mse));

    return {max_err, psnr, fro_err};
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    fs::create_directories(cfg.out_dir);

    std::cout << "Adaptive Warm Residual-Corrected rSVD\n"
              << "  data_dir   : " << cfg.data_dir << "\n"
              << "  out_dir    : " << cfg.out_dir  << "\n"
              << "  vars       : ";
    for (auto& v : cfg.vars) std::cout << v << " ";
    std::cout << "\n"
              << "  timesteps  : " << cfg.start_t << " to " << cfg.end_t << "\n"
              << "  tau        = " << cfg.tau << "\n"
              << "  k_max_init = " << cfg.k_max_init
              << "  k_delta = " << cfg.k_delta
              << "  k_expand = " << cfg.k_expand << "\n"
              << "  p_cold = " << cfg.p_cold << "  p_warm = " << cfg.p_warm
              << "  q = " << cfg.q << "  seed = " << cfg.seed << "\n"
              << "  r_max = " << cfg.r_max << "  r_expand = " << cfg.r_expand
              << "  p_stage2 = " << cfg.p_stage2
              << "  c_entry = " << cfg.c_entry << " bytes\n\n";

    const int c_float = cfg.c_float;
    const int c_entry = cfg.c_entry;

    for (const auto& var : cfg.vars) {
        std::string csv_path = cfg.out_dir + "/" + var + "_adaptive.csv";
        std::cout << "[" << var << "] writing -> " << csv_path << "\n";

        AdaptiveRowWriter writer(csv_path);

        // Warm state across timesteps
        MatF    U1_prev;
        MatF*   U1_prev_ptr = nullptr;
        int     k_prev = 0;

        MatF    U2_prev;
        MatF*   U2_prev_ptr = nullptr;
        int     r_prev = 0;

        // Reusable residual buffer
        MatF R1;

        for (int t = cfg.start_t; t <= cfg.end_t; ++t) {
            std::string fpath = bin_path(cfg.data_dir, var, t);
            std::cout << "  t=" << std::setw(2) << t << "  " << std::flush;

            // ==============================================================
            // Load data
            // ==============================================================
            MatF A;
            try {
                A = load_bin_matrix(fpath);
            } catch (const std::exception& e) {
                std::cerr << "ERROR loading " << fpath << ": " << e.what() << "\n";
                continue;
            }
            const Idx m = A.rows();
            const Idx n = A.cols();
            const int64_t original_bytes = static_cast<int64_t>(m) * n * c_float;
            const float peak_A = A.cwiseAbs().maxCoeff();
            double norm_A_sq = 0.0;
            for (Idx j = 0; j < A.cols(); ++j)
                norm_A_sq += A.col(j).cast<double>().squaredNorm();

            // ==============================================================
            // Stage 1: compute rSVD (cold bootstrap or warm tracking)
            // ==============================================================
            auto t_stage1_start = std::chrono::steady_clock::now();

            bool is_bootstrap = (t == cfg.start_t);
            bool stage1_warm = false;
            bool k_expanded = false;

            // Determine rank search window
            int k_search_lo, k_search_hi;
            if (is_bootstrap) {
                k_search_lo = 1;
                k_search_hi = cfg.k_max_init;
            } else {
                k_search_lo = std::max(1, k_prev - cfg.k_delta);
                k_search_hi = k_prev + cfg.k_delta;
            }

            // Compute rSVD at k_search_hi.
            // For warm rSVD, the sketch has r_prev + p columns. We need
            // r_prev + p >= k_search_hi, so p must be at least
            // k_search_hi - r_prev. Add cfg.p_warm as extra oversampling.
            SVDResult svd1;
            if (is_bootstrap) {
                svd1 = cold_rsvd(A, k_search_hi, cfg.p_cold, cfg.q, cfg.seed);
                stage1_warm = false;
            } else {
                int r_prev_cols = U1_prev_ptr ? static_cast<int>(U1_prev_ptr->cols()) : 0;
                int p_needed = std::max(cfg.p_warm, k_search_hi - r_prev_cols + cfg.p_warm);
                svd1 = warm_rsvd(A, U1_prev_ptr, k_search_hi, p_needed, cfg.q, cfg.seed);
                stage1_warm = svd1.warm_start;
            }

            // ==============================================================
            // Stage 1 rank sweep by truncation
            // ==============================================================
            auto k_candidates = build_stage1_candidates(k_search_lo, k_search_hi);

            // Evaluate each candidate via a running rank-k reconstruction,
            // accumulated one rank-1 term at a time (ascending), so each
            // candidate costs O(m*n) instead of O(m*n*k).
            std::vector<int>     s1_ranks;
            std::vector<int64_t> s1_violations;
            std::vector<int64_t> s1_costs;
            int     k_star = k_candidates.back();
            int64_t best_s1_cost = std::numeric_limits<int64_t>::max();
            int64_t best_s1_violations = 0;

            const int64_t per_rank_bytes = static_cast<int64_t>(m + n + 1) * c_float;
            MatF recon = MatF::Zero(m, n);
            int  recon_rank = 0;

            for (int k_cand : k_candidates) {
                if (k_cand > svd1.s.size()) break;  // can't exceed computed rank
                // Exact prune: cost(k') >= rank_bytes(k') >= rank_bytes(k_cand)
                // for every k' >= k_cand, so none can beat the running best.
                if (static_cast<int64_t>(k_cand) * per_rank_bytes >= best_s1_cost)
                    break;
                while (recon_rank < k_cand) {
                    recon.noalias() += svd1.U.col(recon_rank)
                        * (svd1.s(recon_rank) * svd1.Vt.row(recon_rank));
                    ++recon_rank;
                }
                int64_t viol = count_violations_recon(A, recon, cfg.tau);
                int64_t cost = compute_cost(k_cand, m, n, viol, c_float, c_entry);
                s1_ranks.push_back(k_cand);
                s1_violations.push_back(viol);
                s1_costs.push_back(cost);
                if (cost < best_s1_cost) {
                    best_s1_cost = cost;
                    k_star = k_cand;
                    best_s1_violations = viol;
                }
            }

            // Escape check: if k_star hit the upper boundary and sparse cost is high
            if (!is_bootstrap && k_star == k_search_hi &&
                best_s1_violations * c_entry > compute_cost(1, m, n, 0, c_float, c_entry))
            {
                // Expand once
                k_search_hi += cfg.k_expand;
                k_expanded = true;
                int r_prev_cols2 = U1_prev_ptr ? static_cast<int>(U1_prev_ptr->cols()) : 0;
                int p_needed2 = std::max(cfg.p_warm, k_search_hi - r_prev_cols2 + cfg.p_warm);
                svd1 = warm_rsvd(A, U1_prev_ptr, k_search_hi, p_needed2, cfg.q, cfg.seed);

                // Re-sweep with expanded range (svd1 changed: rebuild recon)
                auto expanded_cands = build_stage1_candidates(k_search_lo, k_search_hi);
                s1_ranks.clear();
                s1_violations.clear();
                s1_costs.clear();
                best_s1_cost = std::numeric_limits<int64_t>::max();
                recon.setZero();
                recon_rank = 0;

                for (int k_cand : expanded_cands) {
                    if (k_cand > svd1.s.size()) break;
                    if (static_cast<int64_t>(k_cand) * per_rank_bytes >= best_s1_cost)
                        break;
                    while (recon_rank < k_cand) {
                        recon.noalias() += svd1.U.col(recon_rank)
                            * (svd1.s(recon_rank) * svd1.Vt.row(recon_rank));
                        ++recon_rank;
                    }
                    int64_t viol = count_violations_recon(A, recon, cfg.tau);
                    int64_t cost = compute_cost(k_cand, m, n, viol, c_float, c_entry);
                    s1_ranks.push_back(k_cand);
                    s1_violations.push_back(viol);
                    s1_costs.push_back(cost);
                    if (cost < best_s1_cost) {
                        best_s1_cost = cost;
                        k_star = k_cand;
                        best_s1_violations = viol;
                    }
                }
            }
            recon.resize(0, 0);  // free the m x n buffer before R1 is materialized

            double t_stage1 = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_stage1_start).count();

            // ==============================================================
            // Stage 1 quality metrics at k_star
            // ==============================================================
            int64_t s0 = best_s1_violations;
            int64_t stage1_rank_bytes = static_cast<int64_t>(k_star) * (m + n + 1) * c_float;

            // Compute fro_error at k_star via identity trick on truncated s
            float warm_fro = fro_error(A, svd1.s.head(k_star));
            float opt_fro  = optimal_fro_error(A, k_star);
            float warm_fro_overhead = (opt_fro > 0) ? warm_fro / opt_fro - 1.0f
                                    : std::numeric_limits<float>::quiet_NaN();

            // Compression metrics at k_star (materializes residual as side effect)
            CompressionMetrics warm_cm = compression_metrics(
                A, svd1.U.leftCols(k_star), svd1.s.head(k_star),
                svd1.Vt.topRows(k_star));

            // ==============================================================
            // Materialize residual R1 at k_star
            // ==============================================================
            materialize_residual(R1, A, svd1.U, svd1.s, svd1.Vt, k_star);
            auto viol_result = count_violations(R1, cfg.tau);
            double residual_fro_sq = viol_result.fro_norm_sq;
            float residual_fro_norm = static_cast<float>(std::sqrt(residual_fro_sq));

            // ==============================================================
            // Stage 2 decision
            // ==============================================================
            auto t_stage2_start = std::chrono::steady_clock::now();

            int     r_star = 0;
            int64_t r_star_violations = s0;
            bool    stage2_skipped = false;
            bool    stage2_warm = false;
            std::string skip_reason = "none";

            // Residual spectrum probe results
            VecF probe_svs;
            float spectral_concentration = std::numeric_limits<float>::quiet_NaN();

            // Sweep data for stage 2
            std::vector<int>     s2_ranks;
            std::vector<int64_t> s2_violations;
            std::vector<int64_t> s2_costs;

            SVDResult svd2;  // stage 2 SVD result (kept for warm state update)

            if (s0 == 0) {
                // Stage 1 already satisfies tau
                stage2_skipped = true;
                skip_reason = "no_violations";
                r_star_violations = 0;
            } else {
                int64_t C0 = s0 * c_entry;
                int64_t C_rank1 = static_cast<int64_t>(m + n + 1) * c_float;

                if (C0 <= C_rank1) {
                    // Sparse-only is cheaper than one residual rank
                    stage2_skipped = true;
                    skip_reason = "sparse_cheap";
                } else {
                    // Probe residual spectrum
                    probe_svs = cold_rsvd(R1, cfg.q_probe, cfg.p_probe, 0,
                                          cfg.seed + 99).s;

                    // Cumulative spectral concentration
                    if (residual_fro_sq > 0 && probe_svs.size() > 0) {
                        double cum_energy = 0.0;
                        for (Idx i = 0; i < probe_svs.size(); ++i)
                            cum_energy += static_cast<double>(probe_svs(i)) * probe_svs(i);
                        spectral_concentration = static_cast<float>(cum_energy / residual_fro_sq);
                    }

                    const float eta = 0.2f;
                    if (spectral_concentration < eta) {
                        stage2_skipped = true;
                        skip_reason = "flat_spectrum";
                    }
                }
            }

            if (!stage2_skipped) {
                // Compute stage 2 rSVD on R1
                int r_max_t = cfg.r_max;
                int r2_prev_cols = U2_prev_ptr ? static_cast<int>(U2_prev_ptr->cols()) : 0;
                int p_stage2_needed = std::max(cfg.p_stage2,
                                               r_max_t - r2_prev_cols + cfg.p_stage2);
                svd2 = warm_rsvd(R1, U2_prev_ptr, r_max_t, p_stage2_needed,
                                 cfg.q, cfg.seed + 1);
                stage2_warm = svd2.warm_start;

                // Candidate rank sweep
                auto r_candidates = build_stage2_candidates(r_max_t);

                int64_t best_s2_cost = s0 * c_entry;  // r=0 baseline
                r_star = 0;
                r_star_violations = s0;

                for (int r_cand : r_candidates) {
                    int64_t viol;
                    if (r_cand == 0) {
                        viol = s0;
                    } else {
                        if (r_cand > svd2.s.size()) break;
                        viol = count_violations_stage2(R1, svd2.U, svd2.s, svd2.Vt,
                                                       r_cand, cfg.tau);
                    }
                    int64_t cost = compute_cost(r_cand, m, n, viol, c_float, c_entry);
                    s2_ranks.push_back(r_cand);
                    s2_violations.push_back(viol);
                    s2_costs.push_back(cost);
                    if (cost < best_s2_cost) {
                        best_s2_cost = cost;
                        r_star = r_cand;
                        r_star_violations = viol;
                    }
                }

                // Escape: if r_star hit r_max_t and sparse cost is still high
                if (r_star == r_max_t &&
                    r_star_violations * c_entry > compute_cost(1, m, n, 0, c_float, c_entry))
                {
                    r_max_t += cfg.r_expand;
                    int r2_prev_cols2 = U2_prev_ptr ? static_cast<int>(U2_prev_ptr->cols()) : 0;
                    int p2_needed2 = std::max(cfg.p_stage2,
                                              r_max_t - r2_prev_cols2 + cfg.p_stage2);
                    svd2 = warm_rsvd(R1, U2_prev_ptr, r_max_t, p2_needed2,
                                     cfg.q, cfg.seed + 1);
                    stage2_warm = svd2.warm_start;

                    auto expanded_r_cands = build_stage2_candidates(r_max_t);
                    s2_ranks.clear();
                    s2_violations.clear();
                    s2_costs.clear();
                    best_s2_cost = s0 * c_entry;
                    r_star = 0;
                    r_star_violations = s0;

                    for (int r_cand : expanded_r_cands) {
                        int64_t viol;
                        if (r_cand == 0) {
                            viol = s0;
                        } else {
                            if (r_cand > svd2.s.size()) break;
                            viol = count_violations_stage2(R1, svd2.U, svd2.s, svd2.Vt,
                                                           r_cand, cfg.tau);
                        }
                        int64_t cost = compute_cost(r_cand, m, n, viol, c_float, c_entry);
                        s2_ranks.push_back(r_cand);
                        s2_violations.push_back(viol);
                        s2_costs.push_back(cost);
                        if (cost < best_s2_cost) {
                            best_s2_cost = cost;
                            r_star = r_cand;
                            r_star_violations = viol;
                        }
                    }
                }
            }

            double t_stage2 = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_stage2_start).count();

            // ==============================================================
            // Combined metrics
            // ==============================================================
            int64_t stage2_rank_bytes = static_cast<int64_t>(r_star) * (m + n + 1) * c_float;
            int64_t sparse_bytes_val = r_star_violations * c_entry;
            int64_t total_bytes = stage1_rank_bytes + stage2_rank_bytes + sparse_bytes_val;
            float compression_ratio = (total_bytes > 0)
                ? static_cast<float>(original_bytes) / static_cast<float>(total_bytes)
                : std::numeric_limits<float>::infinity();

            // Compute combined quality
            CombinedMetrics combined = compute_combined_metrics(
                R1,
                (r_star > 0) ? &svd2.U : nullptr,
                (r_star > 0) ? &svd2.s : nullptr,
                (r_star > 0) ? &svd2.Vt : nullptr,
                r_star, cfg.tau, peak_A, norm_A_sq, m, n);

            // ==============================================================
            // Build CSV row
            // ==============================================================
            AdaptiveRowWriter::RowData row;

            // Identity
            row.var           = var;
            row.timestep      = t;
            row.k_max_init    = cfg.k_max_init;
            row.k_delta       = cfg.k_delta;
            row.p_cold        = cfg.p_cold;
            row.p_warm        = cfg.p_warm;
            row.q             = cfg.q;
            row.seed          = static_cast<int>(cfg.seed);
            row.data_file     = fpath;
            row.run_timestamp = iso_timestamp();
            row.tau           = cfg.tau;
            row.r_max         = cfg.r_max;
            row.c_entry_bytes = cfg.c_entry;

            // Stage 1 adaptive rank
            row.k_star                = k_star;
            row.k_search_lo           = k_search_lo;
            row.k_search_hi           = k_search_hi;
            row.k_expanded            = k_expanded;
            row.s0_violations_at_kstar = s0;
            row.stage1_rank_bytes     = stage1_rank_bytes;
            row.stage1_time           = t_stage1;
            row.stage1_warm_start     = stage1_warm;

            // Stage 1 quality
            row.warm_fro_error          = warm_fro;
            row.warm_max_elem_error     = warm_cm.max_elem_error;
            row.warm_psnr               = warm_cm.psnr;
            row.warm_pctl_99            = warm_cm.pctl_99;
            row.warm_pctl_999           = warm_cm.pctl_999;
            row.cold_fro_error_at_kstar = std::numeric_limits<float>::quiet_NaN(); // not computing cold for speed
            row.optimal_fro_error_at_kstar = opt_fro;
            row.warm_fro_overhead       = warm_fro_overhead;

            // Stage 1 sweep
            row.stage1_sweep_ranks      = join_ints(s1_ranks);
            row.stage1_sweep_violations = join_int64s(s1_violations);
            row.stage1_sweep_costs      = join_int64s(s1_costs);

            // Residual diagnostics
            row.residual_fro_norm               = residual_fro_norm;
            row.residual_spectral_concentration = spectral_concentration;
            if (probe_svs.size() >= 1)  row.residual_sv_1  = probe_svs(0);
            if (probe_svs.size() >= 2)  row.residual_sv_2  = probe_svs(1);
            if (probe_svs.size() >= 5)  row.residual_sv_5  = probe_svs(4);
            if (probe_svs.size() >= 10) row.residual_sv_10 = probe_svs(9);

            // Stage 2 decision
            row.r_star              = r_star;
            row.r_star_violations   = r_star_violations;
            row.stage2_rank_bytes   = stage2_rank_bytes;
            row.sparse_bytes        = sparse_bytes_val;
            row.stage2_time         = t_stage2;
            row.stage2_warm_start   = stage2_warm;
            row.stage2_skipped      = stage2_skipped;
            row.stage2_skip_reason  = skip_reason;

            // Stage 2 sweep
            row.stage2_sweep_ranks      = join_ints(s2_ranks);
            row.stage2_sweep_violations = join_int64s(s2_violations);
            row.stage2_sweep_costs      = join_int64s(s2_costs);

            // Total compression
            row.total_compressed_bytes = total_bytes;
            row.original_bytes         = original_bytes;
            row.compression_ratio      = compression_ratio;
            row.total_sparse_entries   = r_star_violations;

            // Combined quality
            row.combined_max_elem_error = combined.max_elem_error;
            row.combined_psnr           = combined.psnr;
            row.combined_fro_error      = combined.fro_error;

            // Residual spectrum
            if (probe_svs.size() > 0) {
                std::vector<float> sv_vec(probe_svs.data(),
                                          probe_svs.data() + probe_svs.size());
                row.residual_sv_spectrum = join_floats(sv_vec);
            }

            writer.write_row(row);

            // ==============================================================
            // Progress output
            // ==============================================================
            std::cout << std::fixed << std::setprecision(1)
                      << "k*=" << k_star
                      << "  r*=" << r_star
                      << "  s0=" << s0
                      << "  s_r*=" << r_star_violations
                      << "  maxe=" << std::setprecision(3) << combined.max_elem_error
                      << "  ratio=" << std::setprecision(1) << compression_ratio << "x"
                      << "  total=" << std::setprecision(1) << total_bytes / 1e6 << "MB"
                      << "  stage1=" << std::setprecision(0) << t_stage1 * 1000 << "ms"
                      << "  stage2=" << t_stage2 * 1000 << "ms";
            if (stage2_skipped) std::cout << "  [skip:" << skip_reason << "]";
            std::cout << "\n";

            // ==============================================================
            // Update warm state
            // ==============================================================
            U1_prev = svd1.U.leftCols(k_star);
            U1_prev_ptr = &U1_prev;
            k_prev = k_star;

            if (r_star > 0) {
                U2_prev = svd2.U.leftCols(r_star);
                U2_prev_ptr = &U2_prev;
                r_prev = r_star;
            } else {
                U2_prev_ptr = nullptr;
                r_prev = 0;
            }
        }

        std::cout << "  -> done: " << csv_path << "\n\n";
    }

    std::cout << "All done.\n";
    return 0;
}
