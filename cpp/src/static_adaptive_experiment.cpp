// ===========================================================================
// Static (single-snapshot) adaptive residual-corrected rSVD.
//
// Counterpart to adaptive_experiment.cpp, but for the SDRBench datasets (NYX,
// Miranda) which provide ONE 3D volume per variable — no time sequence and
// therefore no warm-start.  Every variable is a single COLD bootstrap of the
// two-stage adaptive compressor:
//
//   Stage 1 : cold rSVD, adaptive rank k* chosen by a cost sweep against tau.
//   Stage 2 : cold rSVD on the residual R1, adaptive residual rank r* (skipped
//             when the residual is cheaper to store as sparse outliers).
//   Sparse  : remaining |error| > tau entries stored explicitly.
//
// The helper routines below are intentionally duplicated from
// adaptive_experiment.cpp so the warm/streaming driver stays untouched.
// ===========================================================================
#include "rsvd.hpp"
#include "metrics.hpp"
#include "data_loader.hpp"
#include "adaptive_csv_writer.hpp"

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
#include <limits>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Dataset presets.  cols = outermost (slowest-varying) cube axis; rows =
// product of the two inner axes.  See load_static_matrix() for the reshape.
// ---------------------------------------------------------------------------
struct Dataset {
    std::string              name;
    Idx                      n_rows;
    int                      n_cols;
    bool                     is_double;
    std::string              ext;
    std::vector<std::string> vars;
    std::string              default_dir;
};

static Dataset make_dataset(const std::string& name) {
    if (name == "nyx") {
        // 512 x 512 x 512 float32 -> (262144, 512)
        return {"nyx", 512 * 512, 512, false, ".f32",
                {"baryon_density", "dark_matter_density", "temperature",
                 "velocity_x", "velocity_y", "velocity_z"},
                "data/NYX_raw"};
    }
    if (name == "miranda") {
        // x:384 (fastest), y:384, z:256 (slowest) float64 -> (147456, 256)
        return {"miranda", 384 * 384, 256, true, ".d64",
                {"density", "diffusivity", "pressure",
                 "velocityx", "velocityy", "velocityz", "viscocity"},
                "data/MIRANDA_raw"};
    }
    throw std::runtime_error("unknown dataset '" + name + "' (expected nyx|miranda)");
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Config {
    std::string              dataset;          // required: nyx|miranda
    std::string              data_dir;         // default: dataset.default_dir
    std::string              out_dir;          // default: results/<dataset>/adaptive
    std::vector<std::string> vars;             // default: dataset.vars

    std::string              tau_mode    = "abs";   // "abs" | "vrel"
    float                    tau         = -1.0f;   // required when tau_mode == abs
    float                    eps         = -1.0f;   // required when tau_mode == vrel
    std::string              eps_str;               // raw --eps string, for filenames
    int                      k_max_init  = 32;       // initial rank cap for the search
    int                      k_expand    = 32;       // window growth when k* hits the cap
    int                      p_cold      = 10;
    int                      q           = 0;
    int                      r_max       = 4;
    int                      r_expand    = 4;
    int                      p_stage2    = 5;
    int                      q_probe     = -1;       // default = r_max
    int                      p_probe     = 5;
    int                      c_entry     = 12;
    uint64_t                 seed        = 42;

    int c_float = 4;  // sizeof(float) — compression baseline representation
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " --dataset <nyx|miranda> --tau <float> [options]\n"
        << "  --dataset     <name>   Dataset preset: nyx | miranda   (REQUIRED)\n"
        << "  --data-dir    <path>   Root data directory             (default: data/<DATASET>_raw)\n"
        << "  --out-dir     <path>   Output CSV directory            (default: results/<dataset>/adaptive)\n"
        << "  --vars        v1 v2    Variable names (space-sep)      (default: all in dataset)\n"
        << "  --tau-mode    <mode>   Tolerance mode: abs | vrel      (default: abs)\n"
        << "  --tau         <float>  Absolute elementwise tolerance  (REQUIRED if tau-mode=abs)\n"
        << "  --eps         <float>  Relative tolerance; tau=eps*range per var (REQUIRED if tau-mode=vrel)\n"
        << "  --k-max-init  <int>    Initial rank cap for the search (default: 32)\n"
        << "  --k-expand    <int>    Window growth when k* caps      (default: 32)\n"
        << "  --p-cold      <int>    Cold rSVD oversampling          (default: 10)\n"
        << "  --q           <int>    Power iterations                (default: 0)\n"
        << "  --r-max       <int>    Max residual rank for stage 2   (default: 4)\n"
        << "  --r-expand    <int>    Residual rank expansion         (default: 4)\n"
        << "  --p-stage2    <int>    Oversampling for stage 2        (default: 5)\n"
        << "  --q-probe     <int>    Probe rank for residual spectrum(default: r_max)\n"
        << "  --p-probe     <int>    Oversampling for spectrum probe (default: 5)\n"
        << "  --c-entry     <int>    Sparse entry cost (bytes)       (default: 12)\n"
        << "  --seed        <uint>   RNG seed                        (default: 42)\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--dataset"    && i+1 < argc) { cfg.dataset    = argv[++i]; }
        else if (arg == "--data-dir"   && i+1 < argc) { cfg.data_dir   = argv[++i]; }
        else if (arg == "--out-dir"    && i+1 < argc) { cfg.out_dir    = argv[++i]; }
        else if (arg == "--tau-mode"   && i+1 < argc) { cfg.tau_mode   = argv[++i]; }
        else if (arg == "--tau"        && i+1 < argc) { cfg.tau        = std::stof(argv[++i]); }
        else if (arg == "--eps"        && i+1 < argc) { cfg.eps_str    = argv[++i]; cfg.eps = std::stof(cfg.eps_str); }
        else if (arg == "--k-max-init" && i+1 < argc) { cfg.k_max_init = std::stoi(argv[++i]); }
        else if (arg == "--k-expand"   && i+1 < argc) { cfg.k_expand   = std::stoi(argv[++i]); }
        else if (arg == "--p-cold"     && i+1 < argc) { cfg.p_cold     = std::stoi(argv[++i]); }
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
    if (cfg.dataset.empty()) { std::cerr << "Error: --dataset is required\n"; print_usage(argv[0]); std::exit(1); }
    if (cfg.tau_mode == "vrel") {
        if (cfg.eps <= 0) { std::cerr << "Error: --eps (>0) is required for --tau-mode vrel\n"; print_usage(argv[0]); std::exit(1); }
    } else if (cfg.tau_mode == "abs") {
        if (cfg.tau < 0)  { std::cerr << "Error: --tau is required for --tau-mode abs\n"; print_usage(argv[0]); std::exit(1); }
    } else {
        std::cerr << "Error: --tau-mode must be 'abs' or 'vrel'\n"; print_usage(argv[0]); std::exit(1);
    }
    if (cfg.q_probe < 0) cfg.q_probe = cfg.r_max;
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

// ---------------------------------------------------------------------------
// Count violations: |R_ij| > tau, plus ||R||_F^2 in double precision.
// ---------------------------------------------------------------------------
struct ViolationResult {
    int64_t count;
    double  fro_norm_sq;
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

// Violations of a precomputed rank-k reconstruction against A:
//   #{ (i,j) : |A_ij - recon_ij| > tau }.
// O(m*n).  The stage-1 search maintains `recon` incrementally (one rank-1 term
// per rank), so this replaces the old O(m*n*k)-per-candidate truncation loop.
static int64_t count_violations_recon(const MatF& A, const MatF& recon, float tau) {
    int64_t cnt = 0;
    for (Idx j = 0; j < A.cols(); ++j)
        for (Idx i = 0; i < A.rows(); ++i)
            if (std::abs(A(i, j) - recon(i, j)) > tau) ++cnt;
    return cnt;
}

// Violations of a stage-2 correction of rank r on residual R1.
static int64_t count_violations_stage2(
    const MatF& R1, const MatF& U2, const VecF& s2, const MatF& Vt2,
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

static int64_t compute_cost(int rank, Idx m, Idx n, int64_t violations,
                            int c_float, int c_entry) {
    return static_cast<int64_t>(rank) * (m + n + 1) * c_float
         + violations * c_entry;
}

static std::vector<int> build_stage1_candidates(int k_lo, int k_hi) {
    std::vector<int> cands;
    for (int k = k_lo; k <= std::min(k_hi, 8); ++k)
        cands.push_back(k);
    for (int k = 12; k <= k_hi; k += 4)
        if (k > 8) cands.push_back(k);
    if (cands.empty() || cands.back() != k_hi)
        cands.push_back(k_hi);
    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
    return cands;
}

static std::vector<int> build_stage2_candidates(int r_max_t) {
    std::vector<int> cands = {0};
    for (int r = 1; r <= r_max_t; r *= 2)
        cands.push_back(r);
    if (cands.back() != r_max_t)
        cands.push_back(r_max_t);
    return cands;
}

static void materialize_residual(MatF& R, const MatF& A,
                                 const MatF& U, const VecF& s, const MatF& Vt,
                                 int k) {
    R.resize(A.rows(), A.cols());
    MatF US = U.leftCols(k) * s.head(k).asDiagonal();
    R.noalias() = A - US * Vt.topRows(k);
}

struct CombinedMetrics {
    float max_elem_error;
    float psnr;
    float fro_error;
};

static CombinedMetrics compute_combined_metrics(
    const MatF& R1,
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
            float final_err = (std::abs(e) > tau) ? 0.0f : std::abs(e);
            if (final_err > max_err) max_err = final_err;
            sum_sq += static_cast<double>(final_err) * static_cast<double>(final_err);
        }
    }
    float fro_err = (norm_A_sq > 0)
        ? static_cast<float>(std::sqrt(sum_sq / norm_A_sq)) : 0.0f;
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

    Dataset ds = make_dataset(cfg.dataset);
    if (cfg.data_dir.empty()) cfg.data_dir = ds.default_dir;
    if (cfg.out_dir.empty())  cfg.out_dir  = "results/" + cfg.dataset + "/adaptive";
    if (cfg.vars.empty())     cfg.vars     = ds.vars;

    fs::create_directories(cfg.out_dir);

    const int   c_float = cfg.c_float;
    const int   c_entry = cfg.c_entry;
    const Idx   m       = ds.n_rows;
    const int   n       = ds.n_cols;
    const std::string src_dtype = ds.is_double ? "float64->float32" : "float32";

    std::cout << "Static Adaptive Residual-Corrected rSVD (cold-only)\n"
              << "  dataset    : " << ds.name
              << "  matrix     : (" << m << " x " << n << ")  dtype=" << src_dtype << "\n"
              << "  data_dir   : " << cfg.data_dir << "\n"
              << "  out_dir    : " << cfg.out_dir  << "\n"
              << "  vars       : ";
    for (auto& v : cfg.vars) std::cout << v << " ";
    std::cout << "\n";
    if (cfg.tau_mode == "vrel")
        std::cout << "  tau_mode   = vrel  eps = " << cfg.eps
                  << "  (tau = eps * value_range, per variable)\n";
    else
        std::cout << "  tau_mode   = abs   tau = " << cfg.tau << "\n";
    std::cout << "  k_max_init = " << cfg.k_max_init
              << "  k_expand = " << cfg.k_expand
              << "  p_cold = " << cfg.p_cold << "  q = " << cfg.q << "\n"
              << "  r_max = " << cfg.r_max << "  r_expand = " << cfg.r_expand
              << "  p_stage2 = " << cfg.p_stage2 << "  c_entry = " << cfg.c_entry
              << " bytes  seed = " << cfg.seed << "\n\n";

    // Filename tag keeps the three eps runs separable when concatenated.
    std::string tag = (cfg.tau_mode == "vrel")
        ? "_eps" + cfg.eps_str
        : "_tau" + std::to_string(cfg.tau);
    std::string csv_path = cfg.out_dir + "/" + cfg.dataset + "_static_adaptive" + tag + ".csv";
    AdaptiveRowWriter writer(csv_path);
    std::cout << "Writing -> " << csv_path << "\n\n";

    MatF R1;  // reusable residual buffer

    for (const auto& var : cfg.vars) {
        std::string fpath = cfg.data_dir + "/" + var + ds.ext;
        std::cout << "[" << std::setw(20) << std::left << var << "] " << std::flush;

        // ------------------------------------------------------------------
        // Load
        // ------------------------------------------------------------------
        MatF A;
        try {
            A = load_static_matrix(fpath, ds.n_rows, ds.n_cols, ds.is_double);
        } catch (const std::exception& e) {
            std::cerr << "ERROR loading " << fpath << ": " << e.what() << "\n";
            continue;
        }
        const int64_t original_bytes = static_cast<int64_t>(m) * n * c_float;
        const float   peak_A = A.cwiseAbs().maxCoeff();
        const float   vmin   = A.minCoeff();
        const float   vmax   = A.maxCoeff();
        const float   value_range = vmax - vmin;
        double norm_A_sq = 0.0;
        for (Idx j = 0; j < A.cols(); ++j)
            norm_A_sq += A.col(j).cast<double>().squaredNorm();

        // Per-variable effective tolerance.
        const float tau = (cfg.tau_mode == "vrel")
            ? cfg.eps * value_range
            : cfg.tau;

        // ------------------------------------------------------------------
        // Stage 1: cold bootstrap rSVD + adaptive rank sweep
        // ------------------------------------------------------------------
        auto t_stage1_start = std::chrono::steady_clock::now();

        const int k_search_lo = 1;
        int       k_search_hi = std::min(cfg.k_max_init, n);  // can't exceed n

        // Cold rSVD computed once at the window top; recomputed only if the
        // optimum lands on the boundary and the window is expanded.
        SVDResult svd1 = cold_rsvd(A, k_search_hi, cfg.p_cold, cfg.q, cfg.seed);

        std::vector<int>     s1_ranks;
        std::vector<int64_t> s1_violations;
        std::vector<int64_t> s1_costs;
        int     k_star = k_search_lo;
        int64_t best_s1_cost = std::numeric_limits<int64_t>::max();
        int64_t best_s1_violations = 0;
        bool    k_expanded = false;

        // Running rank-k reconstruction A_k, accumulated one rank-1 term at a
        // time as  recon += U(:,ell) * (s(ell) * Vt(ell,:)).  Ascending ell
        // reproduces the exact float32 accumulation order of the old per-
        // candidate truncation loop, so selected ranks/metrics are unchanged.
        MatF recon(m, n);
        const int64_t per_rank_bytes = static_cast<int64_t>(m + n + 1) * c_float;
        const int k_grow = std::max(1, cfg.k_expand);

        // Cost C(k) = rank_bytes(k) + violations(k)*c_entry is U-shaped.  Two
        // exact accelerations (neither changes the argmin):
        //   * prune:  once rank_bytes(k) >= best cost, no larger k can win.
        //   * expand: if the best sits on the window top, grow the window
        //             (recompute rSVD) and re-search until the min is interior.
        bool searching = true;
        while (searching) {
            searching = false;
            auto cands = build_stage1_candidates(k_search_lo, k_search_hi);

            s1_ranks.clear();
            s1_violations.clear();
            s1_costs.clear();
            best_s1_cost = std::numeric_limits<int64_t>::max();
            best_s1_violations = 0;
            k_star = cands.front();
            recon.setZero();
            int recon_rank = 0;

            for (int k_cand : cands) {
                if (k_cand > static_cast<int>(svd1.s.size())) break;
                // Exact prune: cost(k') >= rank_bytes(k') >= rank_bytes(k_cand)
                // for every k' >= k_cand, so none can beat the running best.
                if (static_cast<int64_t>(k_cand) * per_rank_bytes >= best_s1_cost)
                    break;
                // Advance the running reconstruction up to rank k_cand.
                while (recon_rank < k_cand) {
                    recon.noalias() += svd1.U.col(recon_rank)
                        * (svd1.s(recon_rank) * svd1.Vt.row(recon_rank));
                    ++recon_rank;
                }
                int64_t viol = count_violations_recon(A, recon, tau);
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

            // Conditional boundary expansion: best at the window top and room
            // to grow => widen and re-search with a fresh, higher-rank rSVD.
            if (k_star == k_search_hi && k_search_hi < n) {
                k_search_hi = std::min(n, k_search_hi + k_grow);
                k_expanded = true;
                svd1 = cold_rsvd(A, k_search_hi, cfg.p_cold, cfg.q, cfg.seed);
                searching = true;
            }
        }
        recon.resize(0, 0);  // free the m x n buffer before R1 is materialized

        double t_stage1 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_stage1_start).count();

        // k* still on the boundary after expansion => capped at the matrix rank.
        bool k_hit_cap = (k_star == k_search_hi);

        // ------------------------------------------------------------------
        // Stage 1 quality at k_star
        // ------------------------------------------------------------------
        int64_t s0 = best_s1_violations;
        int64_t stage1_rank_bytes = static_cast<int64_t>(k_star) * (m + n + 1) * c_float;

        float warm_fro = fro_error(A, svd1.s.head(k_star));
        float opt_fro  = optimal_fro_error(A, k_star);
        float warm_fro_overhead = (opt_fro > 0) ? warm_fro / opt_fro - 1.0f
                                : std::numeric_limits<float>::quiet_NaN();

        CompressionMetrics warm_cm = compression_metrics(
            A, svd1.U.leftCols(k_star), svd1.s.head(k_star),
            svd1.Vt.topRows(k_star));

        // ------------------------------------------------------------------
        // Materialize residual R1 at k_star
        // ------------------------------------------------------------------
        materialize_residual(R1, A, svd1.U, svd1.s, svd1.Vt, k_star);
        auto viol_result = count_violations(R1, tau);
        double residual_fro_sq = viol_result.fro_norm_sq;
        float  residual_fro_norm = static_cast<float>(std::sqrt(residual_fro_sq));

        // ------------------------------------------------------------------
        // Stage 2 decision (cold rSVD on residual)
        // ------------------------------------------------------------------
        auto t_stage2_start = std::chrono::steady_clock::now();

        int     r_star = 0;
        int64_t r_star_violations = s0;
        bool    stage2_skipped = false;
        std::string skip_reason = "none";

        VecF  probe_svs;
        float spectral_concentration = std::numeric_limits<float>::quiet_NaN();

        std::vector<int>     s2_ranks;
        std::vector<int64_t> s2_violations;
        std::vector<int64_t> s2_costs;

        SVDResult svd2;

        if (s0 == 0) {
            stage2_skipped = true;
            skip_reason = "no_violations";
            r_star_violations = 0;
        } else {
            int64_t C0 = s0 * c_entry;
            int64_t C_rank1 = static_cast<int64_t>(m + n + 1) * c_float;
            if (C0 <= C_rank1) {
                stage2_skipped = true;
                skip_reason = "sparse_cheap";
            } else {
                probe_svs = cold_rsvd(R1, cfg.q_probe, cfg.p_probe, 0,
                                      cfg.seed + 99).s;
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
            int r_max_t = cfg.r_max;
            svd2 = cold_rsvd(R1, r_max_t, cfg.p_stage2, cfg.q, cfg.seed + 1);

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
                                                   r_cand, tau);
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

            // Escape: r_star hit r_max and sparse cost still high -> expand once
            if (r_star == r_max_t &&
                r_star_violations * c_entry > compute_cost(1, m, n, 0, c_float, c_entry))
            {
                r_max_t += cfg.r_expand;
                svd2 = cold_rsvd(R1, r_max_t, cfg.p_stage2, cfg.q, cfg.seed + 1);

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
                                                       r_cand, tau);
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

        // ------------------------------------------------------------------
        // Combined metrics + totals
        // ------------------------------------------------------------------
        int64_t stage2_rank_bytes = static_cast<int64_t>(r_star) * (m + n + 1) * c_float;
        int64_t sparse_bytes_val  = r_star_violations * c_entry;
        int64_t total_bytes = stage1_rank_bytes + stage2_rank_bytes + sparse_bytes_val;
        float compression_ratio = (total_bytes > 0)
            ? static_cast<float>(original_bytes) / static_cast<float>(total_bytes)
            : std::numeric_limits<float>::infinity();

        CombinedMetrics combined = compute_combined_metrics(
            R1,
            (r_star > 0) ? &svd2.U : nullptr,
            (r_star > 0) ? &svd2.s : nullptr,
            (r_star > 0) ? &svd2.Vt : nullptr,
            r_star, tau, peak_A, norm_A_sq, m, n);

        // ------------------------------------------------------------------
        // CSV row (reuses the streaming schema; warm/timestep fields are N/A)
        // ------------------------------------------------------------------
        AdaptiveRowWriter::RowData row;

        row.var           = var;
        row.timestep      = 0;            // single snapshot, no time axis
        row.k_max_init    = cfg.k_max_init;
        row.k_delta       = 0;            // N/A (bootstrap searches full range)
        row.p_cold        = cfg.p_cold;
        row.p_warm        = 0;            // N/A (no warm start)
        row.q             = cfg.q;
        row.seed          = static_cast<int>(cfg.seed);
        row.dtype         = src_dtype;
        row.device        = "cpu_cpp";
        row.data_file     = fpath;
        row.run_timestamp = iso_timestamp();
        row.tau           = tau;
        row.r_max         = cfg.r_max;
        row.c_entry_bytes = cfg.c_entry;

        row.k_star                 = k_star;
        row.k_search_lo            = k_search_lo;
        row.k_search_hi            = k_search_hi;
        row.k_expanded             = k_expanded;
        row.s0_violations_at_kstar = s0;
        row.stage1_rank_bytes      = stage1_rank_bytes;
        row.stage1_time            = t_stage1;
        row.stage1_warm_start      = false;

        row.warm_fro_error          = warm_fro;
        row.warm_max_elem_error     = warm_cm.max_elem_error;
        row.warm_psnr               = warm_cm.psnr;
        row.warm_pctl_99            = warm_cm.pctl_99;
        row.warm_pctl_999           = warm_cm.pctl_999;
        row.cold_fro_error_at_kstar = warm_fro;  // cold == the only path here
        row.optimal_fro_error_at_kstar = opt_fro;
        row.warm_fro_overhead       = warm_fro_overhead;

        row.stage1_sweep_ranks      = join_ints(s1_ranks);
        row.stage1_sweep_violations = join_int64s(s1_violations);
        row.stage1_sweep_costs      = join_int64s(s1_costs);

        row.residual_fro_norm               = residual_fro_norm;
        row.residual_spectral_concentration = spectral_concentration;
        if (probe_svs.size() >= 1)  row.residual_sv_1  = probe_svs(0);
        if (probe_svs.size() >= 2)  row.residual_sv_2  = probe_svs(1);
        if (probe_svs.size() >= 5)  row.residual_sv_5  = probe_svs(4);
        if (probe_svs.size() >= 10) row.residual_sv_10 = probe_svs(9);

        row.r_star              = r_star;
        row.r_star_violations   = r_star_violations;
        row.stage2_rank_bytes   = stage2_rank_bytes;
        row.sparse_bytes        = sparse_bytes_val;
        row.stage2_time         = t_stage2;
        row.stage2_warm_start   = false;
        row.stage2_skipped      = stage2_skipped;
        row.stage2_skip_reason  = skip_reason;

        row.stage2_sweep_ranks      = join_ints(s2_ranks);
        row.stage2_sweep_violations = join_int64s(s2_violations);
        row.stage2_sweep_costs      = join_int64s(s2_costs);

        row.total_compressed_bytes = total_bytes;
        row.original_bytes         = original_bytes;
        row.compression_ratio      = compression_ratio;
        row.total_sparse_entries   = r_star_violations;

        row.combined_max_elem_error = combined.max_elem_error;
        row.combined_psnr           = combined.psnr;
        row.combined_fro_error      = combined.fro_error;

        if (probe_svs.size() > 0) {
            std::vector<float> sv_vec(probe_svs.data(),
                                      probe_svs.data() + probe_svs.size());
            row.residual_sv_spectrum = join_floats(sv_vec);
        }

        writer.write_row(row);

        // ------------------------------------------------------------------
        // Progress
        // ------------------------------------------------------------------
        std::cout << "tau=" << std::setprecision(4) << tau << "  "
                  << std::fixed
                  << "k*=" << std::setw(3) << k_star
                  << "  r*=" << r_star
                  << "  s0=" << s0
                  << "  s_r*=" << r_star_violations
                  << "  maxe=" << std::setprecision(3) << combined.max_elem_error
                  << "  ratio=" << std::setprecision(1) << compression_ratio << "x"
                  << "  total=" << std::setprecision(1) << total_bytes / 1e6 << "MB"
                  << "  t1=" << std::setprecision(0) << t_stage1 * 1000 << "ms"
                  << "  t2=" << t_stage2 * 1000 << "ms";
        if (stage2_skipped) std::cout << "  [skip:" << skip_reason << "]";
        if (k_hit_cap)      std::cout << "  [WARN k* hit k_max_init=" << cfg.k_max_init
                                      << "; raise --k-max-init]";
        std::cout << "\n";
    }

    std::cout << "\nAll done -> " << csv_path << "\n";
    return 0;
}
