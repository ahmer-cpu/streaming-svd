#include "warm_rsvd.hpp"
#include "rsvd.hpp"
#include "timer.hpp"
#include "rng.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <chrono>

SVDResult warm_rsvd(const MatF& A, const MatF* U_prev,
                    int k, int p, int q, uint64_t seed) {
    // ------------------------------------------------------------------
    // Fallback: no prior basis → cold start
    // ------------------------------------------------------------------
    if (U_prev == nullptr) {
        // Use cold defaults: p_cold = 10 (caller passes p_warm=5 for warm,
        // but for the very first timestep we want the cold oversampling).
        // The caller should pass p=p_cold when U_prev is null, but to be
        // safe we fall back with whatever p was given.
        SVDResult res = cold_rsvd(A, k, p, q, seed);
        res.warm_start = false;
        res.r_prev     = 0;
        return res;
    }

    const Idx m      = A.rows();
    const Idx n      = A.cols();
    const Idx r_prev = U_prev->cols();   // rank of previous basis
    const Idx sketch = r_prev + p;

    SVDResult res;
    res.warm_start = true;
    res.r_prev     = static_cast<int>(r_prev);

    PerPhaseTimer timer;
    auto t_total_start = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // 1. Warm projection  G = A.T @ U_prev   shape (n, r_prev)
    // -----------------------------------------------------------------------
    timer.start("warm_proj");
    MatF G = A.transpose() * (*U_prev);   // (n, r_prev)
    timer.stop("warm_proj");

    // -----------------------------------------------------------------------
    // 2. Warm matmul  Y1 = A @ G   shape (m, r_prev)
    // -----------------------------------------------------------------------
    timer.start("warm_matmul");
    MatF Y1 = A * G;   // (m, r_prev)
    timer.stop("warm_matmul");

    // -----------------------------------------------------------------------
    // 3. Random exploration sketch  Omega ~ N(0,1)  shape (n, p)
    // -----------------------------------------------------------------------
    timer.start("omega_gen");
    MatF Omega = randn_matrix(n, p, seed);
    timer.stop("omega_gen");

    // -----------------------------------------------------------------------
    // 4. Random matmul  Y2 = A @ Omega   shape (m, p)
    // -----------------------------------------------------------------------
    timer.start("random_matmul");
    MatF Y2 = A * Omega;   // (m, p)
    timer.stop("random_matmul");

    // -----------------------------------------------------------------------
    // 5. Concatenate  Y = [Y1, Y2]   shape (m, r_prev + p)
    // -----------------------------------------------------------------------
    timer.start("concat");
    MatF Y(m, sketch);
    Y.leftCols(r_prev)  = Y1;
    Y.rightCols(p)      = Y2;
    timer.stop("concat");

    // -----------------------------------------------------------------------
    // 6. Power iterations
    // -----------------------------------------------------------------------
    timer.start("power_iter");
    for (int i = 0; i < q; ++i) {
        MatF Z = A.transpose() * Y;   // (n, sketch)
        Y = A * Z;                     // (m, sketch)
    }
    timer.stop("power_iter");

    // -----------------------------------------------------------------------
    // 7. QR decomposition  Q shape (m, sketch)
    // -----------------------------------------------------------------------
    timer.start("qr");
    Eigen::HouseholderQR<MatF> qr_decomp(Y);
    MatF Q = qr_decomp.householderQ() * MatF::Identity(m, sketch);
    timer.stop("qr");

    // -----------------------------------------------------------------------
    // 8. Projection  B = Q.T @ A   shape (sketch, n)
    // -----------------------------------------------------------------------
    timer.start("projection");
    MatF B = Q.transpose() * A;
    timer.stop("projection");

    // -----------------------------------------------------------------------
    // 9. Small SVD
    // -----------------------------------------------------------------------
    timer.start("small_svd");
    Eigen::BDCSVD<MatF, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(B);
    MatF Uhat = svd.matrixU();
    timer.stop("small_svd");

    // -----------------------------------------------------------------------
    // 10. Lift  U = Q @ Uhat   shape (m, sketch)
    // -----------------------------------------------------------------------
    timer.start("lift");
    MatF U_full = Q * Uhat;
    timer.stop("lift");

    // -----------------------------------------------------------------------
    // 11. Truncate to rank k
    // -----------------------------------------------------------------------
    res.U  = U_full.leftCols(k);
    res.s  = svd.singularValues().head(k);
    res.Vt = svd.matrixV().transpose().topRows(k);

    // -----------------------------------------------------------------------
    // Finalize
    // -----------------------------------------------------------------------
    double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_total_start).count();

    res.timings["warm_proj"]     = timer.get("warm_proj");
    res.timings["warm_matmul"]   = timer.get("warm_matmul");
    res.timings["omega_gen"]     = timer.get("omega_gen");
    res.timings["random_matmul"] = timer.get("random_matmul");
    res.timings["concat"]        = timer.get("concat");
    res.timings["power_iter"]    = timer.get("power_iter");
    res.timings["qr"]            = timer.get("qr");
    res.timings["projection"]    = timer.get("projection");
    res.timings["small_svd"]     = timer.get("small_svd");
    res.timings["lift"]          = timer.get("lift");
    res.timings["total"]         = total;

    // Matmul counts (independent of increments; set directly)
    res.matmuls["AX"]  = 2 + q;        // warm_matmul + random_matmul + q*(A@Z)
    res.matmuls["ATX"] = 1 + 1 + q;   // warm_proj + projection + q*(A.T@Y)

    return res;
}
