#include "rsvd.hpp"
#include "timer.hpp"
#include "rng.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>
#include <chrono>

SVDResult cold_rsvd(const MatF& A, int k, int p, int q, uint64_t seed) {
    const Idx m = A.rows();
    const Idx n = A.cols();
    const int sketch = k + p;

    SVDResult res;
    res.warm_start = false;
    res.r_prev     = 0;

    PerPhaseTimer timer;
    auto t_total_start = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // 1. Random sketch matrix  Omega ~ N(0,1)  shape (n, k+p)
    // -----------------------------------------------------------------------
    timer.start("omega_gen");
    MatF Omega = randn_matrix(n, sketch, seed);
    timer.stop("omega_gen");

    // -----------------------------------------------------------------------
    // 2. Initial sampling  Y = A @ Omega   shape (m, k+p)
    // -----------------------------------------------------------------------
    timer.start("initial_matmul");
    MatF Y = A * Omega;
    timer.stop("initial_matmul");

    // -----------------------------------------------------------------------
    // 3. Power iterations: Y = (A @ A.T)^q @ Y
    // -----------------------------------------------------------------------
    timer.start("power_iter");
    for (int i = 0; i < q; ++i) {
        MatF Z = A.transpose() * Y;   // (n, k+p)
        Y = A * Z;                     // (m, k+p)
    }
    timer.stop("power_iter");

    // -----------------------------------------------------------------------
    // 4. QR decomposition  Q, _ = qr(Y)   Q shape (m, k+p)
    // -----------------------------------------------------------------------
    timer.start("qr");
    Eigen::HouseholderQR<MatF> qr_decomp(Y);
    // Extract thin Q: multiply Q * I_{m x sketch}
    MatF Q = qr_decomp.householderQ() * MatF::Identity(m, sketch);
    timer.stop("qr");

    // -----------------------------------------------------------------------
    // 5. Projection  B = Q.T @ A   shape (k+p, n)
    // -----------------------------------------------------------------------
    timer.start("projection");
    MatF B = Q.transpose() * A;   // (sketch, n)
    timer.stop("projection");

    // -----------------------------------------------------------------------
    // 6. Small SVD of B   Uhat (sketch x sketch), s (sketch), Vt (sketch x n)
    // -----------------------------------------------------------------------
    timer.start("small_svd");
    Eigen::BDCSVD<MatF, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(B);
    MatF Uhat = svd.matrixU();   // (sketch, sketch) — thin U of a (sketch×n) matrix
    timer.stop("small_svd");

    // -----------------------------------------------------------------------
    // 7. Lift  U = Q @ Uhat   shape (m, sketch)
    // -----------------------------------------------------------------------
    timer.start("lift");
    MatF U_full = Q * Uhat;   // (m, sketch)
    timer.stop("lift");

    // -----------------------------------------------------------------------
    // 8. Truncate to rank k
    // -----------------------------------------------------------------------
    res.U  = U_full.leftCols(k);
    res.s  = svd.singularValues().head(k);
    // Vt from BDCSVD is (min(sketch,n) × n); we need rows = cols of Uhat cols used
    // matrixV() is (n × min(sketch,n)); Vt = matrixV().T
    res.Vt = svd.matrixV().transpose().topRows(k);   // (k, n)

    // -----------------------------------------------------------------------
    // Finalize timings
    // -----------------------------------------------------------------------
    double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_total_start).count();

    res.timings["omega_gen"]      = timer.get("omega_gen");
    res.timings["initial_matmul"] = timer.get("initial_matmul");
    res.timings["power_iter"]     = timer.get("power_iter");
    res.timings["qr"]             = timer.get("qr");
    res.timings["projection"]     = timer.get("projection");
    res.timings["small_svd"]      = timer.get("small_svd");
    res.timings["lift"]           = timer.get("lift");
    res.timings["total"]          = total;

    // Set final matmul counts (clean, no double-counting):
    res.matmuls["AX"]  = 1 + q;   // initial_matmul + q*(A @ Z in power_iter)
    res.matmuls["ATX"] = 1 + q;   // projection     + q*(A.T @ Y in power_iter)

    return res;
}
