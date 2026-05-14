#ifndef FASTGLM_CHUNK_SOURCE_H
#define FASTGLM_CHUNK_SOURCE_H

#include <RcppEigen.h>
#include <cstdlib>
#include <cstdio>

namespace fglm {

// Default row-chunk size (override with env var FASTGLM_CHUNK_ROWS).
// Choose ~16k so a chunk * p * 8 bytes fits comfortably in L2/L3 for typical p.
inline int default_chunk_rows()
{
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("FASTGLM_CHUNK_ROWS");
        int v = 16384;
        if (env != nullptr) {
            int parsed = std::atoi(env);
            if (parsed > 0) v = parsed;
        }
        cached = v;
    }
    return cached;
}

// ---------------------------------------------------------------------------
// Streaming kernels.
//
// Each kernel processes the rows of X in row-blocks of ~chunk_rows.  Each
// chunk borrows a 0-copy block view via X.middleRows(...).  For an
// Eigen::Map<MatrixXd> backed by mmap'd storage (e.g. a filebacked
// big.matrix), this lets the OS page only the active block, never the whole
// matrix.  No additional n*p workspace is allocated.
// ---------------------------------------------------------------------------

// out += X' diag(w_squared) X.  out is assumed to be sized p x p.
inline void accumulate_xtwx_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,
    Eigen::MatrixXd& out,
    int chunk_rows = -1)
{
    if (chunk_rows <= 0) chunk_rows = default_chunk_rows();
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    out.setZero();

    Eigen::MatrixXd WB(chunk_rows, p);
    for (Eigen::Index r = 0; r < n; r += chunk_rows) {
        Eigen::Index k = std::min<Eigen::Index>(chunk_rows, n - r);
        // WB_k = diag(w_k) * X_k  (no copy of X required outside this small block)
        WB.topRows(k).noalias() = w.segment(r, k).asDiagonal() * X.middleRows(r, k);
        out.selfadjointView<Eigen::Lower>().rankUpdate(WB.topRows(k).adjoint());
    }
}

// out += X' (w_squared .* z).  out is assumed to be sized p.
inline void accumulate_xtwz_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,
    const Eigen::Ref<const Eigen::VectorXd>& z,
    Eigen::VectorXd& out,
    int chunk_rows = -1)
{
    if (chunk_rows <= 0) chunk_rows = default_chunk_rows();
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    out.setZero();

    Eigen::VectorXd wz_block(chunk_rows);
    for (Eigen::Index r = 0; r < n; r += chunk_rows) {
        Eigen::Index k = std::min<Eigen::Index>(chunk_rows, n - r);
        // For consistency with the dense path which uses (W X)' (W z), we form
        // w_k .* (w_k .* z_k) here so out = X' diag(w^2) z.
        wz_block.head(k).array() =
            w.segment(r, k).array().square() * z.segment(r, k).array();
        out.noalias() += X.middleRows(r, k).adjoint() * wz_block.head(k);
    }
    (void)p;
}

// eta = X * beta + offset  (computed in chunks, no n*p materialization).
// 'out' must be sized n on entry.
inline void apply_X_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& offset,
    Eigen::Ref<Eigen::VectorXd> out,
    int chunk_rows = -1)
{
    if (chunk_rows <= 0) chunk_rows = default_chunk_rows();
    const Eigen::Index n = X.rows();

    for (Eigen::Index r = 0; r < n; r += chunk_rows) {
        Eigen::Index k = std::min<Eigen::Index>(chunk_rows, n - r);
        out.segment(r, k).noalias() = X.middleRows(r, k) * beta;
        out.segment(r, k) += offset.segment(r, k);
    }
}

// Fused Firth streaming pass: accumulate X'WX and X'Wz* in a single sweep
// over the rows.  z*_i = z_i + ξ_i where ξ_i is the AS_mean bias-reducing
// adjustment computed with lagged leverages from A_prev = (X'WX)^{-1} of the
// previous IRLS iteration.
//
// Both XtWX_out and Xtwzstar_out are zeroed internally.
inline void accumulate_firth_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,           // sqrt working weights
    const Eigen::Ref<const Eigen::VectorXd>& z,           // working response
    const Eigen::Ref<const Eigen::VectorXd>& d2mu,        // d²μ/dη²
    const Eigen::Ref<const Eigen::VectorXd>& var_mu_vec,  // V(μ)
    const Eigen::Ref<const Eigen::VectorXd>& mu_eta_vec,  // dμ/dη
    const Eigen::Ref<const Eigen::VectorXd>& pw,          // prior weights
    double phi,
    const Eigen::MatrixXd& A_prev,                        // p×p (X'WX)^{-1} from prev iter
    Eigen::MatrixXd& XtWX_out,
    Eigen::VectorXd& Xtwzstar_out,
    int chunk_rows = -1)
{
    if (chunk_rows <= 0) chunk_rows = default_chunk_rows();
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    const double eps = std::numeric_limits<double>::epsilon();

    XtWX_out.setZero();
    Xtwzstar_out.setZero();

    Eigen::MatrixXd WB(chunk_rows, p);
    Eigen::MatrixXd XA(chunk_rows, p);
    Eigen::VectorXd wzs_block(chunk_rows);

    for (Eigen::Index r = 0; r < n; r += chunk_rows) {
        Eigen::Index k = std::min<Eigen::Index>(chunk_rows, n - r);
        auto X_k = X.middleRows(r, k);

        // WB_k = diag(w_k) * X_k
        WB.topRows(k).noalias() = w.segment(r, k).asDiagonal() * X_k;

        // X'WX accumulation
        XtWX_out.selfadjointView<Eigen::Lower>().rankUpdate(WB.topRows(k).adjoint());

        // Lagged leverages: h_i = w_i^2 * x_i' A_prev x_i
        //   = rowSums((X_k * A_prev) .* X_k) .* w_k^2
        XA.topRows(k).noalias() = X_k * A_prev;
        Eigen::ArrayXd h_k =
            (XA.topRows(k).array() * X_k.array()).rowwise().sum()
            * w.segment(r, k).array().square();

        // ξ_i = φ * h_i * d2mu_i * V_i / (2 * pw_i * dmu_i^3)
        Eigen::ArrayXd dmu3 = mu_eta_vec.segment(r, k).array().cube();
        for (Eigen::Index i = 0; i < k; ++i)
            if (std::abs(dmu3[i]) < eps) dmu3[i] = (dmu3[i] >= 0) ? eps : -eps;

        Eigen::ArrayXd xi_k = phi * h_k * d2mu.segment(r, k).array()
                               * var_mu_vec.segment(r, k).array()
                               / (2.0 * pw.segment(r, k).array().max(eps) * dmu3);

        // z*_i = z_i + ξ_i
        // w^2 * z* for X'Wz* accumulation
        wzs_block.head(k).array() =
            w.segment(r, k).array().square()
            * (z.segment(r, k).array() + xi_k);
        Xtwzstar_out.noalias() += X_k.adjoint() * wzs_block.head(k);
    }
}

}  // namespace fglm

#endif  // FASTGLM_CHUNK_SOURCE_H
