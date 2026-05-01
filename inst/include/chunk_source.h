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

}  // namespace fglm

#endif  // FASTGLM_CHUNK_SOURCE_H
