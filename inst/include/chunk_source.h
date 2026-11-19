#ifndef FASTGLM_CHUNK_SOURCE_H
#define FASTGLM_CHUNK_SOURCE_H

#include <RcppEigen.h>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include "parallel_reduce.h"

namespace fglm {

// Default row-chunk size (override with env var FASTGLM_CHUNK_ROWS).
// Choose ~16k so a chunk * p * 8 bytes fits comfortably in L2/L3 for typical p.
inline int default_chunk_rows()
{
    const char* env = std::getenv("FASTGLM_CHUNK_ROWS");
    if (env != nullptr) {
        int parsed = std::atoi(env);
        if (parsed > 0) return parsed;
    }
    return 16384;
}

// ---------------------------------------------------------------------------
// Row-block kernels.
//
// Each kernel processes the rows of X in fixed row blocks. Each block borrows
// a view X.middleRows(...), so for an Eigen::Map backed by mmap'd storage
// (for example a filebacked big.matrix) the OS pages only the active blocks,
// and no n x p workspace is allocated.
//
// The sums over blocks go through the deterministic reductions of
// parallel_reduce.h: lanes when the accumulator dimension p is at most
// kLaneMaxDim, column tiles otherwise. The block size is a fixed function of
// p and FASTGLM_CHUNK_ROWS, never of the thread count, so the results are
// identical for every thread count, including 1.
// ---------------------------------------------------------------------------

// Rows per block of the cross-product kernels for p columns: the chunk size
// (FASTGLM_CHUNK_ROWS or the default), capped so that one weighted block of p
// columns is bounded. In the lane scheme (lanes true) the cap is
// max(tile_block_rows(p), 2^21 / p) rows, at most 16 MB per running thread;
// in the tile scheme it is tile_block_rows(p), at most 8 MB per group slot
// (for p up to 4096). Both are fixed functions of p and the chunk size.
inline Eigen::Index reduction_block_rows(Eigen::Index p, bool lanes, int chunk_rows = -1)
{
    if (chunk_rows <= 0) chunk_rows = default_chunk_rows();
    if (p < 1) p = 1;
    const Eigen::Index cap = lanes
        ? std::max<Eigen::Index>(fglm_par::tile_block_rows(p), Eigen::Index(2097152) / p)
        : fglm_par::tile_block_rows(p);
    return std::min<Eigen::Index>(chunk_rows, cap);
}

// Lane accumulator of the row-block kernels: the lower triangle of a p x p
// cross product and two vector sums.
struct LaneAcc {
    Eigen::MatrixXd A;
    Eigen::VectorXd s1, s2;
};

// Workspace of one row block: the weighted block WB (when the cross product
// is accumulated), a k x p scratch matrix T, and two row vectors v1 and v2
// whose products with the block are added to the vector accumulators.
struct RowBlockWork {
    Eigen::MatrixXd WB;
    Eigen::MatrixXd T;
    Eigen::VectorXd v1, v2;
    Eigen::Index r = 0;   // first row of the block
    Eigen::Index k = 0;   // rows in the block
};

// Block workspaces kept between calls of the kernels below, so that an
// iterative fit does not allocate (and page in) fresh row-block buffers at
// every pass (the lane accumulators alone take up to 64 MiB at p = 1024). A
// pool is owned by the caller (for example the IRLS solver)
// and freed with it. It holds one workspace per running thread in the lane
// scheme and one per group slot in the tile scheme. A workspace never
// carries results from one block to the next, so reusing it does not change
// any result.
struct RowBlockPool {
    std::vector<RowBlockWork> slots;
    std::vector<LaneAcc> lanes;   // lane accumulators, zeroed in place at each call
};

// Sizes a workspace for blocks of brows rows and p columns. Buffers are
// reallocated only when their dimensions change.
inline void size_row_block_work(RowBlockWork& wk, Eigen::Index brows, Eigen::Index p,
                                bool info, Eigen::Index work_cols)
{
    if (info && (wk.WB.rows() != brows || wk.WB.cols() != p)) wk.WB.resize(brows, p);
    if (work_cols > 0 && (wk.T.rows() != brows || wk.T.cols() != work_cols))
        wk.T.resize(brows, work_cols);
    if (wk.v1.size() != brows) wk.v1.resize(brows);
    if (wk.v2.size() != brows) wk.v2.resize(brows);
}

// Raised on the calling thread when a block reduction fails. The block
// callbacks below never report a data error, so a failure is an exception
// inside a parallel region, in practice an allocation failure.
inline void stop_block_reduction_failed()
{
    Rcpp::stop("not enough memory to accumulate the weighted cross product in row blocks");
}

// The common engine of the row-block kernels. For the blocks of rows
// [r, r + k) of an n-row design with p columns:
//   prepare(wk) fills, for the block wk.r, wk.k, the weighted block
//     wk.WB.topRows(k) when info is true, and wk.v1, wk.v2 as vec_add needs;
//   vec_add(wk, s1, s2) adds the block's vector contributions to s1 and s2.
// Adds the lower triangle of sum_b WB_b' WB_b to xtwx (when info is true) and
// the vector contributions to s1 and s2 (each sized p, or empty when unused).
// work_cols is the column count of wk.T (0 when unused). pool, when not
// null, supplies workspaces kept between calls. Neither callback may call
// the R API.
template <class Prepare, class VecAdd>
inline void weighted_block_reduce(Eigen::Index n, Eigen::Index p, int chunk_rows, bool info,
                                  Eigen::Index work_cols, RowBlockPool* pool,
                                  Eigen::MatrixXd& xtwx, Eigen::VectorXd& s1, Eigen::VectorXd& s2,
                                  Prepare prepare, VecAdd vec_add)
{
    if (n <= 0) return;
    const bool lanes = !info || fglm_par::use_lanes(p);
    const Eigen::Index R = reduction_block_rows(p, lanes, chunk_rows);
    const Eigen::Index nblocks = (n + R - 1) / R;
    const Eigen::Index brows = std::min<Eigen::Index>(R, n);
    RowBlockPool local_pool;
    RowBlockPool& pl = pool ? *pool : local_pool;
    // Lanes: one workspace per running thread (thread t uses slot t for all
    // of its lanes). Tiles: one per group slot.
    const Eigen::Index nlanes = std::min<Eigen::Index>(nblocks, fglm_par::kLanes);
    const Eigen::Index nslots = lanes
        ? std::min<Eigen::Index>(nlanes, std::max(1, fglm_par::thread_setting()))
        : std::min<Eigen::Index>(fglm_par::kGroupBlocks, nblocks);
    if (static_cast<Eigen::Index>(pl.slots.size()) < nslots) pl.slots.resize(nslots);
    for (Eigen::Index j = 0; j < nslots; ++j)
        size_row_block_work(pl.slots[j], brows, p, info, work_cols);

    // A lane's workspace: its thread's pool slot, or (should a thread index
    // exceed the pool) a workspace of its own.
    struct WorkHandle {
        RowBlockWork* w = nullptr;
        std::unique_ptr<RowBlockWork> own;
    };
    auto make_work = [&]() {
        WorkHandle h;
        const Eigen::Index id = fglm_par::current_thread();
        if (id < nslots) {
            h.w = &pl.slots[id];
        } else {
            h.own.reset(new RowBlockWork());
            size_row_block_work(*h.own, brows, p, info, work_cols);
            h.w = h.own.get();
        }
        return h;
    };
    auto fill = [&](Eigen::Index b, RowBlockWork& wk) {
        wk.r = b * R;
        wk.k = std::min<Eigen::Index>(R, n - wk.r);
        return prepare(wk);
    };

    bool ok;
    if (lanes) {
        typedef LaneAcc Acc;
        // The outputs are swapped into out (no copy), so the lane sums are
        // added to their current contents in lane order. The guard swaps
        // them back, also when an interrupt unwinds the reduction.
        struct SwapGuard {
            Acc out;
            Eigen::MatrixXd& x; Eigen::VectorXd& v1; Eigen::VectorXd& v2;
            SwapGuard(Eigen::MatrixXd& x_, Eigen::VectorXd& v1_, Eigen::VectorXd& v2_)
                : x(x_), v1(v1_), v2(v2_) { out.A.swap(x); out.s1.swap(v1); out.s2.swap(v2); }
            ~SwapGuard() { out.A.swap(x); out.s1.swap(v1); out.s2.swap(v2); }
        } guard(xtwx, s1, s2);
        Acc& out = guard.out;
        const Eigen::Index pa = info ? p : 0, n1 = out.s1.size(), n2 = out.s2.size();
        ok = fglm_par::lane_reduce_into(
            nblocks, out, pl.lanes,
            [&](Acc& a) {
                if (pa > 0) a.A.setZero(pa, pa);  // keep the p x p lanes across vector-only calls
                a.s1.setZero(n1);
                a.s2.setZero(n2);
            },
            make_work,
            [&](Eigen::Index b, Acc& a, WorkHandle& h) {
                RowBlockWork& wk = *h.w;
                if (!fill(b, wk)) return false;
                if (info)
                    a.A.template selfadjointView<Eigen::Lower>().rankUpdate(wk.WB.topRows(wk.k).adjoint());
                vec_add(static_cast<const RowBlockWork&>(wk), a.s1, a.s2);
                return true;
            },
            [&](Acc& o, const Acc& a) {
                if (info) o.A.template triangularView<Eigen::Lower>() += a.A;
                if (o.s1.size()) o.s1 += a.s1;
                if (o.s2.size()) o.s2 += a.s2;
            });
    } else {
        const Eigen::Index ntiles = fglm_par::tile_count(p);
        ok = fglm_par::tile_reduce(
            nblocks, ntiles, pl.slots, fill,
            // Tile t covers columns c0 to c0 + tw - 1 of the lower triangle:
            // the tw x tw diagonal block by a rank update (lower triangle
            // only) and the rows below it by a product.
            [&](const RowBlockWork& wk, Eigen::Index t) {
                const Eigen::Index c0 = t * fglm_par::kTileWidth;
                const Eigen::Index tw = std::min<Eigen::Index>(fglm_par::kTileWidth, p - c0);
                const Eigen::Index below = p - c0 - tw;
                const auto WBk = wk.WB.topRows(wk.k);
                xtwx.block(c0, c0, tw, tw).selfadjointView<Eigen::Lower>()
                    .rankUpdate(WBk.middleCols(c0, tw).adjoint());
                if (below > 0)
                    xtwx.block(c0 + tw, c0, below, tw).noalias() +=
                        WBk.rightCols(below).adjoint() * WBk.middleCols(c0, tw);
            },
            [&](const RowBlockWork& wk) { vec_add(wk, s1, s2); });
    }
    if (!ok) stop_block_reduction_failed();
}

// out = X' diag(w^2) X (lower triangle). out is resized to p x p.
inline void accumulate_xtwx_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,
    Eigen::MatrixXd& out,
    int chunk_rows = -1,
    RowBlockPool* pool = nullptr)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    out.setZero(p, p);
    Eigen::VectorXd none1, none2;
    weighted_block_reduce(
        n, p, chunk_rows, true, 0, pool, out, none1, none2,
        [&](RowBlockWork& wk) {
            wk.WB.topRows(wk.k).noalias() =
                w.segment(wk.r, wk.k).asDiagonal() * X.middleRows(wk.r, wk.k);
            return true;
        },
        [](const RowBlockWork&, Eigen::VectorXd&, Eigen::VectorXd&) {});
}

// One pass over the rows of X: xtwx += X' diag(w^2) X (lower triangle) and
// xtwz += X' diag(w^2) z, both from the same weighted row block while it is in
// cache; the n x p weighted design is never formed. Adds to xtwx and xtwz
// instead of overwriting them, for accumulation across data blocks. WB and
// wzb are kept for source compatibility and are not used: the workspaces
// come from pool (kept between calls) or are made per call.
inline void add_xtwx_xtwz_blocked(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,
    const Eigen::Ref<const Eigen::VectorXd>& z,
    Eigen::MatrixXd& xtwx,
    Eigen::VectorXd& xtwz,
    Eigen::MatrixXd& /*WB*/,
    Eigen::VectorXd& /*wzb*/,
    int chunk_rows = -1,
    RowBlockPool* pool = nullptr)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    Eigen::VectorXd none;
    weighted_block_reduce(
        n, p, chunk_rows, true, 0, pool, xtwx, xtwz, none,
        [&](RowBlockWork& wk) {
            wk.WB.topRows(wk.k).noalias() =
                w.segment(wk.r, wk.k).asDiagonal() * X.middleRows(wk.r, wk.k);
            wk.v1.head(wk.k) = w.segment(wk.r, wk.k).cwiseProduct(z.segment(wk.r, wk.k));
            return true;
        },
        [](const RowBlockWork& wk, Eigen::VectorXd& s1, Eigen::VectorXd&) {
            s1.noalias() += wk.WB.topRows(wk.k).adjoint() * wk.v1.head(wk.k);
        });
}

// As add_xtwx_xtwz_blocked, but overwrites xtwx (lower triangle) and xtwz.
inline void accumulate_xtwx_xtwz_fused(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,
    const Eigen::Ref<const Eigen::VectorXd>& z,
    Eigen::MatrixXd& xtwx,
    Eigen::VectorXd& xtwz,
    Eigen::MatrixXd& WB,
    Eigen::VectorXd& wzb,
    int chunk_rows = -1,
    RowBlockPool* pool = nullptr)
{
    const Eigen::Index p = X.cols();
    xtwx.setZero(p, p);
    xtwz.setZero(p);
    add_xtwx_xtwz_blocked(X, w, z, xtwx, xtwz, WB, wzb, chunk_rows, pool);
}

// out = WX' WX (lower triangle) for an already weighted design WX. out is
// resized to p x p.
inline void accumulate_crossprod_blocked(
    const Eigen::Ref<const Eigen::MatrixXd>& WX,
    Eigen::MatrixXd& out,
    int chunk_rows = -1,
    RowBlockPool* pool = nullptr)
{
    const Eigen::Index n = WX.rows();
    const Eigen::Index p = WX.cols();
    out.setZero(p, p);
    Eigen::VectorXd none1, none2;
    weighted_block_reduce(
        n, p, chunk_rows, true, 0, pool, out, none1, none2,
        [&](RowBlockWork& wk) {
            wk.WB.topRows(wk.k) = WX.middleRows(wk.r, wk.k);
            return true;
        },
        [](const RowBlockWork&, Eigen::VectorXd&, Eigen::VectorXd&) {});
}

// out = X' (w^2 .* z). out is resized to p.
inline void accumulate_xtwz_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,
    const Eigen::Ref<const Eigen::VectorXd>& z,
    Eigen::VectorXd& out,
    int chunk_rows = -1,
    RowBlockPool* pool = nullptr)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    out.setZero(p);
    Eigen::MatrixXd none;
    Eigen::VectorXd none2;
    weighted_block_reduce(
        n, p, chunk_rows, false, 0, pool, none, out, none2,
        [&](RowBlockWork& wk) {
            // For consistency with the dense path, which uses (W X)' (W z),
            // form w .* (w .* z) so that out = X' diag(w^2) z.
            wk.v1.head(wk.k).array() =
                w.segment(wk.r, wk.k).array().square() * z.segment(wk.r, wk.k).array();
            return true;
        },
        [&](const RowBlockWork& wk, Eigen::VectorXd& s1, Eigen::VectorXd&) {
            s1.noalias() += X.middleRows(wk.r, wk.k).adjoint() * wk.v1.head(wk.k);
        });
}

// out = X' v for a vector v of length n, summed over row blocks. out is
// resized to p.
inline void accumulate_xtv_blocked(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& v,
    Eigen::VectorXd& out,
    int chunk_rows = -1,
    RowBlockPool* pool = nullptr)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    out.setZero(p);
    Eigen::MatrixXd none;
    Eigen::VectorXd none2;
    weighted_block_reduce(
        n, p, chunk_rows, false, 0, pool, none, out, none2,
        [](RowBlockWork&) { return true; },
        [&](const RowBlockWork& wk, Eigen::VectorXd& s1, Eigen::VectorXd&) {
            s1.noalias() += X.middleRows(wk.r, wk.k).adjoint() * v.segment(wk.r, wk.k);
        });
}

// eta = X * beta + offset, computed in fixed row blocks (in parallel when
// there are several), with no n x p materialization. 'out' must be sized n.
inline void apply_X_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& beta,
    const Eigen::Ref<const Eigen::VectorXd>& offset,
    Eigen::Ref<Eigen::VectorXd> out,
    int chunk_rows = -1)
{
    if (chunk_rows <= 0) chunk_rows = default_chunk_rows();
    fglm_par::parallel_blocks(X.rows(), chunk_rows, [&](Eigen::Index r, Eigen::Index k) {
        out.segment(r, k).noalias() = X.middleRows(r, k) * beta;
        out.segment(r, k) += offset.segment(r, k);
    });
}

// Fused Firth pass: accumulate X'WX and X'Wz* in a single sweep over the
// rows. z*_i = z_i + xi_i where xi_i is the AS_mean bias-reducing adjustment
// computed with lagged leverages from A_prev = (X'WX)^{-1} of the previous
// IRLS iteration.
//
// Xtwzstar_out is zeroed internally; XtWX_out is retained when its weights
// are constant. For Gamma's log link, constant_adjustment can retain X'h/2
// in p entries, avoiding an observation-sized leverage cache.
inline void accumulate_firth_streamed(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& w,           // sqrt working weights
    const Eigen::Ref<const Eigen::VectorXd>& z,           // working response
    const Eigen::Ref<const Eigen::VectorXd>& d2mu,        // d2mu/deta2
    const Eigen::Ref<const Eigen::VectorXd>& var_mu_vec,  // V(mu)
    const Eigen::Ref<const Eigen::VectorXd>& mu_eta_vec,  // dmu/deta
    const Eigen::Ref<const Eigen::VectorXd>& pw,          // prior weights
    double phi,
    const Eigen::MatrixXd& A_prev,                        // p x p (X'WX)^{-1} from prev iter
    Eigen::MatrixXd& XtWX_out,
    Eigen::VectorXd& Xtwzstar_out,
    int chunk_rows = -1,
    bool accumulate_information = true,
    Eigen::VectorXd* constant_adjustment = nullptr,
    const Eigen::VectorXd* ordinary_score = nullptr,
    const Eigen::VectorXd* adjustment_ratio = nullptr,
    RowBlockPool* pool = nullptr)
{
    (void) var_mu_vec;
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    if (accumulate_information) XtWX_out.setZero(p, p);
    Xtwzstar_out.setZero(p);
    const bool build_adjustment = constant_adjustment != nullptr &&
        constant_adjustment->size() == 0;
    if (build_adjustment) constant_adjustment->setZero(p);
    const bool need_leverages = constant_adjustment == nullptr || build_adjustment;

    Eigen::VectorXd adjustment_sum;
    if (build_adjustment) adjustment_sum.setZero(p);
    Eigen::MatrixXd unused;
    weighted_block_reduce(
        n, p, chunk_rows, accumulate_information, need_leverages ? p : 0, pool,
        accumulate_information ? XtWX_out : unused, Xtwzstar_out, adjustment_sum,
        [&](RowBlockWork& wk) {
            const Eigen::Index r = wk.r, k = wk.k;
            auto X_k = X.middleRows(r, k);
            if (accumulate_information)
                wk.WB.topRows(k).noalias() = w.segment(r, k).asDiagonal() * X_k;
            wk.v1.head(k).array() = w.segment(r, k).array().square() * z.segment(r, k).array();
            if (ordinary_score) wk.v1.head(k) += ordinary_score->segment(r, k);
            if (need_leverages) {
                wk.T.topRows(k).noalias() = X_k * A_prev;
                const Eigen::ArrayXd h_k =
                    (wk.T.topRows(k).array() * X_k.array()).rowwise().sum()
                    * w.segment(r, k).array().square();
                if (adjustment_ratio) {
                    wk.v1.head(k).array() += .5 * h_k * adjustment_ratio->segment(r, k).array();
                } else if (build_adjustment) {
                    wk.v2.head(k) = (0.5 * h_k).matrix();
                } else {
                    Eigen::ArrayXd xi_k = Eigen::ArrayXd::Zero(k);
                    for (Eigen::Index i = 0; i < k; ++i) {
                        const Eigen::Index j = r + i;
                        if (pw[j] > 0 && w[j] != 0 && mu_eta_vec[j] != 0)
                            xi_k[i] = phi * h_k[i] * (d2mu[j] / mu_eta_vec[j]) / (2 * w[j] * w[j]);
                    }
                    wk.v1.head(k).array() = w.segment(r, k).array().square()
                        * (z.segment(r, k).array() + xi_k);
                }
            }
            return true;
        },
        [&](const RowBlockWork& wk, Eigen::VectorXd& s1, Eigen::VectorXd& s2) {
            auto X_k = X.middleRows(wk.r, wk.k);
            s1.noalias() += X_k.adjoint() * wk.v1.head(wk.k);
            if (build_adjustment) s2.noalias() += X_k.adjoint() * wk.v2.head(wk.k);
        });
    if (build_adjustment) *constant_adjustment = adjustment_sum;
    if (constant_adjustment != nullptr)
        Xtwzstar_out.noalias() += phi * (*constant_adjustment);
}

}  // namespace fglm

#endif  // FASTGLM_CHUNK_SOURCE_H
