// Internal test export for the deterministic reductions in parallel_reduce.h.
// Used only by tests/testthat/test-parallel-reduce.R.

#include <RcppEigen.h>
#include <cmath>
#include <string>
#include "parallel_reduce.h"

namespace {

// Sets the package thread setting for the lifetime of the object and
// restores the previous value on exit, including on error.
struct ThreadSettingGuard {
    int old;
    explicit ThreadSettingGuard(int n) : old(fglm_par::thread_setting())
    {
        fglm_par::thread_setting() = (n < 1 || !fglm_par::has_openmp()) ? 1 : n;
    }
    ~ThreadSettingGuard() { fglm_par::thread_setting() = old; }
};

struct LaneAcc {
    Eigen::MatrixXd xtwx; // lower triangle accumulated
    Eigen::VectorXd xtwz;
};

// Workspace for one block: sqrt(w) * X and sqrt(w) * z. In the lane scheme
// one is made per lane iteration, so at most one per running thread is
// alive; in the tile scheme there is one per group slot.
struct BlockWork {
    Eigen::MatrixXd WB;
    Eigen::VectorXd swz;
    Eigen::Index k = 0;   // rows of WB in use
};

// Fills WB.topRows(k) = sqrt(w) * X and swz.head(k) = sqrt(w) * z for the
// block starting at row r. Returns false on a negative or non-finite weight.
bool fill_block(const Eigen::MatrixXd& X, const Eigen::VectorXd& w,
                Eigen::Index r, Eigen::Index k, BlockWork& wk)
{
    const double n = static_cast<double>(X.rows());
    for (Eigen::Index i = 0; i < k; ++i) {
        const double wi = w[r + i];
        if (!(wi >= 0.0) || !std::isfinite(wi)) return false;
        const double s = std::sqrt(wi);
        wk.swz[i] = s;
    }
    wk.k = k;
    wk.WB.topRows(k).noalias() = wk.swz.head(k).asDiagonal() * X.middleRows(r, k);
    for (Eigen::Index i = 0; i < k; ++i)
        wk.swz[i] *= static_cast<double>(r + i + 1) / n;
    return true;
}

} // namespace

// X' W X (lower triangle, mirrored to the upper triangle on return) and
// X' W z with W = diag(w) and z_i = i / n (i = 1, ..., n), computed in row
// blocks of block_rows through lane_reduce ("lanes") or tile_reduce
// ("tiles") with the given thread setting.
// [[Rcpp::export]]
Rcpp::List fastglm_crossprod_test(const Eigen::MatrixXd& X, const Eigen::VectorXd& w,
                                  int block_rows, int threads, std::string scheme)
{
    const Eigen::Index n = X.rows();
    const Eigen::Index d = X.cols();
    if (w.size() != n) Rcpp::stop("w must have nrow(X) entries");
    if (block_rows < 1) Rcpp::stop("block_rows must be positive");
    if (scheme != "lanes" && scheme != "tiles") Rcpp::stop("scheme must be \"lanes\" or \"tiles\"");

    const Eigen::Index rows = block_rows;
    const Eigen::Index nblocks = (n + rows - 1) / rows;
    const Eigen::Index brows = std::min<Eigen::Index>(rows, std::max<Eigen::Index>(n, 1));

    Eigen::MatrixXd xtwx = Eigen::MatrixXd::Zero(d, d);
    Eigen::VectorXd xtwz = Eigen::VectorXd::Zero(d);
    bool ok;
    Eigen::Index failed = -1;
    {
        ThreadSettingGuard guard(threads);
        if (scheme == "lanes") {
            LaneAcc out;
            out.xtwx = Eigen::MatrixXd::Zero(d, d);
            out.xtwz = Eigen::VectorXd::Zero(d);
            ok = fglm_par::lane_reduce_ws(
                nblocks, out,
                [&]() {
                    LaneAcc a;
                    a.xtwx = Eigen::MatrixXd::Zero(d, d);
                    a.xtwz = Eigen::VectorXd::Zero(d);
                    return a;
                },
                [&]() {
                    BlockWork wk;
                    wk.WB.resize(brows, d);
                    wk.swz.resize(brows);
                    return wk;
                },
                [&](Eigen::Index b, LaneAcc& a, BlockWork& wk) {
                    const Eigen::Index r = b * rows;
                    const Eigen::Index k = std::min<Eigen::Index>(rows, n - r);
                    if (!fill_block(X, w, r, k, wk)) return false;
                    a.xtwx.selfadjointView<Eigen::Lower>().rankUpdate(wk.WB.topRows(k).adjoint());
                    a.xtwz.noalias() += wk.WB.topRows(k).adjoint() * wk.swz.head(k);
                    return true;
                },
                [](LaneAcc& o, const LaneAcc& a) {
                    o.xtwx += a.xtwx;
                    o.xtwz += a.xtwz;
                },
                &failed);
            if (ok) {
                xtwx = out.xtwx;
                xtwz = out.xtwz;
            }
        } else {
            std::vector<BlockWork> work(fglm_par::kGroupBlocks);
            for (auto& wk : work) {
                wk.WB.resize(brows, d);
                wk.swz.resize(brows);
            }
            const Eigen::Index ntiles = fglm_par::tile_count(d);
            ok = fglm_par::tile_reduce(
                nblocks, ntiles, work,
                [&](Eigen::Index b, BlockWork& wk) {
                    const Eigen::Index r = b * rows;
                    return fill_block(X, w, r, std::min<Eigen::Index>(rows, n - r), wk);
                },
                // Tile t covers columns c0 to c0 + tw - 1 of the lower
                // triangle: the tw x tw diagonal block by a rank update (lower
                // triangle only) and the rows below it by a product.
                [&](const BlockWork& wk, Eigen::Index t) {
                    const Eigen::Index c0 = t * fglm_par::kTileWidth;
                    const Eigen::Index tw = std::min<Eigen::Index>(fglm_par::kTileWidth, d - c0);
                    const Eigen::Index below = d - c0 - tw;
                    const auto WBk = wk.WB.topRows(wk.k);
                    xtwx.block(c0, c0, tw, tw).selfadjointView<Eigen::Lower>()
                        .rankUpdate(WBk.middleCols(c0, tw).adjoint());
                    if (below > 0)
                        xtwx.block(c0 + tw, c0, below, tw).noalias() +=
                            WBk.rightCols(below).adjoint() * WBk.middleCols(c0, tw);
                },
                [&](const BlockWork& wk) {
                    xtwz.noalias() += wk.WB.topRows(wk.k).adjoint() * wk.swz.head(wk.k);
                },
                &failed);
        }
    }
    if (!ok)
        Rcpp::stop("negative or non-finite weight in block " +
                   std::to_string(static_cast<long long>(failed) + 1));

    xtwx.triangularView<Eigen::StrictlyUpper>() = xtwx.adjoint();
    return Rcpp::List::create(Rcpp::Named("xtwx") = xtwx, Rcpp::Named("xtwz") = xtwz);
}
