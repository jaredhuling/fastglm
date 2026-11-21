// Internal test export for the deterministic reductions in parallel_reduce.h.
// Used only by tests/testthat/test-parallel-reduce.R and test-threads-glm.R.

#include <RcppEigen.h>
#include <cmath>
#include <string>
#include "parallel_reduce.h"
#include "chunk_source.h"
#include "sparse_products.h"
#include <stdexcept>
#include <new>
#include <vector>

namespace {

// Sets the package thread setting for the lifetime of the object and
// restores the previous value on exit, including on error.
struct ThreadSettingGuard {
    int old;
    long old_process;
    explicit ThreadSettingGuard(int n)
        : old(fglm_par::thread_setting()), old_process(fglm_par::thread_setting_process())
    {
        if (n < 1 || !fglm_par::has_openmp()) n = 1;
#ifdef _OPENMP
        // As for a fit: a team above the OpenMP thread limit cannot be formed.
        const int lim = omp_get_thread_limit();
        if (lim >= 1 && n > lim) n = lim;
#endif
        fglm_par::set_thread_setting(n);
    }
    ~ThreadSettingGuard()
    {
        fglm_par::thread_setting() = old;
        fglm_par::thread_setting_process() = old_process;
    }
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

// The largest OpenMP team size seen by the reductions and parallel_blocks
// since the last reset (1 when nothing ran in parallel). With reset = TRUE
// the record is set back to 1 after it is read.
// [[Rcpp::export]]
int fastglm_max_team_size_test(bool reset)
{
    const int m = fglm_par::max_team_size_seen();
    if (reset) fglm_par::max_team_size_seen() = 1;
    return m;
}

// Runs fglm_par::parallel_blocks over n rows in blocks of block_rows with the
// given thread setting; the blocks listed in fail (1-based) throw inside the
// region, a std::runtime_error for a positive entry b and a std::bad_alloc
// for a negative entry -b. The exception rethrown on the calling thread is
// reported with Rcpp::stop. Returns the number of blocks that ran when none
// failed.
// [[Rcpp::export]]
int fastglm_parallel_blocks_test(int n, int block_rows, Rcpp::IntegerVector fail, int threads)
{
    std::vector<char> failing((n + block_rows - 1) / block_rows + 1, 0);
    for (int f : fail) {
        if (f == NA_INTEGER) continue;
        if (f >= 1 && f < static_cast<int>(failing.size())) failing[f - 1] = 1;
        if (f <= -1 && -f < static_cast<int>(failing.size())) failing[-f - 1] = 2;
    }
    std::vector<int> ran(failing.size(), 0);
    std::string msg;
    {
        ThreadSettingGuard guard(threads);
        try {
            fglm_par::parallel_blocks(n, block_rows, [&](Eigen::Index r, Eigen::Index) {
                const Eigen::Index b = r / block_rows;
                ran[b] = 1;
                if (failing[b] == 2) throw std::bad_alloc();
                if (failing[b])
                    throw std::runtime_error("failure in block " + std::to_string(static_cast<long long>(b) + 1));
            });
        } catch (const std::exception& e) {
            msg = e.what();
        }
    }
    if (!msg.empty()) Rcpp::stop(msg);
    int count = 0;
    for (int r : ran) count += r;
    return count;
}

// X' diag(w^2) X through fglm::accumulate_xtwx_streamed with the given
// thread setting; when fail_block (1-based) is positive, preparing that
// block throws std::bad_alloc inside the region, as an allocation failure
// would.
// [[Rcpp::export]]
Eigen::MatrixXd fastglm_block_reduce_test(const Eigen::MatrixXd& X, const Eigen::VectorXd& w,
                                          int fail_block, int threads)
{
    const Eigen::Index n = X.rows(), p = X.cols();
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(p, p);
    Eigen::VectorXd none1, none2;
    {
        ThreadSettingGuard guard(threads);
        const bool lanes = fglm_par::use_lanes(p);
        const Eigen::Index R = fglm::reduction_block_rows(p, lanes);
        fglm::weighted_block_reduce(
            n, p, -1, true, 0, nullptr, out, none1, none2,
            [&](fglm::RowBlockWork& wk) {
                if (fail_block > 0 && wk.r / R == fail_block - 1) throw std::bad_alloc();
                wk.WB.topRows(wk.k).noalias() =
                    w.segment(wk.r, wk.k).asDiagonal() * X.middleRows(wk.r, wk.k);
                return true;
            },
            [](const fglm::RowBlockWork&, Eigen::VectorXd&, Eigen::VectorXd&) {});
    }
    out.triangularView<Eigen::StrictlyUpper>() = out.adjoint();
    return out;
}

// The products of fglm::SparseDesignProducts with the given thread setting:
// xtvx = X' diag(v) X (with the full pattern of X'X), eta = X b + off, and
// xtv = X' v. With by_columns = TRUE the cross product uses the column
// scheme for every p, for both the pattern and the values (it is otherwise
// used only for p > 1024).
// [[Rcpp::export]]
Rcpp::List fastglm_sparse_products_test(SEXP x, const Eigen::VectorXd& v, const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& off, bool by_columns, int threads)
{
    typedef Eigen::SparseMatrix<double> SpMat;
    const Eigen::Map<SpMat> X(Rcpp::as<Eigen::Map<SpMat> >(x));
    SpMat xtvx;
    Eigen::VectorXd eta, xtv;
    {
        ThreadSettingGuard guard(threads);
        fglm::SparseDesignProducts prod;
        if (by_columns) prod.set_row_sweep_max(0);
        prod.init(X);
        prod.cross_product(X, v, xtvx);
        prod.times(b, off, eta);
        prod.adjoint_times(X, v, xtv);
    }
    return Rcpp::List::create(Rcpp::Named("xtvx") = Rcpp::wrap(xtvx),
                              Rcpp::Named("eta") = eta, Rcpp::Named("xtv") = xtv);
}

// Failure injection for lane_reduce_into ("lanes") and tile_reduce ("tiles")
// over 20 blocks with the given thread setting. Blocks in invalid (1-based)
// report a data failure (the callback returns false); blocks in throws throw
// inside the region, at the place named by where: "add" (lanes: add; tiles:
// prepare), "tile" (tiles: add_tile, for tile 1 of a group holding such a
// block), "reset" (lanes: a lane reset), or "work" (lanes: make_work). kind
// "bad_alloc" throws std::bad_alloc, anything else std::runtime_error
// ("failure in block N"). Returns "invalid block N" for a reported data
// failure and "ok" otherwise; a rethrown exception becomes an R error.
// [[Rcpp::export]]
std::string fastglm_reduce_failure_test(std::string scheme, std::string where,
                                        Rcpp::IntegerVector throws, Rcpp::IntegerVector invalid,
                                        std::string kind, int threads)
{
    const Eigen::Index nblocks = 20;
    std::vector<char> bad(nblocks, 0), inv(nblocks, 0);
    for (int b : throws) if (b >= 1 && b <= nblocks) bad[b - 1] = 1;
    for (int b : invalid) if (b >= 1 && b <= nblocks) inv[b - 1] = 1;
    auto fail = [&](Eigen::Index b) {
        if (kind == "bad_alloc") throw std::bad_alloc();
        throw std::runtime_error("failure in block " + std::to_string(static_cast<long long>(b) + 1));
    };
    bool ok;
    Eigen::Index failed = -1;
    {
        ThreadSettingGuard guard(threads);
        if (scheme == "lanes") {
            double out = 0;
            std::vector<double> lanes;
            struct NoWork {};
            ok = fglm_par::lane_reduce_into(
                nblocks, out, lanes,
                [&](double& a) {
                    if (where == "reset" && bad[0]) fail(0);
                    a = 0;
                },
                [&]() {
                    if (where == "work" && bad[0]) fail(0);
                    return NoWork();
                },
                [&](Eigen::Index b, double& a, NoWork&) {
                    if (where == "add" && bad[b]) fail(b);
                    if (inv[b]) return false;
                    a += static_cast<double>(b);
                    return true;
                },
                [](double& o, const double& a) { o += a; }, &failed);
        } else {
            struct Slot { Eigen::Index b = 0; };
            std::vector<Slot> work(fglm_par::kGroupBlocks);
            std::vector<double> tiles(3, 0.0);
            ok = fglm_par::tile_reduce(
                nblocks, 3, work,
                [&](Eigen::Index b, Slot& s) {
                    s.b = b;
                    if (where == "add" && bad[b]) fail(b);
                    return !inv[b];
                },
                [&](const Slot& s, Eigen::Index t) {
                    if (where == "tile" && bad[s.b] && t == 1) fail(s.b);
                    tiles[t] += static_cast<double>(s.b);
                },
                [](const Slot&) {}, &failed);
        }
    }
    if (!ok) return "invalid block " + std::to_string(static_cast<long long>(failed) + 1);
    return "ok";
}
