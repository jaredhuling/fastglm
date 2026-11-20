#ifndef FASTGLM_SPARSE_PRODUCTS_H
#define FASTGLM_SPARSE_PRODUCTS_H

#include <RcppEigen.h>
#include <Eigen/SparseCore>
#include <algorithm>
#include <limits>
#include <vector>
#include "parallel_reduce.h"
#include "chunk_source.h"

namespace fglm {

// Threaded products with a column-compressed sparse design X (n x p), from
// a row-compressed copy of X (12 bytes per nonzero, kept for the fit in
// place of the transposed copy that Eigen's sparse product made at every
// call) and the structural pattern of X'X, which the product formed as well.
//
// The cross product X' diag(v) X is formed in one of two schemes, chosen by
// p alone:
//   - p <= kRowSweepMaxDim (1024): a sweep over fixed row blocks, reduced
//     in 8 dense p x p lanes by fglm_par::lane_reduce_into and copied into
//     the pattern. Extra memory, kept for the fit: the lanes (8 p^2 doubles,
//     64 MiB at p = 1024; only min(8, row blocks) lanes are allocated) and
//     their sum dense_ (p^2 doubles, 8 MiB), so at most 72 MiB. While the
//     pattern is built, before any lane exists, one p x p byte mark matrix
//     per thread is used and freed (at most 8 MiB at 8 threads). The bound
//     is therefore about 72 MiB at any time, about 80 MiB counting both;
//   - larger p: column by column, column j computed by one thread, which
//     visits the nonzeros x_ij of column j in increasing row order and, for
//     each, the nonzeros x_ik (k >= j) of row i, adding (x_ik v_i) x_ij to
//     entry k of a p-vector. This is the order and the arithmetic of Eigen's
//     SpMat(X') * v.asDiagonal() * X; it needs one p-vector per thread and no
//     dense p x p matrix, so wide and very sparse designs keep their memory.
// In both, the order of every addition is fixed by n, p, and the block size,
// so the result is identical for every thread count. The upper triangle is
// the transpose of the lower one.
//
// X beta is computed by rows and X' v by columns; each output entry is
// computed by one thread in a fixed order. Parallel regions are entered only
// when the n rows form at least 2 row blocks (fglm_par::threads_for), and the
// loop bodies do not call the R API; failures inside them (allocation of a
// lane or a per-thread vector) are raised as an R error after the region.
class SparseDesignProducts {
public:
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef SpMat::StorageIndex StorageIndex;

    bool ready() const { return ready_; }

    // True when out has exactly the pattern of X'X (dimensions, column
    // pointers, and row indices), so its values can be written in place.
    // Any other matrix is replaced by a copy of the pattern first.
    bool has_pattern(const SpMat& out) const
    {
        if (out.rows() != p_ || out.cols() != p_ || !out.isCompressed() ||
            out.nonZeros() != pattern_.nonZeros())
            return false;
        const StorageIndex* o = out.outerIndexPtr();
        const StorageIndex* i = out.innerIndexPtr();
        return std::equal(o, o + p_ + 1, pattern_.outerIndexPtr()) &&
               std::equal(i, i + pattern_.nonZeros(), pattern_.innerIndexPtr());
    }

    // Builds the row-compressed copy of X and the pattern of X'X.
    template <class MapSp>
    void init(const MapSp& X)
    {
        if (!X.isCompressed())
            Rcpp::stop("the sparse design must be column compressed");
        n_ = X.rows();
        p_ = X.cols();
        const StorageIndex* cptr = X.outerIndexPtr();
        const StorageIndex* rix = X.innerIndexPtr();
        const double* xv = X.valuePtr();
        const Eigen::Index nnz = cptr[p_] - cptr[0];

        // Row-compressed copy by a counting sort over the columns in
        // increasing order, so each row's entries are sorted by column.
        rptr_.assign(n_ + 1, 0);
        for (Eigen::Index e = cptr[0]; e < cptr[p_]; ++e) ++rptr_[rix[e] + 1];
        for (Eigen::Index i = 0; i < n_; ++i) rptr_[i + 1] += rptr_[i];
        rcol_.resize(nnz);
        rval_.resize(nnz);
        {
            std::vector<Eigen::Index> cur(rptr_.begin(), rptr_.end() - 1);
            for (Eigen::Index j = 0; j < p_; ++j)
                for (Eigen::Index e = cptr[j]; e < cptr[j + 1]; ++e) {
                    const Eigen::Index q = cur[rix[e]]++;
                    rcol_[q] = static_cast<StorageIndex>(j);
                    rval_[q] = xv[e];
                }
        }

        const int nt = std::max(1, fglm_par::threads_for(row_blocks()));
        scratch_.assign(nt, Eigen::VectorXd());

        // Lower pattern of X'X, column by column: the rows k >= j reached
        // from the nonzeros of column j.
        std::vector<Eigen::Index> lcount(p_ + 1, 0);
        std::vector<StorageIndex> lrow;
        // Columns with a nonzero have their diagonal in the pattern; the
        // full pattern has 2 L - nonempty entries for L lower entries.
        nonempty_ = 0;
        for (Eigen::Index j = 0; j < p_; ++j) if (cptr[j + 1] > cptr[j]) ++nonempty_;
        if (p_ <= row_sweep_max_) lower_pattern_by_rows(lcount, lrow);
        else lower_pattern_by_columns(cptr, rix, nt, lcount, lrow);

        // Full symmetric pattern: column k holds its upper entries (rows
        // j < k with (k, j) in the lower pattern), then its lower entries.
        std::vector<Eigen::Index> upcount(p_, 0);
        Eigen::Index offdiag = 0;
        for (Eigen::Index j = 0; j < p_; ++j)
            for (Eigen::Index c = lcount[j]; c < lcount[j + 1]; ++c)
                if (lrow[c] != j) { ++upcount[lrow[c]]; ++offdiag; }
        const Eigen::Index total = lcount[p_] + offdiag;
        pattern_.resize(p_, p_);
        pattern_.resizeNonZeros(total);
        StorageIndex* outer = pattern_.outerIndexPtr();
        StorageIndex* inner = pattern_.innerIndexPtr();
        outer[0] = 0;
        diag_.resize(p_);
        for (Eigen::Index j = 0; j < p_; ++j) {
            diag_[j] = outer[j] + upcount[j];
            outer[j + 1] = static_cast<StorageIndex>(diag_[j] + (lcount[j + 1] - lcount[j]));
        }
        mirror_.assign(total, -1);
        std::vector<Eigen::Index> upcur(p_);
        for (Eigen::Index k = 0; k < p_; ++k) upcur[k] = outer[k];
        for (Eigen::Index j = 0; j < p_; ++j)
            for (Eigen::Index c = lcount[j]; c < lcount[j + 1]; ++c) {
                const Eigen::Index pos = diag_[j] + (c - lcount[j]);
                const StorageIndex k = lrow[c];
                inner[pos] = k;
                if (k != j) {
                    const Eigen::Index up = upcur[k]++;
                    inner[up] = static_cast<StorageIndex>(j);
                    mirror_[pos] = static_cast<StorageIndex>(up);
                }
            }
        std::fill(pattern_.valuePtr(), pattern_.valuePtr() + total, 0.0);
        ready_ = true;
    }

    // out = X' diag(v) X with the full symmetric pattern of X'X.
    template <class MapSp>
    void cross_product(const MapSp& X, const Eigen::Ref<const Eigen::VectorXd>& v, SpMat& out)
    {
        if (!has_pattern(out)) out = pattern_;
        if (p_ <= row_sweep_max_) { cross_product_by_rows(v, out); return; }
        const StorageIndex* cptr = X.outerIndexPtr();
        const StorageIndex* rix = X.innerIndexPtr();
        const double* xv = X.valuePtr();
        const StorageIndex* outer = out.outerIndexPtr();
        const StorageIndex* inner = out.innerIndexPtr();
        double* val = out.valuePtr();
        run_columns([&](Eigen::Index j, int t) {
            double* acc = scratch_[t].data();
            for (Eigen::Index e = cptr[j]; e < cptr[j + 1]; ++e) {
                const Eigen::Index i = rix[e];
                const double xij = xv[e], vi = v[i];
                for (Eigen::Index q = row_start(i, j); q < rptr_[i + 1]; ++q)
                    acc[rcol_[q]] += (rval_[q] * vi) * xij;
            }
            for (Eigen::Index pos = diag_[j]; pos < outer[j + 1]; ++pos) {
                const StorageIndex k = inner[pos];
                val[pos] = acc[k];
                acc[k] = 0.0;
                if (mirror_[pos] >= 0) val[mirror_[pos]] = val[pos];
            }
        }, [&](int t) {
            if (scratch_[t].size() != p_) scratch_[t].setZero(p_);
        });
    }

    // The largest p for which the cross product is accumulated in a dense
    // p x p matrix by a sweep over the rows (see the memory bound above). The
    // schemes agree to rounding; tests lower the limit to compare them.
    static constexpr Eigen::Index kRowSweepMaxDim = 1024;
    void set_row_sweep_max(Eigen::Index m) { row_sweep_max_ = m; }

    // out = X' diag(v) X for p <= kRowSweepMaxDim, by the lane reduction of
    // parallel_reduce.h over fixed row blocks of default_chunk_rows() rows:
    // block b's rows are swept in storage order and, for each row i and
    // each pair of its nonzeros x_ij, x_ik with k >= j, (x_ik v_i) x_ij is
    // added to entry (k, j) of the dense lower triangle of lane b mod 8; the
    // lanes are then added in lane order. The order of every addition
    // depends on n, p, and the block size only, so the result is identical
    // for every thread count. (It differs from the column scheme used for
    // p > kRowSweepMaxDim, and from 0.1.4, in the last bits only.) The
    // 8 lanes of p^2 doubles (at most 64 MiB at p = 1024) are kept for the
    // fit.
    void cross_product_by_rows(const Eigen::Ref<const Eigen::VectorXd>& v, SpMat& out)
    {
        const Eigen::Index p = p_, n = n_;
        const Eigen::Index R = fglm::default_chunk_rows();
        const Eigen::Index nblocks = (n + R - 1) / R;
        dense_.setZero(p, p);
        const Eigen::Index* rp = rptr_.data();
        const StorageIndex* rc = rcol_.data();
        const double* rv = rval_.data();
        struct NoWork {};
        const bool ok = fglm_par::lane_reduce_into(
            nblocks, dense_, lanes_,
            [p](Eigen::MatrixXd& a) { a.setZero(p, p); },
            []() { return NoWork(); },
            [&](Eigen::Index blk, Eigen::MatrixXd& a, NoWork&) {
                double* A = a.data();
                const Eigen::Index r1 = std::min<Eigen::Index>(n, (blk + 1) * R);
                for (Eigen::Index i = blk * R; i < r1; ++i) {
                    const Eigen::Index end = rp[i + 1];
                    const double vi = v[i];
                    for (Eigen::Index q = rp[i]; q < end; ++q) {
                        const double xij = rv[q];
                        double* col = A + static_cast<Eigen::Index>(rc[q]) * p;
                        for (Eigen::Index b = q; b < end; ++b)
                            col[rc[b]] += (rv[b] * vi) * xij;
                    }
                }
                return true;
            },
            [](Eigen::MatrixXd& o, const Eigen::MatrixXd& a) {
                o.triangularView<Eigen::Lower>() += a;
            });
        if (!ok) Rcpp::stop("not enough memory to form the sparse cross product");
        const StorageIndex* outer = out.outerIndexPtr();
        const StorageIndex* inner = out.innerIndexPtr();
        double* val = out.valuePtr();
        for (Eigen::Index j = 0; j < p; ++j) {
            const double* col = dense_.data() + j * p;
            for (Eigen::Index pos = diag_[j]; pos < outer[j + 1]; ++pos) {
                val[pos] = col[inner[pos]];
                if (mirror_[pos] >= 0) val[mirror_[pos]] = val[pos];
            }
        }
    }

    // out = X b + off, by rows.
    void times(const Eigen::Ref<const Eigen::VectorXd>& b,
               const Eigen::Ref<const Eigen::VectorXd>& off, Eigen::VectorXd& out) const
    {
        out.resize(n_);
        fglm_par::parallel_blocks(n_, fglm::default_chunk_rows(), [&](Eigen::Index r, Eigen::Index k) {
            for (Eigen::Index i = r; i < r + k; ++i) {
                double s = 0.0;
                for (Eigen::Index q = rptr_[i]; q < rptr_[i + 1]; ++q) s += rval_[q] * b[rcol_[q]];
                out[i] = s + off[i];
            }
        });
    }

    // out = X' v, by columns.
    template <class MapSp>
    void adjoint_times(const MapSp& X, const Eigen::Ref<const Eigen::VectorXd>& v,
                       Eigen::VectorXd& out)
    {
        out.resize(p_);
        const StorageIndex* cptr = X.outerIndexPtr();
        const StorageIndex* rix = X.innerIndexPtr();
        const double* xv = X.valuePtr();
        run_columns([&](Eigen::Index j, int) {
            double s = 0.0;
            for (Eigen::Index e = cptr[j]; e < cptr[j + 1]; ++e) s += xv[e] * v[rix[e]];
            out[j] = s;
        }, [](int) {});
    }

private:
    bool ready_ = false;
    Eigen::Index row_sweep_max_ = kRowSweepMaxDim;
    Eigen::Index n_ = 0, p_ = 0;
    std::vector<Eigen::Index> rptr_;
    std::vector<StorageIndex> rcol_;
    std::vector<double> rval_;
    SpMat pattern_;
    std::vector<Eigen::Index> diag_;        // position of the first lower entry of each column
    std::vector<StorageIndex> mirror_;      // position of the transposed entry, or -1
    std::vector<Eigen::VectorXd> scratch_;  // one zeroed p-vector per thread
    Eigen::MatrixXd dense_;                 // p x p sum of the lanes of the row sweep
    std::vector<Eigen::MatrixXd> lanes_;    // lane accumulators of the row sweep

    // The lower pattern by columns: for each column j, the rows k >= j
    // reached from its nonzeros through the row-compressed copy; counted,
    // then filled and sorted.
    void lower_pattern_by_columns(const StorageIndex* cptr, const StorageIndex* rix, int nt,
                                  std::vector<Eigen::Index>& lcount, std::vector<StorageIndex>& lrow)
    {
        std::vector<std::vector<StorageIndex>> mark(nt);
        auto make_mark = [&](int t) { mark[t].assign(p_, -1); };
        run_columns([&](Eigen::Index j, int t) {
            std::vector<StorageIndex>& m = mark[t];
            Eigen::Index c = 0;
            for (Eigen::Index e = cptr[j]; e < cptr[j + 1]; ++e) {
                const Eigen::Index i = rix[e];
                for (Eigen::Index q = row_start(i, j); q < rptr_[i + 1]; ++q) {
                    StorageIndex& mk = m[rcol_[q]];
                    if (mk != j) { mk = static_cast<StorageIndex>(j); ++c; }
                }
            }
            lcount[j + 1] = c;
        }, make_mark);
        for (Eigen::Index j = 0; j < p_; ++j) lcount[j + 1] += lcount[j];
        check_pattern_size(lcount[p_]);
        lrow.assign(lcount[p_], 0);
        run_columns([&](Eigen::Index j, int t) {
            std::vector<StorageIndex>& m = mark[t];
            Eigen::Index c = lcount[j];
            for (Eigen::Index e = cptr[j]; e < cptr[j + 1]; ++e) {
                const Eigen::Index i = rix[e];
                for (Eigen::Index q = row_start(i, j); q < rptr_[i + 1]; ++q) {
                    StorageIndex& mk = m[rcol_[q]];
                    if (mk != j) { mk = static_cast<StorageIndex>(j); lrow[c++] = rcol_[q]; }
                }
            }
            std::sort(lrow.begin() + lcount[j], lrow.begin() + lcount[j + 1]);
        }, make_mark);
        mark.clear();
    }

    // The lower pattern by a sweep over the rows, for p <= the row-sweep
    // limit: each thread marks the pairs (k, j), k >= j, of the nonzeros of
    // the rows of its fixed row blocks in a p x p mark matrix of its own;
    // the marks are then combined (a logical or, so the order does not
    // matter) and read column by column, which gives each column's rows in
    // increasing order.
    void lower_pattern_by_rows(std::vector<Eigen::Index>& lcount, std::vector<StorageIndex>& lrow)
    {
        const Eigen::Index p = p_, n = n_;
        const Eigen::Index R = fglm::default_chunk_rows();
        const Eigen::Index nblocks = (n + R - 1) / R;
        int nt = std::max(1, fglm_par::threads_for(nblocks));
        if (!fglm_par::has_openmp()) nt = 1;
        const std::size_t pp = static_cast<std::size_t>(p) * p;
        std::vector<char> mk(pp * nt, 0);
        const Eigen::Index* rp = rptr_.data();
        const StorageIndex* rc = rcol_.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
#endif
        for (Eigen::Index blk = 0; blk < nblocks; ++blk) {
            if (blk == 0) fglm_par::note_team_size();
            char* M = mk.data() + pp * fglm_par::current_thread();
            const Eigen::Index r1 = std::min<Eigen::Index>(n, (blk + 1) * R);
            for (Eigen::Index i = blk * R; i < r1; ++i) {
                const Eigen::Index end = rp[i + 1];
                for (Eigen::Index q = rp[i]; q < end; ++q) {
                    char* col = M + static_cast<Eigen::Index>(rc[q]) * p;
                    for (Eigen::Index b = q; b < end; ++b) col[rc[b]] = 1;
                }
            }
        }
        char* M = mk.data();
        for (int t = 1; t < nt; ++t) {
            const char* Mt = mk.data() + pp * t;
            for (std::size_t e = 0; e < pp; ++e) M[e] |= Mt[e];
        }
        for (Eigen::Index j = 0; j < p; ++j) {
            Eigen::Index c = 0;
            for (Eigen::Index k = j; k < p; ++k) c += M[j * p + k];
            lcount[j + 1] = lcount[j] + c;
        }
        check_pattern_size(lcount[p]);
        lrow.assign(lcount[p], 0);
        for (Eigen::Index j = 0; j < p; ++j) {
            Eigen::Index c = lcount[j];
            for (Eigen::Index k = j; k < p; ++k)
                if (M[j * p + k]) lrow[c++] = static_cast<StorageIndex>(k);
        }
    }

    Eigen::Index nonempty_ = 0;  // columns of X with at least one stored entry

    // Raised before the pattern is allocated: the full pattern of X'X, with
    // lower entries L, has 2 L - nonempty_ entries, which must fit the
    // sparse matrix's int indices (as in 0.1.4, whose sparse product had the
    // same limit).
    void check_pattern_size(Eigen::Index L) const
    {
        const Eigen::Index total = 2 * L - nonempty_;
        if (total > static_cast<Eigen::Index>(std::numeric_limits<StorageIndex>::max()))
            Rcpp::stop("the pattern of X'X has %.0f entries, too many for a sparse Cholesky "
                       "factorization (at most %d)", static_cast<double>(total),
                       std::numeric_limits<StorageIndex>::max());
    }

    Eigen::Index row_blocks() const
    {
        const Eigen::Index R = fglm::default_chunk_rows();
        return (n_ + R - 1) / R;
    }

    // The first entry of row i with column >= j.
    Eigen::Index row_start(Eigen::Index i, Eigen::Index j) const
    {
        const StorageIndex* b = rcol_.data() + rptr_[i];
        const StorageIndex* e = rcol_.data() + rptr_[i + 1];
        return rptr_[i] + (std::lower_bound(b, e, static_cast<StorageIndex>(j)) - b);
    }

    // f(j, t) for every column j, with t the thread index; setup(t) runs once
    // on each thread before its columns (per-thread allocation is done there,
    // so a failure is caught inside the region). Columns are dealt round
    // robin (a static schedule: R's libomp lacks the entry points of a
    // dynamic schedule in newer compilers); each column's result is computed
    // by one thread in a fixed order, so the assignment does not affect the
    // result. A failed setup is
    // raised as an R error after the region.
    template <class F, class Setup>
    void run_columns(F f, Setup setup)
    {
        int nt = std::max(1, std::min<int>(fglm_par::threads_for(row_blocks()),
                                           static_cast<int>(scratch_.size())));
        if (!fglm_par::has_openmp()) nt = 1;
        if (nt <= 1) {
            setup(0);
            for (Eigen::Index j = 0; j < p_; ++j) f(j, 0);
            return;
        }
        std::vector<char> fail(nt, 0);
#ifdef _OPENMP
#pragma omp parallel num_threads(nt)
#endif
        {
            const int t = fglm_par::current_thread();
            fglm_par::note_team_size();
            try {
                setup(t);
            } catch (...) {
                fail[t] = 1;
            }
            const bool ok = !fail[t];
#ifdef _OPENMP
#pragma omp for schedule(static, 1)
#endif
            for (Eigen::Index j = 0; j < p_; ++j)
                if (ok) f(j, t);
        }
        for (int t = 0; t < nt; ++t)
            if (fail[t]) Rcpp::stop("not enough memory to form the sparse cross product");
    }
};

} // namespace fglm

#endif // FASTGLM_SPARSE_PRODUCTS_H
