#ifndef FASTGLM_VGLM_DATA_H
#define FASTGLM_VGLM_DATA_H
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <vector>
namespace fglm_vglm {

// One block of rows as the engine sees it. design holds one view of the
// block's design per linear predictor (all the same view in 0.1.5; the
// general form leaves room for covariates that differ across linear
// predictors). codes is nullptr for a grouped response, and counts is then
// b by K; for an individual response counts is empty. offset is b by M.
template <class DesignView>
struct VglmChunkT {
    std::vector<DesignView> design;
    const int* codes;
    Eigen::Ref<const Eigen::MatrixXd> counts;
    Eigen::Ref<const Eigen::VectorXd> weights;
    Eigen::Ref<const Eigen::MatrixXd> offset;

    int rows() const { return static_cast<int>(weights.size()); }
};

typedef VglmChunkT<Eigen::Ref<const Eigen::MatrixXd> > VglmChunk;

// Dense in-memory data: an n by p design, either 1-based category codes or
// an n by K count matrix, prior weights, and an n by M offset, visited in
// blocks of block_size rows. Nothing is copied; the chunks are views.
class DenseVglmData {
public:
    typedef Eigen::Ref<const Eigen::MatrixXd> DesignView;
    typedef VglmChunk Chunk;

    DenseVglmData(const double* x, int n, int p, const int* codes,
                  const double* counts, int K, const double* weights,
                  const double* offset, int M, int block_size, int outer_stride = 0)
        : x_(x, n, p, Eigen::OuterStride<>(outer_stride > 0 ? outer_stride : n)),
          counts_(counts, counts ? n : 0, counts ? K : 0),
          weights_(weights, n), offset_(offset, n, M), codes_(codes),
          n_(n), p_(p), M_(M), block_(std::max(1, block_size)) {}

    int n() const { return n_; }
    int p() const { return p_; }
    int M() const { return M_; }
    bool grouped() const { return codes_ == nullptr; }

    template <class F>
    void visit(F&& f) const {
        for (int first = 0; first < n_; first += block_) {
            Rcpp::checkUserInterrupt();
            const int b = std::min(block_, n_ - first);
            Chunk chunk{
                std::vector<DesignView>(),
                grouped() ? nullptr : codes_ + first,
                grouped() ? Eigen::Ref<const Eigen::MatrixXd>(counts_.middleRows(first, b))
                          : Eigen::Ref<const Eigen::MatrixXd>(empty_),
                weights_.segment(first, b),
                offset_.middleRows(first, b)};
            chunk.design.reserve(M_);
            for (int a = 0; a < M_; ++a) chunk.design.emplace_back(x_.middleRows(first, b));
            f(static_cast<const Chunk&>(chunk));
        }
    }

private:
    Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<> > x_;
    Eigen::Map<const Eigen::MatrixXd> counts_;
    Eigen::Map<const Eigen::VectorXd> weights_;
    Eigen::Map<const Eigen::MatrixXd> offset_;
    const int* codes_;
    int n_, p_, M_, block_;
    Eigen::MatrixXd empty_;
};

// A memory-mapped (bigmemory::big.matrix) design. The backing memory of a
// double-typed big.matrix is a dense column-major array whose columns are
// total_rows apart, so it is the dense data with an outer stride: the same
// view type, kernels, and visitor, and nothing is copied. first points at
// element (0, 0) of the (sub)matrix and stride is total_rows.
class BigVglmData : public DenseVglmData {
public:
    BigVglmData(const double* first, int stride, int n, int p, const int* codes,
                const double* counts, int K, const double* weights,
                const double* offset, int M, int block_size)
        : DenseVglmData(first, n, p, codes, counts, K, weights, offset, M, block_size,
                        stride) {}
};

// A block of rows of a sparse design. The rows are stored transposed: t is
// the p by b column-compressed block of X' whose column i is row i of the
// block, a view into the full transposed matrix (no copy).
struct SparseVglmView {
    typedef Eigen::Map<const Eigen::SparseMatrix<double> > Map;
    Map t;

    SparseVglmView(int p, int b, int nnz, const int* outer, const int* inner,
                   const double* values)
        : t(p, b, nnz, outer, inner, values) {}

    int rows() const { return static_cast<int>(t.cols()); }
    int cols() const { return static_cast<int>(t.rows()); }
};

typedef VglmChunkT<SparseVglmView> SparseVglmChunk;

// Sparse in-memory data: the transpose of the n by p design, a p by n
// column-compressed matrix (outer has n + 1 entries, as a dgCMatrix built by
// Matrix::t), so a block of rows is a contiguous range of its columns. The
// response, weights, and offsets are as for DenseVglmData. Nothing is copied.
class SparseVglmData {
public:
    typedef SparseVglmView DesignView;
    typedef SparseVglmChunk Chunk;

    SparseVglmData(const int* outer, const int* inner, const double* values, int n, int p,
                   const int* codes, const double* counts, int K, const double* weights,
                   const double* offset, int M, int block_size)
        : outer_(outer), inner_(inner), values_(values),
          counts_(counts, counts ? n : 0, counts ? K : 0), weights_(weights, n),
          offset_(offset, n, M), codes_(codes), n_(n), p_(p), M_(M),
          block_(std::max(1, block_size)) {}

    int n() const { return n_; }
    int p() const { return p_; }
    int M() const { return M_; }
    bool grouped() const { return codes_ == nullptr; }

    template <class F>
    void visit(F&& f) const {
        for (int first = 0; first < n_; first += block_) {
            Rcpp::checkUserInterrupt();
            const int b = std::min(block_, n_ - first);
            Chunk chunk{
                std::vector<DesignView>(),
                grouped() ? nullptr : codes_ + first,
                grouped() ? Eigen::Ref<const Eigen::MatrixXd>(counts_.middleRows(first, b))
                          : Eigen::Ref<const Eigen::MatrixXd>(empty_),
                weights_.segment(first, b),
                offset_.middleRows(first, b)};
            chunk.design.reserve(M_);
            // The block map starts at outer_ + first, so its first outer index
            // is not 0 and its stored entries are valuePtr()[outer[0], outer[b]).
            // Eigen operations that iterate over inner iterators (products,
            // InnerIterator) are valid on it; operations that read
            // valuePtr()[0, nonZeros()) directly, such as sum() or coeffs() on a
            // compressed map, are not and must not be used on this view.
            const int nnz = outer_[first + b] - outer_[first];
            for (int a = 0; a < M_; ++a)
                chunk.design.emplace_back(p_, b, nnz, outer_ + first, inner_, values_);
            f(static_cast<const Chunk&>(chunk));
        }
    }

private:
    const int* outer_;
    const int* inner_;
    const double* values_;
    Eigen::Map<const Eigen::MatrixXd> counts_;
    Eigen::Map<const Eigen::VectorXd> weights_;
    Eigen::Map<const Eigen::MatrixXd> offset_;
    const int* codes_;
    int n_, p_, M_, block_;
    Eigen::MatrixXd empty_;
};

// ---------------------------------------------------------------------------
// Validation shared by the fitting and evaluation exports of every data
// representation and by each block of a streamed source; the design's own
// finiteness is checked by the caller.
inline void validate_vglm_inputs(int n, int p, const Rcpp::IntegerVector& codes,
                                 const Rcpp::NumericMatrix& counts,
                                 const Rcpp::NumericVector& weights,
                                 const Rcpp::NumericMatrix& offset,
                                 const Rcpp::List& constraints, int K, int M,
                                 int block_size) {
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    if (n < 1 || p < 1) Rcpp::stop("the design must have at least one row and one column");
    if (block_size < 1) Rcpp::stop("block_size must be at least 1");
    if (weights.size() != n) Rcpp::stop("weights must have one entry per row");
    if (offset.nrow() != n || offset.ncol() != M) Rcpp::stop("offset must be n by K - 1");
    if (grouped && (counts.nrow() != n || counts.ncol() != K))
        Rcpp::stop("counts must be n by K");
    if (!grouped && codes.size() != n) Rcpp::stop("codes must have one entry per row");
    if (constraints.size() != p) Rcpp::stop("there must be one constraint matrix per design column");
    for (R_xlen_t i = 0; i < offset.size(); ++i)
        if (!std::isfinite(offset[i])) Rcpp::stop("offsets must be finite");
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(weights[i]) || weights[i] < 0)
            Rcpp::stop("weights must be finite and nonnegative");
        if (!grouped && (codes[i] == NA_INTEGER || codes[i] < 1 || codes[i] > K))
            Rcpp::stop("response category codes must lie in 1, ..., K");
    }
    if (grouped)
        for (R_xlen_t i = 0; i < counts.size(); ++i)
            if (!std::isfinite(counts[i]) || counts[i] < 0)
                Rcpp::stop("counts must be finite and nonnegative");
}

inline void validate_dense_design(const Rcpp::NumericMatrix& x) {
    for (R_xlen_t i = 0; i < x.size(); ++i)
        if (!std::isfinite(x[i])) Rcpp::stop("the design must be finite");
}

// One block of a streamed source, as normalized by .vglm_stream_reader():
// list(X = b by p dense design, codes = 1-based categories (integer(0) for
// grouped data), counts = b by K (0 by 0 for individual data), weights = b
// prior weights, offset = b by M). The block is validated as the in-memory
// inputs are, and data() maps it as dense data without copying, so the dense
// kernels are reused. The R objects are held for the life of the block.
struct VglmSourceBlock {
    Rcpp::NumericMatrix X, counts, offset;
    Rcpp::IntegerVector codes;
    Rcpp::NumericVector weights;
    bool grouped;

    VglmSourceBlock(const Rcpp::List& z, int p, int K, int M, int block_size,
                    const Rcpp::List& constraints)
        : X(Rcpp::as<Rcpp::NumericMatrix>(z["X"])),
          counts(Rcpp::as<Rcpp::NumericMatrix>(z["counts"])),
          offset(Rcpp::as<Rcpp::NumericMatrix>(z["offset"])),
          codes(Rcpp::as<Rcpp::IntegerVector>(z["codes"])),
          weights(Rcpp::as<Rcpp::NumericVector>(z["weights"])),
          grouped(counts.nrow() != 0 || counts.ncol() != 0) {
        if (X.ncol() != p) Rcpp::stop("the source design columns changed between blocks");
        validate_vglm_inputs(X.nrow(), p, codes, counts, weights, offset, constraints, K, M,
                             block_size);
        validate_dense_design(X);
    }

    DenseVglmData data(int K, int M, int block_size) const {
        return DenseVglmData(X.begin(), X.nrow(), X.ncol(), grouped ? nullptr : codes.begin(),
                             grouped ? counts.begin() : nullptr, K, weights.begin(),
                             offset.begin(), M, block_size);
    }
};

// A streamed source: blocks 1, ..., chunks come from the R function fetch
// (the reader of a fastglm_source), one at a time, as in OrdinalSource
// (fit_streaming_ordinal.cpp). Each visit reads every block once and visits it
// as dense data in sub-blocks of block_size rows, so the engine makes one pass
// over the source per visit and only one source block is in memory.
class SourceVglmData {
public:
    typedef Eigen::Ref<const Eigen::MatrixXd> DesignView;
    typedef VglmChunk Chunk;

    SourceVglmData(Rcpp::Function fetch, int chunks, int p, int K, int M, int block_size,
                   const Rcpp::List& constraints)
        : fetch_(fetch), constraints_(constraints), chunks_(chunks), p_(p), K_(K), M_(M),
          block_(std::max(1, block_size)) {}

    int p() const { return p_; }
    int M() const { return M_; }

    template <class F>
    void visit(F&& f) const {
        for (int k = 1; k <= chunks_; ++k) {
            Rcpp::checkUserInterrupt();
            const Rcpp::List z(fetch_(k));
            const VglmSourceBlock block(z, p_, K_, M_, block_, constraints_);
            block.data(K_, M_, block_).visit(f);
        }
    }

private:
    Rcpp::Function fetch_;
    Rcpp::List constraints_;
    int chunks_, p_, K_, M_, block_;
};

} // namespace fglm_vglm
#endif
