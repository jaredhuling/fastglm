#define EIGEN_DONT_PARALLELIZE
// bigmemory's BigMatrix.h relies on transitive includes (see fit_glm_dense.cpp).
#include <algorithm>
#include <iterator>
#include <bigmemory/MatrixAccessor.hpp>
#include <bigmemory/BigMatrix.h>
#include <RcppEigen.h>
#include "vglm_family.h"
#include "vglm_multinomial.h"
#include "vglm_data.h"
#include "vglm_engine.h"

namespace fglm_vglm {

// The one table of vector GLM model types.
std::unique_ptr<VectorFamily> make_vector_family(const std::string& name, int K,
                                                 int reference,
                                                 const Rcpp::List& parameters) {
    (void)parameters;
    if (name == "multinomial")
        return std::unique_ptr<VectorFamily>(new MultinomialFamily(K, reference));
    throw std::invalid_argument("unknown vector GLM model type: " + name);
}

// Test only: a model type that delegates to another and restricts every step
// so that no linear predictor changes by more than bound; bound = 0 allows
// no step. It exercises the max_step_fraction path of the engine, which
// multinomial never uses.
class StepBoundFamily : public VectorFamily {
public:
    StepBoundFamily(const VectorFamily& base, double bound) : base_(base), bound_(bound) {}
    int M() const override { return base_.M(); }
    int K() const override { return base_.K(); }
    std::string name() const override { return base_.name(); }
    bool valid(const VglmBlock& block) const override { return base_.valid(block); }
    double loglik(const VglmBlock& block, VectorXd* rows) const override {
        return base_.loglik(block, rows);
    }
    void score_information(const VglmBlock& block, MatrixXd& u, MatrixXd& W) const override {
        base_.score_information(block, u, W);
    }
    void probabilities(const MatrixXd& eta, MatrixXd& P) const override {
        base_.probabilities(eta, P);
    }
    void bias_terms(const VglmBlock& block, const MatrixXd& A, MatrixXd& t) const override {
        base_.bias_terms(block, A, t);
    }
    VectorXd start_eta(const VectorXd& totals) const override { return base_.start_eta(totals); }
    bool restricts_step() const override { return true; }
    double max_step_fraction(const MatrixXd& eta, const MatrixXd& direction) const override {
        (void)eta;
        const double change = direction.size() ? direction.cwiseAbs().maxCoeff() : 0.0;
        return change > bound_ ? bound_ / change : 1.0;
    }

private:
    const VectorFamily& base_;
    double bound_;
};

} // namespace fglm_vglm

namespace fglm_vglm {
// The transpose of a sparse design: a p by n dgCMatrix, as Matrix::t gives
// (mapped, not copied). The stored values of columns [from, to) (0-based; to
// = -1 means n) are checked for finiteness, so evaluating one block checks
// only that block.
struct SparseTransposed {
    Eigen::Map<Eigen::SparseMatrix<double> > xt;
    explicit SparseTransposed(SEXP x, int from = 0, int to = -1)
        : xt(Rcpp::as<Eigen::Map<Eigen::SparseMatrix<double> > >(x)) {
        if (!xt.isCompressed()) Rcpp::stop("the sparse design must be column compressed");
        if (to < 0) to = n();
        if (from < 0 || from > to || to > n()) Rcpp::stop("invalid column range of the sparse design");
        const int* outer = xt.outerIndexPtr();
        for (R_xlen_t j = outer[from]; j < outer[to]; ++j)
            if (!std::isfinite(xt.valuePtr()[j])) Rcpp::stop("the design must be finite");
    }
    int n() const { return static_cast<int>(xt.cols()); }
    int p() const { return static_cast<int>(xt.rows()); }
};

inline std::vector<MatrixXd> vglm_constraints(const Rcpp::List& constraints) {
    std::vector<MatrixXd> H;
    H.reserve(constraints.size());
    for (R_xlen_t k = 0; k < constraints.size(); ++k) H.push_back(Rcpp::as<MatrixXd>(constraints[k]));
    return H;
}

// The intercept is the first constant nonzero design column, as in the GLM
// path; intercept is -1 when there is none.
inline void dense_intercept(const Rcpp::NumericMatrix& x, int& intercept, double& value) {
    const int n = x.nrow(), p = x.ncol();
    intercept = -1;
    value = 0;
    for (int k = 0; k < p && intercept < 0; ++k) {
        const double first = x(0, k);
        bool constant = first != 0;
        for (int i = 1; i < n && constant; ++i) constant = x(i, k) == first;
        if (constant) { intercept = k; value = first; }
    }
}

// The same rule for the transposed sparse design: design column k is row k of
// xt, constant and nonzero when it has n stored entries, all equal and
// nonzero.
inline void sparse_intercept(const SparseTransposed& s, int& intercept, double& value) {
    const int n = s.n(), p = s.p();
    std::vector<int> stored(p, 0);
    std::vector<double> first(p, 0);
    std::vector<bool> equal(p, true);
    for (int i = 0; i < n; ++i)
        for (Eigen::Map<Eigen::SparseMatrix<double> >::InnerIterator it(s.xt, i); it; ++it) {
            const int k = static_cast<int>(it.row());
            if (stored[k]++ == 0) first[k] = it.value();
            else if (it.value() != first[k]) equal[k] = false;
        }
    intercept = -1;
    value = 0;
    for (int k = 0; k < p && intercept < 0; ++k)
        if (stored[k] == n && equal[k] && first[k] != 0) { intercept = k; value = first[k]; }
}

// A double-typed big.matrix as a dense strided view of its backing memory,
// restricted to rows [from, to) (0-based; to = -1 means all rows). It is
// mapped, not copied. The finiteness of the rows in the range is checked, so
// evaluating one block checks only that block. Only double storage in one
// contiguous allocation is supported, as for fit_big_glm.
struct BigDesign {
    Rcpp::XPtr<BigMatrix> ptr;
    int from, count;
    explicit BigDesign(SEXP x, int from_ = 0, int to = -1) : ptr(checked(x)), from(from_), count(0) {
        if (ptr->matrix_type() != 8)
            Rcpp::stop("the big.matrix must have type double");
        if (ptr->separated_columns())
            Rcpp::stop("a big.matrix with separated columns is not supported; use one contiguous allocation");
        const int total = static_cast<int>(ptr->nrow());
        if (to < 0) to = total;
        if (from < 0 || from > to || to > total) Rcpp::stop("invalid row range of the big.matrix");
        count = to - from;
        const Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<> > v = view();
        for (int j = 0; j < v.cols(); ++j)
            if (!v.col(j).allFinite()) Rcpp::stop("the design must be finite");
    }
    // A null external pointer (a big.matrix restored from serialization)
    // must stop with an error rather than be dereferenced.
    static SEXP checked(SEXP x) {
        if (TYPEOF(x) != EXTPTRSXP || R_ExternalPtrAddr(x) == nullptr)
            Rcpp::stop("the big.matrix is no longer available (a null external pointer, for example after serialization)");
        return x;
    }
    int n() const { return count; }
    int p() const { return static_cast<int>(ptr->ncol()); }
    int stride() const { return static_cast<int>(ptr->total_rows()); }
    const double* first() const {
        const double* base = static_cast<const double*>(ptr->matrix()) +
            static_cast<std::size_t>(ptr->col_offset()) * ptr->total_rows() + ptr->row_offset();
        return base + from;
    }
    Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<> > view() const {
        return Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<> >(
            first(), count, p(), Eigen::OuterStride<>(stride()));
    }
};

// The intercept rule of dense_intercept for a big.matrix.
inline void big_intercept(const BigDesign& b, int& intercept, double& value) {
    const Eigen::Map<const MatrixXd, 0, Eigen::OuterStride<> > x = b.view();
    intercept = -1;
    value = 0;
    for (int k = 0; k < x.cols() && intercept < 0; ++k) {
        const double first = x(0, k);
        if (first != 0 && x.col(k).minCoeff() == first && x.col(k).maxCoeff() == first) {
            intercept = k;
            value = first;
        }
    }
}

// The fit for any data representation: starting values and the test-only
// controls, then the engine. known_totals, when given, are the weighted
// category totals (otherwise computed by a pass over the data).
template <class Data>
Rcpp::List vglm_fit_data(const Data& data, const VectorFamily& fam,
                         const std::vector<MatrixXd>& H, int intercept, double value,
                         int K, bool firth, double tol, int maxit, bool trace,
                         const Rcpp::List& control, const VectorXd* known_totals = nullptr) {
    bool force_mapping = false;
    if (control.containsElementNamed("force_mapping"))
        force_mapping = Rcpp::as<bool>(control["force_mapping"]);
    // A streamed source supplies the totals from its preliminary pass, so
    // that the engine does not read the source once more for them.
    const VectorXd totals = known_totals ? *known_totals : vglm_category_totals(data, K);
    const VectorXd start = vglm_start(fam, totals, H, intercept, value);
    if (control.containsElementNamed("step_bound")) {
        const double bound = Rcpp::as<double>(control["step_bound"]);
        if (!(bound >= 0)) Rcpp::stop("step_bound must be nonnegative");
        const StepBoundFamily bounded(fam, bound);
        return vglm_fit(data, bounded, H, start, firth, tol, maxit, trace, force_mapping);
    }
    return vglm_fit(data, fam, H, start, firth, tol, maxit, trace, force_mapping);
}

// Per-row evaluation for any data representation (see vglm_evaluate_dense).
template <class Data>
Rcpp::List vglm_evaluate_data(const Data& data, const VectorFamily& fam,
                              const std::vector<MatrixXd>& H,
                              const Rcpp::NumericVector& coefficients) {
    const int n = data.n(), M = fam.M(), Q = packed_size(M), K = fam.K();
    const VglmLayout L(H, M, false);
    if (coefficients.size() != L.P) Rcpp::stop("coefficients must have one entry per constrained coefficient");
    VectorXd beta(L.P);
    for (int i = 0; i < L.P; ++i) beta[i] = Rcpp::NumericVector::is_na(coefficients[i]) ? 0.0 : coefficients[i];
    const MatrixXd B = L.predictors(beta);
    MatrixXd eta_all(n, M), u_all(n, M), W_all(n, Q), P_all(n, K);
    VectorXd ll_all(n);
    int first = 0;
    MatrixXd eta, u, W, P;
    VectorXd rows;
    data.visit([&](const typename Data::Chunk& chunk) {
        const int b = chunk.rows();
        const bool shared = vglm_shared_view(chunk, M);
        vglm_linear(chunk, shared, B, eta);
        const VglmBlock block{eta, chunk.codes, chunk.counts, chunk.weights};
        fam.loglik(block, &rows);
        fam.score_information(block, u, W);
        fam.probabilities(eta, P);
        eta_all.middleRows(first, b) = eta;
        u_all.middleRows(first, b) = u;
        W_all.middleRows(first, b) = W;
        P_all.middleRows(first, b) = P;
        ll_all.segment(first, b) = rows;
        first += b;
    });
    return Rcpp::List::create(
        Rcpp::Named("eta") = Rcpp::wrap(eta_all),
        Rcpp::Named("u") = Rcpp::wrap(u_all),
        Rcpp::Named("W") = Rcpp::wrap(W_all),
        Rcpp::Named("P") = Rcpp::wrap(P_all),
        Rcpp::Named("loglik_rows") = Rcpp::wrap(ll_all));
}
} // namespace fglm_vglm

// Evaluate every block method of a model type on one block, for tests.
// codes is used when counts is 0 by 0; A may be 0 by 0, in which case the
// bias terms are computed with A = 0.
// [[Rcpp::export]]
Rcpp::List vglm_family_eval(std::string family, int K, int reference,
                            Rcpp::NumericMatrix eta, Rcpp::IntegerVector codes,
                            Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                            Rcpp::NumericMatrix A) {
    using Eigen::Map;
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference, Rcpp::List());
    const int b = eta.nrow(), M = fam->M(), Q = fglm_vglm::packed_size(M);
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    if (eta.ncol() != M) Rcpp::stop("eta must have K - 1 columns");
    if (weights.size() != b) Rcpp::stop("weights must have one entry per row");
    if (!grouped && codes.size() != b) Rcpp::stop("codes must have one entry per row");

    const Map<const MatrixXd> E(eta.begin(), b, M);
    const Map<const MatrixXd> Y(counts.begin(), counts.nrow(), counts.ncol());
    const Map<const VectorXd> w(weights.begin(), b);
    MatrixXd Amat = MatrixXd::Zero(b, Q);
    if (A.nrow() != 0 || A.ncol() != 0) {
        if (A.nrow() != b || A.ncol() != Q) Rcpp::stop("A must be b by M(M + 1)/2");
        Amat = Map<const MatrixXd>(A.begin(), b, Q);
    }

    const fglm_vglm::VglmBlock block{E, grouped ? nullptr : codes.begin(), Y, w};
    VectorXd rows;
    fam->loglik(block, &rows);
    MatrixXd u, W, P, t;
    fam->score_information(block, u, W);
    fam->probabilities(E, P);
    fam->bias_terms(block, Amat, t);
    return Rcpp::List::create(
        Rcpp::Named("loglik_rows") = Rcpp::wrap(rows),
        Rcpp::Named("u") = Rcpp::wrap(u),
        Rcpp::Named("W") = Rcpp::wrap(W),
        Rcpp::Named("P") = Rcpp::wrap(P),
        Rcpp::Named("t") = Rcpp::wrap(t),
        Rcpp::Named("valid") = fam->valid(block));
}

// Fit a vector GLM to dense in-memory data (internal). codes holds 1-based
// categories when counts is 0 by 0; otherwise counts is the n by K grouped
// response. offset is n by M; constraints holds one M by r_k matrix per design
// column. reference is the 1-based reference category (as in
// .vglm_reference_fit; vglm_family_eval takes it 0-based). Starting values
// come from the model type's start_eta at the weighted category totals,
// placed on the first constant design column.
// control may hold step_bound (test only): restrict every step so that no
// linear predictor changes by more than step_bound (0 allows no step).
// control may hold force_mapping (test only): apply the constraint mapping
// even when every constraint matrix is the identity.
// [[Rcpp::export]]
Rcpp::List vglm_fit_dense(Rcpp::NumericMatrix x, Rcpp::IntegerVector codes,
                          Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                          Rcpp::NumericMatrix offset, Rcpp::List constraints,
                          std::string family, int K, int reference, bool firth,
                          double tol, int maxit, int block_size, bool trace,
                          Rcpp::List control) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const int n = x.nrow(), p = x.ncol(), M = fam->M();
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    fglm_vglm::validate_vglm_inputs(n, p, codes, counts, weights, offset, constraints, K, M, block_size);
    fglm_vglm::validate_dense_design(x);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::DenseVglmData data(x.begin(), n, p, grouped ? nullptr : codes.begin(),
                                        grouped ? counts.begin() : nullptr, K, weights.begin(),
                                        offset.begin(), M, block_size);
    int intercept;
    double value;
    fglm_vglm::dense_intercept(x, intercept, value);
    return fglm_vglm::vglm_fit_data(data, *fam, H, intercept, value, K, firth, tol, maxit,
                                    trace, control);
}

// Fit a vector GLM to a sparse design (internal). xt is the transpose of the
// n by p design, a p by n dgCMatrix (Matrix::t of the design), so that a
// block of rows is a contiguous range of its columns; the other arguments are
// those of vglm_fit_dense. The engine code is the same; only the data concept
// and its design view differ.
// [[Rcpp::export]]
Rcpp::List vglm_fit_sparse(SEXP xt, Rcpp::IntegerVector codes,
                           Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                           Rcpp::NumericMatrix offset, Rcpp::List constraints,
                           std::string family, int K, int reference, bool firth,
                           double tol, int maxit, int block_size, bool trace,
                           Rcpp::List control) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const fglm_vglm::SparseTransposed s(xt);
    const int n = s.n(), p = s.p(), M = fam->M();
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    fglm_vglm::validate_vglm_inputs(n, p, codes, counts, weights, offset, constraints, K, M, block_size);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::SparseVglmData data(s.xt.outerIndexPtr(), s.xt.innerIndexPtr(),
                                         s.xt.valuePtr(), n, p,
                                         grouped ? nullptr : codes.begin(),
                                         grouped ? counts.begin() : nullptr, K, weights.begin(),
                                         offset.begin(), M, block_size);
    int intercept;
    double value;
    fglm_vglm::sparse_intercept(s, intercept, value);
    return fglm_vglm::vglm_fit_data(data, *fam, H, intercept, value, K, firth, tol, maxit,
                                    trace, control);
}

// Evaluate a model type at given coefficients for every row of dense data:
// the linear predictors eta (n by M, offsets included), row scores u (n by M)
// and packed row information W (n by M(M + 1)/2), fitted probabilities P
// (n by K), and the weighted per-row log likelihood. The arguments are those
// of vglm_fit_dense, with reference 1-based and coefficients the term-major
// vector of length sum_k r_k; NA entries (aliased coefficients) are treated
// as 0. It uses the engine's own layout and chunk visitor, so evaluation and
// fitting share one code path.
// [[Rcpp::export]]
Rcpp::List vglm_evaluate_dense(Rcpp::NumericMatrix x, Rcpp::IntegerVector codes,
                               Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                               Rcpp::NumericMatrix offset, Rcpp::List constraints,
                               std::string family, int K, int reference,
                               Rcpp::NumericVector coefficients, int block_size) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const int n = x.nrow(), p = x.ncol(), M = fam->M();
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    fglm_vglm::validate_vglm_inputs(n, p, codes, counts, weights, offset, constraints, K, M, block_size);
    fglm_vglm::validate_dense_design(x);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::DenseVglmData data(x.begin(), n, p, grouped ? nullptr : codes.begin(),
                                        grouped ? counts.begin() : nullptr, K, weights.begin(),
                                        offset.begin(), M, block_size);
    return fglm_vglm::vglm_evaluate_data(data, *fam, H, coefficients);
}

// The per-row evaluation of vglm_evaluate_dense for rows first, ..., last
// (1-based, inclusive) of a sparse design, given as its full transpose xt (a
// p by n dgCMatrix) as for vglm_fit_sparse. The block of rows is a contiguous
// range of columns of xt, mapped without copying, so evaluating every block
// of a large design costs O(nnz) in all rather than a row subset of the whole
// matrix per block. codes, counts, weights, and offset hold the rows of the
// block only.
// [[Rcpp::export]]
Rcpp::List vglm_evaluate_sparse(SEXP xt, int first, int last, Rcpp::IntegerVector codes,
                                Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                                Rcpp::NumericMatrix offset, Rcpp::List constraints,
                                std::string family, int K, int reference,
                                Rcpp::NumericVector coefficients, int block_size) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const Eigen::Map<Eigen::SparseMatrix<double> > whole =
        Rcpp::as<Eigen::Map<Eigen::SparseMatrix<double> > >(xt);
    if (first < 1 || last < first || last > whole.cols())
        Rcpp::stop("the row range must satisfy 1 <= first <= last <= n");
    const fglm_vglm::SparseTransposed s(xt, first - 1, last);
    const int n = last - first + 1, p = s.p(), M = fam->M();
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    fglm_vglm::validate_vglm_inputs(n, p, codes, counts, weights, offset, constraints, K, M, block_size);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::SparseVglmData data(s.xt.outerIndexPtr() + (first - 1), s.xt.innerIndexPtr(),
                                         s.xt.valuePtr(), n, p,
                                         grouped ? nullptr : codes.begin(),
                                         grouped ? counts.begin() : nullptr, K, weights.begin(),
                                         offset.begin(), M, block_size);
    return fglm_vglm::vglm_evaluate_data(data, *fam, H, coefficients);
}

// Fit a vector GLM to a memory-mapped design (internal). x is the external
// pointer of a double-typed big.matrix (its `address` slot); the other
// arguments are those of vglm_fit_dense. The engine maps the backing memory
// and visits it in blocks of block_size rows, so the design is never copied.
// [[Rcpp::export]]
Rcpp::List vglm_fit_big(SEXP x, Rcpp::IntegerVector codes,
                        Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                        Rcpp::NumericMatrix offset, Rcpp::List constraints,
                        std::string family, int K, int reference, bool firth,
                        double tol, int maxit, int block_size, bool trace,
                        Rcpp::List control) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const fglm_vglm::BigDesign b(x);
    const int n = b.n(), p = b.p(), M = fam->M();
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    fglm_vglm::validate_vglm_inputs(n, p, codes, counts, weights, offset, constraints, K, M, block_size);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::BigVglmData data(b.first(), b.stride(), n, p,
                                      grouped ? nullptr : codes.begin(),
                                      grouped ? counts.begin() : nullptr, K, weights.begin(),
                                      offset.begin(), M, block_size);
    int intercept;
    double value;
    fglm_vglm::big_intercept(b, intercept, value);
    return fglm_vglm::vglm_fit_data(data, *fam, H, intercept, value, K, firth, tol, maxit,
                                    trace, control);
}

// The per-row evaluation of vglm_evaluate_dense for rows first, ..., last
// (1-based, inclusive) of a big.matrix design, mapped without copying; only
// the rows of the range are read and checked. codes, counts, weights, and
// offset hold the rows of the range only.
// [[Rcpp::export]]
Rcpp::List vglm_evaluate_big(SEXP x, int first, int last, Rcpp::IntegerVector codes,
                             Rcpp::NumericMatrix counts, Rcpp::NumericVector weights,
                             Rcpp::NumericMatrix offset, Rcpp::List constraints,
                             std::string family, int K, int reference,
                             Rcpp::NumericVector coefficients, int block_size) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    Rcpp::XPtr<BigMatrix> whole(fglm_vglm::BigDesign::checked(x));
    if (first < 1 || last < first || last > static_cast<int>(whole->nrow()))
        Rcpp::stop("the row range must satisfy 1 <= first <= last <= n");
    const fglm_vglm::BigDesign b(x, first - 1, last);
    const int n = b.n(), p = b.p(), M = fam->M();
    const bool grouped = counts.nrow() != 0 || counts.ncol() != 0;
    fglm_vglm::validate_vglm_inputs(n, p, codes, counts, weights, offset, constraints, K, M, block_size);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::BigVglmData data(b.first(), b.stride(), n, p,
                                      grouped ? nullptr : codes.begin(),
                                      grouped ? counts.begin() : nullptr, K, weights.begin(),
                                      offset.begin(), M, block_size);
    return fglm_vglm::vglm_evaluate_data(data, *fam, H, coefficients);
}

// Fit a vector GLM to a streamed source (internal). fetch(k) returns block k
// (1, ..., n_chunks) normalized by .vglm_stream_reader(): list(X, codes,
// counts, weights, offset) as for vglm_fit_dense; every block is validated as
// it is read. The engine reads one source block at a time and visits it in
// sub-blocks of block_size rows with the dense kernels. The starting values
// need the weighted category totals and the intercept column, which the R
// caller computes in its preliminary pass over the source: totals has K
// entries, intercept is the 1-based constant design column (0 for none), and
// value its constant entry. passes counts the engine's passes over the source
// (see vglm_fit), not that preliminary pass.
// [[Rcpp::export]]
Rcpp::List vglm_fit_source(Rcpp::Function fetch, int n_chunks, int p,
                           Rcpp::NumericVector totals, int intercept, double value,
                           Rcpp::List constraints, std::string family, int K, int reference,
                           bool firth, double tol, int maxit, int block_size, bool trace,
                           Rcpp::List control) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const int M = fam->M();
    if (n_chunks < 1 || p < 1 || block_size < 1) Rcpp::stop("invalid source dimensions");
    if (totals.size() != K) Rcpp::stop("totals must have one entry per category");
    if (constraints.size() != p) Rcpp::stop("there must be one constraint matrix per design column");
    if (intercept < 0 || intercept > p) Rcpp::stop("invalid intercept column");
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    const fglm_vglm::SourceVglmData data(fetch, n_chunks, p, K, M, block_size, constraints);
    const Eigen::VectorXd known = Rcpp::as<Eigen::VectorXd>(totals);
    return fglm_vglm::vglm_fit_data(data, *fam, H, intercept - 1, value, K, firth, tol, maxit,
                                    trace, control, &known);
}

// The per-row evaluation of vglm_evaluate_dense for one block of a streamed
// source, the list normalized by .vglm_stream_reader() (see vglm_fit_source).
// The block is validated and mapped as for the fit, so evaluation and fitting
// share one code path; the R caller reads each block once and evaluates it.
// [[Rcpp::export]]
Rcpp::List vglm_evaluate_source(Rcpp::List block, Rcpp::List constraints, std::string family,
                                int K, int reference, Rcpp::NumericVector coefficients,
                                int block_size) {
    std::unique_ptr<fglm_vglm::VectorFamily> fam =
        fglm_vglm::make_vector_family(family, K, reference - 1, Rcpp::List());
    const int M = fam->M();
    const Rcpp::NumericMatrix X = Rcpp::as<Rcpp::NumericMatrix>(block["X"]);
    const fglm_vglm::VglmSourceBlock b(block, X.ncol(), K, M, block_size, constraints);
    const std::vector<Eigen::MatrixXd> H = fglm_vglm::vglm_constraints(constraints);
    return fglm_vglm::vglm_evaluate_data(b.data(K, M, block_size), *fam, H, coefficients);
}
