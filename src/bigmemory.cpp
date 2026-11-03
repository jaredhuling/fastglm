#include <algorithm>
#include <iterator>
#include <bigmemory/MatrixAccessor.hpp>
#include <bigmemory/BigMatrix.h>
#include "../inst/include/bigmemory.h"

using namespace Rcpp;

// All access respects both offsets of sub.big.matrix and separated columns.
// Accumulate in double precision even when the backing matrix stores integers.
template<class Accessor>
SEXP big_operation(Accessor mat, int operation, SEXP value)
{
    const int n = mat.nrow(), p = mat.ncol();
    if (operation == 0) {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(p, p);
        const int block = 4096;
        Eigen::MatrixXd x(std::min(n, block), p);
        for (int first = 0; first < n; first += block) {
            Rcpp::checkUserInterrupt();
            int size = std::min(block, n - first);
            for (int j = 0; j < p; ++j)
                for (int i = 0; i < size; ++i) x(i, j) = mat[j][first + i];
            result.selfadjointView<Eigen::Lower>().rankUpdate(x.topRows(size).transpose());
        }
        return wrap(Eigen::MatrixXd(result.selfadjointView<Eigen::Lower>()));
    }
    NumericVector out(operation == 4 ? n : p);
    if (operation >= 4) {
        NumericVector v = as<NumericVector>(value);
        if (v.size() != (operation == 4 ? p : n)) stop("non-conformant matrices");
        for (int j = 0; j < p; ++j) {
            Rcpp::checkUserInterrupt();
            for (int i = 0; i < n; ++i) {
                if (operation == 4) out[i] += static_cast<double>(mat[j][i]) * v[j];
                else out[j] += v[i] * static_cast<double>(mat[j][i]);
            }
        }
    } else {
        for (int j = 0; j < p; ++j) {
            Rcpp::checkUserInterrupt();
            double total = operation == 2 ? R_NegInf : operation == 3 ? R_PosInf : 0;
            for (int i = 0; i < n; ++i) {
                double v = mat[j][i];
                if (!R_finite(v)) { total = v; break; }
                if (operation == 1) total += v;
                else if (operation == 2) total = std::max(total, v);
                else total = std::min(total, v);
            }
            out[j] = total;
        }
    }
    return out;
}

template<typename T>
SEXP big_dispatch_storage(BigMatrix& mat, int operation, SEXP v)
{
    if (mat.separated_columns()) return big_operation(SepMatrixAccessor<T>(mat), operation, v);
    return big_operation(MatrixAccessor<T>(mat), operation, v);
}
SEXP big_dispatch(SEXP address, int operation, SEXP value = R_NilValue)
{
    XPtr<BigMatrix> mat(address);
    switch (mat->matrix_type()) {
    case 1: return big_dispatch_storage<char>(*mat, operation, value);
    case 2: return big_dispatch_storage<short>(*mat, operation, value);
    case 4: return big_dispatch_storage<int>(*mat, operation, value);
    case 6: return big_dispatch_storage<float>(*mat, operation, value);
    case 8: return big_dispatch_storage<double>(*mat, operation, value);
    default: stop("Unsupported big.matrix storage type");
    }
}
RcppExport SEXP crossprod_big(SEXP x) { BEGIN_RCPP return big_dispatch(x, 0); END_RCPP }
RcppExport SEXP colsums_big(SEXP x) { BEGIN_RCPP return big_dispatch(x, 1); END_RCPP }
RcppExport SEXP colmax_big(SEXP x) { BEGIN_RCPP return big_dispatch(x, 2); END_RCPP }
RcppExport SEXP colmin_big(SEXP x) { BEGIN_RCPP return big_dispatch(x, 3); END_RCPP }
RcppExport SEXP prod_vec_big(SEXP x, SEXP y) { BEGIN_RCPP return big_dispatch(x, 4, y); END_RCPP }
RcppExport SEXP prod_vec_big_right(SEXP x, SEXP y) { BEGIN_RCPP return big_dispatch(y, 5, x); END_RCPP }
