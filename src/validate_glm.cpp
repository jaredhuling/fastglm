#include <Rcpp.h>
#include <cmath>

// One pass over the response, prior weights, and offsets for
// .validate_glm_data(). Returns c(status, number of positive weights):
//   0 valid, 1 non-finite response, 2 non-finite weight, 3 non-finite offset,
//   4 negative weight or no positive weight.
// A zero-length y skips the response check (factor responses are checked
// in R).
// [[Rcpp::export]]
Rcpp::NumericVector validate_glm_vectors(Rcpp::NumericVector y, Rcpp::NumericVector weights,
                                         Rcpp::NumericVector offset)
{
    for (R_xlen_t i = 0; i < y.size(); ++i)
        if (!std::isfinite(y[i])) return Rcpp::NumericVector::create(1, 0);
    double positive = 0; bool negative = false;
    for (R_xlen_t i = 0; i < weights.size(); ++i) {
        const double w = weights[i];
        if (!std::isfinite(w)) return Rcpp::NumericVector::create(2, 0);
        if (w < 0) negative = true; else if (w > 0) ++positive;
    }
    for (R_xlen_t i = 0; i < offset.size(); ++i)
        if (!std::isfinite(offset[i])) return Rcpp::NumericVector::create(3, 0);
    if (negative || positive == 0) return Rcpp::NumericVector::create(4, positive);
    return Rcpp::NumericVector::create(0, positive);
}
