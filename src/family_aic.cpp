#include <Rcpp.h>
#include <Rmath.h>

// AIC (without the 2 * rank term) for the standard binomial, Poisson, and
// Gamma families, computed with the same Rmath densities and the same
// expressions as stats::binomial()$aic, stats::poisson()$aic, and
// stats::Gamma()$aic, without allocating observation-length R vectors.
// The sum uses a long double accumulator, as R's sum() does.
//   code 1 binomial: m = n if any(n > 1) else wt;
//                    -2 * sum(ifelse(m > 0, wt / m, 0) * dbinom(round(m * y), round(m), mu, log = TRUE))
//   code 2 poisson:  -2 * sum(dpois(y, mu, log = TRUE) * wt)
//   code 3 Gamma:    -2 * sum(dgamma(y, 1 / disp, scale = mu * disp, log = TRUE) * wt) + 2,
//                    disp = dev / sum(wt)
// [[Rcpp::export]]
double family_aic_cpp(int code, Rcpp::NumericVector y, Rcpp::NumericVector n,
                      Rcpp::NumericVector mu, Rcpp::NumericVector wt, double dev)
{
    const R_xlen_t len = y.size();
    if (mu.size() != len || wt.size() != len) Rcpp::stop("length mismatch in family_aic_cpp");
    long double total = 0;
    if (code == 1) {
        bool use_n = false;
        for (R_xlen_t i = 0; i < n.size(); ++i) if (n[i] > 1) { use_n = true; break; }
        const R_xlen_t nlen = use_n ? n.size() : wt.size();
        if (nlen != len) Rcpp::stop("length mismatch in family_aic_cpp");
        for (R_xlen_t i = 0; i < len; ++i) {
            const double m = use_n ? n[i] : wt[i];
            const double factor = m > 0 ? wt[i] / m : 0.0;
            const double d = R::dbinom(R::fround(m * y[i], 0), R::fround(m, 0), mu[i], 1);
            total += factor * d;
        }
        return (double)(-2 * total);
    } else if (code == 2) {
        for (R_xlen_t i = 0; i < len; ++i) total += R::dpois(y[i], mu[i], 1) * wt[i];
        return (double)(-2 * total);
    } else if (code == 3) {
        long double wsum = 0;
        for (R_xlen_t i = 0; i < len; ++i) wsum += wt[i];
        const double disp = dev / (double)wsum;
        for (R_xlen_t i = 0; i < len; ++i)
            total += R::dgamma(y[i], 1 / disp, mu[i] * disp, 1) * wt[i];
        return (double)(-2 * total + 2);
    }
    Rcpp::stop("unknown family code in family_aic_cpp");
    return NA_REAL;
}
