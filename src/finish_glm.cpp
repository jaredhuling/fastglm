#include <RcppEigen.h>
#include <cmath>
#include "../inst/include/families.h"

using Eigen::ArrayXd;
using Eigen::Map;

// Post-fit summaries for natively evaluated families, in one pass and with
// the same definitions as .finish_glm():
//   residuals    (y - mu) / mu.eta(eta), zero where the prior weight is 0
//   weights      prior weight * mu.eta^2 / variance(mu), zero where the prior
//                weight is 0 (returned only when want_weights)
//   pearson      sum(w * (y - mu)^2 / variance(mu))
//   null.deviance sum of deviance residuals at mu0, where mu0 is the constant
//                null_mean, or linkinv(offset) when null_mean is NA
// [[Rcpp::export]]
Rcpp::List finish_glm_native(int code, Rcpp::NumericVector params,
                             Rcpp::NumericVector y, Rcpp::NumericVector mu,
                             Rcpp::NumericVector eta, Rcpp::NumericVector w,
                             Rcpp::NumericVector offset, double null_mean,
                             bool want_weights)
{
    fglm::FamilyParams fp;
    if (params.size() >= 1) fp.theta = params[0];
    if (params.size() >= 2) fp.var_power = params[1];
    if (params.size() >= 3) fp.link_power = params[2];
    const Eigen::Index n = y.size();
    Map<const ArrayXd> ya(y.begin(), n), ma(mu.begin(), n), ea(eta.begin(), n), wa(w.begin(), n);
    ArrayXd dmu(n), va(n);
    fglm::mu_eta(code, fp, ea, dmu);
    fglm::variance(code, fp, ma, va);

    Rcpp::NumericVector resid(n);
    Map<ArrayXd> ra(resid.begin(), n);
    ra = (ya - ma) / dmu;
    for (Eigen::Index i = 0; i < n; ++i) if (!(wa[i] > 0)) ra[i] = 0;
    const double pearson = (wa * (ya - ma).square() / va).sum();

    double null_dev;
    if (std::isnan(null_mean)) {
        ArrayXd mu0(n);
        Map<const ArrayXd> oa(offset.begin(), n);
        fglm::linkinv(code, fp, oa, mu0);
        null_dev = fglm::dev_resids_sum(code, fp, ya, mu0, wa);
    } else {
        ArrayXd mu0 = ArrayXd::Constant(n, null_mean);
        null_dev = fglm::dev_resids_sum(code, fp, ya, mu0, wa);
    }
    Rcpp::List out = Rcpp::List::create(Rcpp::_["residuals"] = resid,
                                        Rcpp::_["pearson"] = pearson,
                                        Rcpp::_["null.deviance"] = null_dev);
    if (want_weights) {
        Rcpp::NumericVector wt(n);
        Map<ArrayXd> wv(wt.begin(), n);
        wv = wa * dmu.square() / va;
        for (Eigen::Index i = 0; i < n; ++i) if (!(wa[i] > 0)) wv[i] = 0;
        out["weights"] = wt;
    }
    return out;
}

// stats::binomial()$initialize for a numeric 0/1 (or proportion) response
// vector, in one pass. Returns list(y, mustart, status): status 1 means a
// value outside [0, 1] (the caller stops with the stats message), and
// status 2 means non-integer successes (the caller warns as stats does).
// [[Rcpp::export]]
Rcpp::List binomial_initialize_vector(Rcpp::NumericVector y, Rcpp::NumericVector weights)
{
    const R_xlen_t n = y.size();
    Rcpp::NumericVector yy = Rcpp::clone(y), mustart(n);
    int status = 0; bool noninteger = false;
    for (R_xlen_t i = 0; i < n; ++i) {
        if (weights[i] == 0) yy[i] = 0;
        const double v = yy[i];
        if (v < 0 || v > 1) { status = 1; break; }
        mustart[i] = (weights[i] * v + 0.5) / (weights[i] + 1);
        const double m = weights[i] * v;
        if (std::abs(m - R::fround(m, 0)) > 0.001) noninteger = true;
    }
    if (status == 0 && noninteger) status = 2;
    return Rcpp::List::create(Rcpp::_["y"] = yy, Rcpp::_["mustart"] = mustart, Rcpp::_["status"] = status);
}
