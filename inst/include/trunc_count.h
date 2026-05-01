#ifndef FASTGLM_TRUNC_COUNT_H
#define FASTGLM_TRUNC_COUNT_H
//
// Zero-truncated count regression (Poisson and NB2) via Fisher scoring IRLS
// with Cholesky LLT decomposition.  Used inside the hurdle driver (Phase 4)
// and also exposed for direct use where a positive-only count regression is
// the model of interest.
//
// All numerical work is in C++; no R callbacks.  Parameterization is in
// terms of the underlying (untruncated) rate lambda = exp(X*beta + offset).
// The conditional mean E[Y|Y>0] = mu_T is what the score uses; the Fisher
// information per observation is computed in closed form.
//
// References:
//   * Hilbe (2011) "Negative Binomial Regression" sec. 12.2 (truncated NB)
//   * Cameron & Trivedi (1998) "Regression Analysis of Count Data" sec. 4.5
//   * pscl::hurdle source for the shape of the joint log-likelihood.
//

#include <RcppEigen.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/tools/roots.hpp>
#include "nb_theta.h"
#include <cmath>
#include <limits>
#include <utility>

namespace fglm { namespace trunc {

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct TruncFitResult {
    VectorXd beta;
    VectorXd eta;        // X*beta + offset (linear predictor in the rate-parameter scale)
    VectorXd mu_T;       // E[Y|Y>0]
    VectorXd lambda;     // exp(eta) -- underlying (untruncated) rate
    VectorXd se;
    MatrixXd vcov;       // (X'IX)^{-1} at convergence
    double   loglik;
    int      iter;
    bool     converged;
};

// ---------------------------------------------------------------------------
// Per-observation kernels
// ---------------------------------------------------------------------------

// Truncated Poisson at lambda > 0.  Sets mu_T, dmu_T/deta, q = 1 - exp(-lambda).
// For Poisson, score U_i / (y_i - mu_T_i) = 1, info I_i = dmu_T/deta = V_T.
// Returns: mu_T, dmuT/deta, log(1 - exp(-lambda)) for log-lik accumulation.
//
// Stability: -expm1(-lambda) keeps q accurate for small lambda; mu_T / V_T
// extrapolated from the lambda -> 0 series mu_T = 1 + l/2 + l^2/12 + ...
inline void eval_pois(double lambda,
                      double& muT, double& dmuT_deta, double& log_q)
{
    if (!std::isfinite(lambda) || lambda < 0.0) lambda = 0.0;
    if (lambda < 1e-12) {
        // Taylor: q ~ lambda(1 - lambda/2 + ...), mu_T -> 1, V_T -> 1/2
        muT       = 1.0 + 0.5 * lambda + lambda * lambda / 12.0;
        dmuT_deta = 0.5 + lambda / 6.0;     // (V_T at lambda=0 = 1/2)
        log_q     = std::log(lambda) - 0.5 * lambda;  // log(lambda - lambda^2/2 + ...) ~ log(lambda) - lambda/2
        if (lambda == 0.0) { muT = 1.0; dmuT_deta = 0.5; log_q = -std::numeric_limits<double>::infinity(); }
        return;
    }
    const double q = -std::expm1(-lambda);   // 1 - exp(-lambda), stable
    muT       = lambda / q;
    dmuT_deta = muT * (1.0 + lambda - muT);  // Var(Y|Y>0) for Poisson
    if (dmuT_deta < 1e-300) dmuT_deta = 1e-300;
    log_q     = std::log1p(-std::exp(-lambda));
}

// Truncated NB2 at (lambda, theta).  Sets mu_T, dmu_T/deta, log_q, log_p0,
// and the score multiplier c = theta/(theta+lambda).  Score per obs in eta
// is c*(y - mu_T); info per obs is c * dmu_T/deta.
//
// We use log1p / expm1 around log_p0 for stability (NB zero probability
// can be very close to 1 when theta is small or lambda is small).
inline void eval_nb(double lambda, double theta,
                    double& muT, double& dmuT_deta,
                    double& c, double& log_q, double& log_p0)
{
    if (!std::isfinite(lambda) || lambda < 0.0) lambda = 0.0;
    if (theta <= 0.0) theta = 1e-12;

    const double tlam = theta + lambda;
    c = theta / tlam;
    // log(theta/(theta+lambda)) -- prefer log1p when lambda << theta.
    const double log_c = (lambda < 1e-3 * theta)
                          ? -std::log1p(lambda / theta)
                          : std::log(c);
    log_p0 = theta * log_c;

    // q = 1 - p_0,  log_q = log(1 - exp(log_p0))
    double q;
    if (log_p0 < -1e-8) {
        q     = -std::expm1(log_p0);
        log_q = std::log1p(-std::exp(log_p0));
    } else {
        // log_p0 ~ 0 means p_0 ~ 1: q tiny, lambda very small.  Use expm1.
        q     = -std::expm1(log_p0);
        if (q <= 0.0) q = 1e-300;
        log_q = std::log(q);
    }

    if (q < 1e-300 || lambda < 1e-15) {
        // lambda -> 0 limit:  mu_T -> 1, c -> 1, dmu_T/deta -> small but positive.
        // Fall back to Poisson-like values in that corner; the log-lik handles itself.
        muT       = 1.0;
        dmuT_deta = 1e-12;
        return;
    }

    muT = lambda / q;

    // dmu_T/dlambda = (1/q) * [1 - a]
    // where a = lambda * theta * p_0 / (q * (theta+lambda))
    //         = (1 - c) * theta * p_0 / q
    const double p0 = std::exp(log_p0);
    const double a  = (1.0 - c) * theta * p0 / q;
    const double dmuT_dlambda = (1.0 - a) / q;
    dmuT_deta = lambda * dmuT_dlambda;
    if (dmuT_deta < 1e-300) dmuT_deta = 1e-300;
}

// ---------------------------------------------------------------------------
// Score wrt theta for zero-truncated NB2 (used by joint (beta, theta) MLE).
//
//   s_T(theta) = sum_i wt_i * [ s_NB(theta; y_i, lambda_i)
//                                + p_0_i/(1 - p_0_i) * (log_c_i + lambda_i/tlam_i) ]
//
// where s_NB is the unconditional NB2 score wrt theta and the second term is
// the +d/dtheta log(1 - p_0) truncation correction.  Sign:
//    log f_T = log f_NB - log(1 - p_0)
//    d/dtheta log(1 - p_0) = -p_0/(1-p_0) * (log_c + lam/tlam)
//    so d/dtheta log f_T = s_NB + p_0/(1-p_0) * (log_c + lam/tlam).
//
// All numerically stable expansions used.
// ---------------------------------------------------------------------------
inline double trunc_nb_score_theta(double theta,
                                   const Eigen::Ref<const ArrayXd>& y,
                                   const Eigen::Ref<const ArrayXd>& lambda,
                                   const Eigen::Ref<const ArrayXd>& wt)
{
    using boost::math::digamma;
    const double dig_th = digamma(theta);
    double s = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double yi = y[i], lam = lambda[i], wi = wt[i];
        const double tlam = theta + lam;
        const double lam_over_tlam = lam / tlam;             // 1 - c
        // log_c = log(theta/tlam); use log1p when lambda << theta.
        const double log_c = (lam < 1e-3 * theta)
                              ? -std::log1p(lam / theta)
                              : std::log(theta / tlam);
        const double log_p0 = theta * log_c;
        // Standard NB2 score row: digamma(y+t) - digamma(t) + log_c + (lam - y)/tlam
        const double base = digamma(yi + theta) - dig_th + log_c + (lam - yi) / tlam;

        // Truncation correction: + p_0/(1 - p_0) * [log_c + lam/tlam]
        // Compute p_0/(1 - p_0) = exp(log_p0) / (-expm1(log_p0)) for stability.
        const double inner = log_c + lam_over_tlam;
        double ratio;
        if (log_p0 < -50.0) {
            ratio = std::exp(log_p0);  // negligible truncation
        } else {
            const double q = -std::expm1(log_p0);
            ratio = (q > 1e-300) ? std::exp(log_p0) / q : 1e300;
        }
        s += wi * (base + ratio * inner);
    }
    return s;
}

// Bracket theta around theta0 by multiplicative expansion.  Same shape as
// nb::bracket_theta but for the truncated score.
inline std::pair<double, double>
bracket_trunc_theta(double theta0,
                    const Eigen::Ref<const ArrayXd>& y,
                    const Eigen::Ref<const ArrayXd>& lambda,
                    const Eigen::Ref<const ArrayXd>& wt,
                    double theta_lo = 1e-6, double theta_hi = 1e8,
                    int max_steps = 60)
{
    if (theta0 < theta_lo) theta0 = theta_lo;
    if (theta0 > theta_hi) theta0 = theta_hi;
    double s0 = trunc_nb_score_theta(theta0, y, lambda, wt);
    if (s0 == 0.0) return {theta0, theta0};

    double a = theta0, b = theta0;
    if (s0 > 0) {
        for (int i = 0; i < max_steps; ++i) {
            b *= 2.0;
            if (b > theta_hi) { b = theta_hi; break; }
            if (trunc_nb_score_theta(b, y, lambda, wt) < 0) return {a, b};
        }
        return {a, theta_hi};
    } else {
        for (int i = 0; i < max_steps; ++i) {
            a *= 0.5;
            if (a < theta_lo) { a = theta_lo; break; }
            if (trunc_nb_score_theta(a, y, lambda, wt) > 0) return {a, b};
        }
        return {theta_lo, b};
    }
}

// Brent root-finder on the truncated NB theta score.  Mirrors nb::mle_theta.
inline double mle_trunc_theta(double theta_init,
                              const Eigen::Ref<const ArrayXd>& y,
                              const Eigen::Ref<const ArrayXd>& lambda,
                              const Eigen::Ref<const ArrayXd>& wt,
                              double tol = 1e-8, int maxit = 100,
                              double theta_lo = 1e-6, double theta_hi = 1e8)
{
    auto br = bracket_trunc_theta(theta_init, y, lambda, wt, theta_lo, theta_hi);
    double a = br.first, b = br.second;
    double sa = trunc_nb_score_theta(a, y, lambda, wt);
    double sb = trunc_nb_score_theta(b, y, lambda, wt);
    if (sa == 0.0) return a;
    if (sb == 0.0) return b;
    if (sa * sb > 0) return (std::fabs(sa) < std::fabs(sb)) ? a : b;

    double c = a, sc = sa, d = b - a, e = d;
    for (int iter = 0; iter < maxit; ++iter) {
        if (sb * sc > 0) { c = a; sc = sa; d = b - a; e = d; }
        if (std::fabs(sc) < std::fabs(sb)) {
            a = b; b = c; c = a;
            sa = sb; sb = sc; sc = sa;
        }
        const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::fabs(b) + 0.5 * tol;
        const double xm   = 0.5 * (c - b);
        if (std::fabs(xm) <= tol1 || sb == 0.0) return b;

        if (std::fabs(e) >= tol1 && std::fabs(sa) > std::fabs(sb)) {
            double s = sb / sa, p, q, r;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = sa / sc;
                r = sb / sc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0) q = -q;
            p = std::fabs(p);
            const double min1 = 3.0 * xm * q - std::fabs(tol1 * q);
            const double min2 = std::fabs(e * q);
            if (2.0 * p < std::min(min1, min2)) { e = d; d = p / q; }
            else                                { d = xm; e = d; }
        } else {
            d = xm; e = d;
        }
        a  = b;  sa = sb;
        b += (std::fabs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
        sb = trunc_nb_score_theta(b, y, lambda, wt);
    }
    return b;
}

// Information for theta via numerical differentiation of the score; used
// only for SE.theta reporting.  Step h is chosen relative to theta.
inline double info_trunc_theta(double theta,
                               const Eigen::Ref<const ArrayXd>& y,
                               const Eigen::Ref<const ArrayXd>& lambda,
                               const Eigen::Ref<const ArrayXd>& wt)
{
    const double h  = std::max(1e-5, 1e-5 * theta);
    const double sp = trunc_nb_score_theta(theta + h, y, lambda, wt);
    const double sm = trunc_nb_score_theta(theta - h, y, lambda, wt);
    // info = -ds/dtheta
    return -(sp - sm) / (2.0 * h);
}

// ---------------------------------------------------------------------------
// IRLS driver for zero-truncated count (Poisson if !is_negbin, NB2 otherwise).
// ---------------------------------------------------------------------------

inline TruncFitResult fit_trunc_count(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXd>& wt,
    const Eigen::Ref<const VectorXd>& offset,
    bool   is_negbin,
    double theta,
    VectorXd beta_init,
    double tol = 1e-8,
    int    maxit = 100)
{
    using boost::math::lgamma;

    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();

    VectorXd beta = beta_init;
    VectorXd eta(n), lambda_vec(n), muT(n), dmuT(n);
    ArrayXd  c_vec   = ArrayXd::Ones(n);
    ArrayXd  log_q_v = ArrayXd::Zero(n);

    ArrayXd  wls_w_sqrt(n);
    VectorXd wls_z(n), wls_wz(n);
    MatrixXd WX(n, p), XtWX(p, p);
    Eigen::LLT<MatrixXd> Ch;

    bool converged = false;
    int iter = 0;
    VectorXd beta_prev(p);

    auto refresh = [&]() {
        eta.noalias() = X * beta + offset;
        for (Eigen::Index i = 0; i < n; ++i) {
            const double lam = std::isfinite(eta[i]) ? std::exp(eta[i]) : 1e300;
            lambda_vec[i] = lam;
            if (!is_negbin) {
                double mt, dm, lq;
                eval_pois(lam, mt, dm, lq);
                muT[i]     = mt;
                dmuT[i]    = dm;
                c_vec[i]   = 1.0;
                log_q_v[i] = lq;
            } else {
                double mt, dm, c_, lq, lp0;
                eval_nb(lam, theta, mt, dm, c_, lq, lp0);
                muT[i]     = mt;
                dmuT[i]    = dm;
                c_vec[i]   = c_;
                log_q_v[i] = lq;
            }
        }
    };

    refresh();

    for (iter = 0; iter < maxit; ++iter) {
        beta_prev = beta;

        // Working IRLS:  w_i = wt_i * c_i * dmu_T/deta_i  (= info per obs in eta);
        //                z_i = (eta_i - offset_i) + (y_i - mu_T_i) / dmu_T/deta_i.
        for (Eigen::Index i = 0; i < n; ++i) {
            double dm = dmuT[i];
            if (dm < 1e-12) dm = 1e-12;
            double wi = wt[i] * c_vec[i] * dm;
            if (wi < 0.0 || !std::isfinite(wi)) wi = 0.0;
            wls_w_sqrt[i] = std::sqrt(wi);
            wls_z[i]      = (eta[i] - offset[i]) + (y[i] - muT[i]) / dm;
        }

        WX.noalias() = wls_w_sqrt.matrix().asDiagonal() * X;
        XtWX.setZero();
        XtWX.template selfadjointView<Eigen::Lower>().rankUpdate(WX.adjoint());
        Ch.compute(XtWX.template selfadjointView<Eigen::Lower>());
        if (Ch.info() != Eigen::Success) break;

        wls_wz.noalias() = wls_w_sqrt.matrix().cwiseProduct(wls_z);
        beta = Ch.solve(WX.adjoint() * wls_wz);

        refresh();

        const double db = (beta - beta_prev).cwiseAbs().maxCoeff();
        if (db < tol) { converged = true; break; }
    }

    // ---- log-likelihood ------------------------------------------------------
    double loglik = 0.0;
    if (!is_negbin) {
        // log f_T(y; lambda) = y*log(lambda) - lambda - log_q - lgamma(y+1)
        for (Eigen::Index i = 0; i < n; ++i) {
            const double lam = lambda_vec[i], yi = y[i];
            const double log_lam = std::log(lam > 1e-300 ? lam : 1e-300);
            loglik += wt[i] * (yi * log_lam - lam - log_q_v[i] - lgamma(yi + 1.0));
        }
    } else {
        const double lg_th = lgamma(theta);
        for (Eigen::Index i = 0; i < n; ++i) {
            const double lam = lambda_vec[i], yi = y[i];
            const double tlam = theta + lam;
            const double log_c = (lam < 1e-3 * theta)
                                  ? -std::log1p(lam / theta)
                                  : std::log(theta / tlam);
            const double log_p0   = theta * log_c;
            const double y_part   = (yi > 0.0)
                                     ? yi * (std::log(lam > 1e-300 ? lam : 1e-300) - std::log(tlam))
                                     : 0.0;
            loglik += wt[i] * (lgamma(yi + theta) - lg_th - lgamma(yi + 1.0)
                              + log_p0 + y_part - log_q_v[i]);
        }
    }

    // ---- vcov / se -----------------------------------------------------------
    VectorXd se = VectorXd::Zero(p);
    MatrixXd vcov = MatrixXd::Zero(p, p);
    if (Ch.info() == Eigen::Success) {
        vcov = Ch.solve(MatrixXd::Identity(p, p));
        for (Eigen::Index j = 0; j < p; ++j)
            se[j] = (vcov(j, j) >= 0.0) ? std::sqrt(vcov(j, j))
                                         : std::numeric_limits<double>::quiet_NaN();
    }

    TruncFitResult res;
    res.beta      = beta;
    res.eta       = eta;
    res.mu_T      = muT;
    res.lambda    = lambda_vec;
    res.se        = se;
    res.vcov      = vcov;
    res.loglik    = loglik;
    res.iter      = iter + 1;
    res.converged = converged;
    return res;
}

inline TruncFitResult fit_trunc_pois_log(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXd>& wt,
    const Eigen::Ref<const VectorXd>& offset,
    VectorXd beta_init,
    double tol = 1e-8, int maxit = 100)
{
    return fit_trunc_count(X, y, wt, offset, /*is_negbin=*/false, /*theta=*/0.0,
                           beta_init, tol, maxit);
}

inline TruncFitResult fit_trunc_nb_log(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXd>& wt,
    const Eigen::Ref<const VectorXd>& offset,
    double theta,
    VectorXd beta_init,
    double tol = 1e-8, int maxit = 100)
{
    return fit_trunc_count(X, y, wt, offset, /*is_negbin=*/true, theta,
                           beta_init, tol, maxit);
}

// ---------------------------------------------------------------------------
// Joint (beta, theta) MLE for zero-truncated NB2 with log link.
// Alternates beta-IRLS (at fixed theta) with a Brent-1D MLE on the truncated
// theta score.  Same outer loop structure as fglm::nb::mle_theta + glm.
// ---------------------------------------------------------------------------

struct TruncNbJointResult {
    TruncFitResult inner;     // final IRLS result (beta, etc.)
    double         theta;
    double         se_theta;
    int            outer_iter;
    int            theta_iter;
    bool           outer_converged;
};

inline TruncNbJointResult fit_trunc_nb_joint(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXd>& wt,
    const Eigen::Ref<const VectorXd>& offset,
    double init_theta,
    VectorXd beta_init,
    double tol = 1e-8, int maxit = 100,
    double theta_tol = 1e-8, int theta_maxit = 100,
    double outer_tol = 1e-7, int outer_maxit = 50)
{
    TruncFitResult inner = fit_trunc_pois_log(X, y, wt, offset, beta_init, tol, maxit);
    double theta = (init_theta > 0.0 && std::isfinite(init_theta)) ? init_theta
                   : fglm::nb::init_theta_mom(ArrayXd(y.array()),
                                              ArrayXd(inner.mu_T.array().max(1e-3)),
                                              ArrayXd(wt.array()));
    if (theta <= 0.0) theta = 1.0;

    int outer_iter = 0, theta_iter = 0;
    bool outer_conv = false;
    double theta_prev = theta;
    VectorXd beta_prev = inner.beta;

    for (outer_iter = 0; outer_iter < outer_maxit; ++outer_iter) {
        // beta-step: NB IRLS at fixed theta, warm-started from inner.beta.
        inner = fit_trunc_nb_log(X, y, wt, offset, theta, inner.beta, tol, maxit);

        // theta-step: 1-D MLE on truncated NB score.
        ArrayXd y_arr   = y.array();
        ArrayXd lam_arr = inner.lambda.array().max(1e-12);
        ArrayXd wt_arr  = wt.array();

        theta_prev = theta;
        theta = mle_trunc_theta(theta, y_arr, lam_arr, wt_arr,
                                theta_tol, theta_maxit);
        ++theta_iter;

        const double db = (inner.beta - beta_prev).cwiseAbs().maxCoeff();
        const double dt = std::fabs(theta - theta_prev) / std::max(theta, 1e-12);
        if (db + dt < outer_tol && outer_iter > 0) {
            outer_conv = true;
            // One last beta-IRLS pass at the new theta so vcov and SE
            // reflect the converged value.
            inner = fit_trunc_nb_log(X, y, wt, offset, theta, inner.beta, tol, maxit);
            break;
        }
        beta_prev = inner.beta;
    }

    // SE.theta from numerical info at the joint MLE.
    ArrayXd y_arr   = y.array();
    ArrayXd lam_arr = inner.lambda.array().max(1e-12);
    ArrayXd wt_arr  = wt.array();
    const double info_th = info_trunc_theta(theta, y_arr, lam_arr, wt_arr);
    const double se_th   = (info_th > 0.0)
                             ? std::sqrt(1.0 / info_th)
                             : std::numeric_limits<double>::quiet_NaN();

    TruncNbJointResult res;
    res.inner            = inner;
    res.theta            = theta;
    res.se_theta         = se_th;
    res.outer_iter       = outer_iter + 1;
    res.theta_iter       = theta_iter;
    res.outer_converged  = outer_conv && inner.converged;
    return res;
}

}}  // namespace fglm::trunc

#endif // FASTGLM_TRUNC_COUNT_H
