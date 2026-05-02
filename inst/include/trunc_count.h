#ifndef FASTGLM_TRUNC_COUNT_H
#define FASTGLM_TRUNC_COUNT_H
//
// Zero-truncated count regression (Poisson and NB2) via Fisher scoring IRLS.
// Used inside the hurdle driver (Phase 4) and also exposed for direct use
// where a positive-only count regression is the model of interest.
//
// All numerical work is in C++; no R callbacks.  Parameterization is in
// terms of the underlying (untruncated) rate lambda = exp(X*beta + offset).
// The conditional mean E[Y|Y>0] = mu_T is what the score uses; the Fisher
// information per observation is computed in closed form.
//
// Structure mirrors the main `glm` class in glm.h: the IRLS loop is the
// inherited GlmBase::solve() driver, and the truncated-count specifics are
// localized to overrides of update_var_mu / update_mu_eta / update_z /
// update_w / update_eta / update_mu / update_dev_resids / solve_wls /
// save_vcov / save_se.  Kept on the LLT (type=2) path: the truncated
// likelihood is convex on the rate parameterization and full-rank dense
// problems are the only consumer (zero-truncated counts coming from a
// hurdle subset).
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
#include "glm_base.h"
#include "nb_theta.h"
#include <cmath>
#include <limits>
#include <utility>

namespace fglm { namespace trunc {

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;
using Eigen::Lower;

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
// glm_trunc -- IRLS solver for zero-truncated count regression.
//
// Plugs the truncated-Poisson / truncated-NB family into the GlmBase IRLS
// driver inherited from glm_base.h, exactly the way the main `glm` class
// in glm.h plugs the standard exponential-family kernels in.  The driver
// loop in GlmBase::solve() calls (in order, per iteration):
//   update_var_mu, update_mu_eta, update_z, update_w, solve_wls,
//   update_eta, update_mu, update_dev_resids, run_step_halving.
// then save_se / save_vcov on exit.
//
// The solver carries six working vectors of length n:
//   eta          linear predictor in the rate parameterization
//   mu           E[Y|Y>0] (mu_T)
//   mu_eta       d mu_T / d eta
//   var_mu       (d mu_T / d eta) / c        (so the standard GLM weight
//                 formula w = wts * mu_eta^2 / var_mu reduces to the
//                 closed-form per-observation Fisher info wts * c * dmuT)
//   lambda_      exp(eta) cached for log-lik computation
//   c_vec_       theta/(theta+lambda) for NB; 1 for Poisson
//   log_q_       log(1 - p_0) cached for log-lik computation
// ---------------------------------------------------------------------------
class glm_trunc : public GlmBase<VectorXd, MatrixXd>
{
protected:
    const Eigen::Ref<const MatrixXd>& X_;
    const Eigen::Ref<const VectorXd>& Y_;
    const Eigen::Ref<const VectorXd>& weights_;
    const Eigen::Ref<const VectorXd>& offset_;

    bool   is_negbin_;
    double theta_;

    // Per-observation scratch refreshed by update_mu().  c_vec_ and log_q_
    // are needed by update_var_mu() (next iteration) and compute_loglik()
    // respectively; lambda_ is needed by both compute_loglik() and the
    // joint-theta driver to refit theta at the current fitted lambdas.
    VectorXd lambda_;
    VectorXd c_vec_;
    VectorXd log_q_;

    // LLT (type = 2) WLS workspace -- preallocated like glm.h.
    MatrixXd WX_;
    VectorXd wz_;
    MatrixXd XtWX_buf_;
    LLT<MatrixXd> Ch_;
    bool ch_ok_;

    // Refresh lambda / mu_T / dmu_T-deta / c / log_q from the current eta.
    // Called once in init_parms() and at the bottom of every IRLS iteration
    // by update_mu().  Stores mu_T into the inherited `mu` slot and
    // dmu_T/deta into a scratch that update_mu_eta() copies into `mu_eta`.
    void refresh_kernels()
    {
        for (int i = 0; i < nobs; ++i) {
            const double e_i = eta[i];
            const double lam = std::isfinite(e_i) ? std::exp(e_i) : 1e300;
            lambda_[i] = lam;
            if (!is_negbin_) {
                double mt, dm, lq;
                eval_pois(lam, mt, dm, lq);
                mu[i]      = mt;
                mu_eta[i]  = dm;
                c_vec_[i]  = 1.0;
                log_q_[i]  = lq;
            } else {
                double mt, dm, ci, lq, lp0;
                eval_nb(lam, theta_, mt, dm, ci, lq, lp0);
                mu[i]      = mt;
                mu_eta[i]  = dm;
                c_vec_[i]  = ci;
                log_q_[i]  = lq;
            }
        }
    }

    // ---------------- GlmBase virtual hooks --------------------------------

    virtual void update_eta() override
    {
        eta.noalias() = X_ * beta + offset_;
    }

    virtual void update_mu() override
    {
        // refresh_kernels writes mu (= mu_T) and mu_eta (= dmu_T/deta) plus
        // the lambda_ / c_vec_ / log_q_ scratch used by other hooks.
        refresh_kernels();
    }

    virtual void update_mu_eta() override
    {
        // already populated by refresh_kernels(); nothing to do.
    }

    virtual void update_var_mu() override
    {
        // Choose V(mu_T) so that the inherited update_w() formula
        //   w_i^2 = wts_i * mu_eta_i^2 / var_mu_i
        // collapses to the closed-form Fisher info per observation
        //   I_i = wts_i * c_i * (dmu_T/deta)_i.
        // I.e. var_mu = mu_eta / c.
        for (int i = 0; i < nobs; ++i) {
            double dm = mu_eta[i] < 1e-300 ? 1e-300 : mu_eta[i];
            double ci = c_vec_[i] < 1e-300 ? 1e-300 : c_vec_[i];
            var_mu[i] = dm / ci;
        }
    }

    virtual void update_z() override
    {
        // Standard IRLS working response on the rate-parameter scale.
        z = (eta - offset_).array() + (Y_ - mu).array() / mu_eta.array();
    }

    virtual void update_w() override
    {
        // sqrt(weights * mu_eta^2 / var_mu) = sqrt(weights * c * mu_eta).
        w = (weights_.array() * mu_eta.array().square() / var_mu.array()).sqrt();
    }

    // The truncated-count IRLS converges on the coefficient step rather than
    // the deviance: deviance is expensive (lgamma per obs) and we don't need
    // it for monotonicity checking because Fisher scoring on a convex log-
    // likelihood is descent for sufficiently small steps.  We still return
    // a finite/non-finite signal so run_step_halving can recover blow-ups.
    virtual void update_dev_resids() override
    {
        devold = dev;
        dev = mu.allFinite() && lambda_.allFinite() ? 0.0
                                                    : std::numeric_limits<double>::infinity();
    }

    virtual void update_dev_resids_dont_update_old() override
    {
        dev = mu.allFinite() && lambda_.allFinite() ? 0.0
                                                    : std::numeric_limits<double>::infinity();
    }

    virtual void step_halve() override
    {
        beta = 0.5 * (beta.array() + beta_prev.array());
        update_eta();
        update_mu();
    }

    virtual void run_step_halving(int &iterr) override
    {
        // Only recover from non-finite eta -> non-finite mu / lambda.  No
        // monotonicity halving (Fisher scoring is monotone here, and the
        // true deviance is expensive to compute).
        (void) iterr;
        if (std::isinf(dev)) {
            int itrr = 0;
            while (std::isinf(dev) && itrr < maxit) {
                ++itrr;
                step_halve();
                update_dev_resids_dont_update_old();
            }
        }
    }

    // ||Δβ||_∞ < tol -- matches the prior trunc IRLS criterion.
    virtual bool converged() override
    {
        return (beta - beta_prev).cwiseAbs().maxCoeff() < tol;
    }

    // LLT-only WLS; the full glm.h dispatch table is overkill for the
    // truncated-count consumer, which only ever asks for type=2.
    virtual void solve_wls(int /*iter*/) override
    {
        beta_prev = beta;

        WX_.noalias() = w.asDiagonal() * X_;
        XtWX_buf_.setZero();
        XtWX_buf_.template selfadjointView<Lower>().rankUpdate(WX_.adjoint());
        Ch_.compute(XtWX_buf_.template selfadjointView<Lower>());
        ch_ok_ = (Ch_.info() == Eigen::Success);
        if (!ch_ok_) {
            // Mark the iterate as infeasible so step-halving / outer caller
            // can react.  Leaves beta unchanged.
            dev = std::numeric_limits<double>::infinity();
            return;
        }

        wz_.noalias() = w.cwiseProduct(z);
        beta = Ch_.solve(WX_.adjoint() * wz_);
    }

    virtual void save_vcov() override
    {
        if (ch_ok_) {
            vcov = Ch_.solve(MatrixXd::Identity(nvars, nvars));
        } else {
            vcov.setConstant(std::numeric_limits<double>::quiet_NaN());
        }
    }

    virtual void save_se() override
    {
        if (ch_ok_) {
            se = Ch_.matrixL().solve(MatrixXd::Identity(nvars, nvars)).colwise().norm();
        } else {
            se.setConstant(std::numeric_limits<double>::quiet_NaN());
        }
    }

public:
    glm_trunc(const Eigen::Ref<const MatrixXd>& X,
              const Eigen::Ref<const VectorXd>& Y,
              const Eigen::Ref<const VectorXd>& weights,
              const Eigen::Ref<const VectorXd>& offset,
              bool is_negbin, double theta,
              double tol_ = 1e-8, int maxit_ = 100) :
        GlmBase<VectorXd, MatrixXd>((int)X.rows(), (int)X.cols(), tol_, maxit_),
        X_(X), Y_(Y), weights_(weights), offset_(offset),
        is_negbin_(is_negbin), theta_(theta),
        lambda_(X.rows()), c_vec_(X.rows()), log_q_(X.rows()),
        WX_(X.rows(), X.cols()), wz_(X.rows()),
        XtWX_buf_(X.cols(), X.cols()),
        ch_ok_(false)
    {}

    // Initialize from a starting beta.  We compute eta / mu / kernels
    // ourselves rather than accepting them from the caller (the caller
    // doesn't know mu_T / dmu_T-deta).
    void init_from_beta(const VectorXd& start)
    {
        beta = start;
        beta_prev = start;          // first converged() check shouldn't fire
        update_eta();
        refresh_kernels();          // populates mu, mu_eta, lambda_, c_vec_, log_q_
        dev = 0.0;
        devold = std::numeric_limits<double>::infinity();
    }

    // Allow the joint (beta, theta) outer loop to re-fit at a new theta
    // without reconstructing the solver -- mirrors glm::set_fam_params.
    void set_theta(double t) { theta_ = t; }

    // Cached scratch for downstream consumers (joint NB driver, hurdle
    // result struct).
    const VectorXd& get_lambda() const { return lambda_; }
    const VectorXd& get_log_q()  const { return log_q_; }
    const VectorXd& get_mu_T()   const { return mu; }
    bool   get_chol_ok()         const { return ch_ok_; }

    // Closed-form log-likelihood at the current (beta, theta).  Computed
    // once after solve(); during IRLS we use the cheap dev proxy.
    double compute_loglik() const
    {
        using boost::math::lgamma;
        double ll = 0.0;
        if (!is_negbin_) {
            for (int i = 0; i < nobs; ++i) {
                const double lam = lambda_[i], yi = Y_[i];
                const double log_lam = std::log(lam > 1e-300 ? lam : 1e-300);
                ll += weights_[i] * (yi * log_lam - lam - log_q_[i] - lgamma(yi + 1.0));
            }
        } else {
            const double lg_th = lgamma(theta_);
            for (int i = 0; i < nobs; ++i) {
                const double lam = lambda_[i], yi = Y_[i];
                const double tlam = theta_ + lam;
                const double log_c = (lam < 1e-3 * theta_)
                                      ? -std::log1p(lam / theta_)
                                      : std::log(theta_ / tlam);
                const double log_p0 = theta_ * log_c;
                const double y_part = (yi > 0.0)
                                       ? yi * (std::log(lam > 1e-300 ? lam : 1e-300) - std::log(tlam))
                                       : 0.0;
                ll += weights_[i] * (lgamma(yi + theta_) - lg_th - lgamma(yi + 1.0)
                                     + log_p0 + y_part - log_q_[i]);
            }
        }
        return ll;
    }
};

// ---------------------------------------------------------------------------
// Free-function wrappers -- preserved for callers (fit_glm_hurdle.cpp).
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
    glm_trunc solver(X, y, wt, offset, is_negbin, theta, tol, maxit);
    solver.init_from_beta(beta_init);
    int iters = solver.solve(maxit);

    TruncFitResult res;
    res.beta      = solver.get_beta();
    res.eta       = solver.get_eta();
    res.mu_T      = solver.get_mu_T();
    res.lambda    = solver.get_lambda();
    res.se        = solver.get_se();
    res.vcov      = solver.get_vcov();
    res.loglik    = solver.compute_loglik();
    res.iter      = iters;
    res.converged = solver.get_converged();
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
// theta score.  The solver state is reused across outer iterations via
// set_theta + init_from_beta(warm start), which avoids reallocating WX /
// XtWX_buf and lets every theta update warm-start from the previous beta.
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
    // Pilot fit: zero-truncated Poisson, used to seed beta and (if needed)
    // method-of-moments theta.
    glm_trunc solver(X, y, wt, offset, /*is_negbin=*/false, /*theta=*/0.0,
                     tol, maxit);
    solver.init_from_beta(beta_init);
    int iters = solver.solve(maxit);
    bool inner_conv = solver.get_converged();
    VectorXd beta_curr = solver.get_beta();
    VectorXd lam_curr  = solver.get_lambda();

    double theta = (init_theta > 0.0 && std::isfinite(init_theta))
                    ? init_theta
                    : fglm::nb::init_theta_mom(ArrayXd(y.array()),
                                               ArrayXd(lam_curr.array().max(1e-3)),
                                               ArrayXd(wt.array()));
    if (theta <= 0.0) theta = 1.0;

    int outer_iter = 0, theta_iter = 0;
    bool outer_conv = false;
    double theta_prev = theta;
    VectorXd beta_prev = beta_curr;

    // is_negbin_ is set at construction, so we use a separate NB solver for
    // the joint loop.  beta is warm-started from the Poisson pilot above and
    // theta is updated in-place between outer iterations.
    glm_trunc nb_solver(X, y, wt, offset, /*is_negbin=*/true, theta, tol, maxit);

    for (outer_iter = 0; outer_iter < outer_maxit; ++outer_iter) {
        // beta-step at fixed theta, warm-started from beta_curr.
        nb_solver.set_theta(theta);
        nb_solver.init_from_beta(beta_curr);
        iters     = nb_solver.solve(maxit);
        inner_conv = nb_solver.get_converged();
        beta_curr  = nb_solver.get_beta();
        lam_curr   = nb_solver.get_lambda();

        // theta-step: 1-D MLE on truncated NB score.
        ArrayXd y_arr   = y.array();
        ArrayXd lam_arr = lam_curr.array().max(1e-12);
        ArrayXd wt_arr  = wt.array();

        theta_prev = theta;
        theta = mle_trunc_theta(theta, y_arr, lam_arr, wt_arr,
                                theta_tol, theta_maxit);
        ++theta_iter;

        const double db = (beta_curr - beta_prev).cwiseAbs().maxCoeff();
        const double dt = std::fabs(theta - theta_prev) / std::max(theta, 1e-12);
        if (db + dt < outer_tol && outer_iter > 0) {
            // One last beta-IRLS pass at the converged theta so vcov / se
            // reflect the final value.
            nb_solver.set_theta(theta);
            nb_solver.init_from_beta(beta_curr);
            iters     = nb_solver.solve(maxit);
            inner_conv = nb_solver.get_converged();
            beta_curr  = nb_solver.get_beta();
            lam_curr   = nb_solver.get_lambda();
            outer_conv = inner_conv;
            break;
        }
        beta_prev = beta_curr;
    }

    // SE.theta from numerical info at the joint MLE.
    ArrayXd y_arr   = y.array();
    ArrayXd lam_arr = lam_curr.array().max(1e-12);
    ArrayXd wt_arr  = wt.array();
    const double info_th = info_trunc_theta(theta, y_arr, lam_arr, wt_arr);
    const double se_th   = (info_th > 0.0)
                             ? std::sqrt(1.0 / info_th)
                             : std::numeric_limits<double>::quiet_NaN();

    TruncFitResult inner;
    inner.beta      = beta_curr;
    inner.eta       = nb_solver.get_eta();
    inner.mu_T      = nb_solver.get_mu_T();
    inner.lambda    = lam_curr;
    inner.se        = nb_solver.get_se();
    inner.vcov      = nb_solver.get_vcov();
    inner.loglik    = nb_solver.compute_loglik();
    inner.iter      = iters;
    inner.converged = inner_conv;

    TruncNbJointResult res;
    res.inner            = inner;
    res.theta            = theta;
    res.se_theta         = se_th;
    res.outer_iter       = outer_iter + 1;
    res.theta_iter       = theta_iter;
    res.outer_converged  = outer_conv;
    return res;
}

}}  // namespace fglm::trunc

#endif // FASTGLM_TRUNC_COUNT_H
