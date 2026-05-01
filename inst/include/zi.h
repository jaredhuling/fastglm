#ifndef FASTGLM_ZI_H
#define FASTGLM_ZI_H
//
// Helper kernels for zero-inflated count regression (Poisson or NB2 with
// log link, zero-inflation logit/probit/cloglog/log).  All numerical work
// is in C++ -- the EM loop in src/fit_glm_zi.cpp uses these helpers and
// the existing glm class for the M-step IRLS.
//
// Notation:
//   pi_i = P(Z_i = 1) = inflation probability (zero-inflation latent)
//   mu_i = E[Y_i | Z_i = 0] = exp(eta_count_i)  (Poisson rate / NB mean)
//   f_0_i = P(Y_i = 0 | Z_i = 0)  -- count-component prob of zero
//   tau_i = P(Z_i = 1 | Y_i)
//
// Likelihood:
//   y_i > 0:  ell_i = log(1 - pi_i) + log f(y_i; mu_i)
//   y_i = 0:  ell_i = log( pi_i + (1 - pi_i) * f_0_i )
//
// Score (closed form, holds for any params -- not just MLE):
//   d ell / d eta_z_i = tau_i - pi_i           (zero link only -- logit case;
//                                                generic links use d pi/d eta)
//   d ell / d beta_k  = (1 - tau_i) * d log f / d beta_k
//   d ell / d theta   = (1 - tau_i) * d log f_NB / d theta  (NB only)
//

#include <RcppEigen.h>
#include <boost/math/special_functions/digamma.hpp>
#include "families.h"
#include <cmath>
#include <limits>

namespace fglm { namespace zi {

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// log(1 + exp(-|x|)) -- stable.
static inline double log1pe(double x) {
    if (x > 0) return x + std::log1p(std::exp(-x));
    return std::log1p(std::exp(x));
}

// log(pi_i) and log(1 - pi_i) from eta_z.  Stable for both signs.
inline void log_pi_and_log1m(int zero_fam_code, double eta_z,
                             double& log_pi, double& log_1m_pi)
{
    switch (zero_fam_code) {
    case FAM_BINOMIAL_LOGIT: {
        // log(sigma(x)) = -log(1 + exp(-x));  log(1 - sigma(x)) = -log(1 + exp(x))
        const double a  = log1pe(-eta_z);   // log(1 + exp(-eta))
        const double b  = log1pe( eta_z);   // log(1 + exp(eta))
        log_pi    = -a;                       // -log(1+exp(-eta))
        log_1m_pi = -b;                       // -log(1+exp(eta))
        break;
    }
    case FAM_BINOMIAL_PROBIT: {
        // pi = Phi(eta); use pnorm5 for tails.
        log_pi    = R::pnorm5(eta_z, 0.0, 1.0, /*lower=*/1, /*log=*/1);
        log_1m_pi = R::pnorm5(eta_z, 0.0, 1.0, /*lower=*/0, /*log=*/1);
        break;
    }
    case FAM_BINOMIAL_CLOGLOG: {
        // pi = 1 - exp(-exp(eta));  log(1 - pi) = -exp(eta)
        log_1m_pi = -std::exp(eta_z);
        // log_pi = log(1 - exp(log_1m_pi)) -- stable via log(-expm1(log_1m_pi))
        log_pi    = std::log(-std::expm1(log_1m_pi));
        break;
    }
    case FAM_BINOMIAL_LOG: {
        // pi = exp(eta);  must satisfy eta < 0.
        log_pi    = eta_z;
        log_1m_pi = std::log1p(-std::exp(eta_z));
        break;
    }
    default:
        log_pi    = 0.0;
        log_1m_pi = 0.0;
        break;
    }
}

// Count-side log f(y_i = 0 | mu_i) for the count component.
// dist_code: 0 = Poisson, 1 = NB2 (theta in fam_params).
inline double log_f_zero(int dist_code, double theta, double mu)
{
    if (dist_code == 0) {
        // Poisson: log P(Y = 0) = -mu
        return -mu;
    } else {
        // NB2: log P(Y = 0) = theta * log(theta / (theta + mu))
        // Use log1p when mu << theta:  -theta * log(1 + mu/theta).
        if (theta <= 0) return -std::numeric_limits<double>::infinity();
        if (mu < 1e-3 * theta) return -theta * std::log1p(mu / theta);
        return theta * std::log(theta / (theta + mu));
    }
}

// log f(y_i; mu_i, theta) for y_i > 0  (used for the observed log-lik).
inline double log_f_count(int dist_code, double theta, double y, double mu)
{
    using boost::math::lgamma;
    if (dist_code == 0) {
        // Poisson: log f(y) = y*log(mu) - mu - lgamma(y+1)
        const double log_mu = (mu > 1e-300) ? std::log(mu) : -1e300;
        return y * log_mu - mu - lgamma(y + 1.0);
    } else {
        if (theta <= 0) return -std::numeric_limits<double>::infinity();
        const double tlam = theta + mu;
        const double log_c = (mu < 1e-3 * theta)
                              ? -std::log1p(mu / theta)
                              : std::log(theta / tlam);
        const double log_mu_part = (y > 0)
                                   ? y * (std::log(mu > 1e-300 ? mu : 1e-300) - std::log(tlam))
                                   : 0.0;
        return lgamma(y + theta) - lgamma(theta) - lgamma(y + 1.0)
               + theta * log_c + log_mu_part;
    }
}

// Compute observed log-likelihood, posterior tau, and several useful
// per-observation quantities in one vectorized pass.  Used for both the
// EM convergence check and for analytical score computation.
//
// Inputs: eta_z, mu (count mean), theta, weights, y; dist_code, zero_fam_code.
// Outputs: log_lik (sum over i of w_i ell_i), tau (n).
inline double obs_loglik_and_tau(
    int                       dist_code,
    int                       zero_fam_code,
    double                    theta,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXd>& mu,
    const Eigen::Ref<const VectorXd>& eta_z,
    const Eigen::Ref<const VectorXd>& weights,
    Eigen::Ref<VectorXd>      tau)
{
    double ll = 0.0;
    const Eigen::Index n = y.size();
    for (Eigen::Index i = 0; i < n; ++i) {
        double log_pi, log_1m_pi;
        log_pi_and_log1m(zero_fam_code, eta_z[i], log_pi, log_1m_pi);
        if (y[i] > 0.0) {
            const double lf = log_f_count(dist_code, theta, y[i], mu[i]);
            ll += weights[i] * (log_1m_pi + lf);
            tau[i] = 0.0;
        } else {
            // y_i == 0: posterior tau = pi / (pi + (1-pi)*f_0)
            const double log_f0 = log_f_zero(dist_code, theta, mu[i]);
            // log_pi vs log_1m_pi + log_f0  -- logsumexp
            const double a = log_pi;
            const double b = log_1m_pi + log_f0;
            const double m = std::max(a, b);
            const double log_denom = m + std::log1p(std::exp(std::min(a, b) - m));
            ll += weights[i] * log_denom;
            tau[i] = std::exp(a - log_denom);
        }
    }
    return ll;
}

// Score wrt eta_z (per-observation):  s_z_i = tau_i - pi_i  for logit;
// for non-canonical zero links we have s_z_i = (tau_i - pi_i) * (d pi/d eta) / [pi(1-pi)]
// where the factor reduces to 1 for logit.  (Derivation as in standard GLM
// score for binomial: s = (y - pi) for canonical link, generally
// s = (y - pi) * d_pi / [pi(1-pi)] with pseudo-response y = tau_i.)
inline double score_eta_z(int zero_fam_code, double tau_i, double eta_z_i)
{
    switch (zero_fam_code) {
    case FAM_BINOMIAL_LOGIT: {
        // s = tau - pi
        const double pi = 1.0 / (1.0 + std::exp(-eta_z_i));
        return tau_i - pi;
    }
    case FAM_BINOMIAL_PROBIT: {
        const double pi = R::pnorm5(eta_z_i, 0.0, 1.0, 1, 0);
        const double dpi = R::dnorm4(eta_z_i, 0.0, 1.0, 0);
        const double denom = pi * (1.0 - pi);
        if (denom < 1e-300) return 0.0;
        return (tau_i - pi) * dpi / denom;
    }
    case FAM_BINOMIAL_CLOGLOG: {
        // pi = 1 - exp(-exp(eta));  d pi/d eta = exp(eta - exp(eta))
        const double e_eta = std::exp(eta_z_i);
        const double pi    = -std::expm1(-e_eta);
        const double dpi   = std::exp(eta_z_i - e_eta);
        const double denom = pi * (1.0 - pi);
        if (denom < 1e-300) return 0.0;
        return (tau_i - pi) * dpi / denom;
    }
    case FAM_BINOMIAL_LOG: {
        const double pi = std::exp(eta_z_i);
        const double dpi = pi;
        const double denom = pi * (1.0 - pi);
        if (denom < 1e-300) return 0.0;
        return (tau_i - pi) * dpi / denom;
    }
    default:
        return 0.0;
    }
}

// Score wrt eta_count (per-observation), i.e. d ell / d (X*beta + offset)_i.
// For Poisson (log link):  s_c_i = (1 - tau_i) * (y_i - mu_i)
// For NB2     (log link):  s_c_i = (1 - tau_i) * theta * (y_i - mu_i) / (theta + mu_i)
inline double score_eta_count(int dist_code, double theta,
                              double tau_i, double y_i, double mu_i)
{
    const double w = 1.0 - tau_i;
    if (dist_code == 0) {
        return w * (y_i - mu_i);
    }
    const double tm = theta + mu_i;
    if (tm < 1e-300) return 0.0;
    return w * theta * (y_i - mu_i) / tm;
}

// Score wrt theta (NB only):
//   d log f / d theta = digamma(y+t) - digamma(t) + log(t/(t+mu)) + (mu - y)/(t+mu)
inline double score_theta_one(double theta, double y_i, double mu_i)
{
    using boost::math::digamma;
    const double tm = theta + mu_i;
    const double log_c = (mu_i < 1e-3 * theta)
                          ? -std::log1p(mu_i / theta)
                          : std::log(theta / tm);
    return digamma(y_i + theta) - digamma(theta) + log_c + (mu_i - y_i) / tm;
}

}}  // namespace fglm::zi

#endif  // FASTGLM_ZI_H
