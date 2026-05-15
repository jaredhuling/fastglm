#ifndef FASTGLM_FAMILIES_H
#define FASTGLM_FAMILIES_H

#include <RcppEigen.h>
#include <cmath>
#include <limits>

namespace fglm {

// Family/link codes.  Mirror the R-side mapping in family_code().  -1 means
// "unknown / fall back to R callbacks".
enum FamilyCode {
    FAM_UNKNOWN              = -1,
    FAM_GAUSSIAN_IDENTITY    =  0,
    FAM_GAUSSIAN_LOG         =  1,
    FAM_GAUSSIAN_INVERSE     =  2,
    FAM_BINOMIAL_LOGIT       =  3,
    FAM_BINOMIAL_PROBIT      =  4,
    FAM_BINOMIAL_CLOGLOG     =  5,
    FAM_BINOMIAL_LOG         =  6,
    FAM_POISSON_LOG          =  7,
    FAM_POISSON_IDENTITY     =  8,
    FAM_POISSON_SQRT         =  9,
    FAM_GAMMA_LOG            = 10,
    FAM_GAMMA_INVERSE        = 11,
    FAM_GAMMA_IDENTITY       = 12,
    FAM_INVGAUSS_INVMU2      = 13,
    FAM_INVGAUSS_LOG         = 14,
    FAM_INVGAUSS_IDENTITY    = 15,
    FAM_INVGAUSS_INVERSE     = 16,
    // Negative binomial (NB2: variance = mu + mu^2/theta).  theta is carried
    // on FamilyParams.  Links match Poisson's set.
    FAM_NB_LOG               = 17,
    FAM_NB_SQRT              = 18,
    FAM_NB_IDENTITY          = 19,
    // Tweedie (variance = mu^var_power).  var_power on FamilyParams.
    FAM_TWEEDIE_LOG          = 20,
    FAM_TWEEDIE_IDENTITY     = 21,
    FAM_TWEEDIE_INVERSE      = 22,
    FAM_TWEEDIE_SQRT         = 23
};

// Parameters needed by some families (theta for NB, var_power for Tweedie,
// link_power reserved for Tweedie generic power links).  Defaults are inert
// for all params-free families.
struct FamilyParams {
    double theta      = 1.0;
    double var_power  = 0.0;
    double link_power = 0.0;
};

// Numerical floor used by R's family helpers.
static inline double thresh_eps() { return std::numeric_limits<double>::epsilon(); }
static inline double pmin1(double x) { return x > 1.0 ? 1.0 : x; }

// Logit link cutoffs.  These mirror R's `src/library/stats/src/family.c`
// (logit_linkinv / logit_mu_eta) constants exactly — THRESH = 30 and
// MTHRESH = -30, with the formula switching to DOUBLE_EPS / INVEPS at the
// boundaries.  Matching R is more important than tighter accuracy: under
// extreme weighting, IRLS endpoints are sensitive to small changes in the
// per-observation mu / mu_eta values, and disagreeing with R-callback
// fastglm by O(eps) makes converged log-likelihoods differ by O(1) on
// near-degenerate problems (cf. SEQTaRget regression test).
static constexpr double LOGIT_THRESH  =  30.0;
static constexpr double LOGIT_MTHRESH = -30.0;
static inline double logit_inveps() { return 1.0 / std::numeric_limits<double>::epsilon(); }

// y * log(y/mu)  with the standard 0*log(0) = 0 convention used by R.
static inline double y_log_y(double y, double mu) {
    return y > 0.0 ? y * std::log(y / mu) : 0.0;
}

// ---------------------------------------------------------------------------
// linkinv: mu = g^{-1}(eta)
// ---------------------------------------------------------------------------
inline void linkinv(int code,
                    const FamilyParams& /*params*/,
                    const Eigen::Ref<const Eigen::ArrayXd>& eta,
                    Eigen::Ref<Eigen::ArrayXd> mu)
{
    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_POISSON_IDENTITY:
    case FAM_GAMMA_IDENTITY:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_NB_IDENTITY:
    case FAM_TWEEDIE_IDENTITY:
        mu = eta; break;
    case FAM_GAUSSIAN_LOG:
    case FAM_POISSON_LOG:
    case FAM_GAMMA_LOG:
    case FAM_INVGAUSS_LOG:
    case FAM_BINOMIAL_LOG:
    case FAM_NB_LOG:
    case FAM_TWEEDIE_LOG:
        mu = eta.exp().max(thresh_eps()); break;
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
    case FAM_TWEEDIE_INVERSE:
        // Clamp to +/-1/eps to avoid Inf when eta is near zero.
        // valideta rejects eta==0, but near-zero values produce
        // mu so large that downstream mu^2 / mu^3 overflow.
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            double e = eta[i];
            double m = 1.0 / e;
            if (!std::isfinite(m)) m = (e >= 0.0 ? 1.0 : -1.0) / thresh_eps();
            mu[i] = m;
        }
        break;
    case FAM_BINOMIAL_LOGIT: {
        // Match R's logit_linkinv (src/library/stats/src/family.c) exactly:
        //   tmp = (eta < MTHRESH) ? DOUBLE_EPS
        //        : (eta > THRESH ) ? INVEPS
        //        : exp(eta)
        //   mu  = tmp / (1 + tmp)
        const double eps   = thresh_eps();
        const double inveps = logit_inveps();
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            const double e = eta[i];
            const double tmp = (e < LOGIT_MTHRESH) ? eps
                             : (e > LOGIT_THRESH ) ? inveps
                             : std::exp(e);
            mu[i] = tmp / (1.0 + tmp);
        }
        break;
    }
    case FAM_BINOMIAL_PROBIT: {
        const double thr = -Rf_qnorm5(thresh_eps(), 0.0, 1.0, 1, 0);
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            double e = eta[i];
            if (e < -thr) e = -thr;
            if (e >  thr) e =  thr;
            mu[i] = Rf_pnorm5(e, 0.0, 1.0, 1, 0);
        }
        break;
    }
    case FAM_BINOMIAL_CLOGLOG: {
        // Match R's cloglog_linkinv: -expm1(-exp(eta)), then clamp to
        // [DOUBLE_EPS, 1 - DOUBLE_EPS].  expm1 is mandatory for accuracy at
        // very negative eta where (1 - exp(-exp(eta))) catastrophically
        // cancels (off by ~1e-4 at eta = -30).
        const double eps = thresh_eps();
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            double v = -std::expm1(-std::exp(eta[i]));
            if (v < eps)             v = eps;
            else if (v > 1.0 - eps)  v = 1.0 - eps;
            mu[i] = v;
        }
        break;
    }
    case FAM_INVGAUSS_INVMU2:
        // mu = 1/sqrt(eta); clamp to avoid Inf when eta is near zero.
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            double m = 1.0 / std::sqrt(eta[i]);
            if (!std::isfinite(m)) m = 1.0 / thresh_eps();
            mu[i] = m;
        }
        break;
    case FAM_POISSON_SQRT:
    case FAM_NB_SQRT:
    case FAM_TWEEDIE_SQRT:
        mu = eta.square(); break;
    default:
        // Should not be called for FAM_UNKNOWN
        mu = eta;
        break;
    }
}

// ---------------------------------------------------------------------------
// mu_eta: dmu / deta
// ---------------------------------------------------------------------------
inline void mu_eta(int code,
                   const FamilyParams& /*params*/,
                   const Eigen::Ref<const Eigen::ArrayXd>& eta,
                   Eigen::Ref<Eigen::ArrayXd> dmu)
{
    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_POISSON_IDENTITY:
    case FAM_GAMMA_IDENTITY:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_NB_IDENTITY:
    case FAM_TWEEDIE_IDENTITY:
        dmu.setOnes(); break;
    case FAM_GAUSSIAN_LOG:
    case FAM_POISSON_LOG:
    case FAM_GAMMA_LOG:
    case FAM_INVGAUSS_LOG:
    case FAM_BINOMIAL_LOG:
    case FAM_NB_LOG:
    case FAM_TWEEDIE_LOG:
        dmu = eta.exp().max(thresh_eps()); break;
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
    case FAM_TWEEDIE_INVERSE:
        dmu = -1.0 / eta.square(); break;
    case FAM_BINOMIAL_LOGIT: {
        // Match R's logit_mu_eta (src/library/stats/src/family.c) exactly:
        //   if (eta > THRESH || eta < MTHRESH)  mu_eta = DOUBLE_EPS
        //   else                                mu_eta = exp(eta) / (1+exp(eta))^2
        const double eps = thresh_eps();
        for (Eigen::Index i = 0; i < eta.size(); ++i) {
            const double e = eta[i];
            if (e > LOGIT_THRESH || e < LOGIT_MTHRESH) {
                dmu[i] = eps;
            } else {
                const double ee = std::exp(e);
                const double op = 1.0 + ee;
                dmu[i] = ee / (op * op);
            }
        }
        break;
    }
    case FAM_BINOMIAL_PROBIT: {
        for (Eigen::Index i = 0; i < eta.size(); ++i)
            dmu[i] = std::max(Rf_dnorm4(eta[i], 0.0, 1.0, 0), thresh_eps());
        break;
    }
    case FAM_BINOMIAL_CLOGLOG: {
        Eigen::ArrayXd e = eta;
        for (Eigen::Index i = 0; i < e.size(); ++i) {
            if (e[i] < -700.0) e[i] = -700.0;
            if (e[i] >  700.0) e[i] =  700.0;
        }
        dmu = (e - e.exp()).exp();
        for (Eigen::Index i = 0; i < dmu.size(); ++i)
            if (dmu[i] < thresh_eps()) dmu[i] = thresh_eps();
        break;
    }
    case FAM_INVGAUSS_INVMU2:
        dmu = -1.0 / (2.0 * eta.pow(1.5)); break;
    case FAM_POISSON_SQRT:
    case FAM_NB_SQRT:
    case FAM_TWEEDIE_SQRT:
        dmu = 2.0 * eta; break;
    default:
        dmu.setOnes();
        break;
    }
}

// ---------------------------------------------------------------------------
// variance: V(mu)
// ---------------------------------------------------------------------------
inline void variance(int code,
                     const FamilyParams& params,
                     const Eigen::Ref<const Eigen::ArrayXd>& mu,
                     Eigen::Ref<Eigen::ArrayXd> v)
{
    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_GAUSSIAN_LOG:
    case FAM_GAUSSIAN_INVERSE:
        v.setOnes(); break;
    case FAM_BINOMIAL_LOGIT:
    case FAM_BINOMIAL_PROBIT:
    case FAM_BINOMIAL_CLOGLOG:
    case FAM_BINOMIAL_LOG:
        v = mu * (1.0 - mu); break;
    case FAM_POISSON_LOG:
    case FAM_POISSON_IDENTITY:
    case FAM_POISSON_SQRT:
        v = mu; break;
    case FAM_GAMMA_LOG:
    case FAM_GAMMA_INVERSE:
    case FAM_GAMMA_IDENTITY:
        v = mu.square(); break;
    case FAM_INVGAUSS_INVMU2:
    case FAM_INVGAUSS_LOG:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_INVGAUSS_INVERSE:
        v = mu.pow(3); break;
    case FAM_NB_LOG:
    case FAM_NB_SQRT:
    case FAM_NB_IDENTITY: {
        // V(mu) = mu + mu^2 / theta; for theta -> Inf this reduces to Poisson.
        const double inv_theta = (params.theta > 0.0) ? 1.0 / params.theta : 0.0;
        v = mu + mu.square() * inv_theta;
        break;
    }
    case FAM_TWEEDIE_LOG:
    case FAM_TWEEDIE_IDENTITY:
    case FAM_TWEEDIE_INVERSE:
    case FAM_TWEEDIE_SQRT:
        // V(mu) = mu^p
        v = mu.pow(params.var_power); break;
    default:
        v.setOnes();
        break;
    }
}

// ---------------------------------------------------------------------------
// dvariance: V'(mu) = dV/dmu, derivative of the variance function.
// Used in the general Firth / AS_mean adjustment for non-canonical links.
// ---------------------------------------------------------------------------
inline void dvariance(int code,
                      const FamilyParams& params,
                      const Eigen::Ref<const Eigen::ArrayXd>& mu,
                      Eigen::Ref<Eigen::ArrayXd> dv)
{
    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_GAUSSIAN_LOG:
    case FAM_GAUSSIAN_INVERSE:
        dv.setZero(); break;
    case FAM_BINOMIAL_LOGIT:
    case FAM_BINOMIAL_PROBIT:
    case FAM_BINOMIAL_CLOGLOG:
    case FAM_BINOMIAL_LOG:
        dv = 1.0 - 2.0 * mu; break;
    case FAM_POISSON_LOG:
    case FAM_POISSON_IDENTITY:
    case FAM_POISSON_SQRT:
        dv.setOnes(); break;
    case FAM_GAMMA_LOG:
    case FAM_GAMMA_INVERSE:
    case FAM_GAMMA_IDENTITY:
        dv = 2.0 * mu; break;
    case FAM_INVGAUSS_INVMU2:
    case FAM_INVGAUSS_LOG:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_INVGAUSS_INVERSE:
        dv = 3.0 * mu.square(); break;
    case FAM_NB_LOG:
    case FAM_NB_SQRT:
    case FAM_NB_IDENTITY: {
        const double inv_theta = (params.theta > 0.0) ? 1.0 / params.theta : 0.0;
        dv = 1.0 + 2.0 * mu * inv_theta;
        break;
    }
    case FAM_TWEEDIE_LOG:
    case FAM_TWEEDIE_IDENTITY:
    case FAM_TWEEDIE_INVERSE:
    case FAM_TWEEDIE_SQRT:
        dv = params.var_power * mu.pow(params.var_power - 1.0); break;
    default:
        dv.setZero();
        break;
    }
}

// ---------------------------------------------------------------------------
// dev_resids: per-observation deviance contribution.
//
// Returns the *sum* over all observations.  For binomial with prior weights,
// y is taken as a proportion (the standard glm convention after $initialize).
// ---------------------------------------------------------------------------
inline double dev_resids_sum(int code,
                             const FamilyParams& params,
                             const Eigen::Ref<const Eigen::ArrayXd>& y,
                             const Eigen::Ref<const Eigen::ArrayXd>& mu,
                             const Eigen::Ref<const Eigen::ArrayXd>& wt)
{
    const Eigen::Index n = y.size();
    double s = 0.0;

    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_GAUSSIAN_LOG:
    case FAM_GAUSSIAN_INVERSE:
        for (Eigen::Index i = 0; i < n; ++i) {
            double d = y[i] - mu[i];
            s += wt[i] * d * d;
        }
        break;

    case FAM_BINOMIAL_LOGIT:
    case FAM_BINOMIAL_PROBIT:
    case FAM_BINOMIAL_CLOGLOG:
    case FAM_BINOMIAL_LOG:
        for (Eigen::Index i = 0; i < n; ++i) {
            double yi = y[i], mi = mu[i];
            double a = (yi > 0.0)        ? yi * std::log(yi / mi)             : 0.0;
            double b = ((1.0 - yi) > 0.0) ? (1.0 - yi) * std::log((1.0 - yi) / (1.0 - mi)) : 0.0;
            s += 2.0 * wt[i] * (a + b);
        }
        break;

    case FAM_POISSON_LOG:
    case FAM_POISSON_IDENTITY:
    case FAM_POISSON_SQRT:
        for (Eigen::Index i = 0; i < n; ++i) {
            double yi = y[i], mi = mu[i];
            double a = (yi > 0.0) ? yi * std::log(yi / mi) : 0.0;
            s += 2.0 * wt[i] * (a - (yi - mi));
        }
        break;

    case FAM_GAMMA_LOG:
    case FAM_GAMMA_INVERSE:
    case FAM_GAMMA_IDENTITY:
        for (Eigen::Index i = 0; i < n; ++i) {
            double yi = y[i], mi = mu[i];
            // -2 * (log(ifelse(y == 0, 1, y/mu)) - (y - mu)/mu)
            double r = (yi == 0.0) ? 0.0 : std::log(yi / mi);
            s += -2.0 * wt[i] * (r - (yi - mi) / mi);
        }
        break;

    case FAM_INVGAUSS_INVMU2:
    case FAM_INVGAUSS_LOG:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_INVGAUSS_INVERSE: {
        // (y - mu)^2 / (y * mu^2) rewritten as ((y/mu - 1)^2) / y
        // to avoid mu^2 overflow.
        Eigen::ArrayXd r = y / mu - 1.0;
        s = (wt * r * r / y).sum();
        break;
    }

    case FAM_NB_LOG:
    case FAM_NB_SQRT:
    case FAM_NB_IDENTITY: {
        // Stable NB deviance avoiding cancellation for large mu.
        // d_i = 2*[y*log(y) - (y+t)*log(y+t) + y*log1p(t/mu) + t*log(mu+t)]
        // For y=0: y.max(1).log() = log(1) = 0, so the y*log(y) term vanishes.
        const double theta = params.theta;
        Eigen::ArrayXd yt  = y + theta;
        Eigen::ArrayXd mt  = mu + theta;
        Eigen::ArrayXd ysf = y.max(1.0);
        Eigen::ArrayXd d   = 2.0 * (y * ysf.log() - yt * yt.log()
                                     + y * (theta / mu).log1p()
                                     + theta * mt.log());
        s = (wt * d).sum();
        break;
    }

    case FAM_TWEEDIE_LOG:
    case FAM_TWEEDIE_IDENTITY:
    case FAM_TWEEDIE_INVERSE:
    case FAM_TWEEDIE_SQRT: {
        // d(y, mu) = 2 * [ y * (y^{1-p} - mu^{1-p})/(1-p)
        //               - (y^{2-p} - mu^{2-p})/(2-p) ]
        // y = 0 limit:  d(0, mu) = 2 * mu^{2-p}/(2-p).
        // p = 1 limit (Poisson):  y*log(y/mu) - (y - mu)
        //   - via L'Hopital on the first term.
        // p = 2 limit (Gamma):   -log(y/mu) + (y - mu)/mu
        //   - via L'Hopital on the second term.
        // Tolerance for the limit branches; statmod uses 1e-6.
        const double p   = params.var_power;
        const double a1  = 1.0 - p;
        const double a2  = 2.0 - p;
        const double tol = 1e-6;
        const bool near_one = std::fabs(a1) < tol;     // p ≈ 1
        const bool near_two = std::fabs(a2) < tol;     // p ≈ 2
        for (Eigen::Index i = 0; i < n; ++i) {
            double yi = y[i], mi = mu[i];
            double term1, term2;
            if (near_one) {
                // Poisson limit
                term1 = (yi > 0.0) ? yi * std::log(yi / mi) : 0.0;
                term2 = yi - mi;
            } else if (near_two) {
                // Gamma limit
                term1 = (yi > 0.0)
                          ? yi * (std::pow(yi, a1) - std::pow(mi, a1)) / a1
                          : 0.0;
                term2 = (yi > 0.0) ? std::log(yi / mi) : 0.0;
            } else if (yi > 0.0) {
                term1 = yi * (std::pow(yi, a1) - std::pow(mi, a1)) / a1;
                term2 = (std::pow(yi, a2) - std::pow(mi, a2)) / a2;
            } else {
                term1 = 0.0;
                term2 = -std::pow(mi, a2) / a2;
            }
            s += 2.0 * wt[i] * (term1 - term2);
        }
        break;
    }

    default:
        // Should not be reached.
        break;
    }

    return s;
}

// ---------------------------------------------------------------------------
// d2mu_deta2: d²μ / dη²  (second derivative of the inverse link)
//
// Needed by the generalized Firth / AS_mean bias-reducing penalty.  The
// per-observation adjustment to the working response is
//
//   ξ_i = h_i · (d²μ/dη²)_i · V(μ_i) / (2 · pw_i · (dμ/dη)_i³)
//
// For the canonical logit link this collapses to the familiar
//   ξ_i = h_i · (0.5 - μ_i) / (μ_i (1 - μ_i)).
// ---------------------------------------------------------------------------
inline void d2mu_deta2(int code,
                       const FamilyParams& /*params*/,
                       const Eigen::Ref<const Eigen::ArrayXd>& eta,
                       const Eigen::Ref<const Eigen::ArrayXd>& mu,
                       const Eigen::Ref<const Eigen::ArrayXd>& dmu,
                       Eigen::Ref<Eigen::ArrayXd> d2mu)
{
    switch (code) {
    // Identity link: d²μ/dη² = 0
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_POISSON_IDENTITY:
    case FAM_GAMMA_IDENTITY:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_NB_IDENTITY:
    case FAM_TWEEDIE_IDENTITY:
        d2mu.setZero(); break;

    // Log link: μ = exp(η), d²μ/dη² = exp(η) = dμ/dη
    case FAM_GAUSSIAN_LOG:
    case FAM_POISSON_LOG:
    case FAM_GAMMA_LOG:
    case FAM_INVGAUSS_LOG:
    case FAM_BINOMIAL_LOG:
    case FAM_NB_LOG:
    case FAM_TWEEDIE_LOG:
        d2mu = dmu; break;

    // Inverse link: μ = 1/η, d²μ/dη² = 2/η³
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
    case FAM_TWEEDIE_INVERSE:
        d2mu = 2.0 / (eta * eta * eta); break;

    // Logit: μ = logistic(η), d²μ/dη² = μ(1−μ)(1−2μ)
    case FAM_BINOMIAL_LOGIT:
        d2mu = mu * (1.0 - mu) * (1.0 - 2.0 * mu); break;

    // Probit: μ = Φ(η), d²μ/dη² = −η · φ(η) = −η · dμ/dη
    case FAM_BINOMIAL_PROBIT:
        d2mu = -eta * dmu; break;

    // Cloglog: μ = 1−exp(−exp(η)), d²μ/dη² = dμ/dη · (1 − exp(η))
    case FAM_BINOMIAL_CLOGLOG:
        d2mu = dmu * (1.0 - eta.exp()); break;

    // 1/μ² link: μ = η^{−1/2}, d²μ/dη² = (3/4) · η^{−5/2}
    case FAM_INVGAUSS_INVMU2:
        d2mu = 0.75 / eta.pow(2.5); break;

    // Sqrt link: μ = η², d²μ/dη² = 2
    case FAM_POISSON_SQRT:
    case FAM_NB_SQRT:
    case FAM_TWEEDIE_SQRT:
        d2mu.setConstant(2.0); break;

    default:
        d2mu.setZero();
        break;
    }
}

// ---------------------------------------------------------------------------
// valideta / validmu (boolean predicates)
// ---------------------------------------------------------------------------
inline bool valideta(int code,
                     const FamilyParams& /*params*/,
                     const Eigen::Ref<const Eigen::ArrayXd>& eta)
{
    switch (code) {
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
    case FAM_TWEEDIE_INVERSE:
        return eta.allFinite() && (eta != 0.0).all();
    case FAM_INVGAUSS_INVMU2:
    case FAM_POISSON_SQRT:
    case FAM_NB_SQRT:
        return eta.allFinite() && (eta > 0.0).all();
    case FAM_TWEEDIE_SQRT:
        // statmod::tweedie() does not constrain eta > 0 for the power-0.5
        // link -- mu = eta^2 is valid for any real eta.  Match glm() so
        // step-halving doesn't reject perfectly good iterates.
        return eta.allFinite();
    default:
        return eta.allFinite();
    }
}

inline bool validmu(int code,
                    const FamilyParams& /*params*/,
                    const Eigen::Ref<const Eigen::ArrayXd>& mu)
{
    switch (code) {
    case FAM_BINOMIAL_LOGIT:
    case FAM_BINOMIAL_PROBIT:
    case FAM_BINOMIAL_CLOGLOG:
    case FAM_BINOMIAL_LOG:
        return mu.allFinite() && (mu > 0.0).all() && (mu < 1.0).all();
    case FAM_POISSON_LOG:
    case FAM_POISSON_IDENTITY:
    case FAM_POISSON_SQRT:
    case FAM_GAMMA_LOG:
    case FAM_GAMMA_INVERSE:
    case FAM_GAMMA_IDENTITY:
    case FAM_INVGAUSS_INVMU2:
    case FAM_INVGAUSS_LOG:
    case FAM_INVGAUSS_IDENTITY:
    case FAM_INVGAUSS_INVERSE:
    case FAM_NB_LOG:
    case FAM_NB_SQRT:
    case FAM_NB_IDENTITY:
    case FAM_TWEEDIE_LOG:
    case FAM_TWEEDIE_IDENTITY:
    case FAM_TWEEDIE_INVERSE:
    case FAM_TWEEDIE_SQRT:
        return mu.allFinite() && (mu > 0.0).all();
    default:
        return mu.allFinite();
    }
}

// ---------------------------------------------------------------------------
// Stable IRLS working-weight computation for families where the generic
// formula  w = sqrt(prior_wt * mu_eta^2 / V(mu))  can overflow.
//
// For a log link, mu_eta = mu for all families.  Then:
//
//   Gamma log:     w^2 = mu^2 / mu^2 = 1
//   InvGauss log:  w^2 = mu^2 / mu^3 = 1/mu
//   NB log:        w^2 = mu^2 / (mu + mu^2/theta) = mu*theta/(theta+mu)
//
// Each of these is bounded and avoids computing V(mu) which can overflow.
// Returns TRUE if a stable formula was used (caller should skip the
// generic path).
// ---------------------------------------------------------------------------
inline bool stable_weights(int code,
                           const FamilyParams& params,
                           const Eigen::Ref<const Eigen::ArrayXd>& mu,
                           const Eigen::Ref<const Eigen::ArrayXd>& /*mu_eta*/,
                           const Eigen::Ref<const Eigen::ArrayXd>& prior_wt,
                           Eigen::Ref<Eigen::ArrayXd> w)
{
    switch (code) {
    case FAM_GAMMA_LOG:
        // w^2 = prior_wt * mu^2 / mu^2 = prior_wt
        w = prior_wt.sqrt();
        return true;
    case FAM_INVGAUSS_LOG:
        // w^2 = prior_wt * mu^2 / mu^3 = prior_wt / mu
        w = (prior_wt / mu).sqrt();
        return true;
    case FAM_NB_LOG: {
        const double theta = params.theta;
        if (theta <= 0.0) return false;
        // w^2 = prior_wt * mu * theta / (theta + mu)
        w = (prior_wt * theta * mu / (theta + mu)).sqrt();
        return true;
    }
    default:
        return false;
    }
}

}  // namespace fglm

#endif // FASTGLM_FAMILIES_H
