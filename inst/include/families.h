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
    FAM_INVGAUSS_INVERSE     = 16
};

// Numerical floor used by R's family helpers.
static inline double thresh_eps() { return std::numeric_limits<double>::epsilon(); }
static inline double pmin1(double x) { return x > 1.0 ? 1.0 : x; }

// y * log(y/mu)  with the standard 0*log(0) = 0 convention used by R.
static inline double y_log_y(double y, double mu) {
    return y > 0.0 ? y * std::log(y / mu) : 0.0;
}

// ---------------------------------------------------------------------------
// linkinv: mu = g^{-1}(eta)
// ---------------------------------------------------------------------------
inline void linkinv(int code,
                    const Eigen::Ref<const Eigen::ArrayXd>& eta,
                    Eigen::Ref<Eigen::ArrayXd> mu)
{
    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_POISSON_IDENTITY:
    case FAM_GAMMA_IDENTITY:
    case FAM_INVGAUSS_IDENTITY:
        mu = eta; break;
    case FAM_GAUSSIAN_LOG:
    case FAM_POISSON_LOG:
    case FAM_GAMMA_LOG:
    case FAM_INVGAUSS_LOG:
    case FAM_BINOMIAL_LOG:
        mu = eta.exp(); break;
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
        mu = 1.0 / eta; break;
    case FAM_BINOMIAL_LOGIT: {
        // Numerically stable: pmax(pmin(eta, large), -large) then plogis
        const double thr = -std::log(thresh_eps());
        Eigen::ArrayXd e = eta.max(-thr).min(thr);
        mu = 1.0 / (1.0 + (-e).exp());
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
        Eigen::ArrayXd v = -((-eta.exp()).exp()) + 1.0;  // 1 - exp(-exp(eta))
        // pmax(pmin(v, 1-eps), eps)
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            if (v[i] < thresh_eps()) v[i] = thresh_eps();
            if (v[i] > 1.0 - thresh_eps()) v[i] = 1.0 - thresh_eps();
        }
        mu = v;
        break;
    }
    case FAM_INVGAUSS_INVMU2:
        mu = 1.0 / eta.sqrt(); break;
    case FAM_POISSON_SQRT:
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
                   const Eigen::Ref<const Eigen::ArrayXd>& eta,
                   Eigen::Ref<Eigen::ArrayXd> dmu)
{
    switch (code) {
    case FAM_GAUSSIAN_IDENTITY:
    case FAM_POISSON_IDENTITY:
    case FAM_GAMMA_IDENTITY:
    case FAM_INVGAUSS_IDENTITY:
        dmu.setOnes(); break;
    case FAM_GAUSSIAN_LOG:
    case FAM_POISSON_LOG:
    case FAM_GAMMA_LOG:
    case FAM_INVGAUSS_LOG:
    case FAM_BINOMIAL_LOG:
        dmu = eta.exp().max(thresh_eps()); break;
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
        dmu = -1.0 / eta.square(); break;
    case FAM_BINOMIAL_LOGIT: {
        Eigen::ArrayXd ee = (-eta.abs()).exp();
        dmu = ee / (1.0 + ee).square();
        for (Eigen::Index i = 0; i < dmu.size(); ++i)
            if (dmu[i] < thresh_eps()) dmu[i] = thresh_eps();
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
    default:
        v.setOnes();
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
    case FAM_INVGAUSS_INVERSE:
        for (Eigen::Index i = 0; i < n; ++i) {
            double yi = y[i], mi = mu[i];
            double d  = yi - mi;
            s += wt[i] * (d * d) / (yi * mi * mi);
        }
        break;

    default:
        // Should not be reached.
        break;
    }

    return s;
}

// ---------------------------------------------------------------------------
// valideta / validmu (boolean predicates)
// ---------------------------------------------------------------------------
inline bool valideta(int code, const Eigen::Ref<const Eigen::ArrayXd>& eta)
{
    switch (code) {
    case FAM_GAUSSIAN_INVERSE:
    case FAM_GAMMA_INVERSE:
    case FAM_INVGAUSS_INVERSE:
        return eta.allFinite() && (eta != 0.0).all();
    case FAM_INVGAUSS_INVMU2:
    case FAM_POISSON_SQRT:
        return eta.allFinite() && (eta > 0.0).all();
    default:
        return eta.allFinite();
    }
}

inline bool validmu(int code, const Eigen::Ref<const Eigen::ArrayXd>& mu)
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
        return mu.allFinite() && (mu > 0.0).all();
    default:
        return mu.allFinite();
    }
}

}  // namespace fglm

#endif // FASTGLM_FAMILIES_H
