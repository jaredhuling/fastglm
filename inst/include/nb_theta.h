#ifndef FASTGLM_NB_THETA_H
#define FASTGLM_NB_THETA_H
//
// Negative-binomial dispersion (theta) helpers used by the joint (beta, theta)
// MLE driver of fastglm_nb().  The same routines will be reused inside the
// hurdle and zero-inflated drivers (Phases 4, 5).
//
// All numerical work is in C++; no R callbacks.  We rely on
// boost::math::digamma (BH already linked from DESCRIPTION).
//

#include <RcppEigen.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/tools/roots.hpp>
#include "brent.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace fglm { namespace nb {

// The leading 1/theta terms cancel in the size score. Evaluating the
// remaining series avoids artificial roots near the Poisson boundary.
inline double score_theta_row(double theta, double yi, double mi, double dig_th)
{
    if (theta > 1e4 * std::max(1.0, std::max(yi, mi))) {
        const double c2 = (yi - (yi - mi) * (yi - mi)) / 2.0;
        const double c3 = yi * (yi - 1.0) * (2.0 * yi - 1.0) / 6.0
                          + 2.0 * mi * mi * mi / 3.0 - yi * mi * mi;
        const double c4 = -yi * yi * (yi - 1.0) * (yi - 1.0) / 4.0
                          - 3.0 * mi * mi * mi * mi / 4.0 + yi * mi * mi * mi;
        return (c2 + (c3 + c4 / theta) / theta) / (theta * theta);
    }
    return boost::math::digamma(yi + theta) - dig_th - std::log1p(mi / theta)
           + (mi - yi) / (theta + mi);
}

// Score of the NB2 log-likelihood w.r.t. theta, evaluated at fixed mu:
//
//   s(theta) = sum_i wt_i * [
//                  digamma(y_i + theta) - digamma(theta)
//                  + log(theta / (theta + mu_i))
//                  + (mu_i - y_i) / (theta + mu_i) ]
//
// theta -> Inf reduces this to the Poisson score (which is identically zero
// at the Poisson MLE), so for very large theta the score is small and we
// stop bracketing.  theta -> 0+ blows up; we cap.
inline double score_theta(double theta,
                          const Eigen::Ref<const Eigen::ArrayXd>& y,
                          const Eigen::Ref<const Eigen::ArrayXd>& mu,
                          const Eigen::Ref<const Eigen::ArrayXd>& wt)
{
    using boost::math::digamma;
    const double dig_th = digamma(theta);
    double s = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double yi = y[i], mi = mu[i], wi = wt[i];
        s += wi * score_theta_row(theta, yi, mi, dig_th);
    }
    return s;
}

// Observed information for theta at fixed mu, as in MASS::theta.ml:
//
//   info(theta) = sum_i wt_i * [
//                  - trigamma(y_i + theta) + trigamma(theta)
//                  - 1/theta + 2/(theta + mu_i)
//                  - (y_i + theta) / (theta + mu_i)^2 ]
//
// Joint information additionally includes the coefficient/size cross
// derivatives. Newton updates use score / info.
inline double info_theta(double theta,
                         const Eigen::Ref<const Eigen::ArrayXd>& y,
                         const Eigen::Ref<const Eigen::ArrayXd>& mu,
                         const Eigen::Ref<const Eigen::ArrayXd>& wt)
{
    using boost::math::trigamma;
    const double tri_th = trigamma(theta);
    const double inv_th = 1.0 / theta;
    double I = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double yi = y[i], mi = mu[i], wi = wt[i];
        if (theta > 1e4 * std::max(1.0, std::max(yi, mi))) {
            const double c2 = (yi - (yi - mi) * (yi - mi)) / 2.0;
            const double c3 = yi * (yi - 1.0) * (2.0 * yi - 1.0) / 6.0
                              + 2.0 * mi * mi * mi / 3.0 - yi * mi * mi;
            const double c4 = -yi * yi * (yi - 1.0) * (yi - 1.0) / 4.0
                              - 3.0 * mi * mi * mi * mi / 4.0 + yi * mi * mi * mi;
            I += wi * (2.0 * c2 + (3.0 * c3 + 4.0 * c4 / theta) / theta)
                 / (theta * theta * theta);
            continue;
        }
        const double tm  = theta + mi;
        const double tm2 = tm * tm;
        I += wi * (-trigamma(yi + theta) + tri_th
                   - inv_th + 2.0 / tm
                   - (yi + theta) / tm2);
    }
    return I;
}

// Method-of-moments initializer matching MASS::theta.ml's starting value.
// theta0 = sum(wt) / sum(wt * (y/mu - 1)^2).
inline double init_theta_mom(const Eigen::Ref<const Eigen::ArrayXd>& y,
                             const Eigen::Ref<const Eigen::ArrayXd>& mu,
                             const Eigen::Ref<const Eigen::ArrayXd>& wt)
{
    double num = 0.0, den = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double mi = mu[i] > 0 ? mu[i] : 1e-8;
        const double r  = y[i] / mi - 1.0;
        num += wt[i];
        den += wt[i] * r * r;
    }
    if (den <= 0.0) return 50.0;       // y ~ mu (no overdispersion)
    return num / den;
}

// Adaptive bracketing: expand multiplicatively from `theta0` until score
// signs differ.  Cap at [theta_lo, theta_hi].  Return {a, b} with
// score(a) > 0 and score(b) < 0 (the typical NB sign convention --
// score is monotonically decreasing in theta).
//
// If score is the same sign throughout the cap, return {theta_lo, theta_hi}
// and let the caller handle the boundary case (very small or very large
// theta -- typically meaning the Poisson limit).
inline std::pair<double, double>
bracket_theta(double theta0,
              const Eigen::Ref<const Eigen::ArrayXd>& y,
              const Eigen::Ref<const Eigen::ArrayXd>& mu,
              const Eigen::Ref<const Eigen::ArrayXd>& wt,
              double theta_lo = 1e-6,
              double theta_hi = 1e8,
              int    max_steps = 60)
{
    if (theta0 < theta_lo) theta0 = theta_lo;
    if (theta0 > theta_hi) theta0 = theta_hi;
    double s0 = score_theta(theta0, y, mu, wt);
    if (s0 == 0.0) return {theta0, theta0};

    double a = theta0, b = theta0;
    if (s0 > 0) {
        // Need to grow b until score(b) < 0.
        for (int i = 0; i < max_steps; ++i) {
            b *= 2.0;
            if (b > theta_hi) { b = theta_hi; break; }
            if (score_theta(b, y, mu, wt) < 0) return {a, b};
        }
        // Couldn't find b: theta is at the upper boundary (Poisson-like).
        return {a, theta_hi};
    } else {
        // score < 0 at theta0; need to shrink a until score(a) > 0.
        for (int i = 0; i < max_steps; ++i) {
            a *= 0.5;
            if (a < theta_lo) { a = theta_lo; break; }
            if (score_theta(a, y, mu, wt) > 0) return {a, b};
        }
        return {theta_lo, b};
    }
}

// Brent root-finder for theta given fixed mu.  Returns the MLE of theta.
// Tolerance is on |score| relative to scale, with a max of 100 iterations.
inline double mle_theta(double theta_init,
                        const Eigen::Ref<const Eigen::ArrayXd>& y,
                        const Eigen::Ref<const Eigen::ArrayXd>& mu,
                        const Eigen::Ref<const Eigen::ArrayXd>& wt,
                        double tol = 1e-8,
                        int    maxit = 100,
                        double theta_lo = 1e-6,
                        double theta_hi = 1e8)
{
    auto score = [&](double t) { return score_theta(t, y, mu, wt); };
    auto br = bracket_theta(theta_init, y, mu, wt, theta_lo, theta_hi);
    double a = br.first, b = br.second;
    double sa = score(a);
    double sb = score(b);
    if (sa == 0.0) return a;
    if (sb == 0.0) return b;

    // If the bracket didn't actually trap a root, we're at the boundary.
    // Return the boundary closer to zero score; the caller can detect this
    // via the cap.
    if (sa * sb > 0) {
        return (std::fabs(sa) < std::fabs(sb)) ? a : b;
    }

    return brent_root(score, a, b, sa, sb, tol, maxit);
}

}}  // namespace fglm::nb

#endif // FASTGLM_NB_THETA_H
