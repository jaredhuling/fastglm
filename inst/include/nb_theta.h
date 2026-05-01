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
#include <cmath>
#include <limits>
#include <utility>

namespace fglm { namespace nb {

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
        const double tm = theta + mi;
        // log(theta / (theta + mu)) computed via log1p when mu << theta to
        // retain precision.
        double log_ratio;
        if (mi < 1e-3 * theta) {
            // log(theta / (theta + mu)) = -log(1 + mu/theta)
            log_ratio = -std::log1p(mi / theta);
        } else {
            log_ratio = std::log(theta / tm);
        }
        s += wi * (digamma(yi + theta) - dig_th + log_ratio + (mi - yi) / tm);
    }
    return s;
}

// Information for theta at fixed mu, matching MASS::theta.ml exactly:
//
//   info(theta) = sum_i wt_i * [
//                  - trigamma(y_i + theta) + trigamma(theta)
//                  - 1/theta + 2/(theta + mu_i)
//                  - (y_i + theta) / (theta + mu_i)^2 ]
//
// At the MLE, this is positive and equals the Fisher information; we
// report SE.theta = sqrt(1 / info).  Newton fallback uses score / info
// (matches MASS's iteration sign convention).
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
    auto br = bracket_theta(theta_init, y, mu, wt, theta_lo, theta_hi);
    double a = br.first, b = br.second;
    double sa = score_theta(a, y, mu, wt);
    double sb = score_theta(b, y, mu, wt);
    if (sa == 0.0) return a;
    if (sb == 0.0) return b;

    // If the bracket didn't actually trap a root, we're at the boundary.
    // Return the boundary itself; the caller can detect this via the cap.
    if (sa * sb > 0) {
        // Same sign: return the boundary closer to zero score.
        return (std::fabs(sa) < std::fabs(sb)) ? a : b;
    }

    // Brent's method (textbook Numerical Recipes-style).
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
        sb = score_theta(b, y, mu, wt);
    }
    return b;
}

}}  // namespace fglm::nb

#endif // FASTGLM_NB_THETA_H
