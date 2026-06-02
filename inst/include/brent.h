#ifndef FASTGLM_BRENT_H
#define FASTGLM_BRENT_H
//
// Brent's method scalar root-finder, shared by every theta-MLE driver
// (negative binomial in nb_theta.h, zero-truncated counts in trunc_count.h,
// zero-inflated counts in fit_glm_zi.cpp).  Each of those sites supplies its
// own score function and its own multiplicative bracketing; the inner
// iteration is identical, so it lives here once.
//

#include <algorithm>
#include <cmath>
#include <limits>

namespace fglm {

// Brent's method (Numerical Recipes style) on a scalar function `f` already
// known to bracket a root in [a, b] with fa = f(a), fb = f(b) and
// fa * fb <= 0.  `tol` is the tolerance on the abscissa; after `maxit`
// iterations without convergence the current best estimate is returned.
template <typename F>
inline double brent_root(F&& f, double a, double b,
                         double fa, double fb,
                         double tol, int maxit)
{
    double c = a, fc = fa, d = b - a, e = d;
    for (int iter = 0; iter < maxit; ++iter) {
        if (fb * fc > 0) { c = a; fc = fa; d = b - a; e = d; }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        const double tol1 = 2.0 * std::numeric_limits<double>::epsilon() * std::fabs(b) + 0.5 * tol;
        const double xm   = 0.5 * (c - b);
        if (std::fabs(xm) <= tol1 || fb == 0.0) return b;

        if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
            double s = fb / fa, p, q, r;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
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
        a  = b;  fa = fb;
        b += (std::fabs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
        fb = f(b);
    }
    return b;
}

}  // namespace fglm

#endif // FASTGLM_BRENT_H
