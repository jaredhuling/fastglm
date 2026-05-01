#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include "../inst/include/nb_theta.h"
#include "../inst/include/zi.h"
#include <RcppEigen.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <cmath>
#include <vector>

// Zero-inflated Poisson / NB driver via EM.  All numerical work in C++.
//
// Model:
//   Z_i ~ Bern(pi_i),     g_z(pi_i) = z_i' gamma            (zero link)
//   if Z_i = 0:  Y_i ~ Poisson(mu_i) or NB2(mu_i, theta)    (log link on count)
//   if Z_i = 1:  Y_i = 0
//
// Observed log-lik:
//   y > 0:  log(1 - pi) + log f(y; mu, theta)
//   y = 0:  log( pi + (1 - pi) * f(0; mu, theta) )
//
// EM:
//   E-step:  tau_i = pi_i / (pi_i + (1-pi_i) f_0_i)  for y_i = 0;  tau_i = 0 ow.
//   M-step (gamma):  weighted logit(/probit/cloglog/log) fit on tau response.
//   M-step (beta):   Poisson (or NB) regression on (y, X) with weights w_i*(1-tau_i).
//   M-step (theta):  Brent on the NB-only theta-MLE at fixed beta.
//
// Final vcov: numerical Jacobian of the analytical observed score at the EM
// fixed point (block-structured score for (gamma, beta, theta)).  Cheaper /
// more stable than fully numerical Hessian; reuses the per-observation score
// kernels in zi.h.

using namespace Rcpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::Map;

// ---------------------------------------------------------------------------
// Score helpers (analytical, vectorized).  Used both by the M-steps (only
// weights are needed) and by the final numerical-Jacobian vcov.
// ---------------------------------------------------------------------------

// Score wrt gamma (zero coefs):  s_gamma_k = sum_i Z_ik * s_eta_z_i.
static VectorXd score_gamma(int zero_fam_code,
                            const VectorXd& tau, const VectorXd& eta_z,
                            const VectorXd& wts, const MatrixXd& Z)
{
    const Eigen::Index n = Z.rows();
    const Eigen::Index p = Z.cols();
    VectorXd s_per_obs(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        s_per_obs[i] = wts[i] * fglm::zi::score_eta_z(zero_fam_code, tau[i], eta_z[i]);
    }
    return Z.transpose() * s_per_obs;
}

// Score wrt beta (count coefs):
static VectorXd score_beta(int dist_code, double theta,
                           const VectorXd& tau, const VectorXd& y,
                           const VectorXd& mu,
                           const VectorXd& wts, const MatrixXd& X)
{
    const Eigen::Index n = X.rows();
    VectorXd s_per_obs(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        s_per_obs[i] = wts[i] * fglm::zi::score_eta_count(
            dist_code, theta, tau[i], y[i], mu[i]);
    }
    return X.transpose() * s_per_obs;
}

// Score wrt theta (NB only):
static double score_theta_zi(double theta,
                             const VectorXd& tau, const VectorXd& y,
                             const VectorXd& mu, const VectorXd& wts)
{
    double s = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        s += wts[i] * (1.0 - tau[i])
             * fglm::zi::score_theta_one(theta, y[i], mu[i]);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Theta MLE for the ZI count component (NB only).  At an EM fixed point the
// posterior tau_i is fixed; the M-step for theta is a 1-D root find on
//   sum_i wts_i (1 - tau_i) * score_theta_one(theta, y_i, mu_i).
// We bracket adaptively from the current theta and Brent-solve.
// ---------------------------------------------------------------------------
static double mle_theta_zi(double theta_init,
                           const VectorXd& tau, const VectorXd& y,
                           const VectorXd& mu, const VectorXd& wts,
                           double tol = 1e-8, int maxit = 100,
                           double theta_lo = 1e-6, double theta_hi = 1e8)
{
    auto score = [&](double t) { return score_theta_zi(t, tau, y, mu, wts); };
    double th = theta_init;
    if (!std::isfinite(th) || th <= 0.0) th = 1.0;
    if (th < theta_lo) th = theta_lo;
    if (th > theta_hi) th = theta_hi;
    double s0 = score(th);
    if (s0 == 0.0) return th;

    double a = th, b = th;
    if (s0 > 0) {
        for (int i = 0; i < 60; ++i) {
            b *= 2.0;
            if (b > theta_hi) { b = theta_hi; break; }
            if (score(b) < 0) break;
        }
    } else {
        for (int i = 0; i < 60; ++i) {
            a *= 0.5;
            if (a < theta_lo) { a = theta_lo; break; }
            if (score(a) > 0) break;
        }
    }
    double sa = score(a), sb = score(b);
    if (sa == 0.0) return a;
    if (sb == 0.0) return b;
    if (sa * sb > 0) return (std::fabs(sa) < std::fabs(sb)) ? a : b;

    // Brent
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
            if (a == c) { p = 2.0 * xm * s; q = 1.0 - s; }
            else        { q = sa / sc; r = sb / sc;
                          p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                          q = (q - 1.0) * (r - 1.0) * (s - 1.0); }
            if (p > 0) q = -q;
            p = std::fabs(p);
            const double min1 = 3.0 * xm * q - std::fabs(tol1 * q);
            const double min2 = std::fabs(e * q);
            if (2.0 * p < std::min(min1, min2)) { e = d; d = p / q; }
            else                                { d = xm; e = d; }
        } else { d = xm; e = d; }
        a = b; sa = sb;
        b += (std::fabs(d) > tol1) ? d : (xm > 0 ? tol1 : -tol1);
        sb = score(b);
    }
    return b;
}

// ---------------------------------------------------------------------------
// Single observed-data log-likelihood pass (also returns posterior tau).
// ---------------------------------------------------------------------------
static double obs_loglik(int dist_code, int zero_fam_code, double theta,
                         const VectorXd& y, const VectorXd& mu,
                         const VectorXd& eta_z, const VectorXd& wts,
                         VectorXd& tau)
{
    return fglm::zi::obs_loglik_and_tau(
        dist_code, zero_fam_code, theta, y, mu, eta_z, wts, tau);
}

// ---------------------------------------------------------------------------
// Full analytical score vector at (gamma, beta, theta), evaluated at the
// posterior tau implied by the current params.  Used for the numerical-
// Jacobian observed-information matrix (Louis equivalent) at the EM fixpoint.
// Order of params:  [gamma (p_z), beta (p_c), theta (NB only)].
// ---------------------------------------------------------------------------
static VectorXd full_score(int dist_code, int zero_fam_code,
                           const VectorXd& gamma, const VectorXd& beta,
                           double theta,
                           const MatrixXd& Z, const MatrixXd& X,
                           const VectorXd& y, const VectorXd& wts,
                           const VectorXd& off_z, const VectorXd& off_c)
{
    const Eigen::Index n   = X.rows();
    const Eigen::Index p_z = Z.cols();
    const Eigen::Index p_c = X.cols();
    VectorXd eta_z = Z * gamma + off_z;
    VectorXd eta_c = X * beta  + off_c;
    VectorXd mu = eta_c.array().exp();
    VectorXd tau(n);
    obs_loglik(dist_code, zero_fam_code, theta, y, mu, eta_z, wts, tau);

    VectorXd s_g = score_gamma(zero_fam_code, tau, eta_z, wts, Z);
    VectorXd s_b = score_beta(dist_code, theta, tau, y, mu, wts, X);
    Eigen::Index p = p_z + p_c + (dist_code == 1 ? 1 : 0);
    VectorXd s(p);
    s.head(p_z) = s_g;
    s.segment(p_z, p_c) = s_b;
    if (dist_code == 1) {
        s[p - 1] = score_theta_zi(theta, tau, y, mu, wts);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Numerical Jacobian of full_score (= negative observed information).
// The observed-information matrix is - d s / d theta_full = J -> we return
// (-J) so the caller can invert directly.
// ---------------------------------------------------------------------------
static MatrixXd obs_info_numjac(int dist_code, int zero_fam_code,
                                const VectorXd& gamma, const VectorXd& beta,
                                double theta,
                                const MatrixXd& Z, const MatrixXd& X,
                                const VectorXd& y, const VectorXd& wts,
                                const VectorXd& off_z, const VectorXd& off_c)
{
    const Eigen::Index p_z = Z.cols();
    const Eigen::Index p_c = X.cols();
    const Eigen::Index p   = p_z + p_c + (dist_code == 1 ? 1 : 0);

    VectorXd s0 = full_score(dist_code, zero_fam_code, gamma, beta, theta,
                             Z, X, y, wts, off_z, off_c);
    MatrixXd J(p, p);
    const double rel = 1e-5;
    for (Eigen::Index k = 0; k < p; ++k) {
        VectorXd g_k = gamma;
        VectorXd b_k = beta;
        double   t_k = theta;
        double base, h;
        if (k < p_z) {
            base = g_k[k];
            h = rel * (std::fabs(base) + 1.0);
            g_k[k] = base + h;
        } else if (k < p_z + p_c) {
            Eigen::Index j = k - p_z;
            base = b_k[j];
            h = rel * (std::fabs(base) + 1.0);
            b_k[j] = base + h;
        } else {
            base = t_k;
            h = rel * std::max(t_k, 1.0);
            t_k = base + h;
        }
        VectorXd s_k = full_score(dist_code, zero_fam_code, g_k, b_k, t_k,
                                  Z, X, y, wts, off_z, off_c);
        J.col(k) = (s_k - s0) / h;
    }
    // Symmetrize and negate -> observed information.
    MatrixXd I_obs = -0.5 * (J + J.transpose());
    return I_obs;
}

// ---------------------------------------------------------------------------
// M-step gamma:  weighted (binomial) regression with response = tau,
//   weights = wts, link = zero_fam_code.  Uses the existing glm class with
//   native fam_code dispatch.
// We supply binomial family R-callbacks too as a fallback (the C++ class
// honours fam_code ahead of them).
// ---------------------------------------------------------------------------
static void mstep_gamma(const Map<MatrixXd>& Z, const VectorXd& tau,
                        const Map<VectorXd>& wts, const Map<VectorXd>& off_z,
                        int zero_fam_code,
                        Rcpp::Function var_fun_z, Rcpp::Function mu_eta_fun_z,
                        Rcpp::Function linkinv_fun_z, Rcpp::Function dev_resids_fun_z,
                        Rcpp::Function valideta_fun_z, Rcpp::Function validmu_fun_z,
                        VectorXd& gamma_out, VectorXd& eta_z_out,
                        VectorXd& mu_z_out,
                        const VectorXd& gamma_warm, double tol, int maxit)
{
    const Eigen::Index n = Z.rows();
    NumericVector y_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) y_nv[i] = tau[i];
    Map<VectorXd> y_map(as<Map<VectorXd> >(y_nv));

    NumericVector mu0_nv(n), eta0_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double mi = (y_map[i] + 0.5) / 2.0;
        if (mi <= 1e-6) mi = 1e-6;
        if (mi >= 1.0 - 1e-6) mi = 1.0 - 1e-6;
        mu0_nv[i]  = mi;
        eta0_nv[i] = std::log(mi / (1.0 - mi));
    }
    Map<VectorXd> mu0(as<Map<VectorXd> >(mu0_nv));
    Map<VectorXd> eta0(as<Map<VectorXd> >(eta0_nv));

    NumericVector start_nv(Z.cols());
    for (Eigen::Index k = 0; k < Z.cols(); ++k) start_nv[k] = gamma_warm[k];
    Map<VectorXd> start_map(as<Map<VectorXd> >(start_nv));

    fglm::FamilyParams fp;
    glm solver(Z, y_map, wts, off_z,
               var_fun_z, mu_eta_fun_z, linkinv_fun_z,
               dev_resids_fun_z, valideta_fun_z, validmu_fun_z,
               tol, maxit, /*type=*/2, /*is_big=*/false,
               zero_fam_code, fp);
    solver.init_parms(start_map, mu0, eta0);
    solver.solve(maxit);
    gamma_out = solver.get_beta();
    eta_z_out = solver.get_eta();
    mu_z_out  = solver.get_mu();
}

// ---------------------------------------------------------------------------
// M-step beta:  Poisson or NB IRLS with prior weights wts*(1-tau).  Uses the
// existing glm class with native fam_code dispatch (FAM_POISSON_LOG = 7,
// FAM_NB_LOG = 17).  count link is fixed to log here.
// ---------------------------------------------------------------------------
static void mstep_beta(const Map<MatrixXd>& X, const Map<VectorXd>& y,
                       const VectorXd& w_eff, const Map<VectorXd>& off_c,
                       int dist_code, double theta,
                       Rcpp::Function var_fun_c, Rcpp::Function mu_eta_fun_c,
                       Rcpp::Function linkinv_fun_c, Rcpp::Function dev_resids_fun_c,
                       Rcpp::Function valideta_fun_c, Rcpp::Function validmu_fun_c,
                       VectorXd& beta_out, VectorXd& eta_c_out,
                       VectorXd& mu_c_out,
                       const VectorXd& beta_warm, double tol, int maxit)
{
    const Eigen::Index n = X.rows();
    // Effective weights
    NumericVector w_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) w_nv[i] = w_eff[i];
    Map<VectorXd> w_map(as<Map<VectorXd> >(w_nv));

    // Init mu/eta on full y; for ZI Poisson/NB we use a Poisson-like start
    // matching family$initialize: mu0 = y + 0.1.
    NumericVector mu0_nv(n), eta0_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double mi = y[i] + 0.1;
        mu0_nv[i] = mi;
        eta0_nv[i] = std::log(mi);
    }
    Map<VectorXd> mu0(as<Map<VectorXd> >(mu0_nv));
    Map<VectorXd> eta0(as<Map<VectorXd> >(eta0_nv));

    NumericVector start_nv(X.cols());
    for (Eigen::Index k = 0; k < X.cols(); ++k) start_nv[k] = beta_warm[k];
    Map<VectorXd> start_map(as<Map<VectorXd> >(start_nv));

    fglm::FamilyParams fp;
    fp.theta = theta;
    int fam_code = (dist_code == 0) ? 7 : 17;  // FAM_POISSON_LOG / FAM_NB_LOG
    glm solver(X, y, w_map, off_c,
               var_fun_c, mu_eta_fun_c, linkinv_fun_c,
               dev_resids_fun_c, valideta_fun_c, validmu_fun_c,
               tol, maxit, /*type=*/2, /*is_big=*/false,
               fam_code, fp);
    solver.init_parms(start_map, mu0, eta0);
    solver.solve(maxit);
    beta_out  = solver.get_beta();
    eta_c_out = solver.get_eta();
    mu_c_out  = solver.get_mu();
}

// ---------------------------------------------------------------------------
// Pilot starting values:
//   gamma:   logit(P(Y=0)) ~ Z  (binomial logit, response 1(y == 0))
//   beta:    Poisson regression on the full y on X
//   theta:   1 if NB, else NA
// ---------------------------------------------------------------------------
struct PilotResult {
    VectorXd gamma;
    VectorXd beta;
    double   theta;
};
static PilotResult pilot_init(const Map<MatrixXd>& X, const Map<MatrixXd>& Z,
                              const Map<VectorXd>& y, const Map<VectorXd>& wts,
                              const Map<VectorXd>& off_c, const Map<VectorXd>& off_z,
                              int dist_code,
                              Rcpp::Function var_fun_z, Rcpp::Function mu_eta_fun_z,
                              Rcpp::Function linkinv_fun_z, Rcpp::Function dev_resids_fun_z,
                              Rcpp::Function valideta_fun_z, Rcpp::Function validmu_fun_z,
                              Rcpp::Function var_fun_c, Rcpp::Function mu_eta_fun_c,
                              Rcpp::Function linkinv_fun_c, Rcpp::Function dev_resids_fun_c,
                              Rcpp::Function valideta_fun_c, Rcpp::Function validmu_fun_c,
                              int zero_fam_code,
                              double tol, int maxit)
{
    PilotResult out;
    const Eigen::Index n = X.rows();

    // -- gamma pilot: binomial(zero link) on 1(y == 0) ----------------------
    NumericVector y0_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) y0_nv[i] = (y[i] == 0.0) ? 1.0 : 0.0;
    Map<VectorXd> y0(as<Map<VectorXd> >(y0_nv));

    NumericVector mu0_nv(n), eta0_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double mi = (y0[i] + 0.5) / 2.0;
        mu0_nv[i] = mi;
        eta0_nv[i] = std::log(mi / (1.0 - mi));
    }
    Map<VectorXd> mu0_z(as<Map<VectorXd> >(mu0_nv));
    Map<VectorXd> eta0_z(as<Map<VectorXd> >(eta0_nv));

    NumericVector zero_start_nv(Z.cols(), 0.0);
    Map<VectorXd> zero_start(as<Map<VectorXd> >(zero_start_nv));

    fglm::FamilyParams fp_empty;
    glm gamma_solver(Z, y0, wts, off_z,
                     var_fun_z, mu_eta_fun_z, linkinv_fun_z,
                     dev_resids_fun_z, valideta_fun_z, validmu_fun_z,
                     tol, maxit, 2, false, zero_fam_code, fp_empty);
    gamma_solver.init_parms(zero_start, mu0_z, eta0_z);
    gamma_solver.solve(maxit);
    out.gamma = gamma_solver.get_beta();

    // -- beta pilot: Poisson regression on full y on X ----------------------
    NumericVector mu0c_nv(n), eta0c_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double mi = y[i] + 0.1;
        mu0c_nv[i] = mi;
        eta0c_nv[i] = std::log(mi);
    }
    Map<VectorXd> mu0c(as<Map<VectorXd> >(mu0c_nv));
    Map<VectorXd> eta0c(as<Map<VectorXd> >(eta0c_nv));

    NumericVector beta_start_nv(X.cols(), 0.0);
    Map<VectorXd> beta_start(as<Map<VectorXd> >(beta_start_nv));

    fglm::FamilyParams fp_pilot;
    glm beta_solver(X, y, wts, off_c,
                    var_fun_c, mu_eta_fun_c, linkinv_fun_c,
                    dev_resids_fun_c, valideta_fun_c, validmu_fun_c,
                    tol, maxit, 2, false, 7, fp_pilot);  // FAM_POISSON_LOG
    beta_solver.init_parms(beta_start, mu0c, eta0c);
    beta_solver.solve(maxit);
    out.beta = beta_solver.get_beta();

    // -- theta pilot (NB only): theta.ml on Poisson residuals ---------------
    if (dist_code == 1) {
        VectorXd mu_pilot = beta_solver.get_mu();
        ArrayXd y_arr  = y.array();
        ArrayXd mu_arr = mu_pilot.array().max(1e-12);
        ArrayXd w_arr  = wts.array();
        double th = fglm::nb::init_theta_mom(y_arr, mu_arr, w_arr);
        if (!std::isfinite(th) || th <= 0.0) th = 1.0;
        // 1-D MLE refinement, on the unconditional NB likelihood (the EM
        // will refine further with tau-aware weights later).
        out.theta = fglm::nb::mle_theta(th, y_arr, mu_arr, w_arr,
                                        1e-7, 50);
    } else {
        out.theta = std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main entry point.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
List fit_glm_zi(
    Rcpp::NumericMatrix x_count,
    Rcpp::NumericMatrix z_zero,
    Rcpp::NumericVector y,
    Rcpp::NumericVector weights,
    Rcpp::NumericVector offset_count,
    Rcpp::NumericVector offset_zero,
    int                 dist_code,        // 0 = Poisson, 1 = NegBin
    int                 zero_fam_code,    // FAM_BINOMIAL_*
    double              init_theta,       // <=0  =>  pilot
    double              tol,
    int                 maxit,
    double              em_tol,
    int                 em_maxit,
    double              theta_tol,
    int                 theta_maxit,
    Rcpp::Function      var_fun_zero,
    Rcpp::Function      mu_eta_fun_zero,
    Rcpp::Function      linkinv_fun_zero,
    Rcpp::Function      dev_resids_fun_zero,
    Rcpp::Function      valideta_fun_zero,
    Rcpp::Function      validmu_fun_zero,
    Rcpp::Function      var_fun_count,
    Rcpp::Function      mu_eta_fun_count,
    Rcpp::Function      linkinv_fun_count,
    Rcpp::Function      dev_resids_fun_count,
    Rcpp::Function      valideta_fun_count,
    Rcpp::Function      validmu_fun_count)
{
    Map<MatrixXd> X(as<Map<MatrixXd> >(x_count));
    Map<MatrixXd> Z(as<Map<MatrixXd> >(z_zero));
    Map<VectorXd> Y(as<Map<VectorXd> >(y));
    Map<VectorXd> W(as<Map<VectorXd> >(weights));
    Map<VectorXd> OffC(as<Map<VectorXd> >(offset_count));
    Map<VectorXd> OffZ(as<Map<VectorXd> >(offset_zero));

    const Eigen::Index n   = X.rows();
    const Eigen::Index p_c = X.cols();
    const Eigen::Index p_z = Z.cols();
    if (Z.rows() != n)               Rcpp::stop("z_zero / x_count row mismatch");
    if ((Eigen::Index)Y.size()  != n) Rcpp::stop("y size mismatch");
    if ((Eigen::Index)W.size()  != n) Rcpp::stop("weights size mismatch");

    // ---------------- pilot init --------------------------------------------
    PilotResult pilot = pilot_init(
        X, Z, Y, W, OffC, OffZ, dist_code,
        var_fun_zero, mu_eta_fun_zero, linkinv_fun_zero,
        dev_resids_fun_zero, valideta_fun_zero, validmu_fun_zero,
        var_fun_count, mu_eta_fun_count, linkinv_fun_count,
        dev_resids_fun_count, valideta_fun_count, validmu_fun_count,
        zero_fam_code, tol, maxit);

    VectorXd gamma = pilot.gamma;
    VectorXd beta  = pilot.beta;
    double   theta = (dist_code == 1)
        ? ((init_theta > 0) ? init_theta : pilot.theta)
        : std::numeric_limits<double>::quiet_NaN();

    // ---------------- EM loop -----------------------------------------------
    VectorXd eta_z = Z * gamma + OffZ;
    VectorXd eta_c = X * beta  + OffC;
    VectorXd mu_c  = eta_c.array().exp();
    VectorXd tau(n);
    double ll = obs_loglik(dist_code, zero_fam_code,
                           (dist_code == 1 ? theta : 1.0),
                           Y, mu_c, eta_z, W, tau);
    double ll_prev = ll;
    bool   conv    = false;
    int    em_iter = 0;

    for (em_iter = 0; em_iter < em_maxit; ++em_iter) {
        // E-step (already done above for the first iteration; for subsequent
        // iterations we recompute mu_c and tau from the latest params at the
        // top of the loop).
        // Effective weights for the count IRLS:
        VectorXd w_eff = W.array() * (1.0 - tau.array());
        // Numerical guard: nudge zero weights up so the IRLS QR has a stable
        // factorization; the contribution to score is w*(y - mu_T)*(...) so
        // tiny w doesn't bias the solution.
        for (Eigen::Index i = 0; i < n; ++i)
            if (w_eff[i] < 1e-300) w_eff[i] = 0.0;

        // -- M-step gamma ---------------------------------------------------
        VectorXd gamma_new, eta_z_new, mu_z_new;
        mstep_gamma(Z, tau, W, OffZ, zero_fam_code,
                    var_fun_zero, mu_eta_fun_zero, linkinv_fun_zero,
                    dev_resids_fun_zero, valideta_fun_zero, validmu_fun_zero,
                    gamma_new, eta_z_new, mu_z_new,
                    gamma, tol, maxit);

        // -- M-step beta ----------------------------------------------------
        VectorXd beta_new, eta_c_new, mu_c_new;
        mstep_beta(X, Y, w_eff, OffC, dist_code,
                   (dist_code == 1 ? theta : 1.0),
                   var_fun_count, mu_eta_fun_count, linkinv_fun_count,
                   dev_resids_fun_count, valideta_fun_count, validmu_fun_count,
                   beta_new, eta_c_new, mu_c_new,
                   beta, tol, maxit);

        // -- M-step theta (NB only) -----------------------------------------
        double theta_new = theta;
        if (dist_code == 1) {
            theta_new = mle_theta_zi(theta, tau, Y, mu_c_new, W,
                                     theta_tol, theta_maxit);
        }

        // -- update params + recompute tau / log-lik -----------------------
        gamma = gamma_new;
        beta  = beta_new;
        theta = theta_new;
        eta_z = eta_z_new;
        eta_c = eta_c_new;
        mu_c  = mu_c_new;

        ll_prev = ll;
        ll = obs_loglik(dist_code, zero_fam_code,
                        (dist_code == 1 ? theta : 1.0),
                        Y, mu_c, eta_z, W, tau);

        const double rel = std::fabs(ll - ll_prev) / (std::fabs(ll) + 0.1);
        if (rel < em_tol && em_iter > 0) { conv = true; ++em_iter; break; }
    }

    // ---------------- final vcov via numerical-Jac observed information ----
    MatrixXd vcov_full;
    {
        MatrixXd I_obs = obs_info_numjac(dist_code, zero_fam_code,
                                         gamma, beta,
                                         (dist_code == 1 ? theta : 1.0),
                                         Z, X, Y, W, OffZ, OffC);
        // Solve I^{-1} for the full param block.  If singular, return NaN-filled.
        Eigen::LDLT<MatrixXd> ldlt(I_obs);
        if (ldlt.info() != Eigen::Success) {
            const Eigen::Index pp = I_obs.rows();
            vcov_full = MatrixXd::Constant(pp, pp,
                std::numeric_limits<double>::quiet_NaN());
        } else {
            vcov_full = ldlt.solve(MatrixXd::Identity(I_obs.rows(), I_obs.cols()));
        }
    }

    // Split
    MatrixXd vcov_zero  = vcov_full.topLeftCorner(p_z, p_z);
    MatrixXd vcov_count = vcov_full.block(p_z, p_z, p_c, p_c);
    VectorXd se_zero    = vcov_zero.diagonal().array().abs().sqrt();
    VectorXd se_count   = vcov_count.diagonal().array().abs().sqrt();
    double   se_theta   = (dist_code == 1)
        ? std::sqrt(std::fabs(vcov_full(p_z + p_c, p_z + p_c)))
        : std::numeric_limits<double>::quiet_NaN();

    // Reorder vcov to match output: count first, then zero, then theta.
    const Eigen::Index pp = p_c + p_z + (dist_code == 1 ? 1 : 0);
    MatrixXd vcov_out = MatrixXd::Zero(pp, pp);
    // Count block
    vcov_out.block(0, 0, p_c, p_c) = vcov_count;
    vcov_out.block(p_c, p_c, p_z, p_z) = vcov_zero;
    // Cross-terms gamma <-> beta
    MatrixXd vcov_zb = vcov_full.block(0, p_z, p_z, p_c);  // (gamma, beta)
    vcov_out.block(p_c, 0, p_z, p_c) = vcov_zb;
    vcov_out.block(0, p_c, p_c, p_z) = vcov_zb.transpose();
    if (dist_code == 1) {
        // theta block
        vcov_out(pp - 1, pp - 1) = vcov_full(p_z + p_c, p_z + p_c);
        // theta <-> gamma
        for (Eigen::Index k = 0; k < p_z; ++k) {
            vcov_out(pp - 1, p_c + k) = vcov_full(p_z + p_c, k);
            vcov_out(p_c + k, pp - 1) = vcov_full(k, p_z + p_c);
        }
        // theta <-> beta
        for (Eigen::Index k = 0; k < p_c; ++k) {
            vcov_out(pp - 1, k) = vcov_full(p_z + p_c, p_z + k);
            vcov_out(k, pp - 1) = vcov_full(p_z + k, p_z + p_c);
        }
    }

    return List::create(
        _["coefficients_count"] = beta,
        _["coefficients_zero"]  = gamma,
        _["se_count"]           = se_count,
        _["se_zero"]            = se_zero,
        _["vcov_count"]         = vcov_count,
        _["vcov_zero"]          = vcov_zero,
        _["vcov_full"]          = vcov_out,
        _["theta"]              = theta,
        _["SE.theta"]           = se_theta,
        _["loglik"]             = ll,
        _["mu_count"]           = mu_c,
        _["eta_count"]          = eta_c,
        _["eta_zero"]           = eta_z,
        _["tau"]                = tau,
        _["em_iter"]            = em_iter,
        _["converged"]          = conv,
        _["dist_code"]          = dist_code,
        _["zero_fam_code"]      = zero_fam_code,
        _["n"]                  = (int)n,
        _["df"]                 = (int)(p_c + p_z + (dist_code == 1 ? 1 : 0))
    );
}
