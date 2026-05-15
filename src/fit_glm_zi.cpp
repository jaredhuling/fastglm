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

// Zero-inflated Poisson / NB driver via EM with SQUAREM acceleration.
//
// Model:
//   Z_i ~ Bern(pi_i),     g_z(pi_i) = z_i' gamma            (zero link)
//   if Z_i = 0:  Y_i ~ Poisson(mu_i) or NB2(mu_i, theta)    (log link on count)
//   if Z_i = 1:  Y_i = 0
//
// EM:
//   E-step:  tau_i = pi_i / (pi_i + (1-pi_i) f_0_i)  for y_i = 0;  tau_i = 0 ow.
//   M-step (gamma):  weighted logit(/probit/cloglog/log) fit on tau response.
//   M-step (beta):   Poisson (or NB) regression on (y, X) with weights w_i*(1-tau_i).
//   M-step (theta):  Brent on the NB-only theta-MLE at fixed beta.
//
// SQUAREM (Varadhan & Roland 2008) acceleration converts EM's linear
// convergence to near-quadratic by extrapolating two successive EM maps.
//
// Final vcov: numerical Jacobian of the analytical observed score at the EM
// fixed point.

using namespace Rcpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::Map;

// ---------------------------------------------------------------------------
// Score helpers (analytical, vectorized where possible).
// ---------------------------------------------------------------------------

static VectorXd score_gamma(int zero_fam_code,
                            const VectorXd& tau, const VectorXd& eta_z,
                            const VectorXd& wts, const MatrixXd& Z)
{
    const Eigen::Index n = Z.rows();
    VectorXd s_per_obs(n);
    if (zero_fam_code == fglm::FAM_BINOMIAL_LOGIT) {
        ArrayXd pi = 1.0 / (1.0 + (-eta_z.array()).exp());
        s_per_obs = (wts.array() * (tau.array() - pi)).matrix();
    } else {
        for (Eigen::Index i = 0; i < n; ++i)
            s_per_obs[i] = wts[i] * fglm::zi::score_eta_z(zero_fam_code, tau[i], eta_z[i]);
    }
    return Z.transpose() * s_per_obs;
}

static VectorXd score_beta(int dist_code, double theta,
                           const VectorXd& tau, const VectorXd& y,
                           const VectorXd& mu,
                           const VectorXd& wts, const MatrixXd& X)
{
    ArrayXd w = wts.array() * (1.0 - tau.array());
    VectorXd s_per_obs;
    if (dist_code == 0) {
        s_per_obs = (w * (y.array() - mu.array())).matrix();
    } else {
        s_per_obs = (w * theta * (y.array() - mu.array()) / (theta + mu.array())).matrix();
    }
    return X.transpose() * s_per_obs;
}

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
// Theta MLE for the ZI count component (NB only).
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
// Observed log-likelihood + posterior tau.
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
// Full analytical score vector.
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
    MatrixXd I_obs = -0.5 * (J + J.transpose());
    return I_obs;
}

// ---------------------------------------------------------------------------
// Pilot starting values.
// ---------------------------------------------------------------------------
struct PilotResult {
    VectorXd gamma;
    VectorXd beta;
    VectorXd mu_z;
    VectorXd eta_z;
    VectorXd mu_c;
    VectorXd eta_c;
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

    VectorXd y0(n);
    for (Eigen::Index i = 0; i < n; ++i) y0[i] = (y[i] == 0.0) ? 1.0 : 0.0;

    VectorXd mu0_z(n), eta0_z(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double mi = (y0[i] + 0.5) / 2.0;
        mu0_z[i] = mi;
        eta0_z[i] = std::log(mi / (1.0 - mi));
    }
    VectorXd zero_start = VectorXd::Zero(Z.cols());

    fglm::FamilyParams fp_empty;
    glm gamma_solver(Z,
                     Map<VectorXd>(y0.data(), n),
                     wts, off_z,
                     var_fun_z, mu_eta_fun_z, linkinv_fun_z,
                     dev_resids_fun_z, valideta_fun_z, validmu_fun_z,
                     tol, maxit, 2, false, zero_fam_code, fp_empty);
    gamma_solver.init_parms(Map<VectorXd>(zero_start.data(), zero_start.size()),
                            Map<VectorXd>(mu0_z.data(), n),
                            Map<VectorXd>(eta0_z.data(), n));
    gamma_solver.solve(maxit);
    out.gamma = gamma_solver.get_beta();
    out.mu_z  = gamma_solver.get_mu();
    out.eta_z = gamma_solver.get_eta();

    VectorXd mu0c(n), eta0c(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double mi = y[i] + 0.1;
        mu0c[i] = mi;
        eta0c[i] = std::log(mi);
    }
    VectorXd beta_start = VectorXd::Zero(X.cols());

    fglm::FamilyParams fp_pilot;
    glm beta_solver(X, y, wts, off_c,
                    var_fun_c, mu_eta_fun_c, linkinv_fun_c,
                    dev_resids_fun_c, valideta_fun_c, validmu_fun_c,
                    tol, maxit, 2, false, 7, fp_pilot);
    beta_solver.init_parms(Map<VectorXd>(beta_start.data(), beta_start.size()),
                           Map<VectorXd>(mu0c.data(), n),
                           Map<VectorXd>(eta0c.data(), n));
    beta_solver.solve(maxit);
    out.beta  = beta_solver.get_beta();
    out.mu_c  = beta_solver.get_mu();
    out.eta_c = beta_solver.get_eta();

    if (dist_code == 1) {
        ArrayXd y_arr  = y.array();
        ArrayXd mu_arr = out.mu_c.array().max(1e-12);
        ArrayXd w_arr  = wts.array();
        double th = fglm::nb::init_theta_mom(y_arr, mu_arr, w_arr);
        if (!std::isfinite(th) || th <= 0.0) th = 1.0;
        out.theta = fglm::nb::mle_theta(th, y_arr, mu_arr, w_arr, 1e-7, 50);
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
    int                 dist_code,
    int                 zero_fam_code,
    double              init_theta,
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

    // ---------------- initial E-step ----------------------------------------
    VectorXd eta_z = Z * gamma + OffZ;
    VectorXd eta_c = X * beta  + OffC;
    VectorXd mu_c  = eta_c.array().exp();
    VectorXd tau(n);
    double ll = obs_loglik(dist_code, zero_fam_code,
                           (dist_code == 1 ? theta : 1.0),
                           Y, mu_c, eta_z, W, tau);

    // ---------------- persistent solver buffers -----------------------------
    VectorXd tau_buf(n);
    tau_buf = tau;
    VectorXd w_eff_buf(n);
    w_eff_buf = W.array() * (1.0 - tau.array());

    fglm::FamilyParams fp_gamma;
    glm gamma_solver(Z,
                     Map<VectorXd>(tau_buf.data(), n),
                     W, OffZ,
                     var_fun_zero, mu_eta_fun_zero, linkinv_fun_zero,
                     dev_resids_fun_zero, valideta_fun_zero, validmu_fun_zero,
                     tol, maxit, 2, false, zero_fam_code, fp_gamma);

    fglm::FamilyParams fp_beta;
    fp_beta.theta = (dist_code == 1) ? theta : 1.0;
    int fam_code_beta = (dist_code == 0) ? 7 : 17;
    glm beta_solver(X, Y,
                    Map<VectorXd>(w_eff_buf.data(), n),
                    OffC,
                    var_fun_count, mu_eta_fun_count, linkinv_fun_count,
                    dev_resids_fun_count, valideta_fun_count, validmu_fun_count,
                    tol, maxit, 2, false, fam_code_beta, fp_beta);

    // Warm-start vectors (seeded from pilot)
    VectorXd gamma_warm = gamma;
    VectorXd mu_z_warm  = pilot.mu_z;
    VectorXd eta_z_warm = pilot.eta_z;
    mu_z_warm = mu_z_warm.array().max(1e-6).min(1.0 - 1e-6);

    VectorXd beta_warm  = beta;
    VectorXd mu_c_warm  = pilot.mu_c;
    VectorXd eta_c_warm = pilot.eta_c;

    // ---------------- EM step helper ----------------------------------------
    // Runs one complete EM iteration: updates gamma, beta, theta, warm-starts,
    // eta_z, eta_c, mu_c, tau.  Returns new log-likelihood.
    auto do_em_step = [&]() -> double {
        tau_buf = tau;
        w_eff_buf = W.array() * (1.0 - tau.array());
        for (Eigen::Index i = 0; i < n; ++i)
            if (w_eff_buf[i] < 1e-300) w_eff_buf[i] = 0.0;

        // M-step gamma
        gamma_solver.init_parms(
            Map<VectorXd>(gamma_warm.data(), gamma_warm.size()),
            Map<VectorXd>(mu_z_warm.data(), n),
            Map<VectorXd>(eta_z_warm.data(), n));
        gamma_solver.solve(maxit);
        gamma = gamma_solver.get_beta();
        eta_z = gamma_solver.get_eta();
        gamma_warm = gamma;
        mu_z_warm  = gamma_solver.get_mu();
        eta_z_warm = eta_z;
        mu_z_warm  = mu_z_warm.array().max(1e-6).min(1.0 - 1e-6);

        // M-step beta
        if (dist_code == 1) {
            fp_beta.theta = theta;
            beta_solver.set_fam_params(fp_beta);
        }
        beta_solver.init_parms(
            Map<VectorXd>(beta_warm.data(), beta_warm.size()),
            Map<VectorXd>(mu_c_warm.data(), n),
            Map<VectorXd>(eta_c_warm.data(), n));
        beta_solver.solve(maxit);
        beta  = beta_solver.get_beta();
        eta_c = beta_solver.get_eta();
        mu_c  = beta_solver.get_mu();
        beta_warm  = beta;
        mu_c_warm  = mu_c;
        eta_c_warm = eta_c;

        // M-step theta
        if (dist_code == 1) {
            theta = mle_theta_zi(theta, tau, Y, mu_c, W,
                                 theta_tol, theta_maxit);
        }

        // E-step + log-lik
        return obs_loglik(dist_code, zero_fam_code,
                          (dist_code == 1 ? theta : 1.0),
                          Y, mu_c, eta_z, W, tau);
    };

    // Pack/unpack parameter vector for SQUAREM
    const Eigen::Index n_params = p_z + p_c + (dist_code == 1 ? 1 : 0);

    auto pack_params = [&]() -> VectorXd {
        VectorXd x(n_params);
        x.head(p_z) = gamma;
        x.segment(p_z, p_c) = beta;
        if (dist_code == 1) x[n_params - 1] = theta;
        return x;
    };

    auto unpack_params = [&](const VectorXd& x) {
        gamma = x.head(p_z);
        beta  = x.segment(p_z, p_c);
        if (dist_code == 1) {
            theta = x[n_params - 1];
            if (theta <= 0.0) theta = 1e-4;
        }
    };

    // Recompute eta/mu/warm-starts from current gamma,beta (not from IRLS)
    auto set_warm_from_params = [&]() {
        gamma_warm = gamma;
        eta_z = Z * gamma + OffZ;
        eta_z_warm = eta_z;
        ArrayXd eta_arr = eta_z.array();
        ArrayXd mu_arr(n);
        fglm::linkinv(zero_fam_code, fp_gamma, eta_arr, mu_arr);
        mu_z_warm = mu_arr.array().max(1e-6).min(1.0 - 1e-6);

        beta_warm  = beta;
        eta_c = X * beta + OffC;
        eta_c_warm = eta_c;
        mu_c  = eta_c.array().exp().matrix();
        mu_c_warm  = mu_c;
    };

    // ---------------- EM + SQUAREM loop -------------------------------------
    double ll_prev = ll;
    bool   conv    = false;
    int    em_iter = 0;

    for (; em_iter < em_maxit; ) {
        VectorXd x0 = pack_params();

        // -- First EM step --
        double ll1 = do_em_step();
        ++em_iter;
        double rel = std::fabs(ll1 - ll) / (std::fabs(ll1) + 0.1);
        ll = ll1;
        if (rel < em_tol && em_iter > 1) { conv = true; break; }
        if (em_iter >= em_maxit) break;

        VectorXd x1 = pack_params();

        // -- Second EM step --
        double ll2 = do_em_step();
        ++em_iter;
        rel = std::fabs(ll2 - ll) / (std::fabs(ll2) + 0.1);
        ll = ll2;
        if (rel < em_tol && em_iter > 1) { conv = true; break; }
        if (em_iter >= em_maxit) break;

        VectorXd x2 = pack_params();

        // -- SQUAREM extrapolation (SqS3: Varadhan & Roland 2008) --
        VectorXd r = x1 - x0;
        VectorXd v = (x2 - x1) - r;
        double sr2 = r.squaredNorm();
        double sv2 = v.squaredNorm();

        if (sv2 < 1e-30) continue;

        double alpha = -std::sqrt(sr2 / sv2);
        if (alpha > -1.0) alpha = -1.0;

        bool sq_accepted = false;
        for (int bt = 0; bt < 5 && !sq_accepted; ++bt) {
            VectorXd x_try = x0 - 2.0 * alpha * r + alpha * alpha * v;
            unpack_params(x_try);
            set_warm_from_params();
            double ll_try = obs_loglik(dist_code, zero_fam_code,
                                       (dist_code == 1 ? theta : 1.0),
                                       Y, mu_c, eta_z, W, tau);
            if (std::isfinite(ll_try) && ll_try >= ll2) {
                ll = ll_try;
                sq_accepted = true;
            } else {
                alpha = 0.5 * (alpha - 1.0);
            }
        }

        if (!sq_accepted) {
            unpack_params(x2);
            set_warm_from_params();
            obs_loglik(dist_code, zero_fam_code,
                       (dist_code == 1 ? theta : 1.0),
                       Y, mu_c, eta_z, W, tau);
        }
    }

    // ---------------- final vcov via numerical-Jac observed information ------
    MatrixXd vcov_full;
    {
        MatrixXd I_obs = obs_info_numjac(dist_code, zero_fam_code,
                                         gamma, beta,
                                         (dist_code == 1 ? theta : 1.0),
                                         Z, X, Y, W, OffZ, OffC);
        Eigen::LDLT<MatrixXd> ldlt(I_obs);
        if (ldlt.info() != Eigen::Success) {
            const Eigen::Index pp = I_obs.rows();
            vcov_full = MatrixXd::Constant(pp, pp,
                std::numeric_limits<double>::quiet_NaN());
        } else {
            vcov_full = ldlt.solve(MatrixXd::Identity(I_obs.rows(), I_obs.cols()));
        }
    }

    MatrixXd vcov_zero  = vcov_full.topLeftCorner(p_z, p_z);
    MatrixXd vcov_count = vcov_full.block(p_z, p_z, p_c, p_c);
    VectorXd se_zero    = vcov_zero.diagonal().array().abs().sqrt();
    VectorXd se_count   = vcov_count.diagonal().array().abs().sqrt();
    double   se_theta   = (dist_code == 1)
        ? std::sqrt(std::fabs(vcov_full(p_z + p_c, p_z + p_c)))
        : std::numeric_limits<double>::quiet_NaN();

    const Eigen::Index pp = p_c + p_z + (dist_code == 1 ? 1 : 0);
    MatrixXd vcov_out = MatrixXd::Zero(pp, pp);
    vcov_out.block(0, 0, p_c, p_c) = vcov_count;
    vcov_out.block(p_c, p_c, p_z, p_z) = vcov_zero;
    MatrixXd vcov_zb = vcov_full.block(0, p_z, p_z, p_c);
    vcov_out.block(p_c, 0, p_z, p_c) = vcov_zb;
    vcov_out.block(0, p_c, p_c, p_z) = vcov_zb.transpose();
    if (dist_code == 1) {
        vcov_out(pp - 1, pp - 1) = vcov_full(p_z + p_c, p_z + p_c);
        for (Eigen::Index k = 0; k < p_z; ++k) {
            vcov_out(pp - 1, p_c + k) = vcov_full(p_z + p_c, k);
            vcov_out(p_c + k, pp - 1) = vcov_full(k, p_z + p_c);
        }
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
