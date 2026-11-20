#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include "../inst/include/nb_theta.h"
#include "../inst/include/zi.h"
#include "../inst/include/chunk_source.h"
#include "../inst/include/parallel_reduce.h"
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
typedef Eigen::Ref<const MatrixXd> MatRef;
typedef Eigen::Ref<const VectorXd> VecRef;

// Threading. The row loops below run over fixed row blocks of
// fglm::default_chunk_rows() rows, in parallel when there are several:
// elementwise work with fglm_par::parallel_blocks, sums over rows with
// fglm_par::ordered_block_sum, and products X' v with
// fglm::accumulate_xtv_blocked. The blocks and the order of the additions
// do not depend on the thread count, so every result is the same for every
// thread count. The probit zero link calls R's math library (R::pnorm5,
// R::dnorm4) and is evaluated serially.

// out = A b + off, over fixed row blocks.
static VectorXd lin_pred(const MatRef& A, const VecRef& b,
                         const Eigen::Ref<const VectorXd>& off)
{
    VectorXd out(A.rows());
    fglm::apply_X_streamed(A, b, off, out);
    return out;
}

// exp(eta) elementwise, over fixed row blocks.
static VectorXd exp_blocks(const VectorXd& eta)
{
    VectorXd out(eta.size());
    fglm_par::parallel_blocks(eta.size(), fglm::default_chunk_rows(),
                              [&](Eigen::Index r, Eigen::Index k) {
        out.segment(r, k) = eta.segment(r, k).array().exp().matrix();
    });
    return out;
}

// A' v, summed over row blocks in a fixed order.
static VectorXd xtv(const MatRef& A, const VectorXd& v)
{
    VectorXd out;
    fglm::accumulate_xtv_blocked(A, v, out);
    return out;
}

// ---------------------------------------------------------------------------
// Score helpers (analytical, vectorized where possible).
// ---------------------------------------------------------------------------

static VectorXd score_gamma(int zero_fam_code,
                            const VecRef& tau, const VecRef& eta_z,
                            const VecRef& wts, const MatRef& Z)
{
    const Eigen::Index n = Z.rows();
    VectorXd s_per_obs(n);
    if (zero_fam_code == fglm::FAM_BINOMIAL_LOGIT) {
        ArrayXd pi = 1.0 / (1.0 + (-eta_z.array()).exp());
        s_per_obs = (wts.array() * (tau.array() - pi)).matrix();
    } else {
        const double *wp = wts.data(), *tp = tau.data(), *ep = eta_z.data();
        double* sp = s_per_obs.data();
        auto rows = [=](Eigen::Index r, Eigen::Index k) {
            for (Eigen::Index i = r; i < r + k; ++i)
                sp[i] = wp[i] * fglm::zi::score_eta_z(zero_fam_code, tp[i], ep[i]);
        };
        if (zero_fam_code == fglm::FAM_BINOMIAL_PROBIT) rows(0, n);
        else fglm_par::parallel_blocks(n, fglm::default_chunk_rows(), rows);
    }
    return xtv(Z, s_per_obs);
}

static VectorXd score_beta(int dist_code, double theta,
                           const VecRef& tau, const VecRef& y,
                           const VecRef& mu,
                           const VecRef& wts, const MatRef& X)
{
    ArrayXd w = wts.array() * (1.0 - tau.array());
    VectorXd s_per_obs;
    if (dist_code == 0) {
        s_per_obs = (w * (y.array() - mu.array())).matrix();
    } else {
        s_per_obs = (w * theta * (y.array() - mu.array()) / (theta + mu.array())).matrix();
    }
    return xtv(X, s_per_obs);
}

static double score_theta_zi(double theta,
                             const VecRef& tau, const VecRef& y,
                             const VecRef& mu, const VecRef& wts)
{
    // Cancellation-free per-row score (see fglm::zi::score_theta_one), with
    // digamma(theta) computed once.
    const double dig_th = boost::math::digamma(theta);
    const double *tp = tau.data(), *yp = y.data(), *mp = mu.data(), *wp = wts.data();
    return fglm_par::ordered_block_sum(y.size(), fglm::default_chunk_rows(),
                                       [=](Eigen::Index r, Eigen::Index k) {
        double s = 0.0;
        for (Eigen::Index i = r; i < r + k; ++i)
            s += wp[i] * (1.0 - tp[i]) * fglm::nb::score_theta_row(theta, yp[i], mp[i], dig_th);
        return s;
    });
}

// ---------------------------------------------------------------------------
// Theta MLE for the ZI count component (NB only).
// ---------------------------------------------------------------------------
static double mle_theta_zi(double theta_init,
                           const VecRef& tau, const VecRef& y,
                           const VecRef& mu, const VecRef& wts,
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

    return fglm::brent_root(score, a, b, sa, sb, tol, maxit);
}

// ---------------------------------------------------------------------------
// Observed log-likelihood + posterior tau.
// ---------------------------------------------------------------------------
static double obs_loglik(int dist_code, int zero_fam_code, double theta,
                         const VecRef& y, const VecRef& mu,
                         const VecRef& eta_z, const VecRef& wts,
                         VectorXd& tau)
{
    return fglm::zi::obs_loglik_and_tau(
        dist_code, zero_fam_code, theta, y, mu, eta_z, wts, tau);
}

// ---------------------------------------------------------------------------
// Full analytical score vector.
// ---------------------------------------------------------------------------
static VectorXd full_score(int dist_code, int zero_fam_code,
                           const VecRef& gamma, const VecRef& beta,
                           double theta,
                           const MatRef& Z, const MatRef& X,
                           const VecRef& y, const VecRef& wts,
                           const VecRef& off_z, const VecRef& off_c)
{
    const Eigen::Index n   = X.rows();
    const Eigen::Index p_z = Z.cols();
    const Eigen::Index p_c = X.cols();
    VectorXd eta_z = lin_pred(Z, gamma, off_z);
    VectorXd eta_c = lin_pred(X, beta, off_c);
    VectorXd mu = exp_blocks(eta_c);
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
                                const VecRef& gamma, const VecRef& beta,
                                double theta,
                                const MatRef& Z, const MatRef& X,
                                const VecRef& y, const VecRef& wts,
                                const VecRef& off_z, const VecRef& off_c)
{
    const Eigen::Index p_z = Z.cols();
    const Eigen::Index p_c = X.cols();
    const Eigen::Index p   = p_z + p_c + (dist_code == 1 ? 1 : 0);

    MatrixXd J(p, p);
    const double rel = std::cbrt(std::numeric_limits<double>::epsilon());
    for (Eigen::Index k = 0; k < p; ++k) {
        VectorXd gp = gamma, gm = gamma, bp = beta, bm = beta;
        double tp = theta, tm = theta, h;
        if (k < p_z) {
            h = rel * (1 + std::abs(gamma[k])); gp[k] += h; gm[k] -= h;
        } else if (k < p_z + p_c) {
            Eigen::Index j = k - p_z;
            h = rel * (1 + std::abs(beta[j])); bp[j] += h; bm[j] -= h;
        } else {
            h = rel * theta; tp += h; tm -= h;
        }
        J.col(k) = (full_score(dist_code, zero_fam_code, gp, bp, tp, Z, X, y, wts, off_z, off_c)
                  - full_score(dist_code, zero_fam_code, gm, bm, tm, Z, X, y, wts, off_z, off_c)) / (2*h);
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
        eta0_z[i] = fglm::zi::zero_linkfun(zero_fam_code, mi);   // on the zero link's scale
    }
    // Valid for the zero link (see fglm::zi::zero_model_start).
    VectorXd zero_start = fglm::zi::zero_model_start(zero_fam_code, Z, y0, wts, off_z);

    fglm::FamilyParams fp_empty;
    glm gamma_solver(Z,
                     Map<VectorXd>(y0.data(), n),
                     wts, off_z,
                     var_fun_z, mu_eta_fun_z, linkinv_fun_z,
                     dev_resids_fun_z, valideta_fun_z, validmu_fun_z,
                     tol, maxit, 0, false, zero_fam_code, fp_empty);   // pivoted QR: see below
    gamma_solver.init_parms(Map<VectorXd>(zero_start.data(), zero_start.size()),
                            Map<VectorXd>(mu0_z.data(), n),
                            Map<VectorXd>(eta0_z.data(), n));
    gamma_solver.solve(maxit, false);
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
    beta_solver.solve(maxit, false);
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
    VectorXd eta_z = lin_pred(Z, gamma, OffZ);
    VectorXd eta_c = lin_pred(X, beta, OffC);
    VectorXd mu_c  = exp_blocks(eta_c);
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
                     tol, maxit, 0, false, zero_fam_code, fp_gamma);   // pivoted QR: the zero model can approach pi = 0, where X'WX is too ill-conditioned for the Cholesky factorization

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
        gamma_solver.solve(maxit, false);
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
        beta_solver.solve(maxit, false);
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
        // theta enters on the log scale, so the extrapolation cannot leave
        // (0, inf) and treats large theta multiplicatively.
        if (dist_code == 1) x[n_params - 1] = std::log(theta);
        return x;
    };

    auto unpack_params = [&](const VectorXd& x) {
        gamma = x.head(p_z);
        beta  = x.segment(p_z, p_c);
        if (dist_code == 1)
            theta = std::exp(std::min(std::log(1e8), std::max(std::log(1e-6), x[n_params - 1])));
    };

    // Recompute eta/mu/warm-starts from current gamma,beta (not from IRLS)
    auto set_warm_from_params = [&]() {
        gamma_warm = gamma;
        eta_z = lin_pred(Z, gamma, OffZ);
        eta_z_warm = eta_z;
        ArrayXd eta_arr = eta_z.array();
        ArrayXd mu_arr(n);
        fglm::linkinv(zero_fam_code, fp_gamma, eta_arr, mu_arr);
        mu_z_warm = mu_arr.array().max(1e-6).min(1.0 - 1e-6);

        beta_warm  = beta;
        eta_c = lin_pred(X, beta, OffC);
        eta_c_warm = eta_c;
        mu_c  = exp_blocks(eta_c);
        mu_c_warm  = mu_c;
    };

    // ---------------- EM + SQUAREM loop -------------------------------------
    bool   conv    = false;
    int    em_iter = 0;
    // Trace (trace.h): one line per EM step and per Newton step below, on
    // the calling thread; passes count the log likelihood evaluations.
    const fglm_trace::Tracer tracer;
    int evaluations = 0;

    for (; em_iter < em_maxit; ) {
        Rcpp::checkUserInterrupt();
        VectorXd x0 = pack_params();

        // -- First EM step --
        double ll1 = do_em_step();
        ++em_iter; ++evaluations;
        tracer.iteration(em_iter, evaluations, ll1, 0, "loglik");
        double rel = std::fabs(ll1 - ll) / (std::fabs(ll1) + 0.1);
        ll = ll1;
        if (rel < em_tol && em_iter > 1) { conv = true; break; }
        if (em_iter >= em_maxit) break;

        VectorXd x1 = pack_params();

        // -- Second EM step --
        double ll2 = do_em_step();
        ++em_iter; ++evaluations;
        tracer.iteration(em_iter, evaluations, ll2, 0, "loglik");
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
            ++evaluations;
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

    // ---------------- Newton finish and convergence certificate ---------------
    // The EM stopping rule (relative change in the log likelihood) cannot
    // tell a stalled or slowly creeping iteration from a maximum. Finish with
    // safeguarded Newton steps on the observed information (the numerical
    // Jacobian of the score, as for the covariance below), halved until the
    // log likelihood does not decrease, and report convergence only when the
    // Newton decrement s' I^{-1} s is below em_tol * (0.1 + |2 loglik|), the
    // rule of the GLM solvers. theta is left out of the Newton system when it
    // is at a bound and its score points beyond it (the zero-inflated Poisson
    // limit as theta -> inf).
    bool   theta_at_bound = false;
    double em_decrement = std::numeric_limits<double>::infinity();
    int    newton_iter = 0;
    {
        const double th_lo = 1e-6, th_hi = 1e8;
        auto loglik_at = [&](const VectorXd& g, const VectorXd& b, double th) {
            const VectorXd ez = lin_pred(Z, g, OffZ);
            const VectorXd mc = exp_blocks(lin_pred(X, b, OffC));
            VectorXd tau_tmp(n);
            return obs_loglik(dist_code, zero_fam_code, (dist_code == 1 ? th : 1.0),
                              Y, mc, ez, W, tau_tmp);
        };
        bool certified = false;
        // Newton steps allowed. Where the zero-inflation probability tends to
        // zero, each step gains only a constant fraction (about exp(-1)) of
        // the remaining log likelihood, so certification can take dozens.
        const int newton_max = 100;
        for (int k = 0; k <= newton_max; ++k) {
            Rcpp::checkUserInterrupt();
            VectorXd s = full_score(dist_code, zero_fam_code, gamma, beta,
                                    (dist_code == 1 ? theta : 1.0), Z, X, Y, W, OffZ, OffC);
            MatrixXd I = obs_info_numjac(dist_code, zero_fam_code, gamma, beta,
                                         (dist_code == 1 ? theta : 1.0), Z, X, Y, W, OffZ, OffC);
            const Eigen::Index pt = p_z + p_c;
            // Newton in phi = log(theta): with theta near 1e4 the theta score
            // is O(1/theta^2) and the information O(1/theta^4), so the system
            // in theta is badly scaled. s_phi = theta s_theta,
            // I_phiphi = theta^2 I_thth - theta s_theta, I_phibeta = theta I_thbeta.
            if (dist_code == 1) {
                const double st = s[pt];
                I.row(pt) *= theta; I.col(pt) *= theta;
                I(pt, pt) -= theta * st;
                s[pt] = theta * st;
            }
            theta_at_bound = dist_code == 1 &&
                ((theta >= th_hi * (1 - 1e-12) && s[pt] > 0) || (theta <= th_lo * (1 + 1e-12) && s[pt] < 0));
            Eigen::Index m = (dist_code == 1 && !theta_at_bound) ? pt + 1 : pt;
            if (!s.head(m).allFinite() || !I.topLeftCorner(m, m).allFinite()) break;
            Eigen::LLT<MatrixXd> chol(I.topLeftCorner(m, m));
            // Where the likelihood is not concave in log(theta) (it is flat
            // there for large theta), the information is not positive definite
            // through theta. First maximize over log(theta) with the other
            // parameters fixed (golden section on [log(theta) - 3,
            // log(theta) + 3] within the bounds) and continue; once that no
            // longer raises the likelihood, theta is at a maximum along its
            // own direction and is held while Newton continues with the other
            // parameters. A test on the size of the theta score alone does not
            // suffice: without concavity a larger move in log(theta) can still
            // raise the likelihood by more than the tolerance.
            if (chol.info() != Eigen::Success && m == pt + 1) {
                const double rounding = 64 * std::numeric_limits<double>::epsilon() * (1 + std::abs(ll));
                if (k < newton_max) {
                    const double phi0 = std::log(theta);
                    double a = std::max(std::log(th_lo), phi0 - 3), b = std::min(std::log(th_hi), phi0 + 3);
                    const double gr = 0.5 * (std::sqrt(5.0) - 1);
                    auto f = [&](double phi) { return loglik_at(gamma, beta, std::exp(phi)); };
                    double c = b - gr * (b - a), d = a + gr * (b - a), fc = f(c), fd = f(d);
                    for (int it = 0; it < 80 && b - a > 1e-10 * (1 + std::abs(phi0)); ++it) {
                        if (fc >= fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
                        else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
                    }
                    const double phi = fc >= fd ? c : d, lc = std::max(fc, fd);
                    if (std::isfinite(lc) && lc > ll + rounding) {
                        theta = std::exp(phi); ll = lc; ++newton_iter;
                        tracer.iteration(em_iter + newton_iter, evaluations, ll, 0, "loglik");
                        continue;
                    }
                }
                m = pt;
                chol.compute(I.topLeftCorner(m, m));
            }
            if (chol.info() != Eigen::Success) break;
            const VectorXd step = chol.solve(s.head(m));
            em_decrement = s.head(m).dot(step);
            const bool small = std::isfinite(em_decrement) && em_decrement / (0.1 + std::abs(2 * ll)) < em_tol;
            if (small && certified) break;      // after the polishing step below
            if (k == newton_max) { certified = small; break; }
            // Once certified, take one more (polishing) Newton step: near the
            // maximum it squares the error, which matters where the likelihood
            // is flat (e.g. in theta) and a small decrement still allows a
            // visible coefficient error.
            if (small) certified = true;
            const double rounding = 64 * std::numeric_limits<double>::epsilon() * (1 + std::abs(ll));
            double frac = 1; bool accepted = false; int halvings = 0;
            for (int half = 0; half < 30; ++half, frac *= 0.5) {
                ++evaluations; halvings = half;
                const VectorXd g = gamma + frac * step.head(p_z);
                const VectorXd b = beta + frac * step.segment(p_z, p_c);
                double th = theta;
                if (m > pt) th = std::min(th_hi, std::max(th_lo, theta * std::exp(frac * step[pt])));
                const double lc = loglik_at(g, b, th);
                if (std::isfinite(lc) && lc >= ll - rounding) {
                    gamma = g; beta = b; theta = th; ll = lc; accepted = true; break;
                }
            }
            if (!accepted) break;
            ++newton_iter;
            tracer.iteration(em_iter + newton_iter, evaluations, ll, halvings, "loglik");
            if (certified) break;
        }
        conv = certified;
        // Refresh the fitted quantities and posterior zero probabilities.
        set_warm_from_params();
        ll = obs_loglik(dist_code, zero_fam_code, (dist_code == 1 ? theta : 1.0),
                        Y, mu_c, eta_z, W, tau);
    }

    // ---------------- final vcov via numerical-Jac observed information ------
    MatrixXd vcov_full;
    {
        MatrixXd I_obs = obs_info_numjac(dist_code, zero_fam_code,
                                         gamma, beta,
                                         (dist_code == 1 ? theta : 1.0),
                                         Z, X, Y, W, OffZ, OffC);
        Eigen::LDLT<MatrixXd> ldlt(I_obs);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive() || ldlt.rcond() < 1e-12) {
            const Eigen::Index pp = I_obs.rows();
            vcov_full = MatrixXd::Constant(pp, pp,
                std::numeric_limits<double>::quiet_NaN());
        } else {
            vcov_full = ldlt.solve(MatrixXd::Identity(I_obs.rows(), I_obs.cols()));
        }
    }

    MatrixXd vcov_zero  = vcov_full.topLeftCorner(p_z, p_z);
    MatrixXd vcov_count = vcov_full.block(p_z, p_z, p_c, p_c);
    VectorXd se_zero    = vcov_zero.diagonal().array().sqrt();
    VectorXd se_count   = vcov_count.diagonal().array().sqrt();
    double   se_theta   = (dist_code == 1)
        ? std::sqrt(vcov_full(p_z + p_c, p_z + p_c))
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

    List out = List::create(
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
    out["newton.iter"]    = newton_iter;
    out["em.decrement"]   = em_decrement;
    out["theta.at.bound"] = theta_at_bound;
    return out;
}
