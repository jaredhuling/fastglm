#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include "../inst/include/nb_theta.h"
#include <RcppEigen.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>

using namespace Rcpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::Map;

// NB2 log-likelihood (per-observation, weighted): see e.g. Cameron & Trivedi.
// l_i = lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
//       + theta * log(theta / (theta + mu)) + y * log(mu / (theta + mu))
// Returns sum over i of wt_i * l_i.
static double nb_loglik(double theta,
                        const ArrayXd& y, const ArrayXd& mu, const ArrayXd& wt)
{
    using boost::math::lgamma;
    const double lg_th = lgamma(theta);
    double s = 0.0;
    for (Eigen::Index i = 0; i < y.size(); ++i) {
        const double yi = y[i], mi = mu[i], wi = wt[i];
        const double tm = theta + mi;
        // log(theta / tm) = -log1p(mu/theta) when mu << theta.
        double log_ratio_th = (mi < 1e-3 * theta)
                              ? -std::log1p(mi / theta)
                              : std::log(theta / tm);
        // log(mu / tm) = log(mu) - log(tm); skip if y == 0.
        double y_part = (yi > 0.0) ? yi * (std::log(mi) - std::log(tm)) : 0.0;
        s += wi * (lgamma(yi + theta) - lg_th - lgamma(yi + 1.0)
                   + theta * log_ratio_th + y_part);
    }
    return s;
}

// [[Rcpp::export]]
List fit_glm_nb(Rcpp::NumericMatrix x, Rcpp::NumericVector y,
                Rcpp::NumericVector weights, Rcpp::NumericVector offset,
                Rcpp::NumericVector start, Rcpp::NumericVector mu_init,
                Rcpp::NumericVector eta_init,
                Rcpp::Function var_fun, Rcpp::Function mu_eta_fun,
                Rcpp::Function linkinv_fun, Rcpp::Function dev_resids_fun,
                Rcpp::Function valideta_fun, Rcpp::Function validmu_fun,
                int    type,
                double tol,
                int    maxit,
                int    fam_code,
                double init_theta,
                double theta_tol,
                int    theta_maxit,
                int    outer_maxit,
                double outer_tol)
{
    const Map<MatrixXd>  X(as<Map<MatrixXd> >(x));
    const Map<VectorXd>  Y(as<Map<VectorXd> >(y));
    const Map<VectorXd>  W(as<Map<VectorXd> >(weights));
    const Map<VectorXd>  Off(as<Map<VectorXd> >(offset));
    const Map<VectorXd>  beta0(as<Map<VectorXd> >(start));
    const Map<VectorXd>  mu0(as<Map<VectorXd> >(mu_init));
    const Map<VectorXd>  eta0(as<Map<VectorXd> >(eta_init));

    if (fam_code < 17 || fam_code > 19) {
        // Driver requires the NB native fast path (FAM_NB_LOG=17, _SQRT=18,
        // _IDENTITY=19).  Other links fall back to base glm.nb.
        Rcpp::stop("fit_glm_nb requires a native NB fam_code (17 / 18 / 19).");
    }

    const Eigen::Index n = X.rows();
    if ((Eigen::Index)Y.size() != n) Rcpp::stop("size mismatch");

    // Initial theta -- caller-supplied (typically from theta.ml on a Poisson
    // pilot), or method-of-moments if non-positive.
    double theta = init_theta;
    if (!std::isfinite(theta) || theta <= 0.0) {
        theta = fglm::nb::init_theta_mom(
            ArrayXd(Y.array()),
            ArrayXd(mu0.array().max(1e-3)),
            ArrayXd(W.array()));
    }

    fglm::FamilyParams fam_params;
    fam_params.theta = theta;

    // Build the IRLS solver once; we'll set_fam_params() between sweeps.
    glm solver(X, Y, W, Off,
               var_fun, mu_eta_fun, linkinv_fun, dev_resids_fun,
               valideta_fun, validmu_fun,
               tol, maxit, type,
               /*is_big_matrix=*/false,
               fam_code, fam_params);
    solver.init_parms(beta0, mu0, eta0);

    // Joint Fisher decrement (deviance units) at beta and theta; see (c).
    // The coefficient block is the decrement of the IRLS update the solver
    // would take at the updated theta (glm::trial_decrement): g' I^{-1} g at
    // an interior point, and the decrement of the constrained update at a
    // boundary optimum (sqrt and identity links), where g is not zero.
    auto joint_decrement = [&](const VectorXd& mu, double th) {
        fglm::FamilyParams fp; fp.theta = th;
        solver.set_fam_params(fp);
        const double lam_b = solver.trial_decrement();
        const ArrayXd ya = Y.array(), ma = mu.array().max(1e-12), wa = W.array();
        const double st = fglm::nb::score_theta(th, ya, ma, wa);
        const double it = fglm::nb::info_theta(th, ya, ma, wa);
        double lam_t = (it > 0) ? st * st / it : (st == 0 ? 0.0 : std::numeric_limits<double>::infinity());
        // At the cap the score may point beyond it (the Poisson limit, or
        // theta -> 0); the cap is then where the theta step stops.
        if ((th >= 1e8 * (1 - 1e-12) && st > 0) || (th <= 1e-6 * (1 + 1e-12) && st < 0)) lam_t = 0;
        return lam_b + lam_t;
    };
    double last_decrement = std::numeric_limits<double>::infinity();

    int    outer_iter   = 0;
    int    inner_iters  = 0;
    int    theta_iters  = 0;
    bool   outer_conv   = false;
    bool   inner_conv_last = true;

    for (outer_iter = 0; outer_iter < outer_maxit; ++outer_iter) {
        // (a) beta-step: IRLS at fixed theta.
        inner_iters       += solver.solve(maxit);
        inner_conv_last    = solver.get_converged();

        VectorXd mu_hat = solver.get_mu();
        VectorXd beta_now = solver.get_beta();

        // (b) theta-step: 1-D MLE on the score.
        ArrayXd y_arr  = Y.array();
        ArrayXd mu_arr = mu_hat.array().max(1e-12);   // guard log
        ArrayXd w_arr  = W.array();

        theta = fglm::nb::mle_theta(theta, y_arr, mu_arr, w_arr,
                                    theta_tol, theta_maxit);
        ++theta_iters;

        // (c) convergence: the joint Fisher decrement at (beta_now, theta),
        // the deviance decrease predicted by one more scoring step in both
        // blocks, relative to the deviance (the rule the IRLS solver uses).
        // The expected cross information between beta and theta is zero
        // (each term is proportional to E[y - mu] = 0), so the decrement is
        // the sum of the two block decrements:
        //   beta:  g' I^{-1} g, g = X' [w (y - mu) mu.eta / V],
        //          I = X' diag(w mu.eta^2 / V) X, at the updated theta;
        //   theta: s^2 / I_theta (s is ~0 right after the theta update),
        //          omitted at the cap when the score points beyond it.
        // A test on raw changes in beta cannot be met when the likelihood is
        // flat in some direction of beta (e.g. very small theta): beta then
        // moves by 1e-5 per outer iteration while the log-likelihood changes
        // by 1e-9.
        last_decrement = joint_decrement(mu_hat, theta);
        if (last_decrement / (0.1 + std::abs(solver.get_dev())) < outer_tol) {
            outer_conv = true;
            // Polish: two more (beta, theta) rounds. Each inner solve takes
            // at least one full scoring step, which near the optimum squares
            // the error, while a solve warm-started close to the optimum can
            // stop with its decrement just below tol (coefficient error of
            // order sqrt(tol)). Without these rounds, coefficients agree with
            // a tightly converged reference only to about 1e-6 (relative).
            for (int polish = 0; polish < 2; ++polish) {
                fglm::FamilyParams fp_p; fp_p.theta = theta;
                solver.set_fam_params(fp_p);
                inner_iters += solver.solve(maxit);
                const ArrayXd mu_p = solver.get_mu().array().max(1e-12);
                theta = fglm::nb::mle_theta(theta, ArrayXd(Y.array()), mu_p, ArrayXd(W.array()),
                                            theta_tol, theta_maxit);
                ++theta_iters;
            }
            // Push the final theta through one more IRLS pass so the
            // coefficients and vcov reflect it.
            fglm::FamilyParams fp_final; fp_final.theta = theta;
            solver.set_fam_params(fp_final);
            VectorXd eta_now = solver.get_eta();
            inner_iters += solver.solve(maxit);
            inner_conv_last = solver.get_converged();
            break;
        }

        // (d) re-init solver with new theta, warm-starting from current beta.
        fglm::FamilyParams fp_new; fp_new.theta = theta;
        solver.set_fam_params(fp_new);

        VectorXd eta_now = solver.get_eta();
        Map<VectorXd> b_map(beta_now.data(), beta_now.size());
        Map<VectorXd> m_map(mu_hat.data(),    mu_hat.size());
        Map<VectorXd> e_map(eta_now.data(),   eta_now.size());
        solver.init_parms(b_map, m_map, e_map);

    }

    // Extract final fit.
    VectorXd beta = solver.get_beta();
    VectorXd se   = solver.get_se();
    VectorXd mu   = solver.get_mu();
    VectorXd eta  = solver.get_eta();
    VectorXd wts  = solver.get_w();
    VectorXd pwts = solver.get_weights();
    MatrixXd vcov = solver.get_vcov();
    double   dev  = solver.get_dev();
    int      rank = solver.get_rank();

    // SE and twologlik for theta.
    ArrayXd y_arr  = Y.array();
    ArrayXd mu_arr = mu.array().max(1e-12);
    ArrayXd w_arr  = W.array();
    double info_th = fglm::nb::info_theta(theta, y_arr, mu_arr, w_arr);
    double se_th   = (info_th > 0.0) ? std::sqrt(1.0 / info_th)
                                     : std::numeric_limits<double>::quiet_NaN();
    double twolog  = 2.0 * nb_loglik(theta, y_arr, mu_arr, w_arr);

    return List::create(
        _["coefficients"]      = beta,
        _["se"]                = se,
        _["fitted.values"]     = mu,
        _["linear.predictors"] = eta,
        _["deviance"]          = dev,
        _["weights"]           = wts,
        _["prior.weights"]     = pwts,
        _["rank"]              = rank,
        _["df.residual"]       = (int)X.rows() - rank,
        _["iter"]              = inner_iters,
        _["iter.theta"]        = theta_iters,
        _["outer.iter"]        = outer_iter + 1,
        _["converged"]         = outer_conv && inner_conv_last,
        _["cov.unscaled"]      = vcov,
        _["theta"]             = theta,
        _["SE.theta"]          = se_th,
        _["twologlik"]         = twolog,
        _["outer.decrement"]   = last_decrement,
        // "converged_boundary" when the final coefficient fit held rows near
        // a bound of the sqrt or identity link; otherwise the outer result.
        _["convergence.reason"] = (outer_conv && inner_conv_last && solver.get_stop_reason() == "converged_boundary")
                                   ? std::string("converged_boundary")
                                   : std::string(outer_conv && inner_conv_last ? "converged" : "iteration_limit")
    );
}
