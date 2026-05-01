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

    int    outer_iter   = 0;
    int    inner_iters  = 0;
    int    theta_iters  = 0;
    bool   outer_conv   = false;
    double theta_prev   = theta;
    VectorXd beta_prev  = beta0;
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

        theta_prev = theta;
        theta = fglm::nb::mle_theta(theta, y_arr, mu_arr, w_arr,
                                    theta_tol, theta_maxit);
        ++theta_iters;

        // (c) convergence: ||Δβ||_∞ + |Δθ|/θ.
        double db = (beta_now - beta_prev).cwiseAbs().maxCoeff();
        double dt = std::fabs(theta - theta_prev) / std::max(theta, 1e-12);
        if (db + dt < outer_tol && outer_iter > 0) {
            outer_conv = true;
            // Push the new theta through one final IRLS pass so vcov
            // reflects it.
            fglm::FamilyParams fp_final; fp_final.theta = theta;
            solver.set_fam_params(fp_final);
            VectorXd eta_now = solver.get_eta();
            inner_iters += solver.solve(maxit);
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

        beta_prev = beta_now;
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
        _["twologlik"]         = twolog
    );
}
