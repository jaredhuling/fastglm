#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include "../inst/include/trunc_count.h"
#include <RcppEigen.h>
#include <cmath>

using namespace Rcpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXd;
using Eigen::Map;

// Initial linear predictor for the truncated count IRLS, matching pscl's
// "log(y + 0.5)" pilot.  We use eta_init = log(y) - mean(log(y)) ... no, just
// fit OLS of log(y) on X for a warm start.  This matters less than the IRLS
// algorithm itself but speeds convergence.
static VectorXd warm_start_count(const Map<MatrixXd>& X, const Map<VectorXd>& y,
                                 const Map<VectorXd>& wt, const Map<VectorXd>& off)
{
    // Use OLS regression of log(y + 1) - offset on X with sqrt(wt) weighting.
    // This is a cheap, robust initializer for the count side of the hurdle.
    const Eigen::Index n = X.rows();
    const Eigen::Index p = X.cols();
    VectorXd ly(n);
    for (Eigen::Index i = 0; i < n; ++i)
        ly[i] = std::log((y[i] > 0.0 ? y[i] : 0.5)) - off[i];
    ArrayXd w_sqrt = wt.array().max(0.0).sqrt();
    MatrixXd WX = w_sqrt.matrix().asDiagonal() * X;
    VectorXd Wly = w_sqrt.matrix().cwiseProduct(ly);
    Eigen::HouseholderQR<MatrixXd> qr(WX);
    VectorXd b = qr.solve(Wly);
    if (!b.allFinite()) b = VectorXd::Zero(p);
    return b;
}

// [[Rcpp::export]]
List fit_glm_hurdle(
    Rcpp::NumericMatrix x_count,         // n x p_c (count-part design)
    Rcpp::NumericMatrix z_zero,          // n x p_z (zero-part design)
    Rcpp::NumericVector y,
    Rcpp::NumericVector weights,
    Rcpp::NumericVector offset_count,
    Rcpp::NumericVector offset_zero,
    int                 dist_code,       // 0 = Poisson, 1 = NegBin
    int                 zero_fam_code,   // FAM_BINOMIAL_LOGIT/PROBIT/CLOGLOG/LOG
    double              init_theta,      // <=0 means estimate via Brent
    double              tol,
    int                 maxit,
    double              outer_tol,
    int                 outer_maxit,
    double              theta_tol,
    int                 theta_maxit,
    Rcpp::Function      var_fun_zero,
    Rcpp::Function      mu_eta_fun_zero,
    Rcpp::Function      linkinv_fun_zero,
    Rcpp::Function      dev_resids_fun_zero,
    Rcpp::Function      valideta_fun_zero,
    Rcpp::Function      validmu_fun_zero)
{
    const Map<MatrixXd> X(as<Map<MatrixXd> >(x_count));
    const Map<MatrixXd> Z(as<Map<MatrixXd> >(z_zero));
    const Map<VectorXd> Y(as<Map<VectorXd> >(y));
    const Map<VectorXd> W(as<Map<VectorXd> >(weights));
    const Map<VectorXd> OffC(as<Map<VectorXd> >(offset_count));
    const Map<VectorXd> OffZ(as<Map<VectorXd> >(offset_zero));

    const Eigen::Index n   = X.rows();
    const Eigen::Index p_c = X.cols();
    const Eigen::Index p_z = Z.cols();
    if (Z.rows() != n)               Rcpp::stop("z_zero / x_count row mismatch");
    if ((Eigen::Index)Y.size()  != n) Rcpp::stop("y / x_count row mismatch");
    if ((Eigen::Index)W.size()  != n) Rcpp::stop("weights size mismatch");

    // ---------------------------------------------------------------
    // (1) Zero/non-zero binomial fit on (Z, 1(y > 0)).
    // ---------------------------------------------------------------
    NumericVector y0_nv(n);
    VectorXd y0_eig(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double v = (Y[i] > 0.0) ? 1.0 : 0.0;
        y0_nv[i] = v;
        y0_eig[i] = v;
    }
    Map<VectorXd> y0(as<Map<VectorXd> >(y0_nv));

    // Initial mu_zero, eta_zero via the binomial $initialize:  mu = (y + 0.5)/2.
    NumericVector mu0_nv(n), eta0_nv(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double mi = (y0[i] + 0.5) / 2.0;
        mu0_nv[i]  = mi;
        eta0_nv[i] = std::log(mi / (1.0 - mi));   // logit; close enough for any link
    }
    Map<VectorXd> mu0(as<Map<VectorXd> >(mu0_nv));
    Map<VectorXd> eta0(as<Map<VectorXd> >(eta0_nv));

    NumericVector zero_start_nv(p_z, 0.0);
    Map<VectorXd> zero_start(as<Map<VectorXd> >(zero_start_nv));

    fglm::FamilyParams empty_params;
    glm zero_solver(Z, y0, W, OffZ,
                    var_fun_zero, mu_eta_fun_zero, linkinv_fun_zero,
                    dev_resids_fun_zero, valideta_fun_zero, validmu_fun_zero,
                    tol, maxit, /*type=*/2,
                    /*is_big_matrix=*/false,
                    zero_fam_code, empty_params);
    zero_solver.init_parms(zero_start, mu0, eta0);
    int zero_iter        = zero_solver.solve(maxit);
    bool zero_conv       = zero_solver.get_converged();
    VectorXd beta_zero   = zero_solver.get_beta();
    VectorXd se_zero     = zero_solver.get_se();
    MatrixXd vcov_zero   = zero_solver.get_vcov();
    VectorXd mu_zero     = zero_solver.get_mu();
    VectorXd eta_zero    = zero_solver.get_eta();
    double   dev_zero    = zero_solver.get_dev();

    // Log-likelihood of the zero part:  l_z = sum_i y0_i log(mu_i) + (1-y0_i) log(1-mu_i),
    // weighted by W.  (Standard binomial log-lik with the y0 indicator.)
    double loglik_zero = 0.0;
    for (Eigen::Index i = 0; i < n; ++i) {
        double pi = mu_zero[i];
        if (pi < 1e-300) pi = 1e-300;
        if (pi > 1.0 - 1e-300) pi = 1.0 - 1e-300;
        const double yi = y0[i];
        loglik_zero += W[i] * (yi * std::log(pi) + (1.0 - yi) * std::log(1.0 - pi));
    }

    // ---------------------------------------------------------------
    // (2) Truncated count fit on the y > 0 subset.
    // ---------------------------------------------------------------
    int n_pos = 0;
    for (Eigen::Index i = 0; i < n; ++i) if (Y[i] > 0.0) ++n_pos;
    if (n_pos < (int)p_c)
        Rcpp::stop("hurdle count component has fewer positive observations than parameters");

    MatrixXd Xp(n_pos, p_c);
    VectorXd Yp(n_pos), Wp(n_pos), OffCp(n_pos);
    {
        Eigen::Index k = 0;
        for (Eigen::Index i = 0; i < n; ++i) {
            if (Y[i] > 0.0) {
                Xp.row(k)  = X.row(i);
                Yp[k]      = Y[i];
                Wp[k]      = W[i];
                OffCp[k]   = OffC[i];
                ++k;
            }
        }
    }
    // Map<...> wrappers required by fit_trunc_count's Eigen::Ref<const VectorXd>;
    // we already have plain VectorXd which is fine to pass via .ref().
    VectorXd beta_count_init = warm_start_count(
        Map<MatrixXd>(Xp.data(), Xp.rows(), Xp.cols()),
        Map<VectorXd>(Yp.data(), Yp.size()),
        Map<VectorXd>(Wp.data(), Wp.size()),
        Map<VectorXd>(OffCp.data(), OffCp.size()));

    fglm::trunc::TruncFitResult count_res;
    double theta_hat   = NA_REAL;
    double se_theta    = NA_REAL;
    int    outer_iter  = 0;
    int    theta_iter  = 0;
    bool   count_conv  = true;

    if (dist_code == 0) {
        // Poisson hurdle: single IRLS, no theta.
        count_res = fglm::trunc::fit_trunc_pois_log(
            Xp, Yp, Wp, OffCp, beta_count_init, tol, maxit);
        count_conv = count_res.converged;
    } else {
        // NB hurdle: joint (beta, theta) MLE.
        fglm::trunc::TruncNbJointResult joint = fglm::trunc::fit_trunc_nb_joint(
            Xp, Yp, Wp, OffCp,
            init_theta,
            beta_count_init,
            tol, maxit,
            theta_tol, theta_maxit,
            outer_tol, outer_maxit);
        count_res  = joint.inner;
        theta_hat  = joint.theta;
        se_theta   = joint.se_theta;
        outer_iter = joint.outer_iter;
        theta_iter = joint.theta_iter;
        count_conv = joint.outer_converged;
    }

    // ---------------------------------------------------------------
    // (3) Joint return.
    // ---------------------------------------------------------------
    const double loglik_total = loglik_zero + count_res.loglik;
    const int    df_count     = (int)p_c + ((dist_code == 1) ? 1 : 0);
    const int    df           = df_count + (int)p_z;

    return List::create(
        _["coefficients_count"]   = count_res.beta,
        _["coefficients_zero"]    = beta_zero,
        _["se_count"]             = count_res.se,
        _["se_zero"]              = se_zero,
        _["vcov_count"]           = count_res.vcov,
        _["vcov_zero"]            = vcov_zero,
        _["lambda"]               = count_res.lambda,        // count-part rate, on the y>0 subset
        _["mu_count_truncated"]   = count_res.mu_T,           // E[Y | Y>0] on the y>0 subset
        _["eta_count"]            = count_res.eta,
        _["mu_zero"]              = mu_zero,
        _["eta_zero"]             = eta_zero,
        _["theta"]                = theta_hat,
        _["SE.theta"]             = se_theta,
        _["loglik"]               = loglik_total,
        _["loglik_zero"]          = loglik_zero,
        _["loglik_count"]         = count_res.loglik,
        _["dev_zero"]             = dev_zero,
        _["iter_count"]           = count_res.iter,
        _["iter_zero"]            = zero_iter,
        _["outer_iter"]           = outer_iter,
        _["theta_iter"]           = theta_iter,
        _["converged"]            = (count_conv && zero_conv),
        _["dist_code"]            = dist_code,
        _["zero_fam_code"]        = zero_fam_code,
        _["n"]                    = (int)n,
        _["n_positive"]           = n_pos,
        _["df"]                   = df
    );
}
