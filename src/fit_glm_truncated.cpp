#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
#include "../inst/include/trunc_count.h"
using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

// [[Rcpp::export]]
List fit_glm_truncated(NumericMatrix x, NumericVector y, NumericVector weights,
    NumericVector offset, NumericVector start, bool negbin, double theta,
    bool estimate_theta, double tol, int maxit, double outer_tol, int outer_maxit)
{
    Map<MatrixXd> X(x.begin(), x.nrow(), x.ncol());
    Map<VectorXd> Y(y.begin(), y.size()), W(weights.begin(), weights.size()),
        O(offset.begin(), offset.size()), B(start.begin(), start.size());
    fglm::trunc::TruncFitResult fit;
    MatrixXd full;
    bool converged;
    double se_theta = NA_REAL;
    if (negbin && estimate_theta) {
        auto joint = fglm::trunc::fit_trunc_nb_joint(X, Y, W, O, theta, B,
                        tol, maxit, tol, maxit, outer_tol, outer_maxit);
        fit = joint.inner; theta = joint.theta; se_theta = joint.se_theta;
        full = joint.joint_vcov; converged = joint.outer_converged;
    } else {
        fit = fglm::trunc::fit_trunc_count(X, Y, W, O, negbin, theta, B, tol, maxit);
        full = fit.vcov; converged = fit.converged;
    }
    return List::create(_["coefficients"] = fit.beta, _["cov.unscaled"] = fit.vcov,
        _["vcov.full"] = full, _["se"] = fit.se, _["theta"] = negbin ? theta : NA_REAL,
        _["SE.theta"] = se_theta, _["linear.predictors"] = fit.eta,
        _["fitted.values"] = fit.mu_T, _["lambda"] = fit.lambda,
        _["loglik"] = fit.loglik, _["iter"] = fit.iter, _["converged"] = converged);
}
