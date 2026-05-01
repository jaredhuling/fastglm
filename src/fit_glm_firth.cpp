#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include <RcppEigen.h>

using namespace Rcpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Map;

// Firth-penalized binomial-logit IRLS.  Restricted to the dense path and
// LLT (method = 2): Firth's bias-reducing penalty 0.5 * log|I(beta)| is
// computed from the same Cholesky factor as the WLS update, and the
// leverage h_i used to augment the working response falls out of L^{-1} X.
//
// Returned list mirrors fit_glm() with two extra entries:
//   penalized.deviance : -2 * (l(beta) + 0.5 * log|I(beta)|)
//   log.det.XtWX       : log|I(beta)| at the converged estimate
//
// [[Rcpp::export]]
List fit_glm_firth(Rcpp::NumericMatrix x, Rcpp::NumericVector y,
                   Rcpp::NumericVector weights, Rcpp::NumericVector offset,
                   Rcpp::NumericVector start, Rcpp::NumericVector mu,
                   Rcpp::NumericVector eta,
                   Function var, Function mu_eta, Function linkinv,
                   Function dev_resids, Function valideta, Function validmu,
                   double tol, int maxit, int fam_code,
                   Rcpp::Nullable<Rcpp::NumericVector> fam_params = R_NilValue)
{
    fglm::FamilyParams fp;
    if (!fam_params.isNull()) {
        Rcpp::NumericVector fpv(fam_params.get());
        if (fpv.size() >= 1) fp.theta      = fpv[0];
        if (fpv.size() >= 2) fp.var_power  = fpv[1];
        if (fpv.size() >= 3) fp.link_power = fpv[2];
    }

    // Firth currently only valid for binomial logit.  Higher-level R wrapper
    // already enforces this; double-check at the C++ boundary.
    if (fam_code != fglm::FAM_BINOMIAL_LOGIT) {
        Rcpp::stop("fit_glm_firth currently supports family = binomial(link = \"logit\") only.");
    }

    const Map<MatrixXd>  X(as<Map<MatrixXd> >(x));
    const Map<VectorXd>  Y(as<Map<VectorXd> >(y));
    const Map<VectorXd>  W(as<Map<VectorXd> >(weights));
    const Map<VectorXd>  Off(as<Map<VectorXd> >(offset));
    const Map<VectorXd>  beta_init(as<Map<VectorXd> >(start));
    const Map<VectorXd>  mu_init(as<Map<VectorXd> >(mu));
    const Map<VectorXd>  eta_init(as<Map<VectorXd> >(eta));

    if ((Eigen::Index)Y.size() != X.rows())
        Rcpp::stop("size mismatch");

    // Force method = 2 (LLT).  Firth's leverage is derived from the same
    // Cholesky factor as the beta update, so other decompositions would
    // double the cost.
    glm solver(X, Y, W, Off,
               var, mu_eta, linkinv, dev_resids,
               valideta, validmu,
               tol, maxit, /*type=*/2,
               /*is_big_matrix=*/false,
               fam_code, fp,
               /*firth=*/true);
    solver.init_parms(beta_init, mu_init, eta_init);

    int iters = solver.solve(maxit);

    VectorXd b   = solver.get_beta();
    VectorXd se  = solver.get_se();
    VectorXd m   = solver.get_mu();
    VectorXd e   = solver.get_eta();
    VectorXd wfit = solver.get_w();
    VectorXd pwt  = solver.get_weights();
    MatrixXd vc   = solver.get_vcov();
    double pen_dev = solver.get_dev();
    int rk    = solver.get_rank();
    bool conv = solver.get_converged();
    double log_det = solver.get_log_det_XtWX();

    // Recompute the unpenalized (standard) deviance for reporting.
    Eigen::Map<const Eigen::ArrayXd> y_arr(Y.data(), Y.size());
    Eigen::Map<const Eigen::ArrayXd> mu_arr(m.data(), m.size());
    Eigen::Map<const Eigen::ArrayXd> w_arr(W.data(), W.size());
    double std_dev = fglm::dev_resids_sum(fam_code, fp, y_arr, mu_arr, w_arr);

    return List::create(
        _["coefficients"]      = b,
        _["se"]                = se,
        _["fitted.values"]     = m,
        _["linear.predictors"] = e,
        _["deviance"]          = std_dev,
        _["penalized.deviance"]= pen_dev,
        _["log.det.XtWX"]      = log_det,
        _["weights"]           = wfit,
        _["prior.weights"]     = pwt,
        _["rank"]              = rk,
        _["df.residual"]       = (int)X.rows() - rk,
        _["iter"]              = iters,
        _["converged"]         = conv,
        _["cov.unscaled"]      = vc,
        _["firth"]             = true);
}
