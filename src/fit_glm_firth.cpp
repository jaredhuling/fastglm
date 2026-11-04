#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include <RcppEigen.h>

using namespace Rcpp;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Map;

// Firth-penalized IRLS.  Supports all decomposition types (method 0-5).
// The AS_mean bias-reducing adjustment modifies each working response by
//   xi_i = phi * h_i * d2mu * V(mu) / (2 * pw * dmu^3)
// where h_i is computed from the current decomposition (exact leverages).
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
                   int type, double tol, int maxit, int fam_code,
                   Rcpp::Nullable<Rcpp::NumericVector> fam_params = R_NilValue, bool inference = true,
                   Rcpp::Nullable<Rcpp::List> prepared = R_NilValue)
{
    fglm::FamilyParams fp;
    if (!fam_params.isNull()) {
        Rcpp::NumericVector fpv(fam_params.get());
        if (fpv.size() >= 1) fp.theta      = fpv[0];
        if (fpv.size() >= 2) fp.var_power  = fpv[1];
        if (fpv.size() >= 3) fp.link_power = fpv[2];
    }

    // Bias reduction is supported for all families with a native C++ fast
    // path, and for callback families via finite-difference d2mu/deta2.

    const Map<MatrixXd>  X(as<Map<MatrixXd> >(x));
    const Map<VectorXd>  Y(as<Map<VectorXd> >(y));
    const Map<VectorXd>  W(as<Map<VectorXd> >(weights));
    const Map<VectorXd>  Off(as<Map<VectorXd> >(offset));
    const Map<VectorXd>  beta_init(as<Map<VectorXd> >(start));
    const Map<VectorXd>  mu_init(as<Map<VectorXd> >(mu));
    const Map<VectorXd>  eta_init(as<Map<VectorXd> >(eta));

    if ((Eigen::Index)Y.size() != X.rows())
        Rcpp::stop("size mismatch");

    Rcpp::NumericMatrix lower(0,0), covariance(0,0);
    Rcpp::NumericVector leverage(0),adjustment(0),weight_values(0);
    Rcpp::IntegerVector weight_counts(0);
    double logdet=0;
    if(prepared.isNotNull()) {
        if(type!=2 || !(fam_code==0 || fam_code==9 || fam_code==10))
            Rcpp::stop("prepared information requires a constant-weight LLT family");
        Rcpp::List cache(prepared.get());
        lower=Rcpp::as<NumericMatrix>(cache["lower"]);
        covariance=Rcpp::as<NumericMatrix>(cache["covariance"]);
        leverage=Rcpp::as<NumericVector>(cache["leverage"]);
        adjustment=Rcpp::as<NumericVector>(cache["adjustment"]);
        weight_values=Rcpp::as<NumericVector>(cache["weight.values"]);
        weight_counts=Rcpp::as<IntegerVector>(cache["weight.counts"]);
        logdet=Rcpp::as<double>(cache["logdet"]);
        if(lower.nrow()!=X.cols() || lower.ncol()!=X.cols() || leverage.size()!=X.rows() ||
           (inference && (covariance.nrow()!=X.cols() || covariance.ncol()!=X.cols())))
            Rcpp::stop("prepared information dimensions do not match the design");
    }
    PreparedGlmInfo info(lower,leverage,covariance,adjustment,weight_values,weight_counts,logdet);
    glm solver(X, Y, W, Off,
               var, mu_eta, linkinv, dev_resids,
               valideta, validmu,
               tol, maxit, type,
               /*is_big_matrix=*/false,
               fam_code, fp,
               /*firth=*/true, prepared.isNotNull()?&info:nullptr);
    solver.init_parms(beta_init, mu_init, eta_init);

    int iters = solver.solve(maxit, inference);

    VectorXd b   = solver.get_beta();
    VectorXd se  = solver.get_se();
    const VectorXd &m = solver.get_mu_ref();
    const VectorXd &e = solver.get_eta_ref();
    VectorXd wfit = solver.get_w();
    MatrixXd vc   = solver.get_vcov();
    double pen_dev = solver.get_dev();
    int rk    = solver.get_rank();
    bool conv = solver.get_converged();
    double log_det = solver.get_log_det_XtWX();

    // Recompute the unpenalized (standard) deviance for reporting.
    Eigen::Map<const Eigen::ArrayXd> y_arr(Y.data(), Y.size());
    Eigen::Map<const Eigen::ArrayXd> mu_arr(m.data(), m.size());
    Eigen::Map<const Eigen::ArrayXd> w_arr(W.data(), W.size());
    double std_dev = fam_code >= 0 ? fglm::dev_resids_sum(fam_code, fp, y_arr, mu_arr, w_arr)
            : Rcpp::sum(Rcpp::as<NumericVector>(dev_resids(Rcpp::wrap(y_arr), Rcpp::wrap(mu_arr), Rcpp::wrap(w_arr))));
    if(fglm::bounded_firth_predictors(fam_code))
        std_dev=fglm::binomial_firth_deviance(fam_code,e,Y,W);

    List out = List::create(
        _["coefficients"]      = b,
        _["se"]                = se,
        _["fitted.values"]     = m,
        _["linear.predictors"] = e,
        _["deviance"]          = std_dev,
        _["penalized.deviance"]= pen_dev,
        _["log.det.XtWX"]      = log_det,
        _["weights"]           = wfit,
        _["prior.weights"]     = weights,
        _["rank"]              = rk,
        _["df.residual"]       = (int)X.rows() - rk,
        _["iter"]              = iters,
        _["converged"]         = conv,
        _["cov.unscaled"]      = vc,
        _["bias.dispersion"]   = solver.get_bias_dispersion(),
        _["firth"]             = true);
    solver.append_bias_diagnostics(out, Rcpp::as<double>(out["bias.dispersion"]), inference);
    // A saturated working mean can erase every weighted predictor direction.
    // A zero update in that state does not solve the intended model equations.
    if (X.cols()>0 && rk==0) {
        out["converged"]=false;
        out["convergence.reason"]="singular_working_information";
        out["adjusted.score"]=Rcpp::NumericVector(X.cols(),NA_REAL);
        out["standardized.adjusted.score"]=Rcpp::NumericVector(X.cols(),NA_REAL);
    }
    return out;
}
