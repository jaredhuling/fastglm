#define EIGEN_DONT_PARALLELIZE


// bigmemory's BigMatrix.h uses std::copy and std::back_inserter but relies on
// transitive includes for <algorithm>/<iterator>. libc++ (clang >= 23) no longer
// provides those transitively, so include them explicitly before the header.
#include <algorithm>
#include <iterator>

#include <bigmemory/MatrixAccessor.hpp>
#include <bigmemory/BigMatrix.h>

#include <Rcpp.h>
#include "../inst/include/glm.h"
#include "../inst/include/glm_sparse.h"
#include <RcppEigen.h>
#include <memory>

using namespace Rcpp;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Map;

typedef MatrixXd::Index Index;


List fastglm(Rcpp::NumericMatrix Xs,
             Rcpp::NumericVector ys,
             Rcpp::NumericVector weightss,
             Rcpp::NumericVector offsets,
             Rcpp::NumericVector starts,
             Rcpp::NumericVector mus,
             Rcpp::NumericVector etas,
             Function var,
             Function mu_eta,
             Function linkinv,
             Function dev_resids,
             Function valideta,
             Function validmu,
             int type,
             double tol,
             int maxit,
             int fam_code,
             const fglm::FamilyParams& fam_params)
{
    const Map<MatrixXd>  X(as<Map<MatrixXd> >(Xs));
    const Map<VectorXd>  y(as<Map<VectorXd> >(ys));
    const Map<VectorXd>  weights(as<Map<VectorXd> >(weightss));
    const Map<VectorXd>  offset(as<Map<VectorXd> >(offsets));
    const Map<VectorXd>  beta_init(as<Map<VectorXd> >(starts));
    const Map<VectorXd>  mu_init(as<Map<VectorXd> >(mus));
    const Map<VectorXd>  eta_init(as<Map<VectorXd> >(etas));
    Index                n = X.rows();
    if ((Index)y.size() != n) throw invalid_argument("size mismatch");

    // instantiate fitting class
    bool is_big_matrix = false;

    std::unique_ptr<glm> glm_solver(new glm(X, y, weights, offset,
                         var, mu_eta, linkinv, dev_resids,
                         valideta, validmu, tol, maxit, type,
                         is_big_matrix, fam_code, fam_params));

    // initialize parameters
    glm_solver->init_parms(beta_init, mu_init, eta_init);


    // maximize likelihood
    int iters = glm_solver->solve(maxit);
    
    VectorXd beta      = glm_solver->get_beta();
    VectorXd se        = glm_solver->get_se();
    VectorXd mu        = glm_solver->get_mu();
    VectorXd eta       = glm_solver->get_eta();
    VectorXd wts       = glm_solver->get_w();
    VectorXd pweights  = glm_solver->get_weights();
    MatrixXd vcov      = glm_solver->get_vcov();

    double dev         = glm_solver->get_dev();
    int rank           = glm_solver->get_rank();
    bool converged     = glm_solver->get_converged();

    int df = X.rows() - rank;

    return List::create(_["coefficients"]      = beta,
                        _["se"]                = se,
                        _["fitted.values"]     = mu,
                        _["linear.predictors"] = eta,
                        _["deviance"]          = dev,
                        _["weights"]           = wts,
                        _["prior.weights"]     = pweights,
                        _["rank"]              = rank,
                        _["df.residual"]       = df,
                        _["iter"]              = iters,
                        _["converged"]         = converged,
                        _["cov.unscaled"]      = vcov);
}


// Build a FamilyParams from a Nullable<NumericVector> (length 0..3).  Missing
// entries default to 1.0 (theta), 0.0 (var_power), 0.0 (link_power).
static fglm::FamilyParams parse_fam_params(Rcpp::Nullable<Rcpp::NumericVector> fam_params) {
    fglm::FamilyParams fp;
    if (fam_params.isNull()) return fp;
    Rcpp::NumericVector fpv(fam_params.get());
    if (fpv.size() >= 1) fp.theta      = fpv[0];
    if (fpv.size() >= 2) fp.var_power  = fpv[1];
    if (fpv.size() >= 3) fp.link_power = fpv[2];
    return fp;
}

// [[Rcpp::export]]
List fit_glm(Rcpp::NumericMatrix x, Rcpp::NumericVector y, Rcpp::NumericVector weights, Rcpp::NumericVector offset,
             Rcpp::NumericVector start, Rcpp::NumericVector mu, Rcpp::NumericVector eta,
             Function var, Function mu_eta, Function linkinv, Function dev_resids,
             Function valideta, Function validmu,
             int type, double tol, int maxit, int fam_code = -1,
             Rcpp::Nullable<Rcpp::NumericVector> fam_params = R_NilValue)
{
    fglm::FamilyParams fp = parse_fam_params(fam_params);
    return fastglm(x, y, weights, offset, start, mu, eta, var, mu_eta, linkinv, dev_resids, valideta, validmu, type, tol, maxit, fam_code, fp);
}





List bigfastglm(XPtr<BigMatrix> Xs,
                Rcpp::NumericVector ys,
                Rcpp::NumericVector weightss,
                Rcpp::NumericVector offsets,
                Rcpp::NumericVector starts,
                Rcpp::NumericVector mus,
                Rcpp::NumericVector etas,
                Function var,
                Function mu_eta,
                Function linkinv,
                Function dev_resids,
                Function valideta,
                Function validmu,
                int type,
                double tol,
                int maxit,
                int fam_code,
                const fglm::FamilyParams& fam_params,
                bool firth = false)
{
    //const Map<MatrixXd>  X(as<Map<MatrixXd> >(Xs));
    //XPtr<BigMatrix> bMPtr(Xs);
    
    unsigned int typedata = Xs->matrix_type();
    
    if (typedata != 8)
    {
        throw Rcpp::exception("type for provided big.matrix not available");
    }
    const Map<MatrixXd>  X = Map<MatrixXd>((double *)Xs->matrix(), Xs->nrow(), Xs->ncol()  );
    const Map<VectorXd>  y(as<Map<VectorXd> >(ys));
    
    
    
    const Map<VectorXd>  weights(as<Map<VectorXd> >(weightss));
    const Map<VectorXd>  offset(as<Map<VectorXd> >(offsets));
    const Map<VectorXd>  beta_init(as<Map<VectorXd> >(starts));
    const Map<VectorXd>  mu_init(as<Map<VectorXd> >(mus));
    const Map<VectorXd>  eta_init(as<Map<VectorXd> >(etas));
    Index                n = X.rows();
    if ((Index)y.size() != n) throw invalid_argument("size mismatch");
    
    
    if (type != 2 && type != 3)
    {
        throw invalid_argument("type must be either 2 or 3 for big.matrix objects");
    }
    
    bool is_big_matrix = true;

    std::unique_ptr<glm> solver(new glm(X, y, weights, offset,
                          var, mu_eta, linkinv, dev_resids,
                          valideta, validmu, tol, maxit, type,
                          is_big_matrix, fam_code, fam_params, firth));

    // initialize parameters
    solver->init_parms(beta_init, mu_init, eta_init);


    // maximize likelihood
    int iters = solver->solve(maxit);

    VectorXd beta      = solver->get_beta();
    VectorXd se        = solver->get_se();
    VectorXd mu        = solver->get_mu();
    VectorXd eta       = solver->get_eta();
    VectorXd wts       = solver->get_w();
    VectorXd pweights  = solver->get_weights();
    MatrixXd vcov      = solver->get_vcov();

    double dev         = solver->get_dev();
    int rank           = solver->get_rank();
    bool converged     = solver->get_converged();

    int df = X.rows() - rank;

    List out = List::create(_["coefficients"]      = beta,
                            _["se"]                = se,
                            _["fitted.values"]     = mu,
                            _["linear.predictors"] = eta,
                            _["deviance"]          = dev,
                            _["weights"]           = wts,
                            _["prior.weights"]     = pweights,
                            _["rank"]              = rank,
                            _["df.residual"]       = df,
                            _["iter"]              = iters,
                            _["converged"]         = converged,
                            _["cov.unscaled"]      = vcov);
    if (firth) {
        double log_det = solver->get_log_det_XtWX();
        VectorXd mu_final = solver->get_mu();
        Eigen::Map<const Eigen::ArrayXd> y_arr(y.data(), y.size());
        Eigen::Map<const Eigen::ArrayXd> mu_arr(mu_final.data(), mu_final.size());
        Eigen::Map<const Eigen::ArrayXd> w_arr(weights.data(), weights.size());
        double std_dev = fglm::dev_resids_sum(fam_code, fam_params, y_arr, mu_arr, w_arr);
        out["deviance"]           = std_dev;
        out["penalized.deviance"] = dev;
        out["log.det.XtWX"]       = log_det;
        out["firth"]              = true;
    }

    return out;
}


// [[Rcpp::export]]
List fit_big_glm(SEXP x, Rcpp::NumericVector y, Rcpp::NumericVector weights, Rcpp::NumericVector offset,
                 Rcpp::NumericVector start, Rcpp::NumericVector mu, Rcpp::NumericVector eta,
                 Function var, Function mu_eta, Function linkinv, Function dev_resids,
                 Function valideta, Function validmu,
                 int type, double tol, int maxit, int fam_code = -1,
                 Rcpp::Nullable<Rcpp::NumericVector> fam_params = R_NilValue,
                 bool firth = false)
{
    XPtr<BigMatrix> xpMat(x);
    fglm::FamilyParams fp = parse_fam_params(fam_params);
    return bigfastglm(xpMat, y, weights, offset, start, mu, eta, var, mu_eta, linkinv, dev_resids, valideta, validmu, type, tol, maxit, fam_code, fp, firth);
}


// Fit GLM with sparse design matrix (dgCMatrix).  Methods 2 (LLT) and 3
// (LDLT) only -- QR / SVD on sparse X are not supported by this package.
//
// [[Rcpp::export]]
List fit_sparse_glm(SEXP x_sparse,
                    Rcpp::NumericVector y, Rcpp::NumericVector weights, Rcpp::NumericVector offset,
                    Rcpp::NumericVector start, Rcpp::NumericVector mu, Rcpp::NumericVector eta,
                    Function var, Function mu_eta, Function linkinv, Function dev_resids,
                    Function valideta, Function validmu,
                    int type, double tol, int maxit, int fam_code = -1,
                    Rcpp::Nullable<Rcpp::NumericVector> fam_params = R_NilValue,
                    bool firth = false)
{
    if (type != 2 && type != 3) {
        throw std::invalid_argument("for dgCMatrix objects, 'method' must be 2 (LLT) or 3 (LDLT).");
    }

    fglm::FamilyParams fp = parse_fam_params(fam_params);

    typedef Eigen::SparseMatrix<double> SpMat;
    const Eigen::MappedSparseMatrix<double> Xm(as<Eigen::MappedSparseMatrix<double> >(x_sparse));
    Eigen::Map<const SpMat> X(Xm.rows(), Xm.cols(), Xm.nonZeros(),
                              Xm.outerIndexPtr(), Xm.innerIndexPtr(), Xm.valuePtr());

    const Map<VectorXd> Y(as<Map<VectorXd> >(y));
    const Map<VectorXd> wts(as<Map<VectorXd> >(weights));
    const Map<VectorXd> off(as<Map<VectorXd> >(offset));
    const Map<VectorXd> beta_init(as<Map<VectorXd> >(start));
    const Map<VectorXd> mu_init(as<Map<VectorXd> >(mu));
    const Map<VectorXd> eta_init(as<Map<VectorXd> >(eta));

    Index n = X.rows();
    if ((Index)Y.size() != n) throw std::invalid_argument("size mismatch");

    glm_sparse solver(X, Y, wts, off,
                      var, mu_eta, linkinv, dev_resids,
                      valideta, validmu,
                      tol, maxit, type, fam_code, fp, firth);
    solver.init_parms(beta_init, mu_init, eta_init);
    int iters = solver.solve(maxit);

    VectorXd b   = solver.get_beta();
    VectorXd se  = solver.get_se();
    VectorXd m   = solver.get_mu();
    VectorXd e   = solver.get_eta();
    VectorXd wfit = solver.get_w();
    VectorXd pwt  = solver.get_weights();
    MatrixXd vc   = solver.get_vcov();

    double dev = solver.get_dev();
    int rk = solver.get_rank();
    bool conv = solver.get_converged();
    int df = (int)n - rk;

    List out = List::create(_["coefficients"]      = b,
                            _["se"]                = se,
                            _["fitted.values"]     = m,
                            _["linear.predictors"] = e,
                            _["deviance"]          = dev,
                            _["weights"]           = wfit,
                            _["prior.weights"]     = pwt,
                            _["rank"]              = rk,
                            _["df.residual"]       = df,
                            _["iter"]              = iters,
                            _["converged"]         = conv,
                            _["cov.unscaled"]      = vc);
    if (firth) {
        double log_det = solver.get_log_det_XtWX();
        Eigen::Map<const Eigen::ArrayXd> y_arr(Y.data(), Y.size());
        Eigen::Map<const Eigen::ArrayXd> mu_arr(m.data(), m.size());
        Eigen::Map<const Eigen::ArrayXd> w_arr(wts.data(), wts.size());
        double std_dev = fglm::dev_resids_sum(fam_code, fp, y_arr, mu_arr, w_arr);
        out["deviance"]           = std_dev;
        out["penalized.deviance"] = dev;
        out["log.det.XtWX"]       = log_det;
        out["firth"]              = true;
    }
    return out;
}

