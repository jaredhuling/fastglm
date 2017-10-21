#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include "glm.h"
#include <RcppEigen.h>

using namespace Rcpp;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Map;

typedef MatrixXd::Index Index;


List fastglm(Rcpp::NumericMatrix Xs, Rcpp::NumericVector ys, Rcpp::NumericVector weightss,
                 Rcpp::NumericVector offsets, Function var, Function mu_eta, Function linkinv, Function dev_resids, 
                 int type, double tol, int maxit) 
{
    const Map<MatrixXd>  X(as<Map<MatrixXd> >(Xs));
    const Map<VectorXd>  y(as<Map<VectorXd> >(ys));
    const Map<VectorXd>  weights(as<Map<VectorXd> >(weightss));
    const Map<VectorXd>  offset(as<Map<VectorXd> >(offsets));
    Index                n = X.rows();
    if ((Index)y.size() != n) throw invalid_argument("size mismatch");
    
    
    GlmBase<Eigen::VectorXd, Eigen::MatrixXd> *glm_solver = NULL;
    
    glm_solver = new glm(X, y, weights, offset, var, mu_eta, linkinv, dev_resids, tol, maxit, type);
    
    glm_solver->init_parms();
    
    int iters = glm_solver->solve(maxit);
    
    VectorXd beta = glm_solver->get_beta();
    
    return List::create(_["beta"]   = beta,
                        _["niter"]  = iters);
}



// [[Rcpp::export]]
List fit_glm(Rcpp::NumericMatrix x, Rcpp::NumericVector y, Rcpp::NumericVector weights, Rcpp::NumericVector offset, 
             Function var, Function mu_eta, Function linkinv, Function dev_resids, int type, double tol, int maxit) 
{
    return fastglm(x, y, weights, offset, var, mu_eta, linkinv, dev_resids, type, tol, maxit);
}
