#define EIGEN_DONT_PARALLELIZE


#include <bigmemory/MatrixAccessor.hpp>
#include <bigmemory/BigMatrix.h>

#include <Rcpp.h>
#include "glm.h"
#include <RcppEigen.h>

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
             int maxit) 
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
    GlmBase<Eigen::VectorXd, Eigen::MatrixXd> *glm_solver = NULL;
    
    bool is_big_matrix = false;
    
    glm_solver = new glm(X, y, weights, offset, 
                         var, mu_eta, linkinv, dev_resids, 
                         valideta, validmu, tol, maxit, type,
                         is_big_matrix);
    
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
    
    double dev         = glm_solver->get_dev();
    int rank           = glm_solver->get_rank();
    bool converged     = glm_solver->get_converged();
    
    int df = X.rows() - rank;
    
    delete glm_solver;
    
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
                        _["converged"]         = converged);
}


// [[Rcpp::export]]
List fit_glm(Rcpp::NumericMatrix x, Rcpp::NumericVector y, Rcpp::NumericVector weights, Rcpp::NumericVector offset, 
             Rcpp::NumericVector start, Rcpp::NumericVector mu, Rcpp::NumericVector eta,
             Function var, Function mu_eta, Function linkinv, Function dev_resids, 
             Function valideta, Function validmu,  
             int type, double tol, int maxit) 
{
    return fastglm(x, y, weights, offset, start, mu, eta, var, mu_eta, linkinv, dev_resids, valideta, validmu, type, tol, maxit);
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
                int maxit) 
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
    
    // instantiate fitting class
    GlmBase<Eigen::VectorXd, Eigen::MatrixXd> *glm_solver = NULL;
    
    bool is_big_matrix = true;
    
    glm_solver = new glm(X, y, weights, offset, 
                         var, mu_eta, linkinv, dev_resids, 
                         valideta, validmu, tol, maxit, type,
                         is_big_matrix);
    
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
    
    double dev         = glm_solver->get_dev();
    int rank           = glm_solver->get_rank();
    bool converged     = glm_solver->get_converged();
    
    int df = X.rows() - rank;
    
    delete glm_solver;
    
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
                        _["converged"]         = converged);
}


// [[Rcpp::export]]
List fit_big_glm(SEXP x, Rcpp::NumericVector y, Rcpp::NumericVector weights, Rcpp::NumericVector offset, 
                 Rcpp::NumericVector start, Rcpp::NumericVector mu, Rcpp::NumericVector eta,
                 Function var, Function mu_eta, Function linkinv, Function dev_resids, 
                 Function valideta, Function validmu,  
                 int type, double tol, int maxit) 
{
    XPtr<BigMatrix> xpMat(x);
    
    return bigfastglm(xpMat, y, weights, offset, start, mu, eta, var, mu_eta, linkinv, dev_resids, valideta, validmu, type, tol, maxit);
}

