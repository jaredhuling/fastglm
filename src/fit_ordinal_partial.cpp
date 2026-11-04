#include "../inst/include/ordinal_partial.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace fglm_ordinal_partial;

// [[Rcpp::export]]
Rcpp::List ordinal_partial_native_evaluate(Rcpp::NumericMatrix x,Rcpp::NumericMatrix z,
    Rcpp::IntegerVector y,Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,
    Rcpp::NumericVector offset,int categories,Rcpp::NumericVector coefficients,
    bool firth=false,int block_size=2048,double condition_tol=1e-14,int link=0) {
    PartialOrdinalData data(x,z,y,counts,weights,offset,categories,block_size,link);
    return partial_state_list(partial_evaluate(data,Rcpp::as<VectorXd>(coefficients),firth,condition_tol));
}

// [[Rcpp::export]]
Rcpp::List ordinal_partial_native_fit(Rcpp::NumericMatrix x,Rcpp::NumericMatrix z,
    Rcpp::IntegerVector y,Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,
    Rcpp::NumericVector offset,int categories,Rcpp::NumericVector start,
    bool firth=false,double score_tol=1e-8,double step_tol=1e-8,int maxit=200,
    int block_size=2048,double max_step=4,int max_halves=35,double condition_tol=1e-14,
    bool trace=false,int link=0) {
    PartialOrdinalData data(x,z,y,counts,weights,offset,categories,block_size,link);
    return partial_fit(data,Rcpp::as<VectorXd>(start),firth,score_tol,step_tol,maxit,
        max_step,max_halves,condition_tol,trace);
}
