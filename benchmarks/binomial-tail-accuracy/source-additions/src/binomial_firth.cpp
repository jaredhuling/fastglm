#include <RcppEigen.h>
#include "../inst/include/bias_reduction.h"
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List binomial_firth_moments(Rcpp::NumericVector eta,Rcpp::NumericVector y,
    Rcpp::NumericVector weights,int code) {
    const int n=eta.size();
    if(y.size()!=n || weights.size()!=n || !fglm::bounded_firth_predictors(code))
        Rcpp::stop("incompatible binomial moment inputs");
    Rcpp::NumericVector working(n),score(n),mean(n),derivative(n),logp(n),logq(n),deviance(n);
    for(int i=0;i<n;++i) {
        const auto row=fglm::binomial_firth_terms(code,eta[i],y[i],weights[i]);
        working[i]=row.weight;score[i]=row.score;mean[i]=std::exp(row.logp);
        derivative[i]=row.derivative;logp[i]=row.logp;logq[i]=row.logq;
        if(weights[i]>0) {
            if(y[i]>0) deviance[i]+=2*weights[i]*y[i]*(std::log(y[i])-row.logp);
            if(y[i]<1) deviance[i]+=2*weights[i]*(1-y[i])*(std::log1p(-y[i])-row.logq);
        }
    }
    return Rcpp::List::create(Rcpp::_["working"]=working,Rcpp::_["score"]=score,
        Rcpp::_["mean"]=mean,Rcpp::_["derivative"]=derivative,
        Rcpp::_["logp"]=logp,Rcpp::_["logq"]=logq,Rcpp::_["deviance"]=deviance);
}
