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

// Complete post-fit accounting in one pass without materializing all moment
// vectors. Binomial trial totals retain stats::binomial()$aic conventions.
// [[Rcpp::export]]
Rcpp::List binomial_firth_summary(Rcpp::NumericVector eta,Rcpp::NumericVector y,
    Rcpp::NumericVector weights,Rcpp::NumericVector trials,int code) {
    const int n=eta.size();
    if(y.size()!=n || weights.size()!=n || (trials.size()!=1 && trials.size()!=n) ||
        !fglm::bounded_firth_predictors(code))
        Rcpp::stop("incompatible binomial summary inputs");
    bool grouped=false;
    for(double value:trials) if(value>1) {grouped=true;break;}
    Rcpp::NumericVector working(n),residual(n);
    double deviance=0,aic=0;
    for(int i=0;i<n;++i) {
        const auto row=fglm::binomial_firth_terms(code,eta[i],y[i],weights[i]);
        working[i]=row.weight;
        if(weights[i]==0) continue;
        if(code==fglm::FAM_BINOMIAL_CLOGLOG) {
            if(y[i]>0) residual[i]=y[i]*std::exp(-eta[i]);
            if(y[i]<1) {
                const double t=std::exp(eta[i]);
                const double ratio=t==0 ? 1 : (std::isinf(t) ? R_PosInf : std::expm1(t)/t);
                residual[i]-=(1-y[i])*ratio;
            }
        } else {
            const double logd=-.5*eta[i]*eta[i]-.91893853320467274178;
            if(y[i]>0) residual[i]=y[i]*std::exp(row.logq-logd);
            if(y[i]<1) residual[i]-=(1-y[i])*std::exp(row.logp-logd);
        }
        if(y[i]>0) deviance+=2*weights[i]*y[i]*(std::log(y[i])-row.logp);
        if(y[i]<1) deviance+=2*weights[i]*(1-y[i])*(std::log1p(-y[i])-row.logq);
        const double m=grouped ? trials[trials.size()==1 ? 0 : i] : weights[i];
        if(m>0) {
            const double total=R::fround(m,0),successes=R::fround(m*y[i],0),failures=total-successes;
            double value=(successes==0 || successes==total) ? 0 : R::lchoose(total,successes);
            if(successes>0) value+=successes*row.logp;
            if(failures>0) value+=failures*row.logq;
            aic-=2*(weights[i]/m)*value;
        }
    }
    return Rcpp::List::create(Rcpp::_["working"]=working,Rcpp::_["residuals"]=residual,
        Rcpp::_["deviance"]=deviance,Rcpp::_["aic"]=aic);
}

// [[Rcpp::export]]
Rcpp::NumericVector binomial_firth_derivative(Rcpp::NumericVector eta,int code) {
    if(!fglm::bounded_firth_predictors(code)) Rcpp::stop("unsupported binomial derivative link");
    Rcpp::NumericVector derivative(eta.size());
    for(R_xlen_t i=0;i<eta.size();++i) {
        if(std::isinf(eta[i])) derivative[i]=0;
        else derivative[i]=code==fglm::FAM_BINOMIAL_PROBIT ?
            std::exp(-.5*eta[i]*eta[i]-.91893853320467274178) :
            std::exp(eta[i]-std::exp(eta[i]));
    }
    return derivative;
}
