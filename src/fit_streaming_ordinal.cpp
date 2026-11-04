#include "../inst/include/ordinal.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace fglm_ordinal;

namespace {
struct OrdinalSource {
    Rcpp::Function fetch;
    int chunks,p,K,q,d,block,link;
    OrdinalSource(Rcpp::Function callback,int n_chunks,int predictors,int categories,int block_size,int link_code) :
        fetch(callback),chunks(n_chunks),p(predictors),K(categories),q(K-1),d(p+q),block(block_size),link(link_code) {}
    template<class Function> void visit(Function&& function) const {
        for(int k=1;k<=chunks;++k) {
            Rcpp::checkUserInterrupt();
            Rcpp::List value=fetch(k);
            Rcpp::NumericMatrix x=value["X"],counts=value["counts"];
            Rcpp::IntegerVector codes=value["codes"];
            Rcpp::NumericVector weights=value["weights"],offset=value["offset"];
            if(x.ncol()!=p) Rcpp::stop("ordinal source predictor dimension changed");
            OrdinalData data(x,codes,counts,weights,offset,K,block,link,false);
            function(data);
        }
    }
    double predictor_step(const VectorXd& direction) const {
        double largest=0;
        visit([&](const OrdinalData& data) { largest=std::max(largest,data.predictor_step(direction)); });
        return largest;
    }
};
}

// [[Rcpp::export]]
Rcpp::List ordinal_stream_native_fit(Rcpp::Function fetch,int n_chunks,int predictors,
    int categories,Rcpp::NumericVector start,bool firth=false,double score_tol=1e-8,
    double step_tol=1e-8,int maxit=200,int block_size=2048,double max_step=4,
    int max_halves=35,double condition_tol=1e-14,bool trace=false,int link=0) {
    if(n_chunks<1 || predictors<0 || categories<2 || block_size<1 || link<0 || link>2)
        Rcpp::stop("invalid ordinal source dimensions");
    OrdinalSource data(fetch,n_chunks,predictors,categories,block_size,link);
    return ordinal_fit(data,Rcpp::as<VectorXd>(start),firth,score_tol,step_tol,maxit,
        max_step,max_halves,condition_tol,trace);
}
