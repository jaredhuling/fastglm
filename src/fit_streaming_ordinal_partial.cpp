#include "../inst/include/ordinal_partial.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace fglm_ordinal_partial;

namespace {
struct PartialOrdinalSource {
    Rcpp::Function fetch;
    int chunks,p,pn,K,q,d,block,link;
    PartialOrdinalSource(Rcpp::Function callback,int n_chunks,int common_predictors,
        int nonparallel_predictors,int categories,int block_size,int link_code) :
        fetch(callback),chunks(n_chunks),p(common_predictors),pn(nonparallel_predictors),
        K(categories),q(K-1),d(0),block(block_size),link(link_code) {
        if(chunks<1 || p<0 || pn<0 || K<2 || block<1 || link<0 || link>2)
            Rcpp::stop("invalid nonparallel ordinal source dimensions");
        const double dimension=double(q)+p+double(q)*pn;
        if(dimension>std::numeric_limits<int>::max()) Rcpp::stop("ordinal parameter dimension is too large");
        d=static_cast<int>(dimension);
    }
    template<class Function> void visit(Function&& function) const {
        for(int k=1;k<=chunks;++k) {
            Rcpp::checkUserInterrupt();
            Rcpp::List value=fetch(k);
            Rcpp::NumericMatrix x=value["X"],z=value["Z"],counts=value["counts"];
            Rcpp::IntegerVector codes=value["codes"];
            Rcpp::NumericVector weights=value["weights"],offset=value["offset"];
            if(x.ncol()!=p || z.ncol()!=pn)
                Rcpp::stop("nonparallel ordinal source predictor dimension changed");
            PartialOrdinalData data(x,z,codes,counts,weights,offset,K,block,link,false);
            function(data);
        }
    }
    double predictor_step(const VectorXd& direction) const {
        double largest=0;
        visit([&](const PartialOrdinalData& data) {
            largest=std::max(largest,data.predictor_step(direction));
        });
        return largest;
    }
    double feasible_fraction(const VectorXd& b,const VectorXd& direction) const {
        if(q==1) return 1;
        double fraction=1;
        visit([&](const PartialOrdinalData& data) {
            fraction=std::min(fraction,data.feasible_fraction(b,direction));
        });
        return fraction;
    }
};
}

// [[Rcpp::export]]
Rcpp::List ordinal_partial_stream_native_evaluate(Rcpp::Function fetch,int n_chunks,
    int common_predictors,int nonparallel_predictors,int categories,
    Rcpp::NumericVector coefficients,bool firth=false,int block_size=2048,
    double condition_tol=1e-14,int link=0) {
    PartialOrdinalSource data(fetch,n_chunks,common_predictors,nonparallel_predictors,
        categories,block_size,link);
    return partial_state_list(partial_evaluate(data,Rcpp::as<VectorXd>(coefficients),firth,condition_tol));
}

// [[Rcpp::export]]
Rcpp::List ordinal_partial_stream_native_fit(Rcpp::Function fetch,int n_chunks,
    int common_predictors,int nonparallel_predictors,int categories,
    Rcpp::NumericVector start,bool firth=false,double score_tol=1e-8,
    double step_tol=1e-8,int maxit=200,int block_size=2048,double max_step=4,
    int max_halves=35,double condition_tol=1e-14,bool trace=false,int link=0) {
    PartialOrdinalSource data(fetch,n_chunks,common_predictors,nonparallel_predictors,
        categories,block_size,link);
    return partial_fit(data,Rcpp::as<VectorXd>(start),firth,score_tol,step_tol,maxit,
        max_step,max_halves,condition_tol,trace);
}
