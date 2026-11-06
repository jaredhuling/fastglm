#include "../inst/include/ordinal.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace fglm_ordinal;

// [[Rcpp::export]]
Rcpp::List ordinal_native_evaluate(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector coefficients,bool firth=false,
    int block_size=2048,double condition_tol=1e-14,int link=0) {
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    VectorXd beta=Rcpp::as<VectorXd>(coefficients);
    return ordinal_state_list(ordinal_evaluate(data,beta,firth,condition_tol));
}

// [[Rcpp::export]]
Rcpp::List ordinal_native_fit(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector start,bool firth=false,double score_tol=1e-8,
    double step_tol=1e-8,int maxit=200,int block_size=2048,double max_step=4,
    int max_halves=35,double condition_tol=1e-14,bool trace=false,int link=0,bool observed=true) {
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    return ordinal_fit(data,Rcpp::as<VectorXd>(start),firth,score_tol,step_tol,maxit,
        max_step,max_halves,condition_tol,trace,observed);
}

// [[Rcpp::export]]
Rcpp::NumericVector ordinal_category_totals(Rcpp::IntegerVector codes,Rcpp::NumericVector weights,int categories) {
    Rcpp::NumericVector totals(categories);
    for(R_xlen_t i=0;i<codes.size();++i) totals[codes[i]-1]+=weights[i];
    return totals;
}

// Centers and scales the positive-frequency rows and checks rank in one pass. The
// check uses an unpivoted Cholesky factor of the normalized crossproduct of
// (1, standardized x): its squared diagonal equals the squared QR diagonal relative
// to each column norm, which is the ratio a pivoted QR compares with its tolerance.
// [[Rcpp::export]]
Rcpp::List ordinal_design_native(Rcpp::NumericMatrix x,Rcpp::NumericVector totals,double frequency,bool standardize) {
    const int n=x.nrow(),p=x.ncol();
    const Eigen::Map<const MatrixXd> X(x.begin(),n,p);
    if(!X.allFinite()) Rcpp::stop("ordinal fitting requires a finite dense numeric matrix without a location intercept");
    VectorXd center(p),scale(p);
    int active=0;
    for(int i=0;i<n;++i) if(totals[i]>0) ++active;
    Rcpp::NumericMatrix out(n,p);
    Eigen::Map<MatrixXd> Y(out.begin(),n,p);
    for(int j=0;j<p;++j) {
        double mean=0,variance=0;
        for(int i=0;i<n;++i) if(totals[i]>0) mean+=X(i,j)*(totals[i]/frequency);
        for(int i=0;i<n;++i) if(totals[i]>0) { const double c=X(i,j)-mean;variance+=c*c*(totals[i]/frequency); }
        center[j]=mean;scale[j]=std::sqrt(variance);
        for(int i=0;i<n;++i)
            Y(i,j)=totals[i]>0 ? (standardize ? (X(i,j)-mean)/scale[j] : X(i,j)) : 0;
    }
    bool full_rank=false;
    if((scale.array().isFinite() && scale.array()>0).all() && active>p) {
        MatrixXd gram=MatrixXd::Zero(p+1,p+1);
        const int block=4096;
        MatrixXd rows(block,p+1);
        int filled=0;
        const auto flush=[&]() {
            if(filled) gram.selfadjointView<Eigen::Lower>().rankUpdate(rows.topRows(filled).transpose());
            filled=0;
        };
        for(int i=0;i<n;++i) if(totals[i]>0) {
            rows(filled,0)=1;
            for(int j=0;j<p;++j) rows(filled,j+1)=(X(i,j)-center[j])/scale[j];
            if(++filled==block) flush();
        }
        flush();
        const VectorXd inverse=gram.diagonal().cwiseSqrt().cwiseInverse();
        const MatrixXd full=gram.selfadjointView<Eigen::Lower>();
        const MatrixXd normalized=inverse.asDiagonal()*full*inverse.asDiagonal();
        const Eigen::LLT<MatrixXd> factor(normalized);
        if(factor.info()==Eigen::Success) {
            const VectorXd pivots=MatrixXd(factor.matrixL()).diagonal();
            full_rank=pivots.array().square().minCoeff()>=1e-10;
        }
    }
    return Rcpp::List::create(Rcpp::_["x"]=out,Rcpp::_["center"]=center,Rcpp::_["scale"]=scale,
        Rcpp::_["full.rank"]=full_rank);
}

// Exposes the observed terms to the derivative tests.
// [[Rcpp::export]]
Rcpp::List ordinal_observed_evaluate(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector coefficients,int block_size=2048,int link=0) {
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    const ObservedState state=observed_evaluate(data,Rcpp::as<VectorXd>(coefficients),nullptr);
    return Rcpp::List::create(Rcpp::_["loglik"]=state.loglik,Rcpp::_["score"]=state.score,
        Rcpp::_["information"]=state.information);
}
