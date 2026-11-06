#include "../inst/include/ordinal_constrained.h"
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

// [[Rcpp::export]]
Rcpp::List ordinal_constrained_native(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector start,Rcpp::NumericMatrix basis,Rcpp::LogicalVector positive,
    bool adjusted,int block_size,double condition_tol,int link,int maxit,double score_tol,
    double step_tol,double max_step,int max_halves) {
    fglm_ordinal::OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    const fglm_ordinal_constrained::CommonDomain domain{fglm_ordinal_constrained::positive_rows(
        Rcpp::as<Eigen::MatrixXd>(x),positive),categories-1,x.ncol()};
    const fglm_ordinal_constrained::ConstrainedControl ctl{maxit,max_halves,score_tol,step_tol,max_step};
    const auto evaluate=[&](const Eigen::VectorXd& b) {return fglm_ordinal::ordinal_evaluate(data,b,adjusted,condition_tol);};
    const auto wrap=[](const fglm_ordinal::OrdinalState& state) {return fglm_ordinal::ordinal_state_list(state);};
    return fglm_ordinal_constrained::constrained_fit<fglm_ordinal::OrdinalState>(evaluate,domain,wrap,
        Rcpp::as<Eigen::VectorXd>(start),Rcpp::as<Eigen::MatrixXd>(basis),adjusted,ctl);
}

// [[Rcpp::export]]
Rcpp::List ordinal_partial_constrained_native(Rcpp::NumericMatrix x,Rcpp::NumericMatrix z,
    Rcpp::IntegerVector y,Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,
    Rcpp::NumericVector offset,int categories,Rcpp::NumericVector start,Rcpp::NumericMatrix basis,
    Rcpp::LogicalVector positive,bool adjusted,int block_size,double condition_tol,int link,
    int maxit,double score_tol,double step_tol,double max_step,int max_halves) {
    PartialOrdinalData data(x,z,y,counts,weights,offset,categories,block_size,link);
    const fglm_ordinal_constrained::PartialDomain domain{
        fglm_ordinal_constrained::positive_rows(Rcpp::as<Eigen::MatrixXd>(x),positive),
        fglm_ordinal_constrained::positive_rows(Rcpp::as<Eigen::MatrixXd>(z),positive),
        categories-1,x.ncol(),z.ncol()};
    const fglm_ordinal_constrained::ConstrainedControl ctl{maxit,max_halves,score_tol,step_tol,max_step};
    const auto evaluate=[&](const Eigen::VectorXd& b) {return partial_evaluate(data,b,adjusted,condition_tol);};
    const auto wrap=[](const PartialOrdinalState& state) {return partial_state_list(state);};
    return fglm_ordinal_constrained::constrained_fit<PartialOrdinalState>(evaluate,domain,wrap,
        Rcpp::as<Eigen::VectorXd>(start),Rcpp::as<Eigen::MatrixXd>(basis),adjusted,ctl);
}

// Exposes the compiled constrained geometry to the numerical cancellation tests.
// [[Rcpp::export]]
Rcpp::List ordinal_partial_domain_values(Rcpp::NumericMatrix x,Rcpp::NumericMatrix z,int thresholds,
    Rcpp::NumericVector coefficients) {
    const fglm_ordinal_constrained::PartialDomain domain{Rcpp::as<Eigen::MatrixXd>(x),
        Rcpp::as<Eigen::MatrixXd>(z),thresholds,x.ncol(),z.ncol()};
    const Eigen::VectorXd b=Rcpp::as<Eigen::VectorXd>(coefficients);
    Rcpp::NumericVector gaps(z.nrow()*(thresholds-1));
    for(int j=0;j+1<thresholds;++j) for(int i=0;i<z.nrow();++i) gaps[i+j*z.nrow()]=domain.gap(i,j,b);
    return Rcpp::List::create(Rcpp::_["step"]=domain.predictor_step(b),Rcpp::_["gaps"]=gaps);
}
