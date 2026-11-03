#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
#include "../inst/include/bias_reduction.h"
using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Build the shared information and leverage diagonal using bounded row blocks.
// [[Rcpp::export]]
SEXP prepare_glm_constant(NumericMatrix x, NumericVector weights, bool leverage, int code) {
    Eigen::Map<const MatrixXd> X(x.begin(),x.nrow(),x.ncol());
    Eigen::Map<const VectorXd> W(weights.begin(),weights.size());
    const int n=x.nrow(),p=x.ncol(),block=8192;
    MatrixXd a=MatrixXd::Zero(p,p);
    for(int lo=0;lo<n;lo+=block) {
        Rcpp::checkUserInterrupt();
        int size=std::min(block,n-lo);
        MatrixXd wx=W.segment(lo,size).array().sqrt().matrix().asDiagonal()*X.middleRows(lo,size);
        a.selfadjointView<Eigen::Lower>().rankUpdate(wx.adjoint());
    }
    Eigen::LLT<MatrixXd> factor(a.selfadjointView<Eigen::Lower>());
    if(factor.info()!=Eigen::Success || factor.rcond()<1e-12) return R_NilValue;
    MatrixXd lower=factor.matrixL();
    VectorXd h(leverage?n:0);
    if(leverage) for(int lo=0;lo<n;lo+=block) {
        Rcpp::checkUserInterrupt();
        int size=std::min(block,n-lo);
        MatrixXd rows=X.middleRows(lo,size).transpose();
        lower.triangularView<Eigen::Lower>().solveInPlace(rows);
        h.segment(lo,size)=rows.colwise().squaredNorm().transpose().array()*W.segment(lo,size).array();
    }
    VectorXd adjustment=VectorXd::Zero(p);
    if(leverage && code==10) adjustment.noalias()=X.adjoint()*h/2;
    fglm::GammaWeightGroups groups;
    if(leverage && code==10) groups=fglm::gamma_weight_groups(W);
    NumericVector values(groups.size());IntegerVector counts(groups.size());
    for(size_t j=0;j<groups.size();++j) {values[j]=groups[j].first;counts[j]=groups[j].second;}
    return List::create(_["lower"]=lower,_["leverage"]=h,_["adjustment"]=adjustment,
                        _["weight.values"]=values,_["weight.counts"]=counts,
                        _["logdet"]=2*lower.diagonal().array().log().sum());
}
