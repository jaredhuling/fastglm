#include <RcppEigen.h>
#include <cmath>
#include <algorithm>
// [[Rcpp::depends(RcppEigen)]]

// Evaluate the complete Jeffreys-penalized binomial log likelihood. Tested
// columns remain in the information matrix during every constrained fit.
// [[Rcpp::export]]
Rcpp::List firth_profile_evaluate(Rcpp::NumericMatrix x,
                                  Rcpp::NumericVector y,
                                  Rcpp::NumericVector weights,
                                  Rcpp::NumericVector offset,
                                  Rcpp::NumericVector coefficients) {
    const int n=x.nrow(), p=x.ncol();
    if(n<1 || p<1 || y.size()!=n || weights.size()!=n || offset.size()!=n || coefficients.size()!=p)
        Rcpp::stop("inconsistent profile likelihood dimensions");
    Eigen::Map<const Eigen::MatrixXd> X(x.begin(),n,p);
    Eigen::Map<const Eigen::VectorXd> beta(coefficients.begin(),p);
    Eigen::MatrixXd information=Eigen::MatrixXd::Zero(p,p);
    Eigen::VectorXd gradient=Eigen::VectorXd::Zero(p);
    double likelihood=0;
    const int block=1024;
    for(int first=0;first<n;first+=block) {
        Rcpp::checkUserInterrupt();
        const int count=std::min(block,n-first);
        Eigen::VectorXd weight(count),score(count);
        for(int j=0;j<count;++j) {
            const int i=first+j;
            if(weights[i]==0) {weight[j]=score[j]=0;continue;}
            const double eta=X.row(i).dot(beta)+offset[i];
            const double e=std::exp(-std::fabs(eta));
            const double mu=eta>=0 ? 1/(1+e) : e/(1+e);
            const double lp=std::log1p(e);
            likelihood-=weights[i]*(lp+(eta>=0 ? (1-y[i])*eta : -y[i]*eta));
            weight[j]=weights[i]*e/((1+e)*(1+e));
            score[j]=weights[i]*(y[i]-mu);
        }
        const auto xx=X.middleRows(first,count);
        Eigen::MatrixXd wx=weight.cwiseSqrt().asDiagonal()*xx;
        information.noalias()+=wx.transpose()*wx;
        gradient.noalias()+=xx.transpose()*score;
    }
    Eigen::LLT<Eigen::MatrixXd> chol(information);
    if(chol.info()!=Eigen::Success || !information.allFinite() || !std::isfinite(likelihood))
        return Rcpp::List::create(Rcpp::_["valid"]=false,
            Rcpp::_["message"]="full penalized information is not positive definite at these coefficients");
    Eigen::VectorXd diagonal=chol.matrixL().toDenseMatrix().diagonal();
    if((diagonal.array()<=0).any() || !diagonal.allFinite())
        return Rcpp::List::create(Rcpp::_["valid"]=false,Rcpp::_["message"]="invalid information determinant");
    const double penalty=diagonal.array().log().sum();
    for(int first=0;first<n;first+=block) {
        Rcpp::checkUserInterrupt();
        const int count=std::min(block,n-first);
        const auto xx=X.middleRows(first,count);
        Eigen::MatrixXd standardized=chol.matrixL().solve(xx.transpose());
        Eigen::VectorXd adjustment(count);
        for(int j=0;j<count;++j) {
            const int i=first+j;
            if(weights[i]==0) {adjustment[j]=0;continue;}
            const double eta=X.row(i).dot(beta)+offset[i];
            const double e=std::exp(-std::fabs(eta));
            const double mu=eta>=0 ? 1/(1+e) : e/(1+e);
            const double leverage=weights[i]*e/((1+e)*(1+e))*standardized.col(j).squaredNorm();
            adjustment[j]=leverage*(.5-mu);
        }
        gradient.noalias()+=xx.transpose()*adjustment;
    }
    if(!gradient.allFinite())
        return Rcpp::List::create(Rcpp::_["valid"]=false,
            Rcpp::_["message"]="nonfinite adjusted score in the full penalized model");
    return Rcpp::List::create(Rcpp::_["valid"]=true,
        Rcpp::_["value"]=likelihood+penalty,Rcpp::_["loglik"]=likelihood,
        Rcpp::_["penalty"]=penalty,Rcpp::_["gradient"]=gradient,
        Rcpp::_["information"]=information);
}
