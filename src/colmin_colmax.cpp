#include <Rcpp.h>
#include <RcppEigen.h>

//[[Rcpp::export]]
Eigen::MatrixXd colMax_dense(const Eigen::Map<Eigen::MatrixXd> & A){
    Eigen::VectorXd colM = A.colwise().maxCoeff();
    return colM;
}

//[[Rcpp::export]]
Eigen::MatrixXd colMin_dense(const Eigen::Map<Eigen::MatrixXd> & A){
    Eigen::VectorXd colM = A.colwise().minCoeff();
    return colM;
}


