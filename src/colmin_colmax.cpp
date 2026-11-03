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



// [[Rcpp::export]]
bool all_finite_dense(const Eigen::Map<Eigen::MatrixXd>& x) {
    return x.allFinite();
}

// [[Rcpp::export]]
Rcpp::LogicalVector constant_columns_dense(const Eigen::Map<Eigen::MatrixXd>& x) {
    Rcpp::LogicalVector out(x.cols(), false);
    if (x.rows() == 0) return out;
    for (Eigen::Index j = 0; j < x.cols(); ++j) {
        const double first = x(0, j);
        if (first == 0) continue;
        bool constant = true;
        for (Eigen::Index i = 1; i < x.rows(); ++i) {
            if (x(i, j) != first) { constant = false; break; }
        }
        out[j] = constant;
    }
    return out;
}
