#ifndef FASTGLM_PREPARED_GLM_H
#define FASTGLM_PREPARED_GLM_H

#include <RcppEigen.h>
#include "bias_reduction.h"

// Read-only views of a factor and row leverages shared by response fits.
// R owns the underlying arrays, so the cache remains serializable and no
// predictor matrix is copied into a response-specific solver workspace.
struct PreparedGlmInfo {
    Eigen::Map<const Eigen::MatrixXd> lower;
    Eigen::Map<const Eigen::VectorXd> leverage;
    Eigen::Map<const Eigen::MatrixXd> covariance;
    Eigen::Map<const Eigen::VectorXd> adjustment;
    fglm::GammaWeightGroups weight_groups;
    double logdet;

    PreparedGlmInfo(const Rcpp::NumericMatrix& l, const Rcpp::NumericVector& h,
                    const Rcpp::NumericMatrix& v, const Rcpp::NumericVector& adj,
                    const Rcpp::NumericVector& values, const Rcpp::IntegerVector& counts,
                    double ld) :
        lower(l.begin(),l.nrow(),l.ncol()), leverage(h.begin(),h.size()),
        covariance(v.begin(),v.nrow(),v.ncol()), adjustment(adj.begin(),adj.size()), logdet(ld) {
        for(int j=0;j<values.size();++j) weight_groups.emplace_back(values[j],counts[j]);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const {
        Eigen::VectorXd out = lower.triangularView<Eigen::Lower>().solve(rhs);
        lower.transpose().triangularView<Eigen::Upper>().solveInPlace(out);
        return out;
    }
};

#endif
