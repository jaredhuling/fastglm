#ifndef FASTGLM_BIAS_REDUCTION_H
#define FASTGLM_BIAS_REDUCTION_H
#include <RcppEigen.h>
#include "families.h"
#include "brent.h"
#include <utility>
#include <vector>
namespace fglm {

using GammaWeightGroups = std::vector<std::pair<double, int>>;

// Dispersion normalizers depend on prior weights, which stay fixed during
// IRLS. Equal weights therefore require only one special-function evaluation.
inline GammaWeightGroups gamma_weight_groups(
    const Eigen::Ref<const Eigen::VectorXd>& weights)
{
    GammaWeightGroups groups;
    double first = 0;
    int positive = 0;
    bool equal = true;
    for (int i = 0; i < weights.size(); ++i) {
        if (weights[i] <= 0) continue;
        if (positive == 0) first = weights[i];
        else if (weights[i] != first) equal = false;
        ++positive;
    }
    if (positive == 0) return groups;
    if (equal) {
        groups.emplace_back(first, positive);
        return groups;
    }
    std::vector<double> sorted;
    sorted.reserve(positive);
    for (int i = 0; i < weights.size(); ++i)
        if (weights[i] > 0) sorted.push_back(weights[i]);
    std::sort(sorted.begin(), sorted.end());
    for (double w : sorted) {
        if (groups.empty() || groups.back().first != w) groups.emplace_back(w, 1);
        else ++groups.back().second;
    }
    return groups;
}

// Mean bias-reducing dispersion score, multiplied by 2 phi^2.
// For Gaussian and inverse Gaussian this reduces to D - (n-p) phi.
// Gamma requires the derivatives of its dispersion normalizer.
inline double adjusted_dispersion(int code, double dev,
    const Eigen::Ref<const Eigen::VectorXd>& weights, int rank,
    const GammaWeightGroups* cached_groups = nullptr)
{
    if ((code >= 3 && code <= 9) || (code >= 17 && code <= 19)) return 1.0;
    const int n = (weights.array() > 0).count();
    if (n <= rank) Rcpp::stop("bias-reduced dispersion requires positive residual degrees of freedom");
    if (code < 10 || code > 12) return dev / (n - rank);
    if (dev <= 1e-10) return std::max(0.0, dev) / (n - rank);
    GammaWeightGroups local_groups;
    if (cached_groups == nullptr) {
        local_groups = gamma_weight_groups(weights);
        cached_groups = &local_groups;
    }
    auto score = [&](double logphi) {
        double phi = std::exp(logphi), expected = 0, d2 = 0, d3 = 0;
        for (const auto& group : *cached_groups) {
            double w = group.first;
            double count = group.second;
            double a = w / phi;
            if (a > 1e5) {
                double t = 1 / a;
                expected += count * w * (t + t*t/6 - std::pow(t,4)/60);
                d2 += count * w*w * (t*t + t*t*t/3 - std::pow(t,5)/15);
                d3 += count * w*w*w * (2*t*t*t + std::pow(t,4) - std::pow(t,6)/3);
            } else {
                expected += count * 2*w * (std::log(a) - R::digamma(a));
                d2 += count * w*w * (2*R::trigamma(a) - 2/a);
                d3 += count * w*w*w * (-2*R::psigamma(a, 2) - 2/(a*a));
            }
        }
        return dev - expected + (rank - 2)*phi + d3/d2;
    };
    double center = std::max(-30.0, std::min(30.0, std::log(dev / (n - rank))));
    double lo = std::max(-30.0, center - std::log(2.0));
    double hi = std::min(30.0, center + std::log(2.0));
    double flo = score(lo), fhi = score(hi);
    while (flo < 0 && lo > -30) {
        lo = std::max(-30.0, lo - std::log(2.0));
        flo = score(lo);
    }
    while (fhi > 0 && hi < 30) {
        hi = std::min(30.0, hi + std::log(2.0));
        fhi = score(hi);
    }
    if (!(flo >= 0 && fhi <= 0))
        Rcpp::stop("could not bracket the adjusted Gamma dispersion score");
    // The same bracketed equation is solved to relative dispersion precision.
    // Brent interpolation avoids dozens of redundant full weighted sums.
    return std::exp(brent_root(score, lo, hi, flo, fhi, 1e-12, 100));
}
}
#endif
