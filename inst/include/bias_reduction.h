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

// Terms in the Gamma dispersion score. Grouped prior weights preserve the
// exact weighted equation while avoiding repeated special-function calls.
struct GammaDispersionTerms {
    double expected = 0, d2 = 0, d3 = 0;
    void add(const GammaWeightGroups& groups, double phi) {
        for (const auto& group : groups) {
            const double w = group.first, count = group.second, a = w / phi;
            if (a > 1e5) {
                const double t = 1 / a;
                expected += count*w*(t + t*t/6 - std::pow(t,4)/60);
                d2 += count*w*w*(t*t + t*t*t/3 - std::pow(t,5)/15);
                d3 += count*w*w*w*(2*t*t*t + std::pow(t,4) - std::pow(t,6)/3);
            } else {
                expected += count*2*w*(std::log(a)-R::digamma(a));
                d2 += count*w*w*(2*R::trigamma(a)-2/a);
                d3 += count*w*w*w*(-2*R::psigamma(a,2)-2/(a*a));
            }
        }
    }
    double numerator(double dev, int rank, double phi) const {
        return dev - expected + (rank-2)*phi + d3/d2;
    }
};

inline void append_adjusted_diagnostics(Rcpp::List& out,
    const Eigen::VectorXd& score, const Eigen::VectorXd& information,
    double dispersion_score, double standardized_dispersion,
    const std::string& reason)
{
    Eigen::VectorXd standardized = score.array() / information.array().sqrt();
    out["adjusted.score"] = score;
    out["adjusted.information.diagonal"] = information;
    out["standardized.adjusted.score"] = standardized;
    out["adjusted.dispersion.score"] = dispersion_score;
    out["standardized.dispersion.score"] = standardized_dispersion;
    out["convergence.reason"] = reason;
}

inline void dispersion_diagnostics(int code, double dev, int n, int rank,
    double phi, const GammaDispersionTerms& gamma, double& score, double& standardized)
{
    score = standardized = NA_REAL;
    if ((code >= 3 && code <= 9) || (code >= 17 && code <= 19) || !(phi > 0)) return;
    double numerator = dev - (n-rank)*phi;
    double d2 = n*phi*phi;
    if (code >= 10 && code <= 12) {
        numerator = gamma.numerator(dev, rank, phi);
        d2 = gamma.d2;
    }
    score = numerator / (2*phi*phi);
    standardized = numerator / std::sqrt(2*d2);
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
        const double phi = std::exp(logphi);
        GammaDispersionTerms terms;
        terms.add(*cached_groups, phi);
        return terms.numerator(dev, rank, phi);
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
