#ifndef FASTGLM_BIAS_REDUCTION_H
#define FASTGLM_BIAS_REDUCTION_H
#include <RcppEigen.h>
#include "families.h"
#include "brent.h"
#include <utility>
#include <vector>
namespace fglm {

inline bool bounded_firth_predictors(int code) {
    return code==FAM_BINOMIAL_PROBIT || code==FAM_BINOMIAL_CLOGLOG;
}

// Binomial adjusted scores must use the actual tail probabilities. Combining
// separately floored means and derivatives changes both information and roots.
struct BinomialFirthTerms {
    double weight=0,score=0,ratio=0,logp=0,logq=0,derivative=0;
};

inline BinomialFirthTerms binomial_firth_terms(int code,double eta,double y,double prior) {
    BinomialFirthTerms out;
    double success_rate,failure_rate,log_weight;
    if(code==FAM_BINOMIAL_CLOGLOG) {
        const double t=std::exp(eta);
        if(eta>=-36 && t<32) {
            // Here direct products cannot underflow. Reuse the two tails
            // instead of exponentiating three separate log ratios.
            const double q=std::exp(-t),p=-std::expm1(-t);
            out.logp=t>0.6931471805599453 ? std::log1p(-q) : std::log(p);
            out.logq=-t;out.derivative=t*q;
            if(prior==0) return out;
            const double rate=out.derivative/p;
            out.weight=prior*(t*rate);
            out.score=prior*(y*rate-(1-y)*t);out.ratio=1-t;
            return out;
        }
        out.logp=eta < -36 ? eta : (t>0.6931471805599453 ? std::log1p(-std::exp(-t)) : std::log(-std::expm1(-t)));
        out.logq=-t;
        out.derivative=std::exp(eta-t);
        log_weight=2*eta-t-out.logp;
        success_rate=std::exp(eta-t-out.logp);failure_rate=t;
        out.ratio=1-t;
    } else {
        const double logd=-0.5*eta*eta-0.91893853320467274178;
        out.derivative=std::exp(logd);
        if(std::abs(eta)<8) {
            const double tail=R::pnorm(-std::abs(eta),0,1,true,false);
            out.logp=eta<=0 ? std::log(tail) : std::log1p(-tail);
            out.logq=eta<=0 ? std::log1p(-tail) : std::log(tail);
            if(prior==0) return out;
            const double p=eta<=0 ? tail : 1-tail,q=eta<=0 ? 1-tail : tail;
            out.weight=prior*(out.derivative*out.derivative/(p*q));
            out.score=prior*out.derivative*(y/p-(1-y)/q);out.ratio=-eta;
            return out;
        } else {
            out.logp=R::pnorm(eta,0,1,true,true);
            out.logq=R::pnorm(eta,0,1,false,true);
        }
        log_weight=2*logd-out.logp-out.logq;
        success_rate=std::exp(logd-out.logp);failure_rate=std::exp(logd-out.logq);
        out.ratio=-eta;
    }
    if(prior==0) {out.ratio=0;return out;}
    out.weight=std::exp(std::log(prior)+log_weight);
    // Zero counts must not multiply an infinite tail score.
    out.score=(y>0 ? prior*y*success_rate : 0)-(y<1 ? prior*(1-y)*failure_rate : 0);
    if(out.weight==0) out.ratio=0;
    return out;
}

struct BinomialFirthWorking {
    Eigen::VectorXd weight,score,ratio;
    double deviance=0;
    void update(int code,const Eigen::Ref<const Eigen::VectorXd>& eta,
        const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::VectorXd>& prior,
        bool include_deviance=false,bool check=true) {
        weight.resize(eta.size());score.resize(eta.size());ratio.resize(eta.size());
        deviance=0;
        for(int i=0;i<eta.size();++i) {
            const auto row=binomial_firth_terms(code,eta[i],y[i],prior[i]);
            weight[i]=row.weight;score[i]=row.score;ratio[i]=row.ratio;
            if(include_deviance && prior[i]>0) {
                if(y[i]>0) deviance+=2*prior[i]*y[i]*(std::log(y[i])-row.logp);
                if(y[i]<1) deviance+=2*prior[i]*(1-y[i])*(std::log1p(-y[i])-row.logq);
            }
        }
        if(check) validate();
    }
    void validate() const {
        if(!weight.allFinite() || !score.allFinite() || !ratio.allFinite())
            Rcpp::stop("non-finite binomial adjusted-score quantities; use less extreme starting coefficients");
    }
};

inline double binomial_firth_deviance(int code,const Eigen::Ref<const Eigen::VectorXd>& eta,
    const Eigen::Ref<const Eigen::VectorXd>& y,const Eigen::Ref<const Eigen::VectorXd>& prior) {
    double answer=0;
    for(int i=0;i<eta.size();++i) if(prior[i]>0) {
        double logp,logq;
        if(code==FAM_BINOMIAL_CLOGLOG) {
            const double t=std::exp(eta[i]);logq=-t;
            logp=eta[i]<-36 ? eta[i] : (t>0.6931471805599453 ? std::log1p(-std::exp(-t)) : std::log(-std::expm1(-t)));
        } else {
            logp=R::pnorm(eta[i],0,1,true,true);logq=R::pnorm(eta[i],0,1,false,true);
        }
        if(y[i]>0) answer+=2*prior[i]*y[i]*(std::log(y[i])-logp);
        if(y[i]<1) answer+=2*prior[i]*(1-y[i])*(std::log1p(-y[i])-logq);
    }
    return answer;
}

// Limit a scoring update to one link-scale unit over positive-frequency rows.
// This changes the path, not the adjusted-score root, and is invariant to
// invertible recoding of the predictor columns. Small updates are unchanged.
inline double firth_predictor_step_fraction(const Eigen::VectorXd& change,
    const Eigen::Ref<const Eigen::VectorXd>& weights) {
    double largest=0;
    for(int i=0;i<change.size();++i) if(weights[i]>0) {
        if(!std::isfinite(change[i])) return NA_REAL;
        largest=std::max(largest,std::abs(change[i]));
    }
    return largest>1 ? 1/largest : 1;
}

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
