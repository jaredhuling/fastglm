#ifndef FASTGLM_ORDINAL_H
#define FASTGLM_ORDINAL_H
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
namespace fglm_ordinal {
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct OrdinalEvaluationError : std::runtime_error {
    explicit OrdinalEvaluationError(const std::string& message) : std::runtime_error(message) {}
};

inline double ordinal_log1mexp(double x) {
    return x < -std::log(2.0) ? std::log1p(-std::exp(x)) : std::log(-std::expm1(x));
}

// Factor responses stay as category indexes. A grouped response is already an
// n by K input; neither representation is expanded into an ordinal design.
struct OrdinalData {
    Eigen::Map<const MatrixXd> x;
    Rcpp::IntegerVector codes;
    Rcpp::NumericMatrix counts;
    Rcpp::NumericVector weights, offset;
    VectorXd totals;
    int n,p,K,q,d,block,link;
    bool grouped;

    OrdinalData(Rcpp::NumericMatrix X,Rcpp::IntegerVector y,Rcpp::NumericMatrix Y,
        Rcpp::NumericVector w,Rcpp::NumericVector off,int categories,int block_size,int link_code=0,bool require_positive=true) :
        x(X.begin(),X.nrow(),X.ncol()),codes(y),counts(Y),weights(w),offset(off),
        n(X.nrow()),p(X.ncol()),K(categories),q(categories-1),d(p+q),block(block_size),link(link_code),
        grouped(Y.nrow()!=0 || Y.ncol()!=0) {
        if(n<1 || K<2 || block<1 || link<0 || link>2 || w.size()!=n || off.size()!=n ||
           (grouped && (Y.nrow()!=n || Y.ncol()!=K)) || (!grouped && y.size()!=n))
            Rcpp::stop("incompatible ordinal data dimensions");
        if(!x.allFinite()) Rcpp::stop("ordinal design must be finite");
        totals.resize(n);
        for(int i=0;i<n;++i) {
            if(!std::isfinite(w[i]) || w[i]<0 || !std::isfinite(off[i]))
                Rcpp::stop("ordinal weights and offsets must be finite, with nonnegative weights");
            double total=w[i];
            if(grouped) {
                total=0;
                for(int k=0;k<K;++k) {
                    if(!std::isfinite(Y(i,k)) || Y(i,k)<0)
                        Rcpp::stop("ordinal category counts must be finite and nonnegative");
                    // Effective counts are the model frequencies. Their sum
                    // can be finite even when the unweighted row sum overflows;
                    // zero-frequency rows likewise must not form 0 * Inf.
                    total+=w[i]*Y(i,k);
                }
            } else if(y[i]<1 || y[i]>K) Rcpp::stop("invalid ordinal category index");
            totals[i]=total;
            if(!std::isfinite(totals[i])) Rcpp::stop("ordinal total frequency overflow");
        }
        if((require_positive && !(totals.sum()>0)) || !std::isfinite(totals.sum()))
            Rcpp::stop("ordinal fitting requires a finite positive total frequency");
    }

    template<class Function> void visit(Function&& function) const { function(*this); }

    double linear(int i,const VectorXd& beta) const {
        double sum=0,error=0;
        for(int j=0;j<p;++j) {
            const double product=x(i,j)*beta[j],next=sum+product;
            error+=std::abs(sum)>=std::abs(product) ? (sum-next)+product : (product-next)+sum;
            error+=std::fma(x(i,j),beta[j],-product);
            sum=next;
        }
        return sum+error;
    }

    double predictor_step(const VectorXd& direction) const {
        const VectorXd beta=direction.tail(p);
        const double low=direction.head(q).minCoeff(),high=direction.head(q).maxCoeff();
        double ans=0;
        for(int first=0;first<n;first+=block) {
            Rcpp::checkUserInterrupt();
            const int last=std::min(n,first+block);
            for(int i=first;i<last;++i) if(totals[i]>0) {
                const double dx=linear(i,beta);
                ans=std::max(ans,std::max(std::abs(low-dx),std::abs(high-dx)));
            }
        }
        return ans;
    }
};

struct OrdinalRow {
    VectorXd eta,g,cdf,survival,logp,left,right,neighbor,gprime,logg,logcdf,logsurvival,location,logden;
    VectorXd loggap,inverse_gap,neighbor_gap;
    int link=0;
    explicit OrdinalRow(const VectorXd& alpha) : eta(alpha.size()),g(alpha.size()),cdf(alpha.size()),
        survival(alpha.size()),logp(alpha.size()+1),left(alpha.size()),right(alpha.size()),neighbor(alpha.size()),
        gprime(alpha.size()),logg(alpha.size()),logcdf(alpha.size()),logsurvival(alpha.size()),
        location(alpha.size()+1),logden(alpha.size()),loggap(alpha.size()),inverse_gap(alpha.size()),neighbor_gap(alpha.size()) {
        for(int s=1;s<alpha.size();++s) {
            const double gap=alpha[s]-alpha[s-1];
            loggap[s]=ordinal_log1mexp(-gap);inverse_gap[s]=1/std::expm1(gap);
        }
        for(int s=1;s+1<alpha.size();++s)
            neighbor_gap[s]=-std::expm1(alpha[s-1]-alpha[s+1]);
    }
    void evaluate(const VectorXd& alpha,double linear,int link_code=0) {
        link=link_code;
        const int q=alpha.size();
        if(link!=0) {
            for(int s=0;s<q;++s) {
                eta[s]=alpha[s]-linear;
                if(link==1) {
                    Rf_pnorm_both(eta[s],&logcdf[s],&logsurvival[s],2,1);
                    logg[s]=R::dnorm(eta[s],0,1,true);
                    g[s]=std::exp(logg[s]);gprime[s]=-eta[s]*g[s];
                } else {
                    const double e=std::exp(eta[s]);
                    logcdf[s]=eta[s]<-36 ? eta[s] : ordinal_log1mexp(-e);
                    logsurvival[s]=-e;logg[s]=eta[s]-e;
                    g[s]=std::exp(logg[s]);gprime[s]=std::isfinite(e) ? g[s]*(1-e) : 0;
                }
                cdf[s]=std::exp(logcdf[s]);survival[s]=std::exp(logsurvival[s]);
            }
            logp[0]=logcdf[0];logp[q]=logsurvival[q-1];
            for(int s=1;s<q;++s) {
                if(link==2) {
                    const double ld=eta[s]+std::log(-std::expm1(alpha[s-1]-alpha[s]));
                    logden[s]=ld < -36 ? ld : std::log(-std::expm1(-std::exp(ld)));
                }
                logp[s]=eta[s]<=0 ? logcdf[s]+ordinal_log1mexp(logcdf[s-1]-logcdf[s]) :
                    logsurvival[s-1]+(link==2 ? logden[s] : ordinal_log1mexp(logsurvival[s]-logsurvival[s-1]));
            }
            for(int s=0;s<q;++s) {
                left[s]=std::exp(logg[s]-logp[s]);right[s]=std::exp(logg[s]-logp[s+1]);
            }
            if(link==2) {
                right[q-1]=std::exp(eta[q-1]);
                for(int s=1;s<q;++s) {
                    const double difference=std::exp(eta[s])*(-std::expm1(alpha[s-1]-alpha[s]));
                    left[s]=std::exp(eta[s]-difference-logden[s]);
                    right[s-1]=std::exp(eta[s-1]-logden[s]);
                }
            }
            location[0]=left[0];location[q]=-right[q-1];
            for(int s=1;s<q;++s) {
                const double gap=alpha[s]-alpha[s-1];
                const double delta=link==1 ? -.5*gap*(eta[s]+eta[s-1]) :
                    gap-std::exp(eta[s])*(-std::expm1(-gap));
                location[s]=delta>=0 ? left[s]*(-std::expm1(-delta)) :
                    -right[s-1]*(-std::expm1(delta));
            }
            for(int s=0;s<q;++s) neighbor[s]=location[s]-location[s+1];
            if(!logp.allFinite() || !left.allFinite() || !right.allFinite() ||
               !location.allFinite() || !gprime.allFinite())
                throw OrdinalEvaluationError("non-finite ordinal probabilities or derivatives");
            return;
        }
        for(int s=0;s<q;++s) {
            eta[s]=alpha[s]-linear;
            const double e=std::exp(-std::abs(eta[s])),den=1+e;
            cdf[s]=eta[s]>=0 ? 1/den : e/den;
            survival[s]=eta[s]>=0 ? e/den : 1/den;
            g[s]=e/(den*den);
            const double logden=std::log1p(e);
            logcdf[s]=eta[s]>=0 ? -logden : eta[s]-logden;
            logsurvival[s]=eta[s]>=0 ? -eta[s]-logden : -logden;
        }
        logp[0]=logcdf[0];logp[q]=logsurvival[q-1];
        for(int s=1;s<q;++s)
            logp[s]=logcdf[s]+logsurvival[s-1]+loggap[s];
        for(int s=0;s<q;++s) {
            // g_s / P_s and g_s / P_{s+1} simplify exactly for the logit link.
            // Threshold gaps are common to every row and computed only once.
            left[s]=survival[s]+(s ? inverse_gap[s] : 0);
            right[s]=cdf[s]+(s+1<q ? inverse_gap[s+1] : 0);
            neighbor[s]=q==1 ? 1 : (s==0 ? cdf[1] : (s==q-1 ? survival[q-2] :
                cdf[s+1]*survival[s-1]*neighbor_gap[s]));
        }
        if(!logp.allFinite() || !left.allFinite() || !right.allFinite())
            throw OrdinalEvaluationError("non-finite ordinal probabilities or derivatives");
    }
    double location_score(int category) const {
        if(link!=0) return location[category];
        return (category<eta.size() ? survival[category] : 0)-
            (category>0 ? cdf[category-1] : 0);
    }
};

struct OrdinalState {
    MatrixXd information,inverse;
    VectorXd ordinary,adjustment,score,step;
    double loglik=0,scaled_score=0,predictor_step=0,rcond=0,min_logp=0;
    explicit OrdinalState(int d=0) : information(MatrixXd::Zero(d,d)),inverse(d,d),
        ordinary(VectorXd::Zero(d)),adjustment(VectorXd::Zero(d)),score(d),step(d) {}
};

inline void ordinal_add_information(const OrdinalData& data,const VectorXd& parameters,OrdinalState& result) {
    const int p=data.p,q=data.q;
    const VectorXd alpha=parameters.head(q),beta=parameters.tail(p);
    OrdinalRow row(alpha);
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        const int length=std::min(data.block,data.n-first);
        MatrixXd rowsums=MatrixXd::Zero(length,q);
        VectorXd slope_score=VectorXd::Zero(length),slope_weight=VectorXd::Zero(length);
        for(int r=0;r<length;++r) {
            const int i=first+r;const double m=data.totals[i];
            if(m==0) continue;
            row.evaluate(alpha,data.linear(i,beta)+data.offset[i],data.link);
            result.min_logp=std::min(result.min_logp,row.logp.minCoeff());
            if(data.grouped) {
                for(int k=0;k<data.K;++k) {
                    const double y=data.weights[i]*data.counts(i,k);
                    result.loglik+=y*row.logp[k];
                    slope_score[r]-=y*row.location_score(k);
                    if(k<q) result.ordinary[k]+=y*row.left[k];
                    if(k>0) result.ordinary[k-1]-=y*row.right[k-1];
                }
            } else {
                const int k=data.codes[i]-1;
                result.loglik+=m*row.logp[k];
                slope_score[r]=-m*row.location_score(k);
                if(k<q) result.ordinary[k]+=m*row.left[k];
                if(k>0) result.ordinary[k-1]-=m*row.right[k-1];
            }
            for(int s=0;s<q;++s) {
                result.information(s,s)+=m*row.g[s]*(row.left[s]+row.right[s]);
                rowsums(r,s)=m*row.g[s]*row.neighbor[s];
                slope_weight[r]+=rowsums(r,s);
                if(s+1<q) result.information(s,s+1)-=m*row.g[s+1]*row.right[s];
            }
        }
        if(p) {
            const auto xx=data.x.middleRows(first,length);
            result.ordinary.tail(p).noalias()+=xx.transpose()*slope_score;
            result.information.topRightCorner(q,p).noalias()-=rowsums.transpose()*xx;
            MatrixXd weighted=slope_weight.asDiagonal()*xx;
            result.information.bottomRightCorner(p,p).noalias()+=xx.transpose()*weighted;
        }
    }
}

inline void ordinal_factor_information(OrdinalState& result,int p,int q,double condition_tol) {
    const int d=p+q;
    for(int s=1;s<q;++s) result.information(s,s-1)=result.information(s-1,s);
    if(p) result.information.bottomLeftCorner(p,q)=result.information.topRightCorner(q,p).transpose();
    VectorXd diagonal=result.information.diagonal().array().sqrt();
    if(!diagonal.allFinite() || (diagonal.array()<=0).any() || !std::isfinite(result.loglik))
        throw OrdinalEvaluationError("nonpositive or non-finite ordinal information");
    const VectorXd inverse_scale=diagonal.cwiseInverse();
    MatrixXd correlation=inverse_scale.asDiagonal()*result.information*inverse_scale.asDiagonal();
    Eigen::LLT<MatrixXd> factor(correlation);
    if(factor.info()!=Eigen::Success)
        throw OrdinalEvaluationError("ordinal information is not positive definite");
    result.rcond=factor.rcond();
    if(!std::isfinite(result.rcond) || result.rcond<condition_tol)
        throw OrdinalEvaluationError("ordinal information is numerically singular");
    result.inverse=inverse_scale.asDiagonal()*factor.solve(MatrixXd::Identity(d,d))*inverse_scale.asDiagonal();
    result.inverse=(result.inverse+result.inverse.transpose()).eval()/2;
}

inline void ordinal_add_adjustment(const OrdinalData& data,const VectorXd& parameters,OrdinalState& result) {
    const int p=data.p,q=data.q;
    const VectorXd alpha=parameters.head(q),beta=parameters.tail(p);
    OrdinalRow row(alpha);
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        const int length=std::min(data.block,data.n-first);
        const auto xx=data.x.middleRows(first,length);
        MatrixXd xb,xa;
        if(p) {
            xb.noalias()=xx*result.inverse.bottomRightCorner(p,p);
            xa.noalias()=xx*result.inverse.bottomLeftCorner(p,q);
        }
        VectorXd slope=VectorXd::Zero(length),c(q);
        for(int r=0;r<length;++r) {
            const int i=first+r;const double m=data.totals[i];
            if(m==0) continue;
            row.evaluate(alpha,data.linear(i,beta)+data.offset[i],data.link);
            const double quadratic=p ? xb.row(r).dot(xx.row(r)) : 0;
            for(int s=0;s<q;++s) {
                const double v=result.inverse(s,s)+quadratic-(p ? 2*xa(r,s) : 0);
                c[s]=data.link==0 ? .5*m*row.g[s]*(1-2*row.cdf[s])*v : .5*m*row.gprime[s]*v;
                slope[r]-=c[s]*row.neighbor[s];
            }
            for(int s=0;s<q;++s)
                result.adjustment[s]+=(c[s]-(s ? c[s-1] : 0))*row.left[s]-
                    ((s+1<q ? c[s+1] : 0)-c[s])*row.right[s];
        }
        if(p) result.adjustment.tail(p).noalias()+=xx.transpose()*slope;
    }
}

// A source completes the information pass before rereading rows for the bias
// adjustment. Dense and streamed fits use these same mathematical kernels.
template<class Data>
OrdinalState ordinal_evaluate(const Data& data,const VectorXd& parameters,
    bool firth,double condition_tol) {
    const int p=data.p,q=data.q,d=data.d;
    if(parameters.size()!=d || !parameters.allFinite())
        throw OrdinalEvaluationError("invalid ordinal parameters");
    for(int s=1;s<q;++s) if(!(parameters[s]>parameters[s-1]))
        throw OrdinalEvaluationError("ordinal thresholds are not strictly ordered");
    OrdinalState result(d);
    data.visit([&](const OrdinalData& block) { ordinal_add_information(block,parameters,result); });
    ordinal_factor_information(result,p,q,condition_tol);
    if(firth) data.visit([&](const OrdinalData& block) { ordinal_add_adjustment(block,parameters,result); });
    result.score=result.ordinary+result.adjustment;
    if(!result.score.allFinite() || !result.inverse.allFinite())
        throw OrdinalEvaluationError("non-finite ordinal adjusted score or covariance");
    result.step.noalias()=result.inverse*result.score;
    result.scaled_score=(result.score.array().abs()/result.information.diagonal().array().sqrt()).maxCoeff();
    if(!result.step.allFinite() || !std::isfinite(result.scaled_score))
        throw OrdinalEvaluationError("non-finite ordinal convergence diagnostic");
    result.predictor_step=data.predictor_step(result.step);
    if(!std::isfinite(result.predictor_step))
        throw OrdinalEvaluationError("non-finite ordinal predictor step");
    return result;
}

inline Rcpp::List ordinal_state_list(const OrdinalState& state) {
    return Rcpp::List::create(Rcpp::_["loglik"]=state.loglik,
        Rcpp::_["score"]=state.score,Rcpp::_["ordinary.score"]=state.ordinary,
        Rcpp::_["adjustment"]=state.adjustment,Rcpp::_["information"]=state.information,
        Rcpp::_["cov.unscaled"]=state.inverse,Rcpp::_["step"]=state.step,
        Rcpp::_["predictor.step"]=state.predictor_step,Rcpp::_["scaled.score"]=state.scaled_score,
        Rcpp::_["information.rcond"]=state.rcond,Rcpp::_["minimum.log.probability"]=state.min_logp);
}

// Observed-information Newton iterations for maximum likelihood. A row touches
// only the thresholds bounding its observed categories, so a pass costs far less
// than the expected information, which needs every threshold of every row. The
// iterate is a starting point only: ordinal_fit() still certifies convergence
// with the expected information and continues from here if needed.
struct ObservedTerms {
    double logp=0,left=0,right=0,location=0,upper=0,lower=0,cross=0,upper_x=0,lower_x=0,slope=0;
};

struct ObservedThresholds {
    VectorXd alpha,inverse_gap,loggap,one_minus;
    int q,link;
    ObservedThresholds(const VectorXd& a,int link_code) : alpha(a),inverse_gap(VectorXd::Zero(a.size())),
        loggap(VectorXd::Zero(a.size())),one_minus(VectorXd::Zero(a.size())),q(a.size()),link(link_code) {
        for(int s=1;s<q;++s) {
            const double gap=alpha[s]-alpha[s-1];
            loggap[s]=ordinal_log1mexp(-gap);inverse_gap[s]=1/std::expm1(gap);one_minus[s]=-std::expm1(-gap);
        }
    }
    // Logit terms simplify exactly. With a = f_u/P and b = f_l/P, the location
    // curvature reduces to (a+b)P, so no term cancels in the tails.
    ObservedTerms logit(int k,double linear) const {
        ObservedTerms t;
        double Fu=1,Su=0,logFu=0,Fl=0,Sl=1,logSl=0;
        if(k<q) {
            const double e=alpha[k]-linear,x=std::exp(-std::abs(e)),den=1+x,l=std::log1p(x);
            Fu=e>=0 ? 1/den : x/den;Su=e>=0 ? x/den : 1/den;logFu=e>=0 ? -l : e-l;
        }
        if(k>0) {
            const double e=alpha[k-1]-linear,x=std::exp(-std::abs(e)),den=1+x,l=std::log1p(x);
            Fl=e>=0 ? 1/den : x/den;Sl=e>=0 ? x/den : 1/den;logSl=e>=0 ? -e-l : -l;
        }
        const bool interior=k>0 && k<q;
        const double G=interior ? inverse_gap[k] : 0;
        const double P=interior ? Fu*Sl*one_minus[k] : (k==0 ? Fu : Sl);
        t.logp=logFu+logSl+(interior ? loggap[k] : 0);
        t.left=k<q ? Su+G : 0;t.right=k>0 ? Fl+G : 0;t.location=Su-Fl;
        t.upper=t.left*(G+Fu);t.lower=t.right*(G+Sl);t.cross=-t.left*t.right;
        t.upper_x=-t.left*P;t.lower_x=-t.right*P;t.slope=(t.left+t.right)*P;
        return t;
    }
    ObservedTerms general(int k,double linear) const {
        ObservedTerms t;
        double eu=0,el=0,logFu=0,logSu=0,logFl=0,logSl=0,logGu=0,logGl=0,hu=0,hl=0;
        const auto read=[&](int s,double& e,double& logF,double& logS,double& logG,double& h) {
            e=alpha[s]-linear;
            if(link==1) {
                Rf_pnorm_both(e,&logF,&logS,2,1);
                logG=R::dnorm(e,0,1,true);h=-e;
            } else {
                const double x=std::exp(e);
                logF=e<-36 ? e : ordinal_log1mexp(-x);logS=-x;logG=e-x;h=1-x;
            }
        };
        if(k<q) read(k,eu,logFu,logSu,logGu,hu);
        if(k>0) read(k-1,el,logFl,logSl,logGl,hl);
        if(k==0) {
            t.logp=logFu;t.left=std::exp(logGu-t.logp);t.location=t.left;
        } else if(k==q) {
            t.logp=logSl;t.right=link==2 ? std::exp(el) : std::exp(logGl-t.logp);t.location=-t.right;
        } else {
            const double gap=alpha[k]-alpha[k-1];
            double logden=0;
            if(link==2) {
                const double ld=eu+std::log(one_minus[k]);
                logden=ld<-36 ? ld : std::log(-std::expm1(-std::exp(ld)));
            }
            t.logp=eu<=0 ? logFu+ordinal_log1mexp(logFl-logFu) :
                logSl+(link==2 ? logden : ordinal_log1mexp(logSu-logSl));
            if(link==2) {
                const double difference=std::exp(eu)*one_minus[k];
                t.left=std::exp(eu-difference-logden);t.right=std::exp(el-logden);
            } else {
                t.left=std::exp(logGu-t.logp);t.right=std::exp(logGl-t.logp);
            }
            const double delta=link==1 ? -.5*gap*(eu+el) : gap-std::exp(eu)*one_minus[k];
            t.location=delta>=0 ? t.left*(-std::expm1(-delta)) : -t.right*(-std::expm1(delta));
        }
        t.upper=t.left*(t.left-hu);t.lower=t.right*(t.right+hl);t.cross=-t.left*t.right;
        t.upper_x=t.left*(hu-t.location);t.lower_x=t.right*(t.location-hl);
        t.slope=t.location*t.location-t.left*hu+t.right*hl;
        return t;
    }
    ObservedTerms terms(int k,double linear) const { return link==0 ? logit(k,linear) : general(k,linear); }
};

struct ObservedState {
    MatrixXd information;
    VectorXd score;
    double loglik=0,change=0,row_norm=0;
    explicit ObservedState(int d=0) : information(MatrixXd::Zero(d,d)),score(VectorXd::Zero(d)) {}
};

// Adds one data block. `direction` is optional and records the largest change in
// a cumulative predictor, computed in this pass so a source needs no second read.
inline void observed_add(const OrdinalData& data,const ObservedThresholds& thresholds,const VectorXd& beta,
    const VectorXd* direction,ObservedState& out) {
    const int p=data.p,q=data.q;
    VectorXd eta,change,slope_score,slope_weight,upper_x,lower_x;
    Eigen::VectorXi category;
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        const int length=std::min(data.block,data.n-first);
        const auto xx=data.x.middleRows(first,length);
        eta.resize(length);
        if(p) eta.noalias()=xx*beta; else eta.setZero();
        if(direction && p) {
            const double low=direction->head(q).minCoeff(),high=direction->head(q).maxCoeff();
            change.noalias()=xx*direction->tail(p);
            for(int r=0;r<length;++r) if(data.totals[first+r]>0)
                out.change=std::max(out.change,std::max(std::abs(low-change[r]),std::abs(high-change[r])));
        } else if(direction) out.change=std::max(out.change,direction->head(q).cwiseAbs().maxCoeff());
        slope_score.setZero(length);slope_weight.setZero(length);
        if(data.grouped) {
            MatrixXd rows=MatrixXd::Zero(length,q);
            for(int r=0;r<length;++r) {
                const int i=first+r;
                if(data.totals[i]==0) continue;
                const double linear=eta[r]+data.offset[i];
                for(int k=0;k<data.K;++k) {
                    const double y=data.weights[i]*data.counts(i,k);
                    if(y==0) continue;
                    const ObservedTerms t=thresholds.terms(k,linear);
                    out.loglik+=y*t.logp;slope_score[r]-=y*t.location;slope_weight[r]+=y*t.slope;
                    if(k<q) {
                        out.score[k]+=y*t.left;out.information(k,k)+=y*t.upper;rows(r,k)+=y*t.upper_x;
                    }
                    if(k>0) {
                        out.score[k-1]-=y*t.right;out.information(k-1,k-1)+=y*t.lower;rows(r,k-1)+=y*t.lower_x;
                        if(k<q) out.information(k,k-1)+=y*t.cross;
                    }
                }
            }
            if(p) out.information.topRightCorner(q,p).noalias()+=rows.transpose()*xx;
        } else {
            upper_x.setZero(length);lower_x.setZero(length);category.resize(length);
            for(int r=0;r<length;++r) {
                const int i=first+r;const double y=data.totals[i];
                category[r]=-1;
                if(y==0) continue;
                const int k=data.codes[i]-1;
                category[r]=k;
                const ObservedTerms t=thresholds.terms(k,eta[r]+data.offset[i]);
                out.loglik+=y*t.logp;slope_score[r]=-y*t.location;slope_weight[r]=y*t.slope;
                upper_x[r]=y*t.upper_x;lower_x[r]=y*t.lower_x;
                if(k<q) {out.score[k]+=y*t.left;out.information(k,k)+=y*t.upper;}
                if(k>0) {
                    out.score[k-1]-=y*t.right;out.information(k-1,k-1)+=y*t.lower;
                    if(k<q) out.information(k,k-1)+=y*t.cross;
                }
            }
            for(int j=0;j<p;++j) {
                double* column=out.information.col(q+j).data();
                for(int r=0;r<length;++r) {
                    const int k=category[r];
                    if(k<0) continue;
                    const double value=xx(r,j);
                    if(k<q) column[k]+=upper_x[r]*value;
                    if(k>0) column[k-1]+=lower_x[r]*value;
                }
            }
        }
        if(p) {
            out.score.tail(p).noalias()+=xx.transpose()*slope_score;
            if(slope_weight.minCoeff()<0) throw OrdinalEvaluationError("negative observed curvature");
            const MatrixXd root=slope_weight.cwiseSqrt().asDiagonal()*xx;
            out.information.bottomRightCorner(p,p).selfadjointView<Eigen::Lower>().rankUpdate(root.transpose());
            for(int r=0;r<length;++r) if(data.totals[first+r]>0)
                out.row_norm=std::max(out.row_norm,xx.row(r).norm());
        }
    }
}

template<class Data>
ObservedState observed_evaluate(const Data& data,const VectorXd& parameters,const VectorXd* direction) {
    const int p=data.p,q=data.q;
    if(!parameters.allFinite()) throw OrdinalEvaluationError("invalid ordinal parameters");
    for(int s=1;s<q;++s) if(!(parameters[s]>parameters[s-1]))
        throw OrdinalEvaluationError("ordinal thresholds are not strictly ordered");
    const ObservedThresholds thresholds(parameters.head(q),data.link);
    const VectorXd beta=parameters.tail(p);
    ObservedState out(data.d);
    data.visit([&](const OrdinalData& block) { observed_add(block,thresholds,beta,direction,out); });
    if(!std::isfinite(out.loglik) || !out.score.allFinite() || !out.information.allFinite())
        throw OrdinalEvaluationError("non-finite observed ordinal terms");
    for(int s=1;s<q;++s) out.information(s-1,s)=out.information(s,s-1);
    if(p) {
        out.information.bottomRightCorner(p,p).template triangularView<Eigen::StrictlyUpper>()=
            out.information.bottomRightCorner(p,p).transpose();
        out.information.bottomLeftCorner(p,q)=out.information.topRightCorner(q,p).transpose();
    }
    return out;
}

// Returns the iterations used. Any failure stops here, because the expected
// information loop continues from the last accepted parameters.
template<class Data>
int observed_newton(const Data& data,VectorXd& parameters,double score_tol,double step_tol,int maxit,
    double max_step,int max_halves,bool trace,int& evaluations) {
    const int q=data.q,p=data.p,d=data.d;
    int iterations=0;
    try {
        ObservedState current=observed_evaluate(data,parameters,nullptr);
        ++evaluations;
        double row_norm=current.row_norm;
        for(;iterations<maxit;++iterations) {
            Rcpp::checkUserInterrupt();
            const VectorXd diagonal=current.information.diagonal();
            if((diagonal.array()<=0).any()) return iterations;
            const VectorXd inverse_scale=diagonal.cwiseSqrt().cwiseInverse();
            const MatrixXd correlation=inverse_scale.asDiagonal()*current.information*inverse_scale.asDiagonal();
            const Eigen::LLT<MatrixXd> factor(correlation);
            if(factor.info()!=Eigen::Success) return iterations;
            const VectorXd direction=inverse_scale.asDiagonal()*factor.solve(inverse_scale.cwiseProduct(current.score));
            if(!direction.allFinite()) return iterations;
            const double scaled=current.score.cwiseProduct(inverse_scale).cwiseAbs().maxCoeff();
            const double bound=direction.head(q).cwiseAbs().maxCoeff()+row_norm*(p ? direction.tail(p).norm() : 0);
            if(trace) Rcpp::Rcout<<"observed iteration "<<iterations<<": score "<<scaled<<", step bound "<<bound<<"\n";
            double fraction=1;
            for(int s=1;s<q;++s) {
                const double change=direction[s]-direction[s-1];
                if(change<0) fraction=std::min(fraction,.9*(parameters[s]-parameters[s-1])/(-change));
            }
            if(scaled<score_tol && bound<step_tol) {
                if(fraction==1) parameters+=direction;
                return iterations+1;
            }
            const double ascent=current.score.dot(direction);
            const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(current.loglik));
            bool accepted=false;
            VectorXd candidate(d);
            ObservedState trial;
            for(int h=0;h<=max_halves && !accepted;++h) {
                candidate=parameters+fraction*direction;
                const VectorXd scaled_direction=fraction*direction;
                const bool measure=fraction*bound>max_step;
                try {
                    trial=observed_evaluate(data,candidate,measure ? &scaled_direction : nullptr);
                    ++evaluations;
                    if(measure && trial.change>max_step) { fraction*=max_step/trial.change;continue; }
                    if(trial.loglik>=current.loglik+1e-4*fraction*ascent-rounding) accepted=true;
                    else fraction/=2;
                } catch(const OrdinalEvaluationError&) { ++evaluations;fraction/=2; }
            }
            if(!accepted) return iterations;
            parameters=candidate;current=std::move(trial);
        }
    } catch(const OrdinalEvaluationError&) {}
    return iterations;
}

template<class Data>
Rcpp::List ordinal_fit(const Data& data,const VectorXd& start,bool firth,
    double score_tol,double step_tol,int maxit,double max_step,int max_halves,
    double condition_tol,bool trace,bool observed=true) {
    if(!(score_tol>0) || !(step_tol>0) || maxit<1 || !(max_step>0) || max_halves<1 ||
       !(condition_tol>0) || !std::isfinite(score_tol+step_tol+max_step+condition_tol))
        Rcpp::stop("invalid ordinal optimizer controls");
    VectorXd parameters=start;
    int evaluations=0,fallbacks=0,iterations=0;
    // Bias reduction needs expected-information terms in every iteration.
    const int observed_iterations=!observed || firth || maxit<2 ? 0 :
        observed_newton(data,parameters,score_tol,step_tol,std::min(maxit-1,30),max_step,max_halves,trace,evaluations);
    const int expected_limit=maxit-observed_iterations;
    // Newton can reach a point, such as near separation, where the expected
    // information cannot be evaluated. The scoring loop then starts as if Newton
    // had not run, so it reports the same non-convergence diagnostics.
    const auto evaluate_state=[&](const VectorXd& b) { ++evaluations;return ordinal_evaluate(data,b,firth,condition_tol); };
    const auto first_state=[&]() {
        if(observed_iterations>0) {
            try { return evaluate_state(parameters); }
            catch(const OrdinalEvaluationError&) { parameters=start; }
        }
        return evaluate_state(parameters);
    };
    auto evaluate=[&](const VectorXd& b) { ++evaluations;return ordinal_evaluate(data,b,firth,condition_tol); };
    OrdinalState current=first_state();
    bool converged=false;std::string reason="iteration_limit",last_rejection;
    // A two-category cumulative-link model is a binomial GLM. Use its bounded
    // adjusted scoring iteration: a likelihood-score norm need not decrease
    // along this non-gradient direction, even on a path to a regular root.
    const bool binary_scoring=firth && data.q==1 && data.link!=0;
    MatrixXd history(expected_limit,4);
    for(int iter=1;iter<=expected_limit;++iter) {
        Rcpp::checkUserInterrupt();iterations=iter;
        double gap=std::numeric_limits<double>::infinity();
        for(int s=1;s<data.q;++s) gap=std::min(gap,parameters[s]-parameters[s-1]);
        history.row(iter-1)<<current.loglik,current.scaled_score,current.predictor_step,gap;
        if(trace) Rcpp::Rcout<<"iteration "<<iter<<": score "<<current.scaled_score<<
            ", predictor step "<<current.predictor_step<<"\n";
        if(current.scaled_score<score_tol && current.predictor_step<step_tol) {
            converged=true;reason="score_and_predictor_step";break;
        }
        const MatrixXd metric=current.inverse;
        const VectorXd score=current.score;
        const double merit0=.5*score.dot(metric*score);
        VectorXd accepted_parameters;OrdinalState accepted_state(data.d);
        auto search=[&](VectorXd direction) {
            if(!direction.allFinite()) return false;
            const double bound=binary_scoring ? std::min(1.0,max_step) : max_step;
            direction/=std::max(1.0,data.predictor_step(direction)/bound);
            double fraction=1;
            for(int s=1;s<data.q;++s) {
                const double change=direction[s]-direction[s-1];
                if(change<0) fraction=std::min(fraction,.9*(parameters[s]-parameters[s-1])/(-change));
            }
            for(int h=0;h<=max_halves;++h) {
                const VectorXd candidate=parameters+fraction*direction;
                try {
                    OrdinalState trial=evaluate(candidate);
                    const double merit=.5*trial.score.dot(metric*trial.score);
                    if(std::isfinite(merit) && (binary_scoring || merit<=(1-1e-4*fraction)*merit0)) {
                        accepted_parameters=candidate;accepted_state=std::move(trial);return true;
                    }
                } catch(const OrdinalEvaluationError& error) { last_rejection=error.what(); }
                fraction/=2;
            }
            return false;
        };
        bool accepted=search(current.step);
        if(!accepted) {
            ++fallbacks;MatrixXd jacobian(data.d,data.d);bool valid=true;
            for(int j=0;j<data.d;++j) {
                double h=std::cbrt(std::numeric_limits<double>::epsilon())*(1+std::abs(parameters[j]));
                if(j<data.q) {
                    if(j>0) h=std::min(h,.1*(parameters[j]-parameters[j-1]));
                    if(j+1<data.q) h=std::min(h,.1*(parameters[j+1]-parameters[j]));
                }
                VectorXd upper=parameters,lower=parameters;upper[j]+=h;lower[j]-=h;
                try {
                    const OrdinalState up=evaluate(upper),down=evaluate(lower);
                    jacobian.col(j)=(up.score-down.score)/(2*h);
                } catch(const OrdinalEvaluationError& error) { last_rejection=error.what();valid=false;break; }
            }
            if(valid && jacobian.allFinite()) {
                MatrixXd gram=jacobian.transpose()*metric*jacobian;
                gram=(gram+gram.transpose()).eval()/2;
                const VectorXd rhs=-jacobian.transpose()*metric*score;
                const double damping[]={0,1e-8,1e-6,1e-4,.01,1,100};
                for(double lambda:damping) {
                    MatrixXd system=gram;system.diagonal().array()+=lambda;
                    Eigen::LDLT<MatrixXd> factor(system);
                    if(factor.info()==Eigen::Success) accepted=search(factor.solve(rhs));
                    if(accepted) break;
                }
            }
        }
        if(!accepted) { reason="score_search_failed";break; }
        parameters=accepted_parameters;current=std::move(accepted_state);
    }
    if(current.scaled_score<score_tol && current.predictor_step<step_tol) {
        converged=true;reason="score_and_predictor_step";
    }
    Rcpp::List out=ordinal_state_list(current);
    out["coefficients"]=parameters;out["converged"]=converged;
    out["convergence.reason"]=reason;out["iterations"]=observed_iterations+iterations;
    out["observed.iterations"]=observed_iterations;
    out["evaluations"]=evaluations;out["fallbacks"]=fallbacks;
    out["last.rejected.evaluation"]=last_rejection;
    out["history"]=MatrixXd(history.topRows(iterations));
    return out;
}

}
#endif
