#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
// [[Rcpp::depends(RcppEigen)]]

namespace {
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
        Rcpp::NumericVector w,Rcpp::NumericVector off,int categories,int block_size,int link_code=0) :
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
        if(!(totals.sum()>0) || !std::isfinite(totals.sum()))
            Rcpp::stop("ordinal fitting requires a finite positive total frequency");
    }

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
                    logcdf[s]=R::pnorm(eta[s],0,1,true,true);
                    logsurvival[s]=R::pnorm(eta[s],0,1,false,true);
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

// Global expected information is formed before any bias adjustment. The
// adjustment is Kosmidis (2014), equations (7)-(9), in threshold/slope coordinates.
OrdinalState ordinal_evaluate(const OrdinalData& data,const VectorXd& parameters,
    bool firth,double condition_tol) {
    const int p=data.p,q=data.q,d=data.d;
    if(parameters.size()!=d || !parameters.allFinite())
        throw OrdinalEvaluationError("invalid ordinal parameters");
    const VectorXd alpha=parameters.head(q),beta=parameters.tail(p);
    for(int s=1;s<q;++s) if(!(alpha[s]>alpha[s-1]))
        throw OrdinalEvaluationError("ordinal thresholds are not strictly ordered");
    OrdinalState result(d);OrdinalRow row(alpha);
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

    if(firth) for(int first=0;first<data.n;first+=data.block) {
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
    result.score=result.ordinary+result.adjustment;
    if(!result.score.allFinite() || !result.inverse.allFinite())
        throw OrdinalEvaluationError("non-finite ordinal adjusted score or covariance");
    result.step.noalias()=result.inverse*result.score;
    result.scaled_score=(result.score.array().abs()/diagonal.array()).maxCoeff();
    if(!result.step.allFinite() || !std::isfinite(result.scaled_score))
        throw OrdinalEvaluationError("non-finite ordinal convergence diagnostic");
    result.predictor_step=data.predictor_step(result.step);
    if(!std::isfinite(result.predictor_step))
        throw OrdinalEvaluationError("non-finite ordinal predictor step");
    return result;
}

Rcpp::List ordinal_state_list(const OrdinalState& state) {
    return Rcpp::List::create(Rcpp::_["loglik"]=state.loglik,
        Rcpp::_["score"]=state.score,Rcpp::_["ordinary.score"]=state.ordinary,
        Rcpp::_["adjustment"]=state.adjustment,Rcpp::_["information"]=state.information,
        Rcpp::_["cov.unscaled"]=state.inverse,Rcpp::_["step"]=state.step,
        Rcpp::_["predictor.step"]=state.predictor_step,Rcpp::_["scaled.score"]=state.scaled_score,
        Rcpp::_["information.rcond"]=state.rcond,Rcpp::_["minimum.log.probability"]=state.min_logp);
}
}

// [[Rcpp::export]]
Rcpp::List ordinal_native_evaluate(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector coefficients,bool firth=false,
    int block_size=2048,double condition_tol=1e-14,int link=0) {
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    VectorXd beta=Rcpp::as<VectorXd>(coefficients);
    return ordinal_state_list(ordinal_evaluate(data,beta,firth,condition_tol));
}

// [[Rcpp::export]]
Rcpp::List ordinal_native_fit(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector start,bool firth=false,double score_tol=1e-8,
    double step_tol=1e-8,int maxit=200,int block_size=2048,double max_step=4,
    int max_halves=35,double condition_tol=1e-14,bool trace=false,int link=0) {
    if(!(score_tol>0) || !(step_tol>0) || maxit<1 || !(max_step>0) || max_halves<1 ||
       !(condition_tol>0) || !std::isfinite(score_tol+step_tol+max_step+condition_tol))
        Rcpp::stop("invalid ordinal optimizer controls");
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    VectorXd parameters=Rcpp::as<VectorXd>(start);
    int evaluations=0,fallbacks=0,iterations=0;
    auto evaluate=[&](const VectorXd& b) { ++evaluations;return ordinal_evaluate(data,b,firth,condition_tol); };
    OrdinalState current=evaluate(parameters);
    bool converged=false;std::string reason="iteration_limit",last_rejection;
    // A two-category cumulative-link model is a binomial GLM. Use its bounded
    // adjusted scoring iteration: a likelihood-score norm need not decrease
    // along this non-gradient direction, even on a path to a regular root.
    const bool binary_scoring=firth && data.q==1 && data.link!=0;
    MatrixXd history(maxit,4);
    for(int iter=1;iter<=maxit;++iter) {
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
    out["convergence.reason"]=reason;out["iterations"]=iterations;
    out["evaluations"]=evaluations;out["fallbacks"]=fallbacks;
    out["last.rejected.evaluation"]=last_rejection;
    out["history"]=MatrixXd(history.topRows(iterations));
    return out;
}
