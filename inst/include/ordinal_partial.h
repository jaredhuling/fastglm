#ifndef FASTGLM_ORDINAL_PARTIAL_H
#define FASTGLM_ORDINAL_PARTIAL_H
#include "ordinal.h"
#include <vector>

namespace fglm_ordinal_partial {
using namespace fglm_ordinal;

// Coordinates are thresholds, common slopes, then one nonparallel slope block
// per threshold. Only row-specific cumulative predictors must be ordered.
struct PartialOrdinalData : OrdinalData {
    Eigen::Map<const MatrixXd> z;
    int pn;
    PartialOrdinalData(Rcpp::NumericMatrix X,Rcpp::NumericMatrix Z,
        Rcpp::IntegerVector y,Rcpp::NumericMatrix Y,Rcpp::NumericVector w,
        Rcpp::NumericVector off,int categories,int block_size,int link_code=0) :
        OrdinalData(X,y,Y,w,off,categories,block_size,link_code),
        z(Z.begin(),Z.nrow(),Z.ncol()),pn(Z.ncol()) {
        if(Z.nrow()!=n || !z.allFinite())
            Rcpp::stop("nonparallel ordinal design must be finite and match the common design rows");
        const double dimension=double(q)+p+double(q)*pn;
        if(dimension>std::numeric_limits<int>::max()) Rcpp::stop("ordinal parameter dimension is too large");
        d=static_cast<int>(dimension);
    }
    void predictors(int i,const VectorXd& b,VectorXd& eta,bool with_offset=true) const {
        // Accumulate the complete predictor before rounding. Separately
        // rounded common/nonparallel locations can cancel after losing alpha.
        for(int s=0;s<q;++s) {
            double sum=b[s],error=0;
            auto add=[&](double value,double coefficient) {
                const double product=value*coefficient,next=sum+product;
                error+=std::abs(sum)>=std::abs(product) ? (sum-next)+product : (product-next)+sum;
                error+=std::fma(value,coefficient,-product);sum=next;
            };
            if(with_offset) add(offset[i],-1);
            for(int j=0;j<p;++j) add(x(i,j),-b[q+j]);
            for(int j=0;j<pn;++j) add(z(i,j),-b[q+p+s*pn+j]);
            eta[s]=sum+error;
        }
    }
    double gap(int i,const VectorXd& b,int s) const {
        // Common slopes and offsets cancel exactly. Subtracting two complete
        // predictors can erase a valid gap after a large common location shift.
        double sum=b[s],error=0;
        auto add=[&](double value,double coefficient) {
            const double product=value*coefficient,next=sum+product;
            error+=std::abs(sum)>=std::abs(product) ? (sum-next)+product : (product-next)+sum;
            error+=std::fma(value,coefficient,-product);sum=next;
        };
        add(b[s-1],-1);
        for(int j=0;j<pn;++j) {
            // A rounded difference of coefficients can itself lose bits that
            // matter when multiplied by a large predictor.
            add(z(i,j),-b[q+p+s*pn+j]);
            add(z(i,j),b[q+p+(s-1)*pn+j]);
        }
        return sum+error;
    }
    double predictor_step(const VectorXd& direction) const {
        VectorXd delta(q);double largest=0;
        for(int first=0;first<n;first+=block) {
            Rcpp::checkUserInterrupt();
            for(int i=first;i<std::min(n,first+block);++i) if(totals[i]>0) {
                predictors(i,direction,delta,false);
                if(!delta.allFinite()) return std::numeric_limits<double>::infinity();
                largest=std::max(largest,delta.cwiseAbs().maxCoeff());
            }
        }
        return largest;
    }
    double feasible_fraction(const VectorXd& b,const VectorXd& direction) const {
        if(q==1) return 1;
        double fraction=1;
        for(int first=0;first<n;first+=block) {
            Rcpp::checkUserInterrupt();
            for(int i=first;i<std::min(n,first+block);++i) if(totals[i]>0) {
                for(int s=1;s<q;++s) {
                    const double difference=gap(i,direction,s);
                    if(difference<0)
                        fraction=std::min(fraction,.9*gap(i,b,s)/(-difference));
                }
            }
        }
        return fraction;
    }
};

struct PartialOrdinalState : OrdinalState {
    double minimum_gap=std::numeric_limits<double>::infinity();
    explicit PartialOrdinalState(int d=0) : OrdinalState(d) {}
};

// Reuse the common-link tail formulas, rebuilding the gap terms because gaps
// now depend on the predictor row. No cumulative probabilities are clipped.
inline void partial_row(const PartialOrdinalData& data,const VectorXd& b,int i,
    VectorXd& eta,OrdinalRow& row,PartialOrdinalState& state) {
    data.predictors(i,b,eta);
    if(!eta.allFinite()) throw OrdinalEvaluationError("non-finite ordinal cumulative predictors");
    for(int s=1;s<data.q;++s) {
        const double gap=data.gap(i,b,s);
        if(!(gap>0) || !std::isfinite(gap))
            throw OrdinalEvaluationError("nonparallel ordinal cumulative predictors are not strictly ordered on a positive-frequency row");
        state.minimum_gap=std::min(state.minimum_gap,gap);
        row.loggap[s]=ordinal_log1mexp(-gap);row.inverse_gap[s]=1/std::expm1(gap);
    }
    for(int s=1;s+1<data.q;++s)
        row.neighbor_gap[s]=-std::expm1(-(data.gap(i,b,s)+data.gap(i,b,s+1)));
    row.evaluate(eta,0,data.link);
    state.min_logp=std::min(state.min_logp,row.logp.minCoeff());
}

inline void partial_add_information(const PartialOrdinalData& data,const VectorXd& b,
    PartialOrdinalState& state) {
    const int q=data.q,p=data.p,pn=data.pn;
    VectorXd eta=VectorXd::LinSpaced(q,0,double(q-1)),threshold_score(q),point(pn+1);
    OrdinalRow row(eta);
    std::vector<int> index(pn+1),next(pn+1);
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        for(int i=first;i<std::min(data.n,first+data.block);++i) {
            const double m=data.totals[i];if(m==0) continue;
            partial_row(data,b,i,eta,row,state);
            threshold_score.setZero();double common_score=0;
            if(data.grouped) {
                for(int k=0;k<data.K;++k) {
                    const double y=data.weights[i]*data.counts(i,k);
                    if(y==0) continue;
                    state.loglik+=y*row.logp[k];common_score-=y*row.location_score(k);
                    if(k<q) threshold_score[k]+=y*row.left[k];
                    if(k>0) threshold_score[k-1]-=y*row.right[k-1];
                }
            } else {
                const int k=data.codes[i]-1;
                state.loglik+=m*row.logp[k];common_score=-m*row.location_score(k);
                if(k<q) threshold_score[k]+=m*row.left[k];
                if(k>0) threshold_score[k-1]-=m*row.right[k-1];
            }
            state.ordinary.head(q)+=threshold_score;
            for(int j=0;j<p;++j) state.ordinary[q+j]+=data.x(i,j)*common_score;
            point[0]=1;
            for(int j=0;j<pn;++j) point[j+1]=-data.z(i,j);
            double common_weight=0;
            for(int s=0;s<q;++s) {
                index[0]=s;next[0]=s+1;
                for(int j=0;j<pn;++j) {
                    index[j+1]=q+p+s*pn+j;next[j+1]=q+p+(s+1)*pn+j;
                    state.ordinary[index[j+1]]-=data.z(i,j)*threshold_score[s];
                }
                const double diagonal=m*row.g[s]*(row.left[s]+row.right[s]);
                for(int a=0;a<=pn;++a) for(int c=a;c<=pn;++c)
                    state.information(index[a],index[c])+=diagonal*point[a]*point[c];
                if(s+1<q) {
                    const double off=-m*row.g[s+1]*row.right[s];
                    for(int a=0;a<=pn;++a) for(int c=0;c<=pn;++c) {
                        const int lo=std::min(index[a],next[c]),hi=std::max(index[a],next[c]);
                        state.information(lo,hi)+=off*point[a]*point[c];
                    }
                }
                // These stable row sums avoid subtracting large adjacent
                // derivative ratios when common-slope contributions cancel.
                const double rowsum=m*row.g[s]*row.neighbor[s];common_weight+=rowsum;
                for(int j=0;j<p;++j) {
                    state.information(s,q+j)-=rowsum*data.x(i,j);
                    for(int k=0;k<pn;++k)
                        state.information(q+j,q+p+s*pn+k)+=rowsum*data.x(i,j)*data.z(i,k);
                }
            }
            for(int j=0;j<p;++j) for(int k=j;k<p;++k)
                state.information(q+j,q+k)+=common_weight*data.x(i,j)*data.x(i,k);
        }
    }
    state.information=MatrixXd(state.information.selfadjointView<Eigen::Upper>());
}

inline void partial_add_adjustment(const PartialOrdinalData& data,const VectorXd& b,
    PartialOrdinalState& state) {
    const int q=data.q,p=data.p,pn=data.pn,width=1+p+pn;
    VectorXd eta=VectorXd::LinSpaced(q,0,double(q-1)),values(width),correction(q);
    OrdinalRow row(eta);std::vector<int> indices(width);
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        for(int i=first;i<std::min(data.n,first+data.block);++i) {
            const double m=data.totals[i];if(m==0) continue;
            partial_row(data,b,i,eta,row,state);values[0]=1;
            for(int j=0;j<p;++j) {indices[j+1]=q+j;values[j+1]=-data.x(i,j);}
            for(int j=0;j<pn;++j) values[1+p+j]=-data.z(i,j);
            for(int s=0;s<q;++s) {
                indices[0]=s;
                for(int j=0;j<pn;++j) indices[1+p+j]=q+p+s*pn+j;
                double variance=0;
                for(int a=0;a<width;++a) {
                    variance+=values[a]*values[a]*state.inverse(indices[a],indices[a]);
                    for(int c=a+1;c<width;++c)
                        variance+=2*values[a]*values[c]*state.inverse(indices[a],indices[c]);
                }
                if(!(variance>0) || !std::isfinite(variance))
                    throw OrdinalEvaluationError("nonpositive or non-finite cumulative-predictor variance");
                const double curvature=data.link==0 ? row.g[s]*(1-2*row.cdf[s]) : row.gprime[s];
                correction[s]=.5*m*curvature*variance;
            }
            double common_adjustment=0;
            for(int s=0;s<q;++s) {
                const double adjusted=(correction[s]-(s ? correction[s-1] : 0))*row.left[s]-
                    ((s+1<q ? correction[s+1] : 0)-correction[s])*row.right[s];
                state.adjustment[s]+=adjusted;
                for(int j=0;j<pn;++j) state.adjustment[q+p+s*pn+j]-=data.z(i,j)*adjusted;
                common_adjustment-=correction[s]*row.neighbor[s];
            }
            for(int j=0;j<p;++j) state.adjustment[q+j]+=data.x(i,j)*common_adjustment;
        }
    }
}

inline PartialOrdinalState partial_evaluate(const PartialOrdinalData& data,
    const VectorXd& b,bool firth,double condition_tol) {
    if(b.size()!=data.d || !b.allFinite()) throw OrdinalEvaluationError("invalid nonparallel ordinal parameters");
    PartialOrdinalState state(data.d);
    partial_add_information(data,b,state);
    ordinal_factor_information(state,data.d-data.q,data.q,condition_tol);
    if(firth) partial_add_adjustment(data,b,state);
    state.score=state.ordinary+state.adjustment;
    if(!state.score.allFinite() || !state.inverse.allFinite())
        throw OrdinalEvaluationError("non-finite nonparallel ordinal score or covariance");
    state.step.noalias()=state.inverse*state.score;
    state.scaled_score=(state.score.array().abs()/state.information.diagonal().array().sqrt()).maxCoeff();
    state.predictor_step=data.predictor_step(state.step);
    if(!state.step.allFinite() || !std::isfinite(state.scaled_score) || !std::isfinite(state.predictor_step))
        throw OrdinalEvaluationError("non-finite nonparallel ordinal convergence diagnostic");
    return state;
}

inline Rcpp::List partial_state_list(const PartialOrdinalState& state) {
    Rcpp::List out=ordinal_state_list(state);out["minimum.gap"]=state.minimum_gap;return out;
}

inline Rcpp::List partial_fit(const PartialOrdinalData& data,const VectorXd& start,
    bool firth,double score_tol,double step_tol,int maxit,double max_step,int max_halves,
    double condition_tol,bool trace) {
    if(!(score_tol>0) || !(step_tol>0) || maxit<1 || !(max_step>0) || max_halves<1 ||
       !(condition_tol>0) || !std::isfinite(score_tol+step_tol+max_step+condition_tol))
        Rcpp::stop("invalid nonparallel ordinal optimizer controls");
    VectorXd b=start;int evaluations=0,fallbacks=0,iterations=0;
    auto evaluate=[&](const VectorXd& candidate) {++evaluations;return partial_evaluate(data,candidate,firth,condition_tol);};
    PartialOrdinalState current=evaluate(b);
    bool converged=false;std::string reason="iteration_limit",last_rejection;
    const bool binary_scoring=firth && data.q==1 && data.link!=0;
    MatrixXd history(maxit,4);
    for(int iter=1;iter<=maxit;++iter) {
        Rcpp::checkUserInterrupt();iterations=iter;
        history.row(iter-1)<<current.loglik,current.scaled_score,current.predictor_step,current.minimum_gap;
        if(trace) Rcpp::Rcout<<"iteration "<<iter<<": score "<<current.scaled_score<<
            ", predictor step "<<current.predictor_step<<", minimum gap "<<current.minimum_gap<<"\n";
        if(current.scaled_score<score_tol && current.predictor_step<step_tol) {
            converged=true;reason="score_and_predictor_step";break;
        }
        const MatrixXd metric=current.inverse;const VectorXd score=current.score;
        const double merit0=.5*score.dot(metric*score);
        VectorXd accepted_b;PartialOrdinalState accepted_state(data.d);
        auto search=[&](VectorXd direction) {
            if(!direction.allFinite()) return false;
            const double bound=firth ? std::min(1.0,max_step) : max_step;
            direction/=std::max(1.0,data.predictor_step(direction)/bound);
            const double ascent=score.dot(direction);
            if(!firth && !(ascent>0)) return false;
            double fraction=data.feasible_fraction(b,direction);
            if(!(fraction>0) || !std::isfinite(fraction)) return false;
            for(int h=0;h<=max_halves;++h) {
                const VectorXd candidate=b+fraction*direction;
                try {
                    PartialOrdinalState trial=evaluate(candidate);
                    const double merit=.5*trial.score.dot(metric*trial.score);
                    const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(current.loglik));
                    const bool accepted=firth ? (std::isfinite(merit) &&
                        (binary_scoring || merit<=(1-1e-4*fraction)*merit0)) :
                        trial.loglik-current.loglik>=.1*fraction*ascent-rounding;
                    if(accepted) {accepted_b=candidate;accepted_state=std::move(trial);return true;}
                    last_rejection=firth ? "adjusted-score norm did not decrease sufficiently" :
                        "likelihood did not increase sufficiently";
                } catch(const OrdinalEvaluationError& error) {last_rejection=error.what();}
                fraction/=2;
            }
            return false;
        };
        bool accepted=search(current.step);
        if(!accepted) {
            ++fallbacks;MatrixXd jacobian(data.d,data.d);bool valid=true;
            for(int j=0;j<data.d;++j) {
                double h=std::cbrt(std::numeric_limits<double>::epsilon())*(1+std::abs(b[j]));
                VectorXd direction=VectorXd::Zero(data.d);direction[j]=h;
                h*=std::min(data.feasible_fraction(b,direction),data.feasible_fraction(b,-direction));
                bool differentiated=false;
                for(int attempt=0;attempt<12;++attempt) {
                    if(!(h>0) || b[j]+h==b[j] || b[j]-h==b[j]) break;
                    VectorXd upper=b,lower=b;upper[j]+=h;lower[j]-=h;
                    try {
                        const PartialOrdinalState up=evaluate(upper),down=evaluate(lower);
                        jacobian.col(j)=(up.score-down.score)/(2*h);differentiated=true;break;
                    } catch(const OrdinalEvaluationError& error) {last_rejection=error.what();h/=2;}
                }
                if(!differentiated) {valid=false;break;}
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
        if(!accepted) {reason=firth ? "score_search_failed" : "likelihood_search_failed";break;}
        b=accepted_b;current=std::move(accepted_state);
    }
    if(current.scaled_score<score_tol && current.predictor_step<step_tol) {
        converged=true;reason="score_and_predictor_step";
    }
    Rcpp::List out=partial_state_list(current);
    out["coefficients"]=b;out["converged"]=converged;out["convergence.reason"]=reason;
    out["iterations"]=iterations;out["evaluations"]=evaluations;out["fallbacks"]=fallbacks;
    out["last.rejected.evaluation"]=last_rejection;out["history"]=MatrixXd(history.topRows(iterations));
    return out;
}

}
#endif
