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
        Rcpp::NumericVector off,int categories,int block_size,int link_code=0,bool require_positive=true) :
        OrdinalData(X,y,Y,w,off,categories,block_size,link_code,require_positive),
        z(Z.begin(),Z.nrow(),Z.ncol()),pn(Z.ncol()) {
        if(Z.nrow()!=n || !z.allFinite())
            Rcpp::stop("nonparallel ordinal design must be finite and match the common design rows");
        const double dimension=double(q)+p+double(q)*pn;
        if(dimension>std::numeric_limits<int>::max()) Rcpp::stop("ordinal parameter dimension is too large");
        d=static_cast<int>(dimension);
    }
    template<class Function> void visit(Function&& function) const { function(*this); }
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
    double previous_gap=0;
    for(int s=1;s<data.q;++s) {
        const double gap=data.gap(i,b,s);
        if(!(gap>0) || !std::isfinite(gap))
            throw OrdinalEvaluationError("nonparallel ordinal cumulative predictors are not strictly ordered on a positive-frequency row");
        state.minimum_gap=std::min(state.minimum_gap,gap);
        if(data.link==0) {
            row.loggap[s]=ordinal_log1mexp(-gap);row.inverse_gap[s]=1/std::expm1(gap);
            // The preceding adjacent gap has already been evaluated with the
            // same compensated arithmetic. Preserve the original sum order.
            if(s>1) row.neighbor_gap[s-1]=-std::expm1(-(previous_gap+gap));
            previous_gap=gap;
        }
    }
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

template<class Data>
inline PartialOrdinalState partial_evaluate(const Data& data,
    const VectorXd& b,bool firth,double condition_tol) {
    if(b.size()!=data.d || !b.allFinite()) throw OrdinalEvaluationError("invalid nonparallel ordinal parameters");
    PartialOrdinalState state(data.d);
    data.visit([&](const PartialOrdinalData& block) {partial_add_information(block,b,state);});
    ordinal_factor_information(state,data.d-data.q,data.q,condition_tol);
    if(firth) data.visit([&](const PartialOrdinalData& block) {partial_add_adjustment(block,b,state);});
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

// Observed-information Newton iterations for threshold-specific slopes. Each row
// touches only the thresholds bounding its observed categories, and products use
// vectorized arithmetic. The iterate is only a start: partial_fit() certifies
// convergence with the expected information and compensated predictors.
struct PartialObservedState {
    MatrixXd information;
    VectorXd score;
    double loglik=0,minimum_gap=std::numeric_limits<double>::infinity();
    explicit PartialObservedState(int d=0) : information(MatrixXd::Zero(d,d)),score(VectorXd::Zero(d)) {}
};

inline void partial_observed_add(const PartialOrdinalData& data,const VectorXd& b,PartialObservedState& out) {
    const int q=data.q,p=data.p,pn=data.pn;
    const VectorXd alpha=b.head(q),beta=b.segment(q,p);
    const int width=2+p+2*pn;
    std::vector<int> index(width);
    VectorXd upper(width),lower(width);
    VectorXd location,slope;
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        const int length=std::min(data.block,data.n-first);
        const auto xx=data.x.middleRows(first,length);
        const auto zz=data.z.middleRows(first,length);
        location.resize(length);
        if(p) location.noalias()=xx*beta; else location.setZero();
        // Ordering must hold at every positive-frequency row, not only at observed categories.
        for(int s=1;s<q;++s) {
            const VectorXd rowgap=VectorXd::Constant(length,alpha[s]-alpha[s-1])-
                zz*(b.segment(q+p+s*pn,pn)-b.segment(q+p+(s-1)*pn,pn));
            for(int r=0;r<length;++r) if(data.totals[first+r]>0) {
                if(!(rowgap[r]>0)) throw OrdinalEvaluationError("nonparallel ordinal cumulative predictors are not strictly ordered");
                out.minimum_gap=std::min(out.minimum_gap,rowgap[r]);
            }
        }
        for(int r=0;r<length;++r) {
            const int i=first+r;
            if(data.totals[i]==0) continue;
            const double linear=location[r]+data.offset[i];
            const auto add=[&](int k,double y) {
                double eu=0,el=0,gap=0;
                if(k<q) eu=alpha[k]-linear-zz.row(r).dot(b.segment(q+p+k*pn,pn));
                if(k>0) el=alpha[k-1]-linear-zz.row(r).dot(b.segment(q+p+(k-1)*pn,pn));
                if(k>0 && k<q) gap=alpha[k]-alpha[k-1]-zz.row(r).dot(b.segment(q+p+k*pn,pn)-b.segment(q+p+(k-1)*pn,pn));
                const double one_minus=gap>0 ? -std::expm1(-gap) : 0;
                const ObservedTerms t=data.link==0 ?
                    observed_logit(k,q,eu,el,gap>0 ? 1/std::expm1(gap) : 0,gap>0 ? ordinal_log1mexp(-gap) : 0,one_minus) :
                    observed_general(k,q,data.link,eu,el,gap,one_minus);
                out.loglik+=y*t.logp;
                // Gradients of the upper and lower predictors in the coefficients.
                int used=0;
                const auto entry=[&](int at,double u,double l) { index[used]=at;upper[used]=u;lower[used]=l;++used; };
                if(k<q) entry(k,1,0);
                if(k>0) entry(k-1,0,1);
                for(int j=0;j<p;++j) entry(q+j,k<q ? -xx(r,j) : 0,k>0 ? -xx(r,j) : 0);
                for(int j=0;j<pn;++j) {
                    if(k<q) entry(q+p+k*pn+j,-zz(r,j),0);
                    if(k>0) entry(q+p+(k-1)*pn+j,0,-zz(r,j));
                }
                for(int a=0;a<used;++a) {
                    out.score[index[a]]+=y*(t.left*upper[a]-t.right*lower[a]);
                    for(int c=0;c<used;++c) {
                        const double value=t.upper*upper[a]*upper[c]+t.lower*lower[a]*lower[c]+
                            t.cross*(upper[a]*lower[c]+lower[a]*upper[c]);
                        out.information(index[a],index[c])+=y*value;
                    }
                }
            };
            if(data.grouped) {
                for(int k=0;k<data.K;++k) {
                    const double y=data.weights[i]*data.counts(i,k);
                    if(y!=0) add(k,y);
                }
            } else add(data.codes[i]-1,data.totals[i]);
        }
    }
}

template<class Data>
PartialObservedState partial_observed_evaluate(const Data& data,const VectorXd& b) {
    if(b.size()!=data.d || !b.allFinite()) throw OrdinalEvaluationError("invalid nonparallel ordinal parameters");
    PartialObservedState out(data.d);
    data.visit([&](const PartialOrdinalData& block) { partial_observed_add(block,b,out); });
    if(!std::isfinite(out.loglik) || !out.score.allFinite() || !out.information.allFinite())
        throw OrdinalEvaluationError("non-finite observed nonparallel ordinal terms");
    return out;
}

// The largest predictor change and the largest step that keeps every row ordered.
template<class Data>
void partial_step_limits(const Data& data,const VectorXd& b,const VectorXd& direction,double& change,double& fraction) {
    change=0;fraction=1;
    data.visit([&](const PartialOrdinalData& block) {
        const int q=block.q,p=block.p,pn=block.pn;
        VectorXd location,value,previous;
        for(int first=0;first<block.n;first+=block.block) {
            const int length=std::min(block.block,block.n-first);
            const auto xx=block.x.middleRows(first,length);
            const auto zz=block.z.middleRows(first,length);
            if(p) location.noalias()=xx*direction.segment(q,p); else location=VectorXd::Zero(length);
            for(int s=0;s<q;++s) {
                value=VectorXd::Constant(length,direction[s])-location-zz*direction.segment(q+p+s*pn,pn);
                if(s>0) {
                    const VectorXd current=VectorXd::Constant(length,b[s]-b[s-1])-
                        zz*(b.segment(q+p+s*pn,pn)-b.segment(q+p+(s-1)*pn,pn));
                    for(int r=0;r<length;++r) if(block.totals[first+r]>0) {
                        const double difference=value[r]-previous[r];
                        if(difference<0) fraction=std::min(fraction,.9*current[r]/(-difference));
                    }
                }
                for(int r=0;r<length;++r) if(block.totals[first+r]>0) change=std::max(change,std::abs(value[r]));
                previous=value;
            }
        }
    });
}

template<class Data>
int partial_observed_newton(const Data& data,VectorXd& b,double score_tol,double step_tol,int maxit,
    double max_step,int max_halves,bool trace,int& evaluations) {
    int iterations=0;
    try {
        PartialObservedState current=partial_observed_evaluate(data,b);
        ++evaluations;
        for(;iterations<maxit;++iterations) {
            Rcpp::checkUserInterrupt();
            // Near the ordering boundary the certifying iterations take over.
            if(current.minimum_gap<1e-8) return iterations;
            const VectorXd diagonal=current.information.diagonal();
            if((diagonal.array()<=0).any()) return iterations;
            const VectorXd inverse_scale=diagonal.cwiseSqrt().cwiseInverse();
            const MatrixXd correlation=inverse_scale.asDiagonal()*current.information*inverse_scale.asDiagonal();
            const Eigen::LLT<MatrixXd> factor(correlation);
            if(factor.info()!=Eigen::Success) return iterations;
            VectorXd direction=inverse_scale.cwiseProduct(factor.solve(inverse_scale.cwiseProduct(current.score)));
            if(!direction.allFinite()) return iterations;
            const double scaled=current.score.cwiseProduct(inverse_scale).cwiseAbs().maxCoeff();
            double change=0,fraction=1;
            partial_step_limits(data,b,direction,change,fraction);
            if(trace) Rcpp::Rcout<<"observed iteration "<<iterations<<": score "<<scaled<<", predictor step "<<change<<"\n";
            if(scaled<score_tol && change<step_tol) {
                if(fraction==1) b+=direction;
                return iterations+1;
            }
            if(change>max_step) { direction*=max_step/change;fraction=std::min(1.0,fraction*change/max_step); }
            if(!(fraction>0)) return iterations;
            const double ascent=current.score.dot(direction);
            const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(current.loglik));
            bool accepted=false;
            VectorXd candidate(b.size());
            PartialObservedState trial;
            for(int h=0;h<=max_halves && !accepted;++h) {
                candidate=b+fraction*direction;
                try {
                    trial=partial_observed_evaluate(data,candidate);
                    ++evaluations;
                    if(ascent<1024*rounding || trial.loglik>=current.loglik+1e-4*fraction*ascent-rounding) accepted=true;
                    else fraction/=2;
                } catch(const OrdinalEvaluationError&) { ++evaluations;fraction/=2; }
            }
            if(!accepted) return iterations;
            b=candidate;current=std::move(trial);
        }
    } catch(const OrdinalEvaluationError&) {}
    return iterations;
}

template<class Data>
inline Rcpp::List partial_fit(const Data& data,const VectorXd& start,
    bool firth,double score_tol,double step_tol,int maxit,double max_step,int max_halves,
    double condition_tol,bool trace,bool observed=true) {
    if(!(score_tol>0) || !(step_tol>0) || maxit<1 || !(max_step>0) || max_halves<1 ||
       !(condition_tol>0) || !std::isfinite(score_tol+step_tol+max_step+condition_tol))
        Rcpp::stop("invalid nonparallel ordinal optimizer controls");
    VectorXd b=start;int evaluations=0,fallbacks=0,iterations=0;
    // Bias reduction needs expected-information terms in every iteration.
    const int observed_iterations=!observed || firth || maxit<2 ? 0 :
        partial_observed_newton(data,b,score_tol,step_tol,std::min(maxit-1,30),max_step,max_halves,trace,evaluations);
    const int expected_limit=maxit-observed_iterations;
    auto evaluate=[&](const VectorXd& candidate) {++evaluations;return partial_evaluate(data,candidate,firth,condition_tol);};
    // The certifying iterations restart if the Newton endpoint cannot be evaluated.
    const auto first_state=[&]() {
        if(observed_iterations>0) {
            try { return evaluate(b); }
            catch(const OrdinalEvaluationError&) { b=start; }
        }
        return evaluate(b);
    };
    PartialOrdinalState current=first_state();
    bool converged=false;std::string reason="iteration_limit",last_rejection;
    const bool binary_scoring=firth && data.q==1 && data.link!=0;
    // Maximum likelihood can lie where cumulative probabilities touch at some row.
    // Every step then stops at the ordering constraint, and the search would crawl
    // along it without reaching an interior solution, so it ends here instead.
    int boundary_stall=0;
    bool order_limited=false;
    const int boundary_limit=20;
    const double boundary_gap=1e-8;
    MatrixXd history(expected_limit,4);
    for(int iter=1;iter<=expected_limit;++iter) {
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
        auto search=[&](VectorXd direction,double known_predictor_step=-1.0) {
            if(!direction.allFinite()) return false;
            const double bound=firth ? std::min(1.0,max_step) : max_step;
            // The current state already checked its own scoring direction.
            // Reuse that identical result, including for a repeatable source;
            // fallback directions still require their own predictor scan.
            const double predictor_step=known_predictor_step>=0 ? known_predictor_step :
                data.predictor_step(direction);
            direction/=std::max(1.0,predictor_step/bound);
            const double ascent=score.dot(direction);
            if(!firth && !(ascent>0)) return false;
            double fraction=data.feasible_fraction(b,direction);
            if(!(fraction>0) || !std::isfinite(fraction)) return false;
            order_limited=fraction<1;
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
        bool accepted=search(current.step,current.predictor_step);
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
        if(!accepted) {
            reason=firth ? "score_search_failed" : (current.minimum_gap<boundary_gap ? "ordering_boundary" : "likelihood_search_failed");
            break;
        }
        b=accepted_b;current=std::move(accepted_state);
        boundary_stall=!firth && order_limited && current.minimum_gap<boundary_gap ? boundary_stall+1 : 0;
        if(boundary_stall>=boundary_limit) {reason="ordering_boundary";break;}
    }
    if(current.scaled_score<score_tol && current.predictor_step<step_tol) {
        converged=true;reason="score_and_predictor_step";
    }
    Rcpp::List out=partial_state_list(current);
    out["coefficients"]=b;out["converged"]=converged;out["convergence.reason"]=reason;
    out["iterations"]=observed_iterations+iterations;out["observed.iterations"]=observed_iterations;out["evaluations"]=evaluations;out["fallbacks"]=fallbacks;
    out["last.rejected.evaluation"]=last_rejection;out["history"]=MatrixXd(history.topRows(iterations));
    return out;
}

}
#endif
