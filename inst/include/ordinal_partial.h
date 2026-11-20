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
            if(with_offset) add(off(i),-1);
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
    // The largest change in a cumulative predictor, and the largest step
    // fraction that keeps every row ordered. Blocks run in parallel; a
    // maximum or minimum does not depend on the order of its terms (NaN terms
    // are skipped either way), so the results are the serial ones.
    double predictor_step(const VectorXd& direction) const {
        const Eigen::Index nb=(n+block-1)/block;
        std::vector<double> part(static_cast<std::size_t>(nb),0.0);
        fglm_par::parallel_blocks(n,block,[&](Eigen::Index first,Eigen::Index length) {
            VectorXd delta(q);double largest=0;
            for(Eigen::Index i=first;i<first+length;++i) if(totals[i]>0) {
                predictors(static_cast<int>(i),direction,delta,false);
                if(!delta.allFinite()) { largest=std::numeric_limits<double>::infinity();break; }
                largest=std::max(largest,delta.cwiseAbs().maxCoeff());
            }
            part[static_cast<std::size_t>(first/block)]=largest;
        });
        Rcpp::checkUserInterrupt();
        double largest=0;
        for(double v:part) largest=std::max(largest,v);
        return largest;
    }
    double feasible_fraction(const VectorXd& b,const VectorXd& direction) const {
        if(q==1) return 1;
        const Eigen::Index nb=(n+block-1)/block;
        std::vector<double> part(static_cast<std::size_t>(nb),1.0);
        fglm_par::parallel_blocks(n,block,[&](Eigen::Index first,Eigen::Index length) {
            double fraction=1;
            for(Eigen::Index i=first;i<first+length;++i) if(totals[i]>0) {
                for(int s=1;s<q;++s) {
                    const double difference=gap(static_cast<int>(i),direction,s);
                    if(difference<0)
                        fraction=std::min(fraction,.9*gap(static_cast<int>(i),b,s)/(-difference));
                }
            }
            part[static_cast<std::size_t>(first/block)]=fraction;
        });
        Rcpp::checkUserInterrupt();
        double fraction=1;
        for(double v:part) fraction=std::min(fraction,v);
        return fraction;
    }
};

struct PartialOrdinalState : OrdinalState {
    double minimum_gap=std::numeric_limits<double>::infinity();
    explicit PartialOrdinalState(int d=0) : OrdinalState(d) {}
};

// Reuse the common-link tail formulas, rebuilding the gap terms because gaps
// now depend on the predictor row. No cumulative probabilities are clipped.
// pre and rm, when given, are the row's staged cumulative predictors and
// probit terms (see partial_stage_expected); the result is the same as when
// they are computed here.
inline void partial_row(const PartialOrdinalData& data,const VectorXd& b,int i,
    VectorXd& eta,OrdinalRow& row,double& min_logp,double& min_gap,
    const double* pre=nullptr,const double* rm=nullptr) {
    if(pre) for(int s=0;s<data.q;++s) eta[s]=pre[s];
    else data.predictors(i,b,eta);
    if(!eta.allFinite()) throw OrdinalEvaluationError("non-finite ordinal cumulative predictors");
    double previous_gap=0;
    for(int s=1;s<data.q;++s) {
        const double gap=data.gap(i,b,s);
        if(!(gap>0) || !std::isfinite(gap))
            throw OrdinalEvaluationError("nonparallel ordinal cumulative predictors are not strictly ordered on a positive-frequency row");
        min_gap=std::min(min_gap,gap);
        if(data.link==0) {
            row.loggap[s]=ordinal_log1mexp(-gap);row.inverse_gap[s]=1/std::expm1(gap);
            // The preceding adjacent gap has already been evaluated with the
            // same compensated arithmetic. Preserve the original sum order.
            if(s>1) row.neighbor_gap[s-1]=-std::expm1(-(previous_gap+gap));
            previous_gap=gap;
        }
    }
    row.evaluate(eta,0,data.link,true,rm);
    min_logp=std::min(min_logp,row.logp.minCoeff());
}

// Stage of the nonparallel expected-information and adjustment passes for
// the probit link: for the blocks [g, g + count) of rows rows each (slot j =
// block - g), the cumulative predictors of every positive-frequency row
// (computed in parallel) into ws.eta and, on the calling thread, the probit
// terms at each of them into ws.rmath. partial_row evaluates the row at
// the predictors minus a zero location, so the terms are taken there.
inline void partial_stage_expected(const PartialOrdinalData& data,const VectorXd& b,
    Eigen::Index g,Eigen::Index count,int rows,OrdinalWorkspace& ws) {
    const int q=data.q;
    const Eigen::Index r0=g*rows,r1=std::min<Eigen::Index>(data.n,(g+count)*rows),m=r1-r0;
    const Eigen::Index cap=Eigen::Index(fglm_par::kLanes)*rows;
    if(ws.eta.size()<cap*q) ws.eta.resize(cap*q);
    if(ws.rmath.size()<cap*3*q) ws.rmath.resize(cap*3*q);
    fglm_par::parallel_blocks(m,rows,[&](Eigen::Index r,Eigen::Index k) {
        VectorXd eta(q);
        for(Eigen::Index j=r;j<r+k;++j) {
            const int i=static_cast<int>(r0+j);
            if(!(data.totals[i]>0)) continue;
            data.predictors(i,b,eta);
            for(int s=0;s<q;++s) ws.eta[j*q+s]=eta[s];
        }
    });
    for(Eigen::Index j=0;j<m;++j) if(data.totals[r0+j]>0)
        for(int s=0;s<q;++s) ordinal_probit_terms(ws.eta[j*q+s]-0.0,&ws.rmath[(j*q+s)*3]);
}

inline void partial_size_work(OrdinalWork& w,int q,int width) {
    if(w.predictors.size()!=q) w.predictors.resize(q);
    if(w.threshold_score.size()!=q) w.threshold_score.resize(q);
    if(w.c.size()!=q) w.c.resize(q);
    if(w.values.size()!=width) w.values.resize(width);
    if(static_cast<int>(w.index.size())!=width) w.index.assign(width,0);
    if(static_cast<int>(w.next.size())!=width) w.next.assign(width,0);
}

// One row of the expected information: adds its log likelihood, ordinary
// score, and the upper triangle of its information to I.
template<class Matrix,class Vector>
inline void partial_information_row(const PartialOrdinalData& data,const VectorXd& b,int i,OrdinalRow& row,
    OrdinalWork& w,Matrix& I,Vector& ordinary,double& loglik,double& min_logp,double& min_gap,
    const double* pre,const double* rm) {
    const int q=data.q,p=data.p,pn=data.pn;
    const double m=data.totals[i];
    VectorXd& eta=w.predictors;VectorXd& threshold_score=w.threshold_score;VectorXd& point=w.values;
    std::vector<int>& index=w.index;std::vector<int>& next=w.next;
    partial_row(data,b,i,eta,row,min_logp,min_gap,pre,rm);
    threshold_score.setZero();double common_score=0;
    if(data.grouped) {
        for(int k=0;k<data.K;++k) {
            const double y=data.weight(i)*data.count(i,k);
            if(y==0) continue;
            loglik+=y*row.logp[k];common_score-=y*row.location_score(k);
            if(k<q) threshold_score[k]+=y*row.left[k];
            if(k>0) threshold_score[k-1]-=y*row.right[k-1];
        }
    } else {
        const int k=data.code(i)-1;
        loglik+=m*row.logp[k];common_score=-m*row.location_score(k);
        if(k<q) threshold_score[k]+=m*row.left[k];
        if(k>0) threshold_score[k-1]-=m*row.right[k-1];
    }
    ordinary.head(q)+=threshold_score;
    for(int j=0;j<p;++j) ordinary[q+j]+=data.x(i,j)*common_score;
    point[0]=1;
    for(int j=0;j<pn;++j) point[j+1]=-data.z(i,j);
    double common_weight=0;
    for(int s=0;s<q;++s) {
        index[0]=s;next[0]=s+1;
        for(int j=0;j<pn;++j) {
            index[j+1]=q+p+s*pn+j;next[j+1]=q+p+(s+1)*pn+j;
            ordinary[index[j+1]]-=data.z(i,j)*threshold_score[s];
        }
        const double diagonal=m*row.g[s]*(row.left[s]+row.right[s]);
        for(int a=0;a<=pn;++a) for(int c=a;c<=pn;++c)
            I(index[a],index[c])+=diagonal*point[a]*point[c];
        if(s+1<q) {
            const double off=-m*row.g[s+1]*row.right[s];
            for(int a=0;a<=pn;++a) for(int c=0;c<=pn;++c) {
                const int lo=std::min(index[a],next[c]),hi=std::max(index[a],next[c]);
                I(lo,hi)+=off*point[a]*point[c];
            }
        }
        // These stable row sums avoid subtracting large adjacent
        // derivative ratios when common-slope contributions cancel.
        const double rowsum=m*row.g[s]*row.neighbor[s];common_weight+=rowsum;
        for(int j=0;j<p;++j) {
            I(s,q+j)-=rowsum*data.x(i,j);
            for(int k=0;k<pn;++k)
                I(q+j,q+p+s*pn+k)+=rowsum*data.x(i,j)*data.z(i,k);
        }
    }
    for(int j=0;j<p;++j) for(int k=j;k<p;++k)
        I(q+j,q+k)+=common_weight*data.x(i,j)*data.x(i,k);
}

// The expected information. Lanes of d by d accumulators when d <= 1024;
// above that the rows are added on the calling thread in order (each row
// adds to scattered entries of the d by d matrix, which a column-tile
// scheme does not fit).
inline void partial_add_information(const PartialOrdinalData& data,const VectorXd& b,
    PartialOrdinalState& state,OrdinalWorkspace* workspace=nullptr) {
    const int q=data.q,p=data.p,pn=data.pn,d=data.d,n=data.n;
    const VectorXd seed=VectorXd::LinSpaced(q,0,double(q-1));
    if(!fglm_par::use_lanes(d)) {
        OrdinalRow row(seed);OrdinalWork w;partial_size_work(w,q,pn+1);
        for(int first=0;first<n;first+=data.block) {
            Rcpp::checkUserInterrupt();
            for(int i=first;i<std::min(n,first+data.block);++i) {
                if(data.totals[i]==0) continue;
                partial_information_row(data,b,i,row,w,state.information,state.ordinary,state.loglik,
                    state.min_logp,state.minimum_gap,nullptr,nullptr);
            }
        }
        state.information=MatrixXd(state.information.selfadjointView<Eigen::Upper>());
        return;
    }
    OrdinalWorkspace local;
    OrdinalWorkspace& ws=workspace ? *workspace : local;
    const int rows=ordinal_block_rows(data.block,d,true,data.link,q);
    const Eigen::Index nb=(Eigen::Index(n)+rows-1)/rows;
    const bool staged=data.link==1;
    ordinal_size_lanes(ws,nb,d,q,true);
    ordinal_lanes(nb,ws,seed,
        [&](OrdinalAcc& a) { ordinal_reset_acc(a,true); },
        [&](Eigen::Index blk,OrdinalAcc& a,OrdinalWorkHandle& h) {
            OrdinalWork& w=*h.w;partial_size_work(w,q,pn+1);
            const int first=static_cast<int>(blk*rows),length=std::min(rows,n-first);
            const std::ptrdiff_t at=static_cast<std::ptrdiff_t>(blk%fglm_par::kLanes)*rows;
            for(int r=0;r<length;++r) {
                const int i=first+r;
                if(data.totals[i]==0) continue;
                partial_information_row(data,b,i,h.row,w,a.M,a.v,a.loglik,a.min_logp,a.min_gap,
                    staged ? ws.eta.data()+(at+r)*q : nullptr,staged ? ws.rmath.data()+(at+r)*3*q : nullptr);
            }
            return true;
        },
        [&](OrdinalAcc&,const OrdinalAcc& a) {
            state.information+=a.M;
            state.ordinary+=a.v;
            state.loglik+=a.loglik;
            state.min_logp=std::min(state.min_logp,a.min_logp);
            state.minimum_gap=std::min(state.minimum_gap,a.min_gap);
        },staged,
        [&](Eigen::Index g,Eigen::Index count) {
            if(staged) partial_stage_expected(data,b,g,count,rows,ws);
        });
    state.information=MatrixXd(state.information.selfadjointView<Eigen::Upper>());
}

// The bias adjustment needs only vector sums, so it always uses lanes.
inline void partial_add_adjustment(const PartialOrdinalData& data,const VectorXd& b,
    PartialOrdinalState& state,OrdinalWorkspace* workspace=nullptr) {
    const int q=data.q,p=data.p,pn=data.pn,width=1+p+pn,d=data.d,n=data.n;
    const VectorXd seed=VectorXd::LinSpaced(q,0,double(q-1));
    OrdinalWorkspace local;
    OrdinalWorkspace& ws=workspace ? *workspace : local;
    const int rows=ordinal_block_rows(data.block,d,true,data.link,q);
    const Eigen::Index nb=(Eigen::Index(n)+rows-1)/rows;
    const bool staged=data.link==1;
    ordinal_size_lanes(ws,nb,d,q,false);
    ordinal_lanes(nb,ws,seed,
        [&](OrdinalAcc& a) { ordinal_reset_acc(a,false); },
        [&](Eigen::Index blk,OrdinalAcc& a,OrdinalWorkHandle& h) {
            OrdinalWork& w=*h.w;partial_size_work(w,q,width);
            OrdinalRow& row=h.row;
            VectorXd& eta=w.predictors;VectorXd& values=w.values;VectorXd& correction=w.c;
            std::vector<int>& indices=w.index;
            const int first=static_cast<int>(blk*rows),length=std::min(rows,n-first);
            const std::ptrdiff_t at=static_cast<std::ptrdiff_t>(blk%fglm_par::kLanes)*rows;
            for(int r=0;r<length;++r) {
                const int i=first+r;
                const double m=data.totals[i];if(m==0) continue;
                partial_row(data,b,i,eta,row,a.min_logp,a.min_gap,
                    staged ? ws.eta.data()+(at+r)*q : nullptr,staged ? ws.rmath.data()+(at+r)*3*q : nullptr);
                values[0]=1;
                for(int j=0;j<p;++j) {indices[j+1]=q+j;values[j+1]=-data.x(i,j);}
                for(int j=0;j<pn;++j) values[1+p+j]=-data.z(i,j);
                for(int s=0;s<q;++s) {
                    indices[0]=s;
                    for(int j=0;j<pn;++j) indices[1+p+j]=q+p+s*pn+j;
                    double variance=0;
                    for(int c1=0;c1<width;++c1) {
                        variance+=values[c1]*values[c1]*state.inverse(indices[c1],indices[c1]);
                        for(int c2=c1+1;c2<width;++c2)
                            variance+=2*values[c1]*values[c2]*state.inverse(indices[c1],indices[c2]);
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
                    a.v[s]+=adjusted;
                    for(int j=0;j<pn;++j) a.v[q+p+s*pn+j]-=data.z(i,j)*adjusted;
                    common_adjustment-=correction[s]*row.neighbor[s];
                }
                for(int j=0;j<p;++j) a.v[q+j]+=data.x(i,j)*common_adjustment;
            }
            return true;
        },
        [&](OrdinalAcc&,const OrdinalAcc& a) {
            state.adjustment+=a.v;
            state.min_logp=std::min(state.min_logp,a.min_logp);
            state.minimum_gap=std::min(state.minimum_gap,a.min_gap);
        },staged,
        [&](Eigen::Index g,Eigen::Index count) {
            if(staged) partial_stage_expected(data,b,g,count,rows,ws);
        });
}

template<class Data>
inline PartialOrdinalState partial_evaluate(const Data& data,
    const VectorXd& b,bool firth,double condition_tol,OrdinalWorkspace* workspace=nullptr) {
    if(b.size()!=data.d || !b.allFinite()) throw OrdinalEvaluationError("invalid nonparallel ordinal parameters");
    OrdinalWorkspace local;
    OrdinalWorkspace* ws=workspace ? workspace : &local;
    PartialOrdinalState state(data.d);
    data.visit([&](const PartialOrdinalData& block) {partial_add_information(block,b,state,ws);});
    ordinal_factor_information(state,data.d-data.q,data.q,condition_tol);
    if(firth) data.visit([&](const PartialOrdinalData& block) {partial_add_adjustment(block,b,state,ws);});
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

// The cumulative predictor of row r of a block at threshold s, given the
// row's location (offset included); the observed pass and its probit stage
// both use it, so they evaluate the same number.
template<class Rows>
inline double partial_threshold_predictor(const Rows& zz,int r,const VectorXd& alpha,double linear,
    const VectorXd& b,int q,int p,int pn,int s) {
    return alpha[s]-linear-zz.row(r).dot(b.segment(q+p+s*pn,pn));
}

// Stage of the nonparallel observed pass for the probit link: for the
// blocks [g, g + count) of rows rows each (slot j = block - g), in parallel
// the location of every positive-frequency row (offset included) into
// ws.linear and its cumulative predictors at the thresholds that bound its
// observed categories (every threshold for a grouped row) into ws.eta, then
// on the calling thread the probit terms there into ws.rmath.
inline void partial_stage_observed(const PartialOrdinalData& data,const VectorXd& b,
    Eigen::Index g,Eigen::Index count,int rows,OrdinalWorkspace& ws) {
    const int q=data.q,p=data.p,pn=data.pn;
    const VectorXd alpha=b.head(q),beta=b.segment(q,p);
    const Eigen::Index r0=g*rows,r1=std::min<Eigen::Index>(data.n,(g+count)*rows),m=r1-r0;
    const Eigen::Index cap=Eigen::Index(fglm_par::kLanes)*rows;
    if(ws.linear.size()<cap) ws.linear.resize(cap);
    if(ws.eta.size()<cap*q) ws.eta.resize(cap*q);
    if(ws.rmath.size()<cap*3*q) ws.rmath.resize(cap*3*q);
    // Whether row i needs the terms at threshold s.
    const auto needed=[&](int i,int s) {
        if(data.grouped) return true;
        const int k=data.code(i)-1;
        return s==k || s==k-1;
    };
    fglm_par::parallel_blocks(m,rows,[&](Eigen::Index r,Eigen::Index k) {
        const int first=static_cast<int>(r0+r),length=static_cast<int>(k);
        const auto xx=data.x.middleRows(first,length);
        const auto zz=data.z.middleRows(first,length);
        VectorXd location(length);
        if(p) location.noalias()=xx*beta; else location.setZero();
        for(int j=0;j<length;++j) {
            const int i=first+j;
            if(data.totals[i]==0) continue;
            const double linear=location[j]+data.off(i);
            ws.linear[r+j]=linear;
            for(int s=0;s<q;++s) if(needed(i,s))
                ws.eta[(r+j)*q+s]=partial_threshold_predictor(zz,j,alpha,linear,b,q,p,pn,s);
        }
    });
    for(Eigen::Index j=0;j<m;++j) {
        const int i=static_cast<int>(r0+j);
        if(data.totals[i]==0) continue;
        for(int s=0;s<q;++s) if(needed(i,s))
            ordinal_probit_terms(ws.eta[j*q+s],&ws.rmath[(j*q+s)*3]);
    }
}

// The rows of the observed pass for the block of `length` rows from
// `first`, added to I, score, loglik, and min_gap. lin, pre, and rm, when
// given, are the block's staged locations, threshold predictors, and probit
// terms.
template<class Matrix,class Vector>
inline void partial_observed_rows(const PartialOrdinalData& data,const VectorXd& b,int first,int length,
    OrdinalWork& w,Matrix& I,Vector& score,double& loglik,double& min_gap,
    const double* lin,const double* pre,const double* rm) {
    const int q=data.q,p=data.p,pn=data.pn;
    const VectorXd alpha=b.head(q),beta=b.segment(q,p);
    const int width=2+p+2*pn;
    if(static_cast<int>(w.index.size())!=width) w.index.assign(width,0);
    if(w.upper_x.size()!=width) w.upper_x.resize(width);
    if(w.lower_x.size()!=width) w.lower_x.resize(width);
    std::vector<int>& index=w.index;VectorXd& upper=w.upper_x;VectorXd& lower=w.lower_x;
    const auto xx=data.x.middleRows(first,length);
    const auto zz=data.z.middleRows(first,length);
    if(!lin) {
        w.eta.resize(length);
        if(p) w.eta.noalias()=xx*beta; else w.eta.setZero();
    }
    // Ordering must hold at every positive-frequency row, not only at observed categories.
    for(int s=1;s<q;++s) {
        const VectorXd rowgap=VectorXd::Constant(length,alpha[s]-alpha[s-1])-
            zz*(b.segment(q+p+s*pn,pn)-b.segment(q+p+(s-1)*pn,pn));
        for(int r=0;r<length;++r) if(data.totals[first+r]>0) {
            if(!(rowgap[r]>0)) throw OrdinalEvaluationError("nonparallel ordinal cumulative predictors are not strictly ordered");
            min_gap=std::min(min_gap,rowgap[r]);
        }
    }
    for(int r=0;r<length;++r) {
        const int i=first+r;
        if(data.totals[i]==0) continue;
        const double linear=lin ? lin[r] : w.eta[r]+data.off(i);
        const auto add=[&](int k,double y) {
            double eu=0,el=0,gap=0;
            if(pre) {
                if(k<q) eu=pre[static_cast<std::ptrdiff_t>(r)*q+k];
                if(k>0) el=pre[static_cast<std::ptrdiff_t>(r)*q+k-1];
            } else {
                if(k<q) eu=partial_threshold_predictor(zz,r,alpha,linear,b,q,p,pn,k);
                if(k>0) el=partial_threshold_predictor(zz,r,alpha,linear,b,q,p,pn,k-1);
            }
            if(k>0 && k<q) gap=alpha[k]-alpha[k-1]-zz.row(r).dot(b.segment(q+p+k*pn,pn)-b.segment(q+p+(k-1)*pn,pn));
            const double one_minus=gap>0 ? -std::expm1(-gap) : 0;
            const double* rr=rm ? rm+static_cast<std::ptrdiff_t>(r)*3*q : nullptr;
            const ObservedTerms t=data.link==0 ?
                observed_logit(k,q,eu,el,gap>0 ? 1/std::expm1(gap) : 0,gap>0 ? ordinal_log1mexp(-gap) : 0,one_minus) :
                observed_general(k,q,data.link,eu,el,gap,one_minus,rr && k<q ? rr+3*k : nullptr,
                    rr && k>0 ? rr+3*(k-1) : nullptr);
            loglik+=y*t.logp;
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
                score[index[a]]+=y*(t.left*upper[a]-t.right*lower[a]);
                for(int c=0;c<used;++c) {
                    const double value=t.upper*upper[a]*upper[c]+t.lower*lower[a]*lower[c]+
                        t.cross*(upper[a]*lower[c]+lower[a]*upper[c]);
                    I(index[a],index[c])+=y*value;
                }
            }
        };
        if(data.grouped) {
            for(int k=0;k<data.K;++k) {
                const double y=data.weight(i)*data.count(i,k);
                if(y!=0) add(k,y);
            }
        } else add(data.code(i)-1,data.totals[i]);
    }
}

// The observed information. Lanes of d by d accumulators when d <= 1024;
// above that the blocks are added on the calling thread in order.
inline void partial_observed_add(const PartialOrdinalData& data,const VectorXd& b,PartialObservedState& out,
    OrdinalWorkspace* workspace=nullptr) {
    const int q=data.q,d=data.d,n=data.n;
    if(!fglm_par::use_lanes(d)) {
        OrdinalWork w;
        for(int first=0;first<n;first+=data.block) {
            Rcpp::checkUserInterrupt();
            const int length=std::min(data.block,n-first);
            partial_observed_rows(data,b,first,length,w,out.information,out.score,out.loglik,out.minimum_gap,
                nullptr,nullptr,nullptr);
        }
        return;
    }
    OrdinalWorkspace local;
    OrdinalWorkspace& ws=workspace ? *workspace : local;
    const int rows=ordinal_block_rows(data.block,d,true,data.link,q);
    const Eigen::Index nb=(Eigen::Index(n)+rows-1)/rows;
    const bool staged=data.link==1;
    const VectorXd seed=VectorXd::LinSpaced(q,0,double(q-1));
    ordinal_size_lanes(ws,nb,d,q,true);
    ordinal_lanes(nb,ws,seed,
        [&](OrdinalAcc& a) { ordinal_reset_acc(a,true); },
        [&](Eigen::Index blk,OrdinalAcc& a,OrdinalWorkHandle& h) {
            const int first=static_cast<int>(blk*rows),length=std::min(rows,n-first);
            const std::ptrdiff_t at=static_cast<std::ptrdiff_t>(blk%fglm_par::kLanes)*rows;
            partial_observed_rows(data,b,first,length,*h.w,a.M,a.v,a.loglik,a.min_gap,
                staged ? ws.linear.data()+at : nullptr,staged ? ws.eta.data()+at*q : nullptr,
                staged ? ws.rmath.data()+at*3*q : nullptr);
            return true;
        },
        [&](OrdinalAcc&,const OrdinalAcc& a) {
            out.information+=a.M;
            out.score+=a.v;
            out.loglik+=a.loglik;
            out.minimum_gap=std::min(out.minimum_gap,a.min_gap);
        },staged,
        [&](Eigen::Index g,Eigen::Index count) {
            if(staged) partial_stage_observed(data,b,g,count,rows,ws);
        });
}

template<class Data>
PartialObservedState partial_observed_evaluate(const Data& data,const VectorXd& b,OrdinalWorkspace* workspace=nullptr) {
    if(b.size()!=data.d || !b.allFinite()) throw OrdinalEvaluationError("invalid nonparallel ordinal parameters");
    PartialObservedState out(data.d);
    data.visit([&](const PartialOrdinalData& block) { partial_observed_add(block,b,out,workspace); });
    if(!std::isfinite(out.loglik) || !out.score.allFinite() || !out.information.allFinite())
        throw OrdinalEvaluationError("non-finite observed nonparallel ordinal terms");
    return out;
}

// The largest predictor change and the largest step that keeps every row
// ordered. Blocks run in parallel; the maximum and minimum do not depend on
// the order of their terms.
template<class Data>
void partial_step_limits(const Data& data,const VectorXd& b,const VectorXd& direction,double& change,double& fraction) {
    change=0;fraction=1;
    data.visit([&](const PartialOrdinalData& block) {
        const int q=block.q,p=block.p,pn=block.pn;
        const Eigen::Index nb=(block.n+block.block-1)/block.block;
        std::vector<double> changes(static_cast<std::size_t>(nb),0.0),fractions(static_cast<std::size_t>(nb),1.0);
        fglm_par::parallel_blocks(block.n,block.block,[&](Eigen::Index first_,Eigen::Index length_) {
            const int first=static_cast<int>(first_),length=static_cast<int>(length_);
            double ch=0,fr=1;
            VectorXd location,value,previous;
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
                        if(difference<0) fr=std::min(fr,.9*current[r]/(-difference));
                    }
                }
                for(int r=0;r<length;++r) if(block.totals[first+r]>0) ch=std::max(ch,std::abs(value[r]));
                previous=value;
            }
            changes[static_cast<std::size_t>(first/block.block)]=ch;
            fractions[static_cast<std::size_t>(first/block.block)]=fr;
        });
        for(double v:changes) change=std::max(change,v);
        for(double v:fractions) fraction=std::min(fraction,v);
    });
}

template<class Data>
int partial_observed_newton(const Data& data,VectorXd& b,double score_tol,double step_tol,int maxit,
    double max_step,int max_halves,bool trace,int& evaluations,OrdinalWorkspace* workspace=nullptr) {
    int iterations=0;
    // Trace: one line per iteration (trace.h), at its start; passes are the
    // evaluations over the data so far.
    const fglm_trace::Tracer tracer(trace);
    int halvings=0;
    try {
        PartialObservedState current=partial_observed_evaluate(data,b,workspace);
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
            tracer.iteration(iterations+1,evaluations,current.loglik,halvings,"loglik");
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
                halvings=h;
                candidate=b+fraction*direction;
                try {
                    trial=partial_observed_evaluate(data,candidate,workspace);
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
    // Lane accumulators and scratch kept for every pass of the fit.
    OrdinalWorkspace workspace;
    // Bias reduction needs expected-information terms in every iteration.
    const int observed_iterations=!observed || firth || maxit<2 ? 0 :
        partial_observed_newton(data,b,score_tol,step_tol,std::min(maxit-1,30),max_step,max_halves,trace,evaluations,
            &workspace);
    const int expected_limit=maxit-observed_iterations;
    auto evaluate=[&](const VectorXd& candidate) {++evaluations;return partial_evaluate(data,candidate,firth,condition_tol,&workspace);};
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
    // Boundary-aware scoring direction (maximum likelihood only). When the
    // Fisher step would make the cumulative predictors of some row cross, the
    // row-pair crossed first is held at half its current gap, and the step is
    // recomputed on the remaining directions (null-space method), repeating
    // until the step keeps every row ordered. Each ordering constraint is
    // linear in the coefficients:
    //   gap(i,s) = alpha_s - alpha_{s-1} - z_i'(gamma_s - gamma_{s-1}) > 0.
    // Halving the gap, instead of holding the row on the boundary, keeps the
    // evaluation finite; a row whose optimum is on the boundary approaches it
    // geometrically. Near the boundary the expected information is dominated
    // by rank-one terms along the held constraint's normal (the touched
    // category's probability goes to zero), and the null space removes them.
    // held = number of held row-pairs (0: the Fisher step is feasible).
    auto boundary_direction=[&](VectorXd& direction,int& held) {
        direction=current.step;held=0;
        std::vector<VectorXd> normals;std::vector<double> targets;
        for(int round=0;round<data.d;++round) {
            double first=std::numeric_limits<double>::infinity();VectorXd normal;double gap0=0;
            data.visit([&](const PartialOrdinalData& block) {
                const int q=block.q,p=block.p,pn=block.pn;
                for(int i=0;i<block.n;++i) if(block.totals[i]>0) for(int s=1;s<q;++s) {
                    const double change=block.gap(i,direction,s);
                    if(!(change<0)) continue;
                    const double g=block.gap(i,b,s),t=g/(-change);
                    if(t<first) {
                        first=t;gap0=g;normal=VectorXd::Zero(block.d);
                        normal[s]=1;normal[s-1]=-1;
                        for(int j=0;j<pn;++j) {normal[q+p+s*pn+j]=-block.z(i,j);normal[q+p+(s-1)*pn+j]=block.z(i,j);}
                    }
                }
            });
            // feasible_fraction() stops a step at 0.9 of the distance to a
            // crossing, so a pair that the full step brings within that
            // margin (t < 1/0.9) limits the step as well and is held too.
            if(!(first<1/.9)) return true;
            for(const VectorXd& a:normals) if((a-normal).cwiseAbs().maxCoeff()==0) return false;
            // Below 1e-10 a further halving of the gap is beyond the
            // resolution of O(1) coefficients; hold the row where it is, so
            // the step lies on the boundary face and the face problem can
            // converge.
            normals.push_back(normal);targets.push_back(gap0<1e-10 ? 0.0 : -.5*gap0);
            const int k=normals.size(),d=data.d;
            MatrixXd At(d,k);VectorXd r(k);
            for(int j=0;j<k;++j) {At.col(j)=normals[j];r[j]=targets[j];}
            Eigen::ColPivHouseholderQR<MatrixXd> qa(At);
            if(qa.rank()<k) return false;
            const MatrixXd Q=qa.householderQ()*MatrixXd::Identity(d,d);
            const MatrixXd R=qa.matrixQR().topLeftCorner(k,k).template triangularView<Eigen::Upper>();
            const VectorXd base=Q.leftCols(k)*R.transpose().triangularView<Eigen::Lower>().solve(
                VectorXd(qa.colsPermutation().transpose()*r));
            if(k==d) {direction=base;held=k;return direction.allFinite();}
            const MatrixXd Q2=Q.rightCols(d-k);
            const MatrixXd reduced=Q2.transpose()*current.information*Q2;
            Eigen::LDLT<MatrixXd> factor(reduced);
            if(factor.info()!=Eigen::Success) return false;
            direction=base+Q2*factor.solve(Q2.transpose()*(current.score-current.information*base));
            held=k;
            if(!direction.allFinite()) return false;
        }
        return false;
    };
    // The interior certificate (small scaled score and Fisher predictor step)
    // is not used where a constraint binds: the expected information diverges
    // there and makes both quantities small without stationarity. At the
    // boundary, convergence is certified by the constrained step instead.
    // Face certificate for a maximum at which some rows touch but the Fisher
    // step no longer crosses: the expected information diverges along the
    // touching constraints, so the step along them vanishes while the score
    // keeps a component along them (the Lagrange multipliers), and the
    // interior certificate cannot pass. Hold every distinct touching
    // constraint (rows sharing a constraint, for example all rows with the
    // same nonparallel covariate value, contribute one normal), solve the
    // scoring step on the remaining directions, and certify when that step is
    // negligible and the score points into the constraints (nonnegative
    // multipliers), so that no move back into the interior can increase the
    // likelihood.
    auto face_certificate=[&]() {
        std::vector<VectorXd> normals;
        data.visit([&](const PartialOrdinalData& block) {
            const int q=block.q,p=block.p,pn=block.pn;
            for(int i=0;i<block.n;++i) if(block.totals[i]>0) for(int s=1;s<q;++s) {
                if(!(block.gap(i,b,s)<boundary_gap)) continue;
                VectorXd normal=VectorXd::Zero(block.d);
                normal[s]=1;normal[s-1]=-1;
                for(int j=0;j<pn;++j) {normal[q+p+s*pn+j]=-block.z(i,j);normal[q+p+(s-1)*pn+j]=block.z(i,j);}
                bool seen=false;
                for(const VectorXd& a:normals) if((a-normal).cwiseAbs().maxCoeff()==0) {seen=true;break;}
                if(!seen) normals.push_back(normal);
            }
        });
        const int k=normals.size(),d=data.d;
        if(k==0 || k>=d) return false;
        MatrixXd A(d,k);
        for(int j=0;j<k;++j) A.col(j)=normals[j];
        Eigen::ColPivHouseholderQR<MatrixXd> qa(A);
        const int r=qa.rank();
        if(r==0 || r>=d) return false;
        const MatrixXd Q=qa.householderQ()*MatrixXd::Identity(d,d);
        const MatrixXd Q2=Q.rightCols(d-r);
        Eigen::LDLT<MatrixXd> factor(Q2.transpose()*current.information*Q2);
        if(factor.info()!=Eigen::Success) return false;
        const VectorXd direction=Q2*factor.solve(Q2.transpose()*current.score);
        if(!direction.allFinite()) return false;
        // The face step is negligible, or its predicted gain in log likelihood
        // (one half of score'direction) is below the rounding level at which the
        // line search can measure an increase, so no further step can be
        // verified as an improvement in double precision.
        const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(current.loglik));
        if(!(data.predictor_step(direction)<step_tol) && !(.5*current.score.dot(direction)<=rounding)) return false;
        // At a constrained maximum the score is minus a nonnegative
        // combination of the constraint normals (gaps increase along them).
        const VectorXd residual=current.score-current.information*direction;
        const VectorXd multipliers=-qa.solve(residual);
        if(!multipliers.allFinite()) return false;
        const double scale=std::max(1.0,multipliers.cwiseAbs().maxCoeff());
        return multipliers.minCoeff()>=-1e-6*scale;
    };
    auto certify=[&]() {
        if(!firth && current.minimum_gap<boundary_gap) {
            VectorXd direction;int held=0;
            if(boundary_direction(direction,held) && held>0) {
                if(data.predictor_step(direction)<step_tol) {converged=true;reason="converged_boundary";return true;}
                return false;
            }
            if(face_certificate()) {converged=true;reason="converged_boundary";return true;}
        }
        if(current.scaled_score<score_tol && current.predictor_step<step_tol) {
            // A fit whose rows touch to within boundary_gap is on the
            // ordering boundary even when its Fisher step no longer crosses
            // (the step is then tiny because the information diverges).
            converged=true;
            reason=!firth && current.minimum_gap<boundary_gap ? "converged_boundary" : "score_and_predictor_step";
            return true;
        }
        return false;
    };
    // Trace: one line per iteration at its start (numbered after the
    // observed Newton iterations); passes are the evaluations so far and
    // halvings those of the step that led to the current estimate.
    const fglm_trace::Tracer tracer(trace);
    int halvings=0;
    for(int iter=1;iter<=expected_limit;++iter) {
        Rcpp::checkUserInterrupt();iterations=iter;
        history.row(iter-1)<<current.loglik,current.scaled_score,current.predictor_step,current.minimum_gap;
        tracer.iteration(observed_iterations+iter,evaluations,current.loglik,halvings,"loglik");
        if(certify()) break;
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
                    if(accepted) {accepted_b=candidate;accepted_state=std::move(trial);halvings=h;return true;}
                    last_rejection=firth ? "adjusted-score norm did not decrease sufficiently" :
                        "likelihood did not increase sufficiently";
                } catch(const OrdinalEvaluationError& error) {last_rejection=error.what();}
                fraction/=2;
            }
            return false;
        };
        bool accepted=false;
        if(!firth) {
            VectorXd direction;int held=0;
            const bool built=boundary_direction(direction,held);
            if(built && held>0) accepted=search(direction);
        }
        if(!accepted) accepted=search(current.step,current.predictor_step);
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
    if(!converged) certify();
    Rcpp::List out=partial_state_list(current);
    out["coefficients"]=b;out["converged"]=converged;out["convergence.reason"]=reason;
    out["iterations"]=observed_iterations+iterations;out["observed.iterations"]=observed_iterations;out["evaluations"]=evaluations;out["fallbacks"]=fallbacks;
    out["last.rejected.evaluation"]=last_rejection;out["history"]=MatrixXd(history.topRows(iterations));
    return out;
}

}
#endif
