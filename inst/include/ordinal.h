#ifndef FASTGLM_ORDINAL_H
#define FASTGLM_ORDINAL_H
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "parallel_reduce.h"
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
    // Raw views of the R vectors, read inside parallel regions (Rcpp element
    // access can call the R API when an index is out of bounds).
    const int* code_ptr=nullptr;
    const double *weight_ptr=nullptr,*offset_ptr=nullptr,*count_ptr=nullptr;

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
        weight_ptr=w.begin();offset_ptr=off.begin();
        if(grouped) count_ptr=Y.begin(); else code_ptr=y.begin();
    }
    int code(int i) const { return code_ptr[i]; }
    double weight(int i) const { return weight_ptr[i]; }
    double off(int i) const { return offset_ptr[i]; }
    double count(int i,int k) const { return count_ptr[i+static_cast<std::ptrdiff_t>(n)*k]; }

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

    // The largest change in a cumulative predictor. Blocks run in parallel;
    // a maximum does not depend on the order of its terms (NaN terms are
    // skipped by std::max either way), so the result is the serial one.
    double predictor_step(const VectorXd& direction) const {
        const VectorXd beta=direction.tail(p);
        const double low=direction.head(q).minCoeff(),high=direction.head(q).maxCoeff();
        const Eigen::Index nb=(n+block-1)/block;
        std::vector<double> part(static_cast<std::size_t>(nb),0.0);
        fglm_par::parallel_blocks(n,block,[&](Eigen::Index first,Eigen::Index length) {
            double ans=0;
            for(Eigen::Index i=first;i<first+length;++i) if(totals[i]>0) {
                const double dx=linear(static_cast<int>(i),beta);
                ans=std::max(ans,std::max(std::abs(low-dx),std::abs(high-dx)));
            }
            part[static_cast<std::size_t>(first/block)]=ans;
        });
        Rcpp::checkUserInterrupt();
        double ans=0;
        for(double v:part) ans=std::max(ans,v);
        return ans;
    }
};

// Probit terms from R's math library at eta: out[0] = log Phi(eta),
// out[1] = log(1 - Phi(eta)), out[2] = log phi(eta). These calls must stay
// outside parallel regions; threaded passes compute them on the calling
// thread into a buffer (a stage) before the rows are accumulated.
inline void ordinal_probit_terms(double eta,double* out) {
    Rf_pnorm_both(eta,&out[0],&out[1],2,1);
    out[2]=R::dnorm(eta,0,1,true);
}

struct OrdinalRow {
    VectorXd eta,g,cdf,survival,logp,left,right,neighbor,gprime,logg,logcdf,logsurvival,location,logden;
    VectorXd loggap,inverse_gap,neighbor_gap;
    // Logit link without per-pair logarithms (evaluate(..., need_logs=false)):
    // expo holds exp(-|eta|); exp(+-alpha) and the gap probabilities
    // 1 - exp(-(alpha_s - alpha_{s-1})) are computed once per parameter value.
    VectorXd expo,exp_alpha,exp_neg_alpha,gap_probability,inverse_scratch,left_gap,right_gap;
    bool products_ok=false;
    int link=0;
    explicit OrdinalRow(const VectorXd& alpha) : eta(alpha.size()),g(alpha.size()),cdf(alpha.size()),
        survival(alpha.size()),logp(alpha.size()+1),left(alpha.size()),right(alpha.size()),neighbor(alpha.size()),
        gprime(alpha.size()),logg(alpha.size()),logcdf(alpha.size()),logsurvival(alpha.size()),
        location(alpha.size()+1),logden(alpha.size()),loggap(alpha.size()),inverse_gap(alpha.size()),neighbor_gap(alpha.size()),
        expo(alpha.size()),exp_alpha(alpha.size()),exp_neg_alpha(alpha.size()),gap_probability(alpha.size()),
        inverse_scratch(alpha.size()),left_gap(VectorXd::Zero(alpha.size())),right_gap(VectorXd::Zero(alpha.size())) {
        for(int s=1;s<alpha.size();++s) {
            const double gap=alpha[s]-alpha[s-1];
            loggap[s]=ordinal_log1mexp(-gap);inverse_gap[s]=1/std::expm1(gap);
            gap_probability[s]=-std::expm1(-gap);
        }
        for(int s=1;s+1<alpha.size();++s)
            neighbor_gap[s]=-std::expm1(alpha[s-1]-alpha[s+1]);
        // exp(-|alpha_s - l|) = exp(-alpha_s) exp(l) or exp(alpha_s) exp(-l). With
        // |alpha_s| and |l| at most 300 neither factor overflows or underflows.
        // left = survival + left_gap and right = cdf + right_gap, as below.
        for(int s=1;s<alpha.size();++s) { left_gap[s]=inverse_gap[s];right_gap[s-1]=inverse_gap[s]; }
        products_ok=alpha.size()==0 || alpha.cwiseAbs().maxCoeff()<=300;
        if(products_ok) { exp_alpha=alpha.array().exp();exp_neg_alpha=(-alpha.array()).exp(); }
    }
    // Log cumulative and survival probabilities and the log category
    // probability for the logit link, from eta and expo; the same expressions
    // as the full evaluation below.
    double logit_log_cdf(int s) const { const double d=std::log1p(expo[s]);return eta[s]>=0 ? -d : eta[s]-d; }
    double logit_log_survival(int s) const { const double d=std::log1p(expo[s]);return eta[s]>=0 ? -eta[s]-d : -d; }
    double logit_logp(int k) const {
        const int q=eta.size();
        if(k==0) return logit_log_cdf(0);
        if(k==q) return logit_log_survival(q-1);
        return logit_log_cdf(k)+logit_log_survival(k-1)+loggap[k];
    }
    // The smallest log category probability. The category is located on the
    // probability scale, P_s = F_s (1 - F_{s-1}) (1 - exp(-gap_s)), and its
    // logarithm evaluated once; near underflow every category is evaluated.
    double logit_min_logp() const {
        const int q=eta.size();
        int best=0;double v=cdf[0];
        if(q>1) {
            Eigen::Index at;
            const double interior=(cdf.tail(q-1).array()*survival.head(q-1).array()*
                gap_probability.tail(q-1).array()).minCoeff(&at);
            if(interior<v) { v=interior;best=static_cast<int>(at)+1; }
        }
        if(survival[q-1]<v) { v=survival[q-1];best=q; }
        if(v>1e-280 && std::isfinite(v)) return logit_logp(best);
        double m=std::numeric_limits<double>::infinity();
        for(int k=0;k<=q;++k) m=std::min(m,logit_logp(k));
        return m;
    }
    // rmath, when given for the probit link, holds ordinal_probit_terms() at
    // alpha[s] - linear for s = 0, ..., q - 1 (3 values each), computed by a
    // stage outside the parallel region; the result is the same as when the
    // terms are computed here.
    void evaluate(const VectorXd& alpha,double linear,int link_code=0,bool need_logs=true,
                  const double* rmath=nullptr) {
        link=link_code;
        const int q=alpha.size();
        if(link==0 && !need_logs) {
            // The information pass needs no logarithms except at the observed
            // categories (logit_logp) and the smallest probability.
            const bool products=products_ok && std::abs(linear)<=300;
            for(int s=0;s<q;++s) eta[s]=alpha[s]-linear;
            if(products) {
                // The thresholds increase, so eta changes sign once: below
                // `split` eta < 0 and exp(-|eta|) = exp(alpha) exp(-l), from
                // `split` on eta >= 0 and exp(-|eta|) = exp(-alpha) exp(l).
                const double el=std::exp(linear),enl=std::exp(-linear);
                int split=0;
                while(split<q && eta[split]<0) ++split;
                // Array expressions over each range so that they vectorize.
                if(split>0) {
                    expo.head(split).array()=exp_alpha.head(split).array()*enl;
                    inverse_scratch.head(split).array()=(1+expo.head(split).array()).inverse();
                    cdf.head(split).array()=expo.head(split).array()*inverse_scratch.head(split).array();
                    survival.head(split)=inverse_scratch.head(split);
                }
                if(split<q) {
                    const int m=q-split;
                    expo.tail(m).array()=exp_neg_alpha.tail(m).array()*el;
                    inverse_scratch.tail(m).array()=(1+expo.tail(m).array()).inverse();
                    cdf.tail(m)=inverse_scratch.tail(m);
                    survival.tail(m).array()=expo.tail(m).array()*inverse_scratch.tail(m).array();
                }
                g.array()=expo.array()*inverse_scratch.array().square();
            } else {
                for(int s=0;s<q;++s) {
                    const double e=std::exp(-std::abs(eta[s])),inverse=1/(1+e);
                    expo[s]=e;
                    cdf[s]=eta[s]>=0 ? inverse : e*inverse;
                    survival[s]=eta[s]>=0 ? e*inverse : inverse;
                    g[s]=e*inverse*inverse;
                }
            }
            left=survival+left_gap;right=cdf+right_gap;
            if(q==1) neighbor[0]=1;
            else {
                neighbor[0]=cdf[1];neighbor[q-1]=survival[q-2];
                if(q>2) neighbor.segment(1,q-2).array()=cdf.tail(q-2).array()*survival.head(q-2).array()*
                    neighbor_gap.segment(1,q-2).array();
            }
            if(!left.allFinite() || !right.allFinite())
                throw OrdinalEvaluationError("non-finite ordinal probabilities or derivatives");
            return;
        }
        if(link!=0) {
            for(int s=0;s<q;++s) {
                eta[s]=alpha[s]-linear;
                if(link==1) {
                    if(rmath) { logcdf[s]=rmath[3*s];logsurvival[s]=rmath[3*s+1];logg[s]=rmath[3*s+2]; }
                    else {
                        Rf_pnorm_both(eta[s],&logcdf[s],&logsurvival[s],2,1);
                        logg[s]=R::dnorm(eta[s],0,1,true);
                    }
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

// ---------------------------------------------------------------------------
// Threading (OpenMP, through the deterministic reductions of
// parallel_reduce.h). A pass reaches the rows of each data segment (the whole
// in-memory data, or one block of a streamed source at a time, read on the
// calling thread) and splits them into fixed row blocks of
// ordinal_block_rows() rows. The block sums are added by one of two schemes,
// chosen from the dimensions alone:
//
// * lanes (fglm_par::lane_reduce_into), when the accumulator dimension d is
//   at most kLaneMaxDim (1024): each of up to 8 lane accumulators holds the
//   sums of the blocks l, l + 8, ... (the d by d information, the score, and
//   the scalar sums), and the lanes are then added in lane order;
// * column tiles (fglm_par::tile_reduce), for the common-slope information
//   and observed passes when d > 1024: blocks are prepared in groups of 8, the
//   p columns of the slope blocks are split into tiles of width 64, each tile
//   adding the group's blocks in order, and the vector and scalar sums are
//   added on the calling thread in block order.
//
// Every choice is a fixed function of the dimensions, the block size, and the
// segment lengths, so results are identical for every thread count.
//
// No R API is called inside a parallel region. The probit link's cumulative
// probabilities and densities come from R's math library, so a probit pass
// runs a stage on the calling thread before each round of 8 blocks (lanes)
// or group of 8 blocks (tiles): the cumulative predictors of the round are
// computed in parallel, then the R math terms serially into a buffer, which
// the blocks then read (fglm_par::lane_reduce_staged, tile_reduce_staged).
// The values are those an inline call would compute. The logit and
// complementary log-log links use only the C++ standard library and are
// evaluated inside the region.
//
// An exception thrown inside a region (OrdinalEvaluationError for rows that
// cannot be evaluated, or std::bad_alloc) is captured by the reductions and
// rethrown on the calling thread when its block is the smallest failing one,
// which is the block a serial pass would have stopped at.

// Sums of a pass (a lane, the part of one block in the tile scheme, or the
// total of a segment). Which fields a pass uses depends on its kind; the d by
// d matrix M is kept for the fit by the workspace's lanes:
//   information: M (tridiagonal threshold entries, threshold-slope and slope
//     blocks), v (ordinary score), loglik, min_logp;
//   adjustment: v;
//   observed: M (cross block in rows 0..q-1 of columns q.., lower slope
//     block), v (score), diag, off, loglik, change, row_norm;
//   nonparallel passes: M, v, loglik, min_logp, min_gap.
struct OrdinalAcc {
    MatrixXd M;
    VectorXd v,diag,off;
    double loglik=0,min_logp=0,change=0,row_norm=0,
        min_gap=std::numeric_limits<double>::infinity();
};

// Scratch of one thread (lanes) or one group slot (tiles). Nothing in it
// carries a result from one block to the next except the fields of a block
// prepared for the tile scheme, which are read before the slot is reused.
struct OrdinalWork {
    MatrixXd rowsums,weighted,xb,xa,rows;
    VectorXd slope_score,slope_weight,diag,off,eta,change,upper_x,lower_x,c;
    // Nonparallel passes: the row's cumulative predictors, threshold score,
    // coefficient values, and coordinates.
    VectorXd predictors,threshold_score,values;
    std::vector<int> index,next;
    Eigen::VectorXi category;
    std::unique_ptr<OrdinalRow> row;   // tile slots: the slot's row evaluator
    int first=0,length=0;              // tile slots: the prepared block
    OrdinalAcc part;                   // tile slots: the block's vector and scalar sums
};

// Accumulators and scratch kept for a fit, so that its passes do not
// allocate (and page in) them again: the lane accumulators (at most 8 of
// d by d doubles, d <= 1024, at most 64 MiB in all), the thread or group-slot
// scratch, and the staged probit buffers (at most 8 blocks of rows with
// 3 (K - 1) doubles per row, at most 16 MB, plus one predictor per row and
// threshold).
struct OrdinalWorkspace {
    std::vector<OrdinalAcc> lanes;
    std::vector<OrdinalWork> slots;
    VectorXd linear,eta,rmath;
};

// Rows per block: block_size, capped (ruling R17) at
// max(tile_block_rows(d), 2^21 / d) rows in the lane scheme and at
// tile_block_rows(d) in the tile scheme, so that a block buffer of d columns
// takes at most 16 MB (8 MB for tiles). For the probit link the blocks are
// also at most max(64, 2^18 / (3 (K - 1))) rows, so that a round of 8 staged
// blocks takes at most 16 MB. A fixed function of the dimensions and the
// block size; it is block_size whenever block_size is below these caps (at
// the default 2048 whenever d <= 1024 and K <= 43).
inline int ordinal_block_rows(int block,Eigen::Index d,bool lanes,int link,int q) {
    const Eigen::Index dd=std::max<Eigen::Index>(1,d);
    Eigen::Index cap=lanes ? std::max<Eigen::Index>(fglm_par::tile_block_rows(dd),Eigen::Index(2097152)/dd) :
        fglm_par::tile_block_rows(dd);
    if(link==1) cap=std::min<Eigen::Index>(cap,std::max<Eigen::Index>(64,Eigen::Index(262144)/(3*std::max(q,1))));
    return static_cast<int>(std::max<Eigen::Index>(1,std::min<Eigen::Index>(block,cap)));
}

// The scratch of the calling thread in a lane reduction: its slot of the
// workspace, or (should the thread index exceed the slots) its own, and a
// row evaluator for the pass's thresholds.
struct OrdinalWorkHandle {
    OrdinalWork* w=nullptr;
    std::unique_ptr<OrdinalWork> own;
    OrdinalRow row;
    explicit OrdinalWorkHandle(const VectorXd& alpha) : row(alpha) {}
};

// Runs a lane reduction over nblocks blocks with the workspace's lanes and
// one slot per running thread; staged (probit) passes call stage(first,
// count) on the calling thread before each round of 8 blocks. reset, add, and
// combine are as for fglm_par::lane_reduce_into; add receives the thread's
// OrdinalWorkHandle, whose row is made from row_alpha.
template<class Reset,class Add,class Combine,class Stage>
void ordinal_lanes(Eigen::Index nblocks,OrdinalWorkspace& ws,const VectorXd& row_alpha,
    Reset reset,Add add,Combine combine,bool staged,Stage stage) {
    if(nblocks<=0) return;
    const Eigen::Index nlanes=std::min<Eigen::Index>(nblocks,fglm_par::kLanes);
    const Eigen::Index ns=std::min<Eigen::Index>(nlanes,std::max(1,fglm_par::thread_setting()));
    if(static_cast<Eigen::Index>(ws.slots.size())<ns) ws.slots.resize(ns);
    if(static_cast<Eigen::Index>(ws.lanes.size())<nlanes) ws.lanes.resize(nlanes);
    auto make_work=[&]() {
        OrdinalWorkHandle h(row_alpha);
        const Eigen::Index id=fglm_par::current_thread();
        if(id<ns) h.w=&ws.slots[id];
        else { h.own.reset(new OrdinalWork());h.w=h.own.get(); }
        return h;
    };
    OrdinalAcc unused;
    const bool ok=staged ?
        fglm_par::lane_reduce_staged(nblocks,unused,ws.lanes,reset,make_work,stage,add,combine) :
        fglm_par::lane_reduce_into(nblocks,unused,ws.lanes,reset,make_work,add,combine);
    // add never reports a data failure (it throws instead); a guard.
    if(!ok) Rcpp::stop("ordinal block reduction failed");
}

// Sizes the lane accumulators used by a pass on the calling thread, so that
// the resets inside the region do not allocate.
inline void ordinal_size_lanes(OrdinalWorkspace& ws,Eigen::Index nblocks,int d,int q,bool matrix) {
    const Eigen::Index nlanes=std::min<Eigen::Index>(nblocks,fglm_par::kLanes);
    if(static_cast<Eigen::Index>(ws.lanes.size())<nlanes) ws.lanes.resize(nlanes);
    for(Eigen::Index l=0;l<nlanes;++l) {
        OrdinalAcc& a=ws.lanes[l];
        if(matrix && (a.M.rows()!=d || a.M.cols()!=d)) a.M.resize(d,d);
        if(a.v.size()!=d) a.v.resize(d);
        if(a.diag.size()!=q) a.diag.resize(q);
        if(a.off.size()!=std::max(q-1,0)) a.off.resize(std::max(q-1,0));
    }
}

inline void ordinal_reset_acc(OrdinalAcc& a,bool matrix) {
    if(matrix) a.M.setZero();
    a.v.setZero();a.diag.setZero();a.off.setZero();
    a.loglik=0;a.min_logp=0;a.change=0;a.row_norm=0;
    a.min_gap=std::numeric_limits<double>::infinity();
}

// Stage of the common-slope expected-information and adjustment passes for
// the probit link: for the blocks [g, g + count) of rows rows each (slot j =
// block - g), the cumulative predictor of every positive-frequency row
// (computed in parallel, as the passes compute it) into ws.linear and the
// probit terms at each threshold (on the calling thread) into ws.rmath.
inline void ordinal_stage_expected(const OrdinalData& data,const VectorXd& alpha,const VectorXd& beta,
    Eigen::Index g,Eigen::Index count,int rows,OrdinalWorkspace& ws) {
    const int q=data.q;
    const Eigen::Index r0=g*rows,r1=std::min<Eigen::Index>(data.n,(g+count)*rows),m=r1-r0;
    const Eigen::Index cap=Eigen::Index(fglm_par::kLanes)*rows;
    if(ws.linear.size()<cap) ws.linear.resize(cap);
    if(ws.rmath.size()<cap*3*q) ws.rmath.resize(cap*3*q);
    fglm_par::parallel_blocks(m,rows,[&](Eigen::Index r,Eigen::Index k) {
        for(Eigen::Index j=r;j<r+k;++j) {
            const int i=static_cast<int>(r0+j);
            if(data.totals[i]>0) ws.linear[j]=data.linear(i,beta)+data.off(i);
        }
    });
    for(Eigen::Index j=0;j<m;++j) if(data.totals[r0+j]>0)
        for(int s=0;s<q;++s) ordinal_probit_terms(alpha[s]-ws.linear[j],&ws.rmath[(j*q+s)*3]);
}

// The rows of the information pass for the block of `length` rows from
// `first`: adds the log likelihood, smallest log probability, and threshold
// score to a, and fills the block's tridiagonal sums (w.diag, w.off), the
// threshold-slope row sums (w.rowsums, threshold-major), and the slope score
// and weight of each row. lin and rm, when given, are the staged predictors
// and probit terms of the block's rows.
inline void ordinal_information_rows(const OrdinalData& data,const VectorXd& alpha,const VectorXd& beta,
    int first,int length,OrdinalRow& row,OrdinalWork& w,OrdinalAcc& a,const double* lin,const double* rm) {
    const int q=data.q;
    if(w.rowsums.rows()!=q || w.rowsums.cols()<length) w.rowsums.resize(q,length);
    if(w.slope_score.size()<length) w.slope_score.resize(length);
    if(w.slope_weight.size()<length) w.slope_weight.resize(length);
    // Per-row threshold terms are stored threshold-major (column r holds
    // row r), and the tridiagonal information is accumulated in vectors:
    // writing the diagonal of the dense information for every row and
    // threshold touches a new cache line each time.
    w.rowsums.leftCols(length).setZero();
    w.diag.setZero(q);w.off.setZero(std::max(q-1,0));
    w.slope_score.head(length).setZero();w.slope_weight.head(length).setZero();
    for(int r=0;r<length;++r) {
        const int i=first+r;const double m=data.totals[i];
        if(m==0) continue;
        // For the logit link the row is evaluated without per-pair
        // logarithms; log probabilities are formed only where needed.
        const bool lean=data.link==0;
        const double linear=lin ? lin[r] : data.linear(i,beta)+data.off(i);
        row.evaluate(alpha,linear,data.link,!lean,rm ? rm+static_cast<std::ptrdiff_t>(r)*3*q : nullptr);
        const double row_min=lean ? row.logit_min_logp() : row.logp.minCoeff();
        if(!std::isfinite(row_min))
            throw OrdinalEvaluationError("non-finite ordinal probabilities or derivatives");
        a.min_logp=std::min(a.min_logp,row_min);
        if(data.grouped) {
            for(int k=0;k<data.K;++k) {
                const double y=data.weight(i)*data.count(i,k);
                if(lean) { if(y!=0) a.loglik+=y*row.logit_logp(k); }
                else a.loglik+=y*row.logp[k];
                w.slope_score[r]-=y*row.location_score(k);
                if(k<q) a.v[k]+=y*row.left[k];
                if(k>0) a.v[k-1]-=y*row.right[k-1];
            }
        } else {
            const int k=data.code(i)-1;
            a.loglik+=m*(lean ? row.logit_logp(k) : row.logp[k]);
            w.slope_score[r]=-m*row.location_score(k);
            if(k<q) a.v[k]+=m*row.left[k];
            if(k>0) a.v[k-1]-=m*row.right[k-1];
        }
        double* terms=w.rowsums.col(r).data();
        double weight=0;
        for(int s=0;s<q;++s) {
            w.diag[s]+=m*row.g[s]*(row.left[s]+row.right[s]);
            terms[s]=m*row.g[s]*row.neighbor[s];
            weight+=terms[s];
            if(s+1<q) w.off[s]-=m*row.g[s+1]*row.right[s];
        }
        w.slope_weight[r]+=weight;
    }
}

inline void ordinal_add_information(const OrdinalData& data,const VectorXd& parameters,OrdinalState& result,
    OrdinalWorkspace* workspace=nullptr) {
    const int p=data.p,q=data.q,d=p+q,n=data.n;
    const VectorXd alpha=parameters.head(q),beta=parameters.tail(p);
    OrdinalWorkspace local;
    OrdinalWorkspace& ws=workspace ? *workspace : local;
    const bool lanes=fglm_par::use_lanes(d);
    const int rows=ordinal_block_rows(data.block,d,lanes,data.link,q);
    const Eigen::Index nb=(Eigen::Index(n)+rows-1)/rows;
    const bool staged=data.link==1;
    auto stage=[&](Eigen::Index g,Eigen::Index count) {
        if(staged) ordinal_stage_expected(data,alpha,beta,g,count,rows,ws);
    };
    // The staged predictors and probit terms of block b (slot b mod 8).
    auto staged_rows=[&](Eigen::Index b,const double*& lin,const double*& rm) {
        lin=rm=nullptr;
        if(!staged) return;
        const std::ptrdiff_t at=static_cast<std::ptrdiff_t>(b%fglm_par::kLanes)*rows;
        lin=ws.linear.data()+at;rm=ws.rmath.data()+at*3*q;
    };
    if(lanes) {
        ordinal_size_lanes(ws,nb,d,q,true);
        ordinal_lanes(nb,ws,alpha,
            [&](OrdinalAcc& a) { ordinal_reset_acc(a,true); },
            [&](Eigen::Index b,OrdinalAcc& a,OrdinalWorkHandle& h) {
                OrdinalWork& w=*h.w;
                const int first=static_cast<int>(b*rows),length=std::min(rows,n-first);
                const double *lin,*rm;staged_rows(b,lin,rm);
                ordinal_information_rows(data,alpha,beta,first,length,h.row,w,a,lin,rm);
                for(int s=0;s<q;++s) {
                    a.M(s,s)+=w.diag[s];
                    if(s+1<q) a.M(s,s+1)+=w.off[s];
                }
                if(p) {
                    const auto xx=data.x.middleRows(first,length);
                    a.v.tail(p).noalias()+=xx.transpose()*w.slope_score.head(length);
                    a.M.topRightCorner(q,p).noalias()-=w.rowsums.leftCols(length)*xx;
                    if(w.weighted.rows()<length || w.weighted.cols()!=p) w.weighted.resize(length,p);
                    w.weighted.topRows(length).noalias()=w.slope_weight.head(length).asDiagonal()*xx;
                    a.M.bottomRightCorner(p,p).noalias()+=xx.transpose()*w.weighted.topRows(length);
                }
                return true;
            },
            [&](OrdinalAcc&,const OrdinalAcc& a) {
                result.information+=a.M;
                result.ordinary+=a.v;
                result.loglik+=a.loglik;
                result.min_logp=std::min(result.min_logp,a.min_logp);
            },staged,stage);
        return;
    }
    // Column tiles over the p slope columns (d > kLaneMaxDim).
    const Eigen::Index nslots=std::min<Eigen::Index>(fglm_par::kGroupBlocks,nb);
    if(static_cast<Eigen::Index>(ws.slots.size())<nslots) ws.slots.resize(nslots);
    for(Eigen::Index j=0;j<nslots;++j) {
        OrdinalWork& w=ws.slots[j];
        w.row.reset(new OrdinalRow(alpha));
        if(w.part.v.size()!=d) w.part.v.resize(d);
    }
    Eigen::Index failed=-1;
    const bool ok=fglm_par::tile_reduce_staged(nb,fglm_par::tile_count(p),ws.slots,stage,
        [&](Eigen::Index b,OrdinalWork& w) {
            w.first=static_cast<int>(b*rows);w.length=std::min(rows,n-w.first);
            ordinal_reset_acc(w.part,false);
            const double *lin,*rm;staged_rows(b,lin,rm);
            ordinal_information_rows(data,alpha,beta,w.first,w.length,*w.row,w,w.part,lin,rm);
            if(w.weighted.rows()<w.length || w.weighted.cols()!=p) w.weighted.resize(w.length,p);
            w.weighted.topRows(w.length).noalias()=w.slope_weight.head(w.length).asDiagonal()*
                data.x.middleRows(w.first,w.length);
            return true;
        },
        [&](const OrdinalWork& w,Eigen::Index t) {
            const Eigen::Index c0=t*fglm_par::kTileWidth;
            const Eigen::Index tw=std::min<Eigen::Index>(fglm_par::kTileWidth,p-c0);
            const auto xx=data.x.middleRows(w.first,w.length);
            result.information.block(q,q+c0,p,tw).noalias()+=
                xx.transpose()*w.weighted.topRows(w.length).middleCols(c0,tw);
            result.information.block(0,q+c0,q,tw).noalias()-=
                w.rowsums.leftCols(w.length)*xx.middleCols(c0,tw);
        },
        [&](const OrdinalWork& w) {
            result.loglik+=w.part.loglik;
            result.min_logp=std::min(result.min_logp,w.part.min_logp);
            result.ordinary.head(q)+=w.part.v.head(q);
            for(int s=0;s<q;++s) {
                result.information(s,s)+=w.diag[s];
                if(s+1<q) result.information(s,s+1)+=w.off[s];
            }
            result.ordinary.tail(p).noalias()+=data.x.middleRows(w.first,w.length).transpose()*
                w.slope_score.head(w.length);
        },&failed);
    if(!ok) Rcpp::stop("ordinal block reduction failed");
}

// Solves a symmetric system whose threshold block is tridiagonal with a unit
// diagonal. Eliminating thresholds first is the same operation as an unpivoted
// Cholesky factor of the whole matrix, at far lower cost with many categories.
struct OrdinalBlockSystem {
    VectorXd pivot,sub;
    MatrixXd reduced;
    Eigen::LLT<MatrixXd> schur;
    int q=0,p=0;
    bool ok=false;
    void tridiagonal_solve(double* x) const {
        x[0]/=pivot[0];
        for(int s=1;s<q;++s) x[s]=(x[s]-sub[s-1]*x[s-1])/pivot[s];
        x[q-1]/=pivot[q-1];
        for(int s=q-2;s>=0;--s) x[s]=(x[s]-sub[s]*x[s+1])/pivot[s];
    }
    OrdinalBlockSystem(const VectorXd& off,const MatrixXd& cross,const MatrixXd& slopes) :
        pivot(cross.rows()),sub(std::max<Eigen::Index>(cross.rows()-1,0)),reduced(cross),
        q(cross.rows()),p(cross.cols()) {
        double previous=0;
        for(int s=0;s<q;++s) {
            const double value=1-previous*previous;
            if(!(value>0) || !std::isfinite(value)) return;
            pivot[s]=std::sqrt(value);
            if(s+1<q) { sub[s]=off[s]/pivot[s];previous=sub[s]; }
        }
        for(int j=0;j<p;++j) tridiagonal_solve(reduced.col(j).data());
        if(p) {
            MatrixXd difference=slopes;
            difference.noalias()-=cross.transpose()*reduced;
            schur.compute(difference);
            if(schur.info()!=Eigen::Success) return;
        }
        ok=true;
    }
    VectorXd solve(const VectorXd& right,const MatrixXd& cross) const {
        VectorXd out=right;
        tridiagonal_solve(out.data());
        if(p) {
            const VectorXd slopes=schur.solve(right.tail(p)-cross.transpose()*out.head(q));
            out.head(q).noalias()-=reduced*slopes;
            out.tail(p)=slopes;
        }
        return out;
    }
    MatrixXd inverse() const {
        MatrixXd out=MatrixXd::Zero(q+p,q+p);
        for(int s=0;s<q;++s) { out(s,s)=1;tridiagonal_solve(out.col(s).data()); }
        if(p) {
            const MatrixXd slopes=schur.solve(MatrixXd::Identity(p,p));
            const MatrixXd mixed=reduced*slopes;
            out.topLeftCorner(q,q).noalias()+=mixed*reduced.transpose();
            out.topRightCorner(q,p)=-mixed;
            out.bottomLeftCorner(p,q)=-mixed.transpose();
            out.bottomRightCorner(p,p)=slopes;
        }
        return out;
    }
};

inline void ordinal_factor_information(OrdinalState& result,int p,int q,double condition_tol) {
    for(int s=1;s<q;++s) result.information(s,s-1)=result.information(s-1,s);
    if(p) result.information.bottomLeftCorner(p,q)=result.information.topRightCorner(q,p).transpose();
    const VectorXd diagonal=result.information.diagonal().array().sqrt();
    if(!diagonal.allFinite() || (diagonal.array()<=0).any() || !std::isfinite(result.loglik))
        throw OrdinalEvaluationError("nonpositive or non-finite ordinal information");
    const VectorXd inverse_scale=diagonal.cwiseInverse();
    VectorXd off(std::max(q-1,0));
    for(int s=1;s<q;++s) off[s-1]=result.information(s,s-1)*inverse_scale[s]*inverse_scale[s-1];
    const MatrixXd cross=inverse_scale.head(q).asDiagonal()*result.information.topRightCorner(q,p)*
        inverse_scale.tail(p).asDiagonal();
    const MatrixXd slopes=inverse_scale.tail(p).asDiagonal()*result.information.bottomRightCorner(p,p)*
        inverse_scale.tail(p).asDiagonal();
    const OrdinalBlockSystem factor(off,cross,slopes);
    if(!factor.ok) throw OrdinalEvaluationError("ordinal information is not positive definite");
    const MatrixXd correlation_inverse=factor.inverse();
    // One-norm condition number of the correlation-scaled information.
    double norm=0;
    for(int s=0;s<q;++s) norm=std::max(norm,1+(s>0 ? std::abs(off[s-1]) : 0)+(s+1<q ? std::abs(off[s]) : 0)+
        (p ? cross.row(s).cwiseAbs().sum() : 0));
    for(int j=0;j<p;++j) norm=std::max(norm,cross.col(j).cwiseAbs().sum()+slopes.col(j).cwiseAbs().sum());
    result.rcond=1/(norm*correlation_inverse.cwiseAbs().colwise().sum().maxCoeff());
    if(!std::isfinite(result.rcond) || result.rcond<condition_tol)
        throw OrdinalEvaluationError("ordinal information is numerically singular");
    result.inverse=inverse_scale.asDiagonal()*correlation_inverse*inverse_scale.asDiagonal();
    result.inverse=(result.inverse+result.inverse.transpose()).eval()/2;
}

// The bias adjustment needs only vector sums (d entries), so it always uses
// lanes; the per-thread scratch holds the block's products with the inverse
// information.
inline void ordinal_add_adjustment(const OrdinalData& data,const VectorXd& parameters,OrdinalState& result,
    OrdinalWorkspace* workspace=nullptr) {
    const int p=data.p,q=data.q,d=p+q,n=data.n;
    const VectorXd alpha=parameters.head(q),beta=parameters.tail(p);
    OrdinalWorkspace local;
    OrdinalWorkspace& ws=workspace ? *workspace : local;
    const int rows=ordinal_block_rows(data.block,d,true,data.link,q);
    const Eigen::Index nb=(Eigen::Index(n)+rows-1)/rows;
    const bool staged=data.link==1;
    ordinal_size_lanes(ws,nb,d,q,false);
    ordinal_lanes(nb,ws,alpha,
        [&](OrdinalAcc& a) { ordinal_reset_acc(a,false); },
        [&](Eigen::Index b,OrdinalAcc& a,OrdinalWorkHandle& h) {
            OrdinalWork& w=*h.w;OrdinalRow& row=h.row;
            const int first=static_cast<int>(b*rows),length=std::min(rows,n-first);
            const std::ptrdiff_t at=static_cast<std::ptrdiff_t>(b%fglm_par::kLanes)*rows;
            const auto xx=data.x.middleRows(first,length);
            if(p) {
                if(w.xb.rows()<length || w.xb.cols()!=p) w.xb.resize(length,p);
                if(w.xa.rows()<length || w.xa.cols()!=q) w.xa.resize(length,q);
                w.xb.topRows(length).noalias()=xx*result.inverse.bottomRightCorner(p,p);
                w.xa.topRows(length).noalias()=xx*result.inverse.bottomLeftCorner(p,q);
            }
            if(w.slope_score.size()<length) w.slope_score.resize(length);
            w.slope_score.head(length).setZero();
            w.c.resize(q);
            for(int r=0;r<length;++r) {
                const int i=first+r;const double m=data.totals[i];
                if(m==0) continue;
                const double linear=staged ? ws.linear[at+r] : data.linear(i,beta)+data.off(i);
                row.evaluate(alpha,linear,data.link,true,staged ? ws.rmath.data()+(at+r)*3*q : nullptr);
                const double quadratic=p ? w.xb.row(r).head(p).dot(xx.row(r)) : 0;
                for(int s=0;s<q;++s) {
                    const double v=result.inverse(s,s)+quadratic-(p ? 2*w.xa(r,s) : 0);
                    w.c[s]=data.link==0 ? .5*m*row.g[s]*(1-2*row.cdf[s])*v : .5*m*row.gprime[s]*v;
                    w.slope_score[r]-=w.c[s]*row.neighbor[s];
                }
                for(int s=0;s<q;++s)
                    a.v[s]+=(w.c[s]-(s ? w.c[s-1] : 0))*row.left[s]-
                        ((s+1<q ? w.c[s+1] : 0)-w.c[s])*row.right[s];
            }
            if(p) a.v.tail(p).noalias()+=xx.transpose()*w.slope_score.head(length);
            return true;
        },
        [&](OrdinalAcc&,const OrdinalAcc& a) { result.adjustment+=a.v; },
        staged,
        [&](Eigen::Index g,Eigen::Index count) {
            if(staged) ordinal_stage_expected(data,alpha,beta,g,count,rows,ws);
        });
}

// A source completes the information pass before rereading rows for the bias
// adjustment. Dense and streamed fits use these same mathematical kernels.
template<class Data>
OrdinalState ordinal_evaluate(const Data& data,const VectorXd& parameters,
    bool firth,double condition_tol,OrdinalWorkspace* workspace=nullptr) {
    OrdinalWorkspace local;
    OrdinalWorkspace* ws=workspace ? workspace : &local;
    const int p=data.p,q=data.q,d=data.d;
    if(parameters.size()!=d || !parameters.allFinite())
        throw OrdinalEvaluationError("invalid ordinal parameters");
    for(int s=1;s<q;++s) if(!(parameters[s]>parameters[s-1]))
        throw OrdinalEvaluationError("ordinal thresholds are not strictly ordered");
    OrdinalState result(d);
    data.visit([&](const OrdinalData& block) { ordinal_add_information(block,parameters,result,ws); });
    ordinal_factor_information(result,p,q,condition_tol);
    if(firth) data.visit([&](const OrdinalData& block) { ordinal_add_adjustment(block,parameters,result,ws); });
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

// Logit terms simplify exactly. With a = f_u/P and b = f_l/P, the location
// curvature reduces to (a+b)P, so no term cancels in the tails. G is 1/expm1(gap).
inline ObservedTerms observed_logit(int k,int q,double eu,double el,double G,double loggap,double one_minus) {
    ObservedTerms t;
    double Fu=1,Su=0,logFu=0,Fl=0,Sl=1,logSl=0;
    if(k<q) {
        const double x=std::exp(-std::abs(eu)),den=1+x,l=std::log1p(x);
        Fu=eu>=0 ? 1/den : x/den;Su=eu>=0 ? x/den : 1/den;logFu=eu>=0 ? -l : eu-l;
    }
    if(k>0) {
        const double x=std::exp(-std::abs(el)),den=1+x,l=std::log1p(x);
        Fl=el>=0 ? 1/den : x/den;Sl=el>=0 ? x/den : 1/den;logSl=el>=0 ? -el-l : -l;
    }
    const bool interior=k>0 && k<q;
    const double gap_term=interior ? G : 0;
    const double P=interior ? Fu*Sl*one_minus : (k==0 ? Fu : Sl);
    t.logp=logFu+logSl+(interior ? loggap : 0);
    t.left=k<q ? Su+gap_term : 0;t.right=k>0 ? Fl+gap_term : 0;t.location=Su-Fl;
    t.upper=t.left*(gap_term+Fu);t.lower=t.right*(gap_term+Sl);t.cross=-t.left*t.right;
    t.upper_x=-t.left*P;t.lower_x=-t.right*P;t.slope=(t.left+t.right)*P;
    return t;
}

// ru and rl, when given for the probit link, hold ordinal_probit_terms() at
// eu and el, computed by a stage outside the parallel region.
inline ObservedTerms observed_general(int k,int q,int link,double eu,double el,double gap,double one_minus,
    const double* ru=nullptr,const double* rl=nullptr) {
    ObservedTerms t;
    double logFu=0,logSu=0,logFl=0,logSl=0,logGu=0,logGl=0,hu=0,hl=0;
    const auto read=[&](double e,double& logF,double& logS,double& logG,double& h,const double* r) {
        if(link==1) {
            if(r) { logF=r[0];logS=r[1];logG=r[2]; }
            else {
                Rf_pnorm_both(e,&logF,&logS,2,1);
                logG=R::dnorm(e,0,1,true);
            }
            h=-e;
        } else {
            const double x=std::exp(e);
            logF=e<-36 ? e : ordinal_log1mexp(-x);logS=-x;logG=e-x;h=1-x;
        }
    };
    if(k<q) read(eu,logFu,logSu,logGu,hu,ru);
    if(k>0) read(el,logFl,logSl,logGl,hl,rl);
    if(k==0) {
        t.logp=logFu;t.left=std::exp(logGu-t.logp);t.location=t.left;
    } else if(k==q) {
        t.logp=logSl;t.right=link==2 ? std::exp(el) : std::exp(logGl-t.logp);t.location=-t.right;
    } else {
        double logden=0;
        if(link==2) {
            const double ld=eu+std::log(one_minus);
            logden=ld<-36 ? ld : std::log(-std::expm1(-std::exp(ld)));
        }
        t.logp=eu<=0 ? logFu+ordinal_log1mexp(logFl-logFu) :
            logSl+(link==2 ? logden : ordinal_log1mexp(logSu-logSl));
        if(link==2) {
            const double difference=std::exp(eu)*one_minus;
            t.left=std::exp(eu-difference-logden);t.right=std::exp(el-logden);
        } else {
            t.left=std::exp(logGu-t.logp);t.right=std::exp(logGl-t.logp);
        }
        const double delta=link==1 ? -.5*gap*(eu+el) : gap-std::exp(eu)*one_minus;
        t.location=delta>=0 ? t.left*(-std::expm1(-delta)) : -t.right*(-std::expm1(delta));
    }
    t.upper=t.left*(t.left-hu);t.lower=t.right*(t.right+hl);t.cross=-t.left*t.right;
    t.upper_x=t.left*(hu-t.location);t.lower_x=t.right*(t.location-hl);
    t.slope=t.location*t.location-t.left*hu+t.right*hl;
    return t;
}

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
    // rm, when given for the probit link, holds ordinal_probit_terms() at
    // alpha[s] - linear in entries 3 s, 3 s + 1, 3 s + 2 for the thresholds
    // s = k - 1 and k that bound category k.
    ObservedTerms terms(int k,double linear,const double* rm=nullptr) const {
        const double eu=k<q ? alpha[k]-linear : 0,el=k>0 ? alpha[k-1]-linear : 0;
        const bool interior=k>0 && k<q;
        if(link==0) return observed_logit(k,q,eu,el,interior ? inverse_gap[k] : 0,interior ? loggap[k] : 0,interior ? one_minus[k] : 0);
        return observed_general(k,q,link,eu,el,interior ? alpha[k]-alpha[k-1] : 0,interior ? one_minus[k] : 0,
            rm && k<q ? rm+3*k : nullptr,rm && k>0 ? rm+3*(k-1) : nullptr);
    }
};

struct ObservedState {
    VectorXd diagonal,off,score;
    MatrixXd cross,slopes;
    double loglik=0,change=0,row_norm=0;
    ObservedState(int q=0,int p=0) : diagonal(VectorXd::Zero(q)),off(VectorXd::Zero(std::max(q-1,0))),
        score(VectorXd::Zero(q+p)),cross(MatrixXd::Zero(q,p)),slopes(MatrixXd::Zero(p,p)) {}
    MatrixXd information() const {
        const int q=diagonal.size(),p=slopes.rows();
        MatrixXd out=MatrixXd::Zero(q+p,q+p);
        out.diagonal().head(q)=diagonal;
        for(int s=1;s<q;++s) out(s,s-1)=out(s-1,s)=off[s-1];
        out.topRightCorner(q,p)=cross;out.bottomLeftCorner(p,q)=cross.transpose();
        out.bottomRightCorner(p,p)=slopes;
        return out;
    }
};

// Stage of the observed pass for the probit link: for the blocks [g, g +
// count) of rows rows each (slot j = block - g), the cumulative predictor of
// every positive-frequency row (from the block's product with beta, computed
// in parallel as the pass computes it) into ws.linear, and on the calling
// thread the probit terms at the thresholds that bound the row's observed
// categories (every threshold for a grouped row) into ws.rmath.
inline void ordinal_stage_observed(const OrdinalData& data,const ObservedThresholds& thresholds,
    const VectorXd& beta,Eigen::Index g,Eigen::Index count,int rows,OrdinalWorkspace& ws) {
    const int q=data.q,p=data.p;
    const Eigen::Index r0=g*rows,r1=std::min<Eigen::Index>(data.n,(g+count)*rows),m=r1-r0;
    const Eigen::Index cap=Eigen::Index(fglm_par::kLanes)*rows;
    if(ws.linear.size()<cap) ws.linear.resize(cap);
    if(ws.rmath.size()<cap*3*q) ws.rmath.resize(cap*3*q);
    fglm_par::parallel_blocks(m,rows,[&](Eigen::Index r,Eigen::Index k) {
        VectorXd eta(k);
        if(p) eta.noalias()=data.x.middleRows(r0+r,k)*beta; else eta.setZero();
        for(Eigen::Index j=0;j<k;++j) {
            const int i=static_cast<int>(r0+r+j);
            if(data.totals[i]>0) ws.linear[r+j]=eta[j]+data.off(i);
        }
    });
    for(Eigen::Index j=0;j<m;++j) {
        const int i=static_cast<int>(r0+j);
        if(!(data.totals[i]>0)) continue;
        const double linear=ws.linear[j];
        double* rm=ws.rmath.data()+j*3*q;
        if(data.grouped) {
            for(int s=0;s<q;++s) ordinal_probit_terms(thresholds.alpha[s]-linear,rm+3*s);
        } else {
            const int k=data.code(i)-1;
            if(k<q) ordinal_probit_terms(thresholds.alpha[k]-linear,rm+3*k);
            if(k>0) ordinal_probit_terms(thresholds.alpha[k-1]-linear,rm+3*(k-1));
        }
    }
}

// The rows of the observed pass for the block of `length` rows from `first`:
// adds the log likelihood, threshold score, tridiagonal curvature, and the
// largest predictor change to a, and fills the slope score and weight of
// each row and the threshold-slope terms (w.rows for grouped counts, else
// w.upper_x, w.lower_x, and the category of each row). lin and rm, when
// given, are the staged predictors and probit terms of the block's rows.
inline void observed_rows(const OrdinalData& data,const ObservedThresholds& thresholds,const VectorXd& beta,
    const VectorXd* direction,int first,int length,OrdinalWork& w,OrdinalAcc& a,const double* lin,const double* rm) {
    const int p=data.p,q=data.q;
    const auto xx=data.x.middleRows(first,length);
    if(!lin) {
        w.eta.resize(length);
        if(p) w.eta.noalias()=xx*beta; else w.eta.setZero();
    }
    if(direction && p) {
        const double low=direction->head(q).minCoeff(),high=direction->head(q).maxCoeff();
        w.change.resize(length);
        w.change.noalias()=xx*direction->tail(p);
        for(int r=0;r<length;++r) if(data.totals[first+r]>0)
            a.change=std::max(a.change,std::max(std::abs(low-w.change[r]),std::abs(high-w.change[r])));
    } else if(direction) a.change=std::max(a.change,direction->head(q).cwiseAbs().maxCoeff());
    w.slope_score.setZero(length);w.slope_weight.setZero(length);
    const auto row_rm=[&](int r) { return rm ? rm+static_cast<std::ptrdiff_t>(r)*3*q : nullptr; };
    if(data.grouped) {
        w.rows.setZero(length,q);
        for(int r=0;r<length;++r) {
            const int i=first+r;
            if(data.totals[i]==0) continue;
            const double linear=lin ? lin[r] : w.eta[r]+data.off(i);
            for(int k=0;k<data.K;++k) {
                const double y=data.weight(i)*data.count(i,k);
                if(y==0) continue;
                const ObservedTerms t=thresholds.terms(k,linear,row_rm(r));
                a.loglik+=y*t.logp;w.slope_score[r]-=y*t.location;w.slope_weight[r]+=y*t.slope;
                if(k<q) {
                    a.v[k]+=y*t.left;a.diag[k]+=y*t.upper;w.rows(r,k)+=y*t.upper_x;
                }
                if(k>0) {
                    a.v[k-1]-=y*t.right;a.diag[k-1]+=y*t.lower;w.rows(r,k-1)+=y*t.lower_x;
                    if(k<q) a.off[k-1]+=y*t.cross;
                }
            }
        }
    } else {
        w.upper_x.setZero(length);w.lower_x.setZero(length);w.category.resize(length);
        for(int r=0;r<length;++r) {
            const int i=first+r;const double y=data.totals[i];
            w.category[r]=-1;
            if(y==0) continue;
            const int k=data.code(i)-1;
            w.category[r]=k;
            const ObservedTerms t=thresholds.terms(k,lin ? lin[r] : w.eta[r]+data.off(i),row_rm(r));
            a.loglik+=y*t.logp;w.slope_score[r]=-y*t.location;w.slope_weight[r]=y*t.slope;
            w.upper_x[r]=y*t.upper_x;w.lower_x[r]=y*t.lower_x;
            if(k<q) {a.v[k]+=y*t.left;a.diag[k]+=y*t.upper;}
            if(k>0) {
                a.v[k-1]-=y*t.right;a.diag[k-1]+=y*t.lower;
                if(k<q) a.off[k-1]+=y*t.cross;
            }
        }
    }
    if(p && w.slope_weight.head(length).minCoeff()<0) throw OrdinalEvaluationError("negative observed curvature");
}

// Adds the threshold-slope cross terms of a block prepared by observed_rows to
// the q by p matrix cross for the slope columns [c0, c0 + width).
template<class Cross>
inline void observed_add_cross(const OrdinalData& data,const OrdinalWork& w,int first,int length,
    Eigen::Index c0,Eigen::Index width,Cross&& cross) {
    const int q=data.q;
    const auto xx=data.x.middleRows(first,length);
    if(data.grouped) {
        cross.middleCols(c0,width).noalias()+=w.rows.transpose()*xx.middleCols(c0,width);
        return;
    }
    for(Eigen::Index j=c0;j<c0+width;++j) {
        double* column=cross.col(j).data();
        for(int r=0;r<length;++r) {
            const int k=w.category[r];
            if(k<0) continue;
            const double value=xx(r,j);
            if(k<q) column[k]+=w.upper_x[r]*value;
            if(k>0) column[k-1]+=w.lower_x[r]*value;
        }
    }
}

// Adds one data segment. `direction` is optional and records the largest change in
// a cumulative predictor, computed in this pass so a source needs no second read.
inline void observed_add(const OrdinalData& data,const ObservedThresholds& thresholds,const VectorXd& beta,
    const VectorXd* direction,ObservedState& out,OrdinalWorkspace* workspace=nullptr) {
    const int p=data.p,q=data.q,d=p+q,n=data.n;
    OrdinalWorkspace local;
    OrdinalWorkspace& ws=workspace ? *workspace : local;
    const bool lanes=fglm_par::use_lanes(d);
    const int rows=ordinal_block_rows(data.block,d,lanes,data.link,q);
    const Eigen::Index nb=(Eigen::Index(n)+rows-1)/rows;
    const bool staged=data.link==1;
    auto stage=[&](Eigen::Index g,Eigen::Index count) {
        if(staged) ordinal_stage_observed(data,thresholds,beta,g,count,rows,ws);
    };
    auto staged_rows=[&](Eigen::Index b,const double*& lin,const double*& rm) {
        lin=rm=nullptr;
        if(!staged) return;
        const std::ptrdiff_t at=static_cast<std::ptrdiff_t>(b%fglm_par::kLanes)*rows;
        lin=ws.linear.data()+at;rm=ws.rmath.data()+at*3*q;
    };
    if(lanes) {
        ordinal_size_lanes(ws,nb,d,q,true);
        ordinal_lanes(nb,ws,thresholds.alpha,
            [&](OrdinalAcc& a) { ordinal_reset_acc(a,true); },
            [&](Eigen::Index b,OrdinalAcc& a,OrdinalWorkHandle& h) {
                OrdinalWork& w=*h.w;
                const int first=static_cast<int>(b*rows),length=std::min(rows,n-first);
                const double *lin,*rm;staged_rows(b,lin,rm);
                observed_rows(data,thresholds,beta,direction,first,length,w,a,lin,rm);
                if(p) {
                    const auto xx=data.x.middleRows(first,length);
                    observed_add_cross(data,w,first,length,0,p,a.M.topRightCorner(q,p));
                    a.v.tail(p).noalias()+=xx.transpose()*w.slope_score.head(length);
                    if(w.weighted.rows()<length || w.weighted.cols()!=p) w.weighted.resize(length,p);
                    w.weighted.topRows(length).noalias()=w.slope_weight.head(length).cwiseSqrt().asDiagonal()*xx;
                    a.M.bottomRightCorner(p,p).selfadjointView<Eigen::Lower>().rankUpdate(
                        w.weighted.topRows(length).transpose());
                    for(int r=0;r<length;++r) if(data.totals[first+r]>0)
                        a.row_norm=std::max(a.row_norm,xx.row(r).norm());
                }
                return true;
            },
            [&](OrdinalAcc&,const OrdinalAcc& a) {
                out.loglik+=a.loglik;
                out.score+=a.v;
                out.diagonal+=a.diag;
                out.off+=a.off;
                if(p) {
                    out.cross+=a.M.topRightCorner(q,p);
                    out.slopes.triangularView<Eigen::Lower>()+=a.M.bottomRightCorner(p,p);
                }
                out.change=std::max(out.change,a.change);
                out.row_norm=std::max(out.row_norm,a.row_norm);
            },staged,stage);
        return;
    }
    // Column tiles over the p slope columns (d > kLaneMaxDim).
    const Eigen::Index nslots=std::min<Eigen::Index>(fglm_par::kGroupBlocks,nb);
    if(static_cast<Eigen::Index>(ws.slots.size())<nslots) ws.slots.resize(nslots);
    for(Eigen::Index j=0;j<nslots;++j) {
        OrdinalAcc& part=ws.slots[j].part;
        if(part.v.size()!=d) part.v.resize(d);
        if(part.diag.size()!=q) part.diag.resize(q);
        if(part.off.size()!=std::max(q-1,0)) part.off.resize(std::max(q-1,0));
    }
    Eigen::Index failed=-1;
    const bool ok=fglm_par::tile_reduce_staged(nb,fglm_par::tile_count(p),ws.slots,stage,
        [&](Eigen::Index b,OrdinalWork& w) {
            w.first=static_cast<int>(b*rows);w.length=std::min(rows,n-w.first);
            ordinal_reset_acc(w.part,false);
            const double *lin,*rm;staged_rows(b,lin,rm);
            observed_rows(data,thresholds,beta,direction,w.first,w.length,w,w.part,lin,rm);
            const auto xx=data.x.middleRows(w.first,w.length);
            if(w.weighted.rows()<w.length || w.weighted.cols()!=p) w.weighted.resize(w.length,p);
            w.weighted.topRows(w.length).noalias()=w.slope_weight.head(w.length).cwiseSqrt().asDiagonal()*xx;
            for(int r=0;r<w.length;++r) if(data.totals[w.first+r]>0)
                w.part.row_norm=std::max(w.part.row_norm,xx.row(r).norm());
            return true;
        },
        // Tile t covers slope columns c0 to c0 + tw - 1: the lower triangle of
        // the slope block (the tw x tw diagonal block by a rank update and the
        // rows below it by a product) and the threshold-slope cross terms.
        [&](const OrdinalWork& w,Eigen::Index t) {
            const Eigen::Index c0=t*fglm_par::kTileWidth;
            const Eigen::Index tw=std::min<Eigen::Index>(fglm_par::kTileWidth,p-c0);
            const Eigen::Index below=p-c0-tw;
            const auto root=w.weighted.topRows(w.length);
            out.slopes.block(c0,c0,tw,tw).selfadjointView<Eigen::Lower>().rankUpdate(root.middleCols(c0,tw).adjoint());
            if(below>0) out.slopes.block(c0+tw,c0,below,tw).noalias()+=root.rightCols(below).adjoint()*root.middleCols(c0,tw);
            observed_add_cross(data,w,w.first,w.length,c0,tw,out.cross);
        },
        [&](const OrdinalWork& w) {
            out.loglik+=w.part.loglik;
            out.score.head(q)+=w.part.v.head(q);
            out.diagonal+=w.part.diag;
            out.off+=w.part.off;
            out.score.tail(p).noalias()+=data.x.middleRows(w.first,w.length).transpose()*w.slope_score.head(w.length);
            out.change=std::max(out.change,w.part.change);
            out.row_norm=std::max(out.row_norm,w.part.row_norm);
        },&failed);
    if(!ok) Rcpp::stop("ordinal block reduction failed");
}

template<class Data>
ObservedState observed_evaluate(const Data& data,const VectorXd& parameters,const VectorXd* direction,
    OrdinalWorkspace* workspace=nullptr) {
    const int p=data.p,q=data.q;
    if(!parameters.allFinite()) throw OrdinalEvaluationError("invalid ordinal parameters");
    for(int s=1;s<q;++s) if(!(parameters[s]>parameters[s-1]))
        throw OrdinalEvaluationError("ordinal thresholds are not strictly ordered");
    const ObservedThresholds thresholds(parameters.head(q),data.link);
    const VectorXd beta=parameters.tail(p);
    ObservedState out(q,p);
    data.visit([&](const OrdinalData& block) { observed_add(block,thresholds,beta,direction,out,workspace); });
    if(!std::isfinite(out.loglik) || !out.score.allFinite() || !out.diagonal.allFinite() ||
       !out.off.allFinite() || !out.cross.allFinite() || !out.slopes.allFinite())
        throw OrdinalEvaluationError("non-finite observed ordinal terms");
    if(p) out.slopes.template triangularView<Eigen::StrictlyUpper>()=out.slopes.transpose();
    return out;
}

// Returns the iterations used. Any failure stops here, because the expected
// information loop continues from the last accepted parameters.
template<class Data>
int observed_newton(const Data& data,VectorXd& parameters,double score_tol,double step_tol,int maxit,
    double max_step,int max_halves,bool trace,int& evaluations,OrdinalWorkspace* workspace=nullptr) {
    const int q=data.q,p=data.p,d=data.d;
    int iterations=0;
    try {
        ObservedState current=observed_evaluate(data,parameters,nullptr,workspace);
        ++evaluations;
        double row_norm=current.row_norm;
        for(;iterations<maxit;++iterations) {
            Rcpp::checkUserInterrupt();
            VectorXd diagonal(d);
            diagonal.head(q)=current.diagonal;
            if(p) diagonal.tail(p)=current.slopes.diagonal();
            if((diagonal.array()<=0).any()) return iterations;
            const VectorXd inverse_scale=diagonal.cwiseSqrt().cwiseInverse();
            VectorXd off(std::max(q-1,0));
            for(int s=1;s<q;++s) off[s-1]=current.off[s-1]*inverse_scale[s]*inverse_scale[s-1];
            const MatrixXd cross=inverse_scale.head(q).asDiagonal()*current.cross*inverse_scale.tail(p).asDiagonal();
            const MatrixXd slopes=inverse_scale.tail(p).asDiagonal()*current.slopes*inverse_scale.tail(p).asDiagonal();
            const OrdinalBlockSystem factor(off,cross,slopes);
            if(!factor.ok) return iterations;
            const VectorXd direction=inverse_scale.cwiseProduct(factor.solve(inverse_scale.cwiseProduct(current.score),cross));
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
                    trial=observed_evaluate(data,candidate,measure ? &scaled_direction : nullptr,workspace);
                    ++evaluations;
                    if(measure && trial.change>max_step) { fraction*=max_step/trial.change;continue; }
                    // A predicted gain near the rounding error of a long sum cannot be
                    // measured, so comparing likelihoods would reject valid steps.
                    if(ascent<1024*rounding || trial.loglik>=current.loglik+1e-4*fraction*ascent-rounding) accepted=true;
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
    // Lane accumulators and scratch kept for every pass of the fit.
    OrdinalWorkspace workspace;
    // Bias reduction needs expected-information terms in every iteration.
    const int observed_iterations=!observed || firth || maxit<2 ? 0 :
        observed_newton(data,parameters,score_tol,step_tol,std::min(maxit-1,30),max_step,max_halves,trace,evaluations,
            &workspace);
    const int expected_limit=maxit-observed_iterations;
    // Newton can reach a point, such as near separation, where the expected
    // information cannot be evaluated. The scoring loop then starts as if Newton
    // had not run, so it reports the same non-convergence diagnostics.
    const auto evaluate_state=[&](const VectorXd& b) { ++evaluations;return ordinal_evaluate(data,b,firth,condition_tol,&workspace); };
    const auto first_state=[&]() {
        if(observed_iterations>0) {
            try { return evaluate_state(parameters); }
            catch(const OrdinalEvaluationError&) { parameters=start; }
        }
        return evaluate_state(parameters);
    };
    auto evaluate=[&](const VectorXd& b) { ++evaluations;return ordinal_evaluate(data,b,firth,condition_tol,&workspace); };
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
