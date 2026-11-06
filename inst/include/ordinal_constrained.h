#ifndef FASTGLM_ORDINAL_CONSTRAINED_H
#define FASTGLM_ORDINAL_CONSTRAINED_H
#include "ordinal_partial.h"
#include <limits>
#include <string>
#include <vector>
// Damped Newton iterations for ordinal coefficients restricted to an affine
// subspace. The columns of `basis` span the free directions of the working
// coefficients, and each iterate stays in the subspace containing `start`.
namespace fglm_ordinal_constrained {
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct ConstrainedControl {
    int maxit,max_halves;
    double score_tol,step_tol,max_step;
};

inline MatrixXd positive_rows(const MatrixXd& x,const Rcpp::LogicalVector& positive) {
    std::vector<int> rows;
    for(int i=0;i<positive.size();++i) if(positive[i]==TRUE) rows.push_back(i);
    MatrixXd out(rows.size(),x.cols());
    for(int i=0;i<(int)rows.size();++i) out.row(i)=x.row(rows[i]);
    return out;
}

// Common-slope models share one predictor per row, so threshold order depends
// only on the thresholds.
struct CommonDomain {
    MatrixXd x;
    int q,p;
    double predictor_step(const VectorXd& delta) const {
        const double low=delta.head(q).minCoeff(),high=delta.head(q).maxCoeff();
        if(!p) return std::max(std::abs(low),std::abs(high));
        const VectorXd shift=x*delta.tail(p);
        double out=0;
        for(int i=0;i<shift.size();++i)
            out=std::max(out,std::max(std::abs(low-shift[i]),std::abs(high-shift[i])));
        return out;
    }
    double order_fraction(const VectorXd& gamma,const VectorXd& delta) const {
        double out=1;
        for(int j=1;j<q;++j) {
            const double change=delta[j]-delta[j-1];
            if(change<0) out=std::min(out,-.9*(gamma[j]-gamma[j-1])/change);
        }
        return out;
    }
};

// Threshold-specific slopes make the cumulative predictors row dependent. Sums
// are compensated because order gaps can cancel to rounding size.
struct PartialDomain {
    MatrixXd common,varying;
    int q,pc,pn;
    double row_sum(int row,double intercept,const VectorXd& a,double a_sign,
                   const double* c,double c_sign,const double* v,double v_sign) const {
        double sum=intercept,error=0;
        const auto add=[&](double product,double exact) {
            const double next=sum+product;
            error+=std::abs(sum)>=std::abs(product) ? (sum-next)+product : (product-next)+sum;
            error+=exact;
            sum=next;
        };
        for(int j=0;j<a.size();++j) {
            const double value=a_sign*common(row,j)*a[j];
            add(value,std::fma(a_sign*common(row,j),a[j],-value));
        }
        for(int j=0;j<pn;++j) {
            const double first=c_sign*varying(row,j)*c[j];
            add(first,std::fma(c_sign*varying(row,j),c[j],-first));
            if(v) {
                const double second=v_sign*varying(row,j)*v[j];
                add(second,std::fma(v_sign*varying(row,j),v[j],-second));
            }
        }
        return sum+error;
    }
    double predictor_step(const VectorXd& delta) const {
        const VectorXd slopes=delta.segment(q,pc);
        double out=0;
        for(int j=0;j<q;++j) {
            const double* s=delta.data()+q+pc+j*pn;
            for(int i=0;i<varying.rows();++i)
                out=std::max(out,std::abs(row_sum(i,delta[j],slopes,-1,s,-1,nullptr,0)));
        }
        return out;
    }
    double gap(int row,int j,const VectorXd& b) const {
        const double* lower=b.data()+q+pc+j*pn;
        const double* upper=lower+pn;
        const VectorXd none(0);
        double sum=b[j+1],error=-b[j];
        const double next=sum+error;
        error=std::abs(sum)>=std::abs(error) ? (sum-next)+error : (error-next)+sum;
        return row_sum(row,next,none,1,lower,1,upper,-1)+error;
    }
    double order_fraction(const VectorXd& gamma,const VectorXd& delta) const {
        double out=1;
        for(int j=0;j+1<q;++j) for(int i=0;i<varying.rows();++i) {
            const double change=gap(i,j,delta);
            if(change<0) out=std::min(out,-.9*gap(i,j,gamma)/change);
        }
        return out;
    }
};

template<class State,class Evaluate,class Domain,class Wrap>
Rcpp::List constrained_fit(const Evaluate& evaluate,const Domain& domain,const Wrap& wrap,
    const VectorXd& start,const MatrixXd& basis,bool adjusted,const ConstrainedControl& ctl) {
    int calls=0;
    const auto attempt=[&](const VectorXd& candidate,State& out,std::string& message) {
        ++calls;
        try { out=evaluate(candidate);return true; }
        catch(const std::exception& failure) { message=failure.what();return false; }
    };
    const double infinity=std::numeric_limits<double>::infinity();
    const auto finish=[&](const VectorXd& gamma,const State* state,int code,const std::string& message,
                          int iterations,double scaled,double step) {
        return Rcpp::List::create(Rcpp::_["gamma"]=gamma,
            Rcpp::_["state"]=state ? Rcpp::wrap(wrap(*state)) : R_NilValue,Rcpp::_["code"]=code,
            Rcpp::_["message"]=message,Rcpp::_["iterations"]=iterations,Rcpp::_["evaluations"]=calls,
            Rcpp::_["score"]=scaled,Rcpp::_["step"]=step);
    };
    VectorXd gamma=start;
    State current,trial;
    std::string failure;
    if(!attempt(gamma,current,failure)) return finish(gamma,nullptr,2,failure,0,infinity,infinity);
    const int d=gamma.size(),free=basis.cols();
    for(int iter=0;iter<=ctl.maxit;++iter) {
        VectorXd direction=VectorXd::Zero(d);
        double scaled=0,step=0;
        if(free) {
            const MatrixXd information=basis.transpose()*current.information*basis;
            const VectorXd score=basis.transpose()*current.score;
            bool solved=information.allFinite() && score.allFinite();
            if(solved) {
                const Eigen::PartialPivLU<MatrixXd> lu(information);
                // Matches the tolerance that solve() applies to the same system.
                solved=lu.rcond()>=std::numeric_limits<double>::epsilon();
                if(solved) direction.noalias()=basis*lu.solve(score);
            }
            scaled=-infinity;
            for(int j=0;j<free;++j) {
                const double value=std::abs(score[j])/std::sqrt(information(j,j));
                if(std::isnan(value)) { scaled=value;break; }
                scaled=std::max(scaled,value);
            }
            step=solved && direction.allFinite() ? domain.predictor_step(direction) : infinity;
        }
        if(std::isfinite(scaled) && scaled<=ctl.score_tol && step<=ctl.step_tol)
            return finish(gamma,&current,0,"",iter,scaled,step);
        if(!std::isfinite(step) || !std::isfinite(scaled))
            return finish(gamma,&current,3,"non-finite constrained score or information",iter,scaled,step);
        if(iter==ctl.maxit)
            return finish(gamma,&current,5,"constrained iteration limit reached",iter,scaled,step);
        direction/=std::max(1.0,step/(adjusted ? std::min(1.0,ctl.max_step) : ctl.max_step));
        double fraction=domain.order_fraction(gamma,direction);
        const double ascent=current.score.dot(direction);
        bool accepted=false;
        std::string rejected;
        VectorXd candidate(d);
        for(int half=0;half<=ctl.max_halves;++half) {
            candidate=gamma+fraction*direction;
            if(attempt(candidate,trial,rejected)) {
                const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(current.loglik));
                if(adjusted || trial.loglik-current.loglik>=.1*fraction*ascent-rounding) { accepted=true;break; }
                rejected="likelihood step did not increase sufficiently";
            }
            fraction/=2;
        }
        if(!accepted) return finish(gamma,&current,4,"constrained step search failed: "+rejected,iter+1,scaled,step);
        gamma=candidate;current=trial;
    }
    return finish(gamma,&current,5,"constrained iteration limit reached",ctl.maxit,infinity,infinity);
}
}
#endif
