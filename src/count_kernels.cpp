#include <RcppEigen.h>
#include <cmath>
#include <limits>
// [[Rcpp::depends(RcppEigen)]]
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
// Link codes follow the order log, sqrt, identity.
inline double link_inverse(double eta,int link) {
    if(link==0) return std::max(std::exp(eta),std::numeric_limits<double>::epsilon());
    return link==1 ? eta*eta : eta;
}
inline double link_derivative(double eta,int link) {
    if(link==0) return std::max(std::exp(eta),std::numeric_limits<double>::epsilon());
    return link==1 ? 2*eta : 1;
}

// Score for the size parameter. Above the large-size threshold, digamma
// differences lose accuracy, so a series in 1/theta replaces them.
inline double theta_score(double y,double mu,double theta,double digamma_theta,bool truncated) {
    double a;
    if(theta>1e4*std::max(1.0,std::max(y,mu))) {
        const double c2=(y-(y-mu)*(y-mu))/2;
        const double c3=y*(y-1)*(2*y-1)/6+2*mu*mu*mu/3-y*mu*mu;
        const double c4=-y*y*(y-1)*(y-1)/4-3*mu*mu*mu*mu/4+y*mu*mu*mu;
        a=(c2+(c3+c4/theta)/theta)/(theta*theta);
    } else a=R::digamma(y+theta)-digamma_theta-std::log1p(mu/theta)+(mu-y)/(theta+mu);
    if(truncated) {
        const double lp0=-theta*std::log1p(mu/theta),r=mu/theta;
        const double d=r<1e-4 ? r*r*(-.5+r*(2.0/3+r*(-.75+r*.8))) : -std::log1p(r)+r/(1+r);
        a+=std::exp(lp0)/(-std::expm1(lp0))*d;
    }
    return a;
}

// Accumulates x' diag(d) x. A nonnegative weight permits a symmetric rank update.
void add_weighted_crossproduct(const Map<MatrixXd>& x,const Eigen::VectorXi& rows,const VectorXd& weight,MatrixXd& out) {
    const int kept=rows.size(),p=x.cols();
    MatrixXd left(kept,p);
    for(int j=0;j<p;++j) for(int r=0;r<kept;++r) left(r,j)=x(rows[r],j);
    if(weight.size() && weight.minCoeff()>=0) {
        const MatrixXd root=weight.cwiseSqrt().asDiagonal()*left;
        MatrixXd lower=MatrixXd::Zero(p,p);
        lower.selfadjointView<Eigen::Lower>().rankUpdate(root.transpose());
        out+=MatrixXd(lower.selfadjointView<Eigen::Lower>());
    } else {
        const MatrixXd right=weight.asDiagonal()*left;
        out.noalias()+=left.transpose()*right;
    }
}
}

// One data block of the count likelihood, its derivatives for coefficients, and
// the log-size derivatives. Rows with zero weight do not contribute.
// [[Rcpp::export]]
Rcpp::List count_block_summary(Eigen::Map<Eigen::MatrixXd> x,Rcpp::NumericVector y,Rcpp::NumericVector weights,
    Rcpp::NumericVector offset,Eigen::Map<Eigen::VectorXd> beta,double theta,bool truncated,int link,bool system,bool derivatives) {
    const bool nb=!ISNAN(theta);
    const int n=x.rows(),p=x.cols();
    VectorXd eta=x*beta;
    Eigen::VectorXi rows(n);
    int kept=0;
    for(int i=0;i<n;++i) if(weights[i]>0 && (!truncated || y[i]>0)) rows[kept++]=i;
    rows.conservativeResize(kept);
    double loglik=0,theta_total=0,theta_information=0;
    VectorXd score=VectorXd::Zero(p),curvature(system ? kept : 0);
    MatrixXd information=MatrixXd::Zero(p,p);
    const double h=std::cbrt(std::numeric_limits<double>::epsilon());
    const double plus=nb ? theta*std::exp(h) : 0,minus=nb ? theta*std::exp(-h) : 0;
    const double digamma_theta=nb && derivatives ? R::digamma(theta) : 0;
    const double digamma_plus=nb && derivatives ? R::digamma(plus) : 0;
    const double digamma_minus=nb && derivatives ? R::digamma(minus) : 0;
    VectorXd row_score(system ? n : 0);
    if(system) row_score.setZero();
    for(int r=0;r<kept;++r) {
        const int i=rows[r];
        const double linear=eta[i]+offset[i],mu=link_inverse(linear,link),yi=y[i],w=weights[i];
        if(!std::isfinite(mu) || mu<=0) return Rcpp::List::create(Rcpp::_["valid"]=false);
        double density=nb ? R::dnbinom_mu(yi,theta,mu,1) : R::dpois(yi,mu,1);
        double lp0=0,q=1;
        if(truncated) {
            lp0=nb ? -theta*std::log1p(mu/theta) : -mu;
            q=-std::expm1(lp0);density-=std::log(q);
        }
        loglik+=w*density;
        if(system) {
            const double variance=nb ? mu+mu*mu/theta : mu;
            double u,iw;
            if(truncated) {
                const double mean=mu==0 ? 1 : mu/q;
                const double moment=std::max(0.0,(variance+mu*mu)/q-mean*mean);
                const double multiplier=nb ? theta/(theta+mu) : 1;
                u=multiplier*(yi-mean);iw=multiplier*multiplier*moment;
            } else {
                const double dm=link_derivative(linear,link);
                u=dm*(yi-mu)/variance;iw=dm*dm/variance;
            }
            row_score[i]=w*u;curvature[r]=w*iw;
        }
        if(derivatives) {
            theta_total+=w*theta*theta_score(yi,mu,theta,digamma_theta,truncated);
            const double d=(plus*theta_score(yi,mu,plus,digamma_plus,truncated)-
                minus*theta_score(yi,mu,minus,digamma_minus,truncated))/(2*h);
            theta_information-=w*d;
        }
    }
    if(system && kept) {
        score.noalias()=x.transpose()*row_score;
        add_weighted_crossproduct(x,rows,curvature,information);
    }
    return Rcpp::List::create(Rcpp::_["valid"]=true,Rcpp::_["loglik"]=loglik,Rcpp::_["score"]=score,
        Rcpp::_["information"]=information,Rcpp::_["theta.score"]=theta_total,
        Rcpp::_["theta.information"]=theta_information);
}

// One data block of the joint observed information for coefficients and log size.
// [[Rcpp::export]]
Rcpp::NumericMatrix count_block_nb_information(Eigen::Map<Eigen::MatrixXd> x,Rcpp::NumericVector y,Rcpp::NumericVector weights,
    Rcpp::NumericVector offset,Eigen::Map<Eigen::VectorXd> beta,double theta,int link) {
    const int n=x.rows(),p=x.cols();
    const VectorXd eta=x*beta;
    Eigen::VectorXi rows(n);
    int kept=0;
    for(int i=0;i<n;++i) if(weights[i]>0) rows[kept++]=i;
    rows.conservativeResize(kept);
    VectorXd curvature(kept),cross=VectorXd::Zero(n);
    double size=0;
    const double digamma_theta=R::digamma(theta),trigamma_theta=R::trigamma(theta);
    for(int r=0;r<kept;++r) {
        const int i=rows[r];
        const double linear=eta[i]+offset[i],mu=link_inverse(linear,link),dm=link_derivative(linear,link);
        const double yi=y[i],w=weights[i];
        const double d2=link==0 ? mu : (link==1 ? 2 : 0);
        const double variance=mu+mu*mu/theta,vp=1+2*mu/theta,residual=yi-mu;
        curvature[r]=w*(dm*dm/variance+residual*(dm*dm*vp/(variance*variance)-d2/variance));
        cross[i]=-w*dm*theta*residual/((theta+mu)*(theta+mu));
        double ii;
        if(theta>1e4*std::max(1.0,std::max(yi,mu))) {
            const double c2=(yi-residual*residual)/2;
            const double c3=yi*(yi-1)*(2*yi-1)/6+2*mu*mu*mu/3-yi*mu*mu;
            const double c4=-yi*yi*(yi-1)*(yi-1)/4-3*mu*mu*mu*mu/4+yi*mu*mu*mu;
            ii=(c2+(2*c3+3*c4/theta)/theta)/theta;
        } else {
            const double sc=theta_score(yi,mu,theta,digamma_theta,false);
            const double ds=R::trigamma(yi+theta)-trigamma_theta+mu/(theta*(theta+mu))+residual/((theta+mu)*(theta+mu));
            ii=-theta*sc-theta*theta*ds;
        }
        size+=w*ii;
    }
    Rcpp::NumericMatrix out(p+1,p+1);
    Map<MatrixXd> info(out.begin(),p+1,p+1);
    MatrixXd block=MatrixXd::Zero(p,p);
    if(kept) add_weighted_crossproduct(x,rows,curvature,block);
    info.topLeftCorner(p,p)=block;
    info.topRightCorner(p,1)=x.transpose()*cross;
    info(p,p)=size;
    return out;
}
