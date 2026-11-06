#include "../inst/include/ordinal.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace fglm_ordinal;

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
    int max_halves=35,double condition_tol=1e-14,bool trace=false,int link=0,bool observed=true) {
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    return ordinal_fit(data,Rcpp::as<VectorXd>(start),firth,score_tol,step_tol,maxit,
        max_step,max_halves,condition_tol,trace,observed);
}

// [[Rcpp::export]]
Rcpp::NumericVector ordinal_category_totals(Rcpp::IntegerVector codes,Rcpp::NumericVector weights,int categories) {
    Rcpp::NumericVector totals(categories);
    for(R_xlen_t i=0;i<codes.size();++i) totals[codes[i]-1]+=weights[i];
    return totals;
}

// Centers and scales the positive-frequency rows and checks rank in one pass. The
// check uses an unpivoted Cholesky factor of the normalized crossproduct of
// (1, standardized x): its squared diagonal equals the squared QR diagonal relative
// to each column norm, which is the ratio a pivoted QR compares with its tolerance.
// [[Rcpp::export]]
Rcpp::List ordinal_design_native(Rcpp::NumericMatrix x,Rcpp::NumericVector totals,double frequency,bool standardize) {
    const int n=x.nrow(),p=x.ncol();
    const Eigen::Map<const MatrixXd> X(x.begin(),n,p);
    if(!X.allFinite()) Rcpp::stop("ordinal fitting requires a finite dense numeric matrix without a location intercept");
    VectorXd center(p),scale(p);
    int active=0;
    for(int i=0;i<n;++i) if(totals[i]>0) ++active;
    Rcpp::NumericMatrix out(n,p);
    Eigen::Map<MatrixXd> Y(out.begin(),n,p);
    for(int j=0;j<p;++j) {
        double mean=0,variance=0;
        for(int i=0;i<n;++i) if(totals[i]>0) mean+=X(i,j)*(totals[i]/frequency);
        for(int i=0;i<n;++i) if(totals[i]>0) { const double c=X(i,j)-mean;variance+=c*c*(totals[i]/frequency); }
        center[j]=mean;scale[j]=std::sqrt(variance);
        for(int i=0;i<n;++i)
            Y(i,j)=totals[i]>0 ? (standardize ? (X(i,j)-mean)/scale[j] : X(i,j)) : 0;
    }
    bool full_rank=false;
    if((scale.array().isFinite() && scale.array()>0).all() && active>p) {
        MatrixXd gram=MatrixXd::Zero(p+1,p+1);
        const int block=4096;
        MatrixXd rows(block,p+1);
        int filled=0;
        const auto flush=[&]() {
            if(filled) gram.selfadjointView<Eigen::Lower>().rankUpdate(rows.topRows(filled).transpose());
            filled=0;
        };
        for(int i=0;i<n;++i) if(totals[i]>0) {
            rows(filled,0)=1;
            for(int j=0;j<p;++j) rows(filled,j+1)=(X(i,j)-center[j])/scale[j];
            if(++filled==block) flush();
        }
        flush();
        const VectorXd inverse=gram.diagonal().cwiseSqrt().cwiseInverse();
        const MatrixXd full=gram.selfadjointView<Eigen::Lower>();
        const MatrixXd normalized=inverse.asDiagonal()*full*inverse.asDiagonal();
        const Eigen::LLT<MatrixXd> factor(normalized);
        if(factor.info()==Eigen::Success) {
            const VectorXd pivots=MatrixXd(factor.matrixL()).diagonal();
            full_rank=pivots.array().square().minCoeff()>=1e-10;
        }
    }
    return Rcpp::List::create(Rcpp::_["x"]=out,Rcpp::_["center"]=center,Rcpp::_["scale"]=scale,
        Rcpp::_["full.rank"]=full_rank);
}

// Exposes the observed terms to the derivative tests.
// [[Rcpp::export]]
Rcpp::List ordinal_observed_evaluate(Rcpp::NumericMatrix x,Rcpp::IntegerVector y,
    Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,Rcpp::NumericVector offset,
    int categories,Rcpp::NumericVector coefficients,int block_size=2048,int link=0) {
    OrdinalData data(x,y,counts,weights,offset,categories,block_size,link);
    const ObservedState state=observed_evaluate(data,Rcpp::as<VectorXd>(coefficients),nullptr);
    return Rcpp::List::create(Rcpp::_["loglik"]=state.loglik,Rcpp::_["score"]=state.score,
        Rcpp::_["information"]=state.information());
}

// Moves covariance and information from centered and scaled coordinates to the
// original ones. The change of coordinates adds one slope combination to every
// threshold and rescales slopes, so no dense matrix product is needed.
// [[Rcpp::export]]
Rcpp::List ordinal_effective_native(Eigen::Map<Eigen::MatrixXd> covariance,Eigen::Map<Eigen::MatrixXd> information,
    Eigen::Map<Eigen::VectorXd> center,Eigen::Map<Eigen::VectorXd> scale,int thresholds) {
    const int q=thresholds,p=center.size(),d=q+p;
    if(covariance.rows()!=d || covariance.cols()!=d || information.rows()!=d || information.cols()!=d || scale.size()!=p)
        Rcpp::stop("incompatible ordinal coordinate dimensions");
    MatrixXd V=covariance,I=information;
    const VectorXd shift=center.cwiseQuotient(scale);
    if(p) {
        // Covariance becomes T V T', where T adds shift'beta to each threshold.
        for(int side=0;side<2;++side) {
            if(side==1) V.transposeInPlace();
            const VectorXd combination=V.bottomRows(p).transpose()*shift;
            V.topRows(q).rowwise()+=combination.transpose();
            for(int j=0;j<p;++j) V.row(q+j)/=scale[j];
        }
        // Information becomes A' I A, the matching inverse transformation.
        for(int side=0;side<2;++side) {
            if(side==1) I.transposeInPlace();
            const VectorXd thresholds_sum=I.leftCols(q).rowwise().sum();
            for(int j=0;j<p;++j) I.col(q+j)=scale[j]*I.col(q+j)-center[j]*thresholds_sum;
        }
    }
    return Rcpp::List::create(Rcpp::_["covariance"]=V,Rcpp::_["information"]=I);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix ordinal_center_scale(Rcpp::NumericMatrix x,Rcpp::NumericVector center,Rcpp::NumericVector scale) {
    const int n=x.nrow(),p=x.ncol();
    if(center.size()!=p || scale.size()!=p) Rcpp::stop("incompatible ordinal scaling dimensions");
    Rcpp::NumericMatrix out(Rcpp::no_init(n,p));
    for(int j=0;j<p;++j) {
        const double* from=x.begin()+static_cast<R_xlen_t>(j)*n;
        double* to=out.begin()+static_cast<R_xlen_t>(j)*n;
        const double c=center[j],s=scale[j];
        for(int i=0;i<n;++i) to[i]=(from[i]-c)/s;
    }
    if(x.hasAttribute("dimnames")) out.attr("dimnames")=x.attr("dimnames");
    return out;
}

// Observed information and empirical score moments for one data block, on the
// frequency-normalized scale of the sandwich covariance. Logit and probit only:
// cloglog tail curvature needs direct derivatives that the R path computes.
// [[Rcpp::export]]
Rcpp::List ordinal_sandwich_native(Rcpp::NumericMatrix x,Rcpp::NumericVector alpha,Rcpp::NumericVector beta,
    Rcpp::NumericVector offset,Rcpp::IntegerVector codes,Rcpp::NumericMatrix counts,Rcpp::NumericVector weights,
    double frequency,bool independent,int link) {
    const int n=x.nrow(),p=x.ncol(),q=alpha.size(),d=q+p;
    const Eigen::Map<const MatrixXd> X(x.begin(),n,p);
    const ObservedThresholds thresholds(Rcpp::as<VectorXd>(alpha),link);
    const bool grouped=counts.nrow()>0 || counts.ncol()>0;
    const VectorXd eta=p ? VectorXd(X*Rcpp::as<VectorXd>(beta)) : VectorXd::Zero(n);
    MatrixXd information=MatrixXd::Zero(d,d),meat=MatrixXd::Zero(d,d);
    MatrixXd row_threshold=MatrixXd::Zero(n,q),info_x=MatrixXd::Zero(n,q),meat_x=MatrixXd::Zero(independent ? n : 0,independent ? q : 0);
    VectorXd row_location=VectorXd::Zero(n),info_slope=VectorXd::Zero(n),meat_slope=VectorXd::Zero(independent ? n : 0);
    const auto add=[&](int i,int k,double y) {
        const ObservedTerms t=thresholds.terms(k,eta[i]+offset[i]);
        const double location=-t.location;
        row_location[i]+=y*location;info_slope[i]+=y*t.slope;
        if(independent) meat_slope[i]+=y*location*location;
        if(k<q) {
            row_threshold(i,k)+=y*t.left;information(k,k)+=y*t.upper;info_x(i,k)+=y*t.upper_x;
            if(independent) { meat(k,k)+=y*t.left*t.left;meat_x(i,k)+=y*t.left*location; }
        }
        if(k>0) {
            row_threshold(i,k-1)-=y*t.right;information(k-1,k-1)+=y*t.lower;info_x(i,k-1)+=y*t.lower_x;
            if(independent) { meat(k-1,k-1)+=y*t.right*t.right;meat_x(i,k-1)-=y*t.right*location; }
            if(k<q) {
                information(k-1,k)+=y*t.cross;
                if(independent) meat(k-1,k)-=y*t.right*t.left;
            }
        }
    };
    for(int i=0;i<n;++i) {
        if(grouped) { for(int k=0;k<=q;++k) { const double y=weights[i]*counts(i,k)/frequency;if(y>0) add(i,k,y); } }
        else { const double y=weights[i]/frequency;if(y>0) add(i,codes[i]-1,y); }
    }
    for(int s=1;s<q;++s) { information(s,s-1)=information(s-1,s);if(independent) meat(s,s-1)=meat(s-1,s); }
    if(p) {
        information.topRightCorner(q,p).noalias()=info_x.transpose()*X;
        information.bottomLeftCorner(p,q)=information.topRightCorner(q,p).transpose();
        information.bottomRightCorner(p,p).noalias()=X.transpose()*(info_slope.asDiagonal()*X);
        if(independent) {
            meat.topRightCorner(q,p).noalias()=meat_x.transpose()*X;
            meat.bottomLeftCorner(p,q)=meat.topRightCorner(q,p).transpose();
            meat.bottomRightCorner(p,p).noalias()=X.transpose()*(meat_slope.asDiagonal()*X);
        }
    }
    VectorXd score(d);
    score.head(q)=row_threshold.colwise().sum();
    if(p) score.tail(p).noalias()=X.transpose()*row_location;
    Rcpp::List out=Rcpp::List::create(Rcpp::_["information"]=information,Rcpp::_["meat"]=meat,Rcpp::_["score"]=score);
    if(independent) out["rows"]=R_NilValue;
    else {
        MatrixXd rows(n,d);
        rows.leftCols(q)=row_threshold;
        if(p) rows.rightCols(p)=row_location.asDiagonal()*X;
        out["rows"]=rows;
    }
    return out;
}

// Score and expected information for the threshold-specific slopes that a
// common-slope null fit omits. The null blocks are the usual expected information.
// [[Rcpp::export]]
Rcpp::List ordinal_po_moments_native(Rcpp::NumericMatrix x,Rcpp::IntegerVector codes,Rcpp::NumericMatrix counts,
    Rcpp::NumericVector weights,Rcpp::NumericVector offset,Rcpp::NumericVector alpha_values,Rcpp::NumericVector beta_values,
    Rcpp::IntegerVector selected,double frequency,int block_size,int link) {
    const int p=x.ncol(),q=alpha_values.size(),s=selected.size(),d0=q+p,d=d0+s*(q-1);
    OrdinalData data(x,codes,counts,weights,offset,q+1,block_size,link,false);
    const VectorXd alpha=Rcpp::as<VectorXd>(alpha_values),beta=Rcpp::as<VectorXd>(beta_values);
    VectorXd U=VectorXd::Zero(d);
    MatrixXd I=MatrixXd::Zero(d,d);
    OrdinalRow row(alpha);
    const auto index=[&](int j) { return d0+(j-1)*s; };
    for(int first=0;first<data.n;first+=data.block) {
        Rcpp::checkUserInterrupt();
        const int length=std::min(data.block,data.n-first);
        const auto xx=data.x.middleRows(first,length);
        MatrixXd zz(length,s);
        for(int j=0;j<s;++j) zz.col(j)=xx.col(selected[j]-1);
        MatrixXd threshold=MatrixXd::Zero(length,q),diagonal=MatrixXd::Zero(length,q),rowsum=MatrixXd::Zero(length,q);
        MatrixXd adjacent=MatrixXd::Zero(length,std::max(q-1,0));
        VectorXd location=VectorXd::Zero(length);
        for(int r=0;r<length;++r) {
            const int i=first+r;
            try { row.evaluate(alpha,data.linear(i,beta)+data.offset[i],data.link); }
            catch(const OrdinalEvaluationError&) {
                Rcpp::stop("the fitted null has a category probability outside the representable positive range");
            }
            if(row.logp.minCoeff()<std::log(std::numeric_limits<double>::denorm_min()))
                Rcpp::stop("the fitted null has a category probability outside the representable positive range");
            const double mass=data.totals[i]/frequency;
            for(int k=0;k<=q;++k) {
                const double c=data.grouped ? data.weights[i]*data.counts(i,k)/frequency :
                    (data.codes[i]-1==k ? data.weights[i]/frequency : 0);
                if(c==0) continue;
                location[r]-=c*row.location_score(k);
                if(k<q) threshold(r,k)+=c*row.left[k];
                if(k>0) threshold(r,k-1)-=c*row.right[k-1];
            }
            for(int j=0;j<q;++j) {
                diagonal(r,j)=mass*row.g[j]*(row.left[j]+row.right[j]);
                rowsum(r,j)=mass*row.g[j]*row.neighbor[j];
                if(j+1<q) adjacent(r,j)=-mass*row.g[j+1]*row.right[j];
            }
        }
        U.head(q)+=threshold.colwise().sum().transpose();
        if(p) U.segment(q,p).noalias()+=xx.transpose()*location;
        for(int j=0;j<q;++j) {
            I(j,j)+=diagonal.col(j).sum();
            if(j+1<q) I(j,j+1)+=adjacent.col(j).sum();
        }
        if(p) {
            I.block(0,q,q,p).noalias()-=rowsum.transpose()*xx;
            const VectorXd total=rowsum.rowwise().sum();
            I.block(q,q,p,p).noalias()+=xx.transpose()*(total.asDiagonal()*xx);
        }
        const MatrixXd dg=diagonal.transpose()*zz,ad=q>1 ? MatrixXd(adjacent.transpose()*zz) : MatrixXd();
        for(int j=1;j<q;++j) {
            const int at=index(j);
            U.segment(at,s).noalias()-=zz.transpose()*threshold.col(j);
            I.block(j,at,1,s)-=dg.row(j);
            I.block(j-1,at,1,s)-=ad.row(j-1);
            if(j+1<q) I.block(j+1,at,1,s)-=ad.row(j);
            I.block(q,at,p,s).noalias()+=xx.transpose()*(rowsum.col(j).asDiagonal()*zz);
            I.block(at,at,s,s).noalias()+=zz.transpose()*(diagonal.col(j).asDiagonal()*zz);
            if(j+1<q) I.block(at,index(j+1),s,s).noalias()+=zz.transpose()*(adjacent.col(j).asDiagonal()*zz);
        }
    }
    I.triangularView<Eigen::StrictlyLower>()=I.transpose();
    return Rcpp::List::create(Rcpp::_["score"]=U,Rcpp::_["information"]=I);
}
