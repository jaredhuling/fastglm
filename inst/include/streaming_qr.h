#ifndef FASTGLM_STREAMING_QR_H
#define FASTGLM_STREAMING_QR_H
#include <RcppEigen.h>
#include <vector>

namespace fglm {
// Two-double arithmetic is used only by the difficult-design convergence
// verification. Keeping quotient and product residuals prevents division by a
// small triangular diagonal from magnifying intermediate rounding errors.
struct QRAccurateScalar {
    double hi,lo;
    explicit QRAccurateScalar(double x=0):hi(x),lo(0) {}
    QRAccurateScalar(double h,double l):hi(h),lo(l) {}
    static QRAccurateScalar sum(double a,double b) {
        const double s=a+b,z=s-a;
        return QRAccurateScalar(s,(a-(s-z))+(b-z));
    }
    QRAccurateScalar plus(const QRAccurateScalar& b) const {
        QRAccurateScalar s=sum(hi,b.hi);
        return sum(s.hi,s.lo+lo+b.lo);
    }
    QRAccurateScalar times(double b) const {
        const double product=hi*b;
        return sum(product,std::fma(hi,b,-product)+lo*b);
    }
    QRAccurateScalar divided(double b) const {
        const double first=hi/b;
        QRAccurateScalar residual=plus(QRAccurateScalar(first).times(-b));
        return sum(first,(residual.hi+residual.lo)/b);
    }
};

// A compensated dot product prevents large, cancelling coefficients from
// injecting avoidable rounding noise into deviance-based step acceptance.
inline Eigen::VectorXd qr_linear_predictor(const Eigen::MatrixXd& X,
    const Eigen::VectorXd& beta,const Eigen::VectorXd& offset)
{
    Eigen::VectorXd eta(X.rows());
    for(int i=0;i<X.rows();++i) {
        double sum=offset[i], error=0;
        for(int j=0;j<X.cols();++j) {
            const double product=X(i,j)*beta[j];
            const double next=sum+product;
            error += std::abs(sum)>=std::abs(product) ? (sum-next)+product : (product-next)+sum;
            error += std::fma(X(i,j),beta[j],-product);
            sum=next;
        }
        eta[i]=sum+error;
    }
    return eta;
}

// Block QR retains a triangular factor and transformed response only. The
// small factor is scaled and pivoted after the complete pass; early blocks
// are allowed to have too few rows or no positive-weight observations.
class StreamingQR {
public:
    Eigen::MatrixXd raw_R, R;
    Eigen::VectorXd raw_qty, qty, scale;
    Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> pivot;
    bool projection = false;
    bool ready = false;

    void reset(int p) {
        raw_R.setZero(p,p); raw_qty.setZero(p);
        projection=false; ready=false;
    }
    void project_with(const StreamingQR& factor) {
        *this=factor; qty.setZero(); projection=true;
    }
    void add(const Eigen::MatrixXd& X, const Eigen::VectorXd& z,
             const Eigen::VectorXd& weights) {
        const int p=X.cols(), n=X.rows();
        if(!weights.allFinite() || (weights.array()<0).any())
            Rcpp::stop("streaming QR requires finite non-negative working weights");
        Eigen::VectorXd sw=weights.array().sqrt();
        Eigen::VectorXd b=sw.array()*z.array();
        for(int i=0;i<n;++i) if(weights[i]==0) b[i]=0;
        if(projection) {
            Eigen::MatrixXd basis=row_basis(X);
            basis.array().colwise()*=sw.array();
            qty.noalias()+=basis.adjoint()*b;
            return;
        }
        Eigen::MatrixXd stacked(p+n,p);
        stacked.topRows(p)=raw_R;
        stacked.bottomRows(n)=X.array().colwise()*sw.array();
        Eigen::VectorXd response(p+n);
        response.head(p)=raw_qty; response.tail(n)=b;
        Eigen::HouseholderQR<Eigen::MatrixXd> block(stacked);
        Eigen::VectorXd transformed=block.householderQ().adjoint()*response;
        raw_R=block.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
        raw_qty=transformed.head(p);
    }
    void finish() {
        if(projection) return;
        const int p=raw_R.cols();
        scale.resize(p);
        for(int j=0;j<p;++j) {
            scale[j]=raw_R.col(j).stableNorm();
            if(!(scale[j]>0) || !std::isfinite(scale[j]))
                Rcpp::stop("streaming QR requires full column rank; a weighted design column is zero or non-finite");
        }
        Eigen::MatrixXd normalized=raw_R*scale.cwiseInverse().asDiagonal();
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> check(normalized);
        check.setThreshold(64*std::numeric_limits<double>::epsilon()*std::max(1,p));
        if(check.rank()!=p)
            Rcpp::stop("streaming QR requires full column rank; the weighted design is rank deficient at working precision");
        R=check.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
        pivot=check.colsPermutation();
        qty=check.householderQ().adjoint()*raw_qty;
        ready=true;
    }
    Eigen::VectorXd solve() const {
        return solve_transformed(qty);
    }
    Eigen::VectorXd solve_transformed(const Eigen::VectorXd& rhs) const {
        Eigen::VectorXd b=R.template triangularView<Eigen::Upper>().solve(rhs);
        return (pivot*b).cwiseQuotient(scale);
    }
    // X / scale, with the QR column permutation, times R^{-1}.
    Eigen::MatrixXd row_basis(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd scaled=X*scale.cwiseInverse().asDiagonal()*pivot;
        Eigen::MatrixXd t=R.transpose().template triangularView<Eigen::Lower>().solve(scaled.transpose());
        return t.transpose();
    }
    Eigen::MatrixXd accurate_row_basis(const Eigen::MatrixXd& X) const {
        const int p=X.cols();
        Eigen::MatrixXd basis(X.rows(),p);
        std::vector<QRAccurateScalar> row(p);
        for(int i=0;i<X.rows();++i) for(int j=0;j<p;++j) {
            const int original=pivot.indices()[j];
            QRAccurateScalar value=QRAccurateScalar(X(i,original)).divided(scale[original]);
            for(int k=0;k<j;++k) value=value.plus(row[k].times(-R(k,j)));
            row[j]=value.divided(R(j,j));
            basis(i,j)=row[j].hi+row[j].lo;
        }
        return basis;
    }
    double condition_bound() const {
        Eigen::MatrixXd inverse=R.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(R.rows(),R.cols()));
        return R.cwiseAbs().colwise().sum().maxCoeff()*inverse.cwiseAbs().colwise().sum().maxCoeff();
    }
    Eigen::VectorXd coefficient_coordinates(const Eigen::VectorXd& b) const {
        return R*(pivot.transpose()*scale.cwiseProduct(b));
    }
    Eigen::VectorXd original_score(const Eigen::VectorXd& transformed) const {
        return scale.cwiseProduct(pivot*(R.transpose()*transformed));
    }
    Eigen::VectorXd information_diagonal() const {
        Eigen::VectorXd d=R.array().square().colwise().sum().transpose().matrix();
        return scale.array().square().matrix().cwiseProduct(pivot*d);
    }
    Eigen::MatrixXd covariance() const {
        Eigen::MatrixXd inverse=R.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(R.rows(),R.cols()));
        Eigen::MatrixXd mapped=pivot*inverse;
        mapped.array().colwise()/=scale.array();
        return mapped*mapped.adjoint();
    }
    double logdet() const {
        return 2*(R.diagonal().array().abs().log().sum()+scale.array().log().sum());
    }
    Rcpp::List factor() const {
        Rcpp::IntegerVector order(pivot.indices().size());
        for(int j=0;j<order.size();++j) order[j]=pivot.indices()[j]+1;
        return Rcpp::List::create(Rcpp::_["R"]=R,Rcpp::_["pivot"]=order,Rcpp::_["scale"]=scale);
    }
};
}
#endif
