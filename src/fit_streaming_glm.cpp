#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include <RcppEigen.h>
#include "../inst/include/families.h"
#include "../inst/include/chunk_source.h"
#include "../inst/include/bias_reduction.h"
#include "../inst/include/streaming_qr.h"

#include <stdexcept>
#include <string>

using namespace Rcpp;

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

namespace {

// Holds either a native (FamilyCode-dispatched) or a callback-driven family.
struct FamilyOps {
    int code;
    bool native;
    fglm::FamilyParams params;
    // Only used when !native
    Function var_fn;
    Function mueta_fn;
    Function linkinv_fn;
    Function devresids_fn;
    Function valideta_fn;
    Function validmu_fn;

    FamilyOps(int code_,
              const fglm::FamilyParams& params_,
              Function var_,
              Function mueta_,
              Function linkinv_,
              Function devresids_,
              Function valideta_,
              Function validmu_)
        : code(code_),
          native(code_ != fglm::FAM_UNKNOWN),
          params(params_),
          var_fn(var_),
          mueta_fn(mueta_),
          linkinv_fn(linkinv_),
          devresids_fn(devresids_),
          valideta_fn(valideta_),
          validmu_fn(validmu_)
    {}

    void linkinv(const ArrayXd& eta, ArrayXd& mu) const {
        if (native) {
            mu.resize(eta.size());
            fglm::linkinv(code, params, eta, mu);
        } else {
            NumericVector r = linkinv_fn(NumericVector(eta.data(), eta.data() + eta.size()));
            mu = Map<const ArrayXd>(r.begin(), r.size());
        }
    }
    void mu_eta(const ArrayXd& eta, ArrayXd& dmu) const {
        if (native) {
            dmu.resize(eta.size());
            fglm::mu_eta(code, params, eta, dmu);
        } else {
            NumericVector r = mueta_fn(NumericVector(eta.data(), eta.data() + eta.size()));
            dmu = Map<const ArrayXd>(r.begin(), r.size());
        }
    }
    void variance(const ArrayXd& mu, ArrayXd& v) const {
        if (native) {
            v.resize(mu.size());
            fglm::variance(code, params, mu, v);
        } else {
            NumericVector r = var_fn(NumericVector(mu.data(), mu.data() + mu.size()));
            v = Map<const ArrayXd>(r.begin(), r.size());
        }
    }
    double dev_resids_sum(const ArrayXd& y, const ArrayXd& mu, const ArrayXd& w) const {
        if (native) {
            return fglm::dev_resids_sum(code, params, y, mu, w);
        } else {
            NumericVector r = devresids_fn(
                NumericVector(y.data(),  y.data()  + y.size()),
                NumericVector(mu.data(), mu.data() + mu.size()),
                NumericVector(w.data(),  w.data()  + w.size()));
            double s = 0.0;
            for (R_xlen_t i = 0; i < r.size(); ++i) s += r[i];
            return s;
        }
    }
    bool valideta(const ArrayXd& eta) const {
        if (native) return fglm::valideta(code, params, eta);
        SEXP r = valideta_fn(NumericVector(eta.data(), eta.data() + eta.size()));
        return Rf_asLogical(r) == TRUE;
    }
    bool validmu(const ArrayXd& mu) const {
        if (native) return fglm::validmu(code, params, mu);
        SEXP r = validmu_fn(NumericVector(mu.data(), mu.data() + mu.size()));
        return Rf_asLogical(r) == TRUE;
    }
    void d2mu_deta2(const ArrayXd& eta, const ArrayXd& mu,
                    const ArrayXd& dmu, ArrayXd& d2mu) const {
        if (native) {
            d2mu.resize(eta.size());
            fglm::d2mu_deta2(code, params, eta, mu, dmu, d2mu);
        } else {
            const double h = std::pow(std::numeric_limits<double>::epsilon(), 1.0/3.0);
            ArrayXd ep = eta + h;
            ArrayXd em = eta - h;
            ArrayXd dp, dm;
            mu_eta(ep, dp);
            mu_eta(em, dm);
            d2mu = (dp - dm) / (2.0 * h);
        }
    }
    bool is_unit_dispersion() const {
        return (code >= 3 && code <= 9) || (code >= 17 && code <= 19);
    }
};

struct Chunk {
    MatrixXd X;
    VectorXd y;
    VectorXd w;       // prior weights, all 1 if not supplied
    VectorXd off;     // offset, all 0 if not supplied
    bool has_weights;
    bool has_offset;
};

// Pull the k-th chunk via the user's R callback.  k is 1-based for the
// closure, matching the R-side documentation.
static Chunk pull_chunk(Function& callback, int k, int expected_p)
{
    Rcpp::checkUserInterrupt();
    SEXP res = callback(k);
    if (TYPEOF(res) != VECSXP)
        Rcpp::stop("chunk_callback must return a list");
    List L(res);
    if (!L.containsElementNamed("X") || !L.containsElementNamed("y"))
        Rcpp::stop("chunk must contain elements 'X' and 'y'");

    Chunk c;
    NumericMatrix Xs = as<NumericMatrix>(L["X"]);
    NumericVector ys = as<NumericVector>(L["y"]);
    int n_k = Xs.nrow();
    int p_k = Xs.ncol();
    if ((int)ys.size() != n_k)
        Rcpp::stop("chunk$X has rows != length(y)");
    if (expected_p > 0 && p_k != expected_p)
        Rcpp::stop("chunk$X has unexpected number of columns");

    c.X = Map<MatrixXd>(Xs.begin(), n_k, p_k);
    c.y = Map<VectorXd>(ys.begin(), n_k);

    c.has_weights = L.containsElementNamed("weights") && !Rf_isNull(L["weights"]);
    if (c.has_weights) {
        NumericVector ws = as<NumericVector>(L["weights"]);
        if ((int)ws.size() != n_k)
            Rcpp::stop("chunk$weights has wrong length");
        c.w = Map<VectorXd>(ws.begin(), n_k);
    } else {
        c.w = VectorXd::Ones(n_k);
    }

    c.has_offset = L.containsElementNamed("offset") && !Rf_isNull(L["offset"]);
    if (c.has_offset) {
        NumericVector os = as<NumericVector>(L["offset"]);
        if ((int)os.size() != n_k)
            Rcpp::stop("chunk$offset has wrong length");
        c.off = Map<VectorXd>(os.begin(), n_k);
    } else {
        c.off = VectorXd::Zero(n_k);
    }

    return c;
}

// One streaming pass at the supplied beta.
//
//   accumulate     : if true, also build XtWX and Xtwz at this beta.
//   intercept_mask : if non-null, AND-folds per-chunk "constant column" mask
//                    into it (used once to detect the intercept column).
//
// Returns the total deviance.  Throws when valideta/validmu fails on any chunk.
//
// XtWX / Xtwz come back through the supplied refs; caller is responsible for
// zeroing them before the call.
static double stream_pass(Function& callback,
                          int n_chunks, int p,
                          const VectorXd& beta,
                          const FamilyOps& fam,
                          bool accumulate,
                          MatrixXd* XtWX_out,
                          VectorXd* Xtwz_out,
                          std::vector<bool>* intercept_mask,
                          int* n_total_out)
{
    double dev_sum = 0.0;
    int n_total = 0;

    if (accumulate) {
        XtWX_out->setZero();
        Xtwz_out->setZero();
    }

    for (int k = 1; k <= n_chunks; ++k) {
        Chunk c = pull_chunk(callback, k, p);
        const int n_k = c.y.size();
        n_total += n_k;

        ArrayXd eta = (c.X * beta).array() + c.off.array();
        if (!fam.valideta(eta))
            throw std::domain_error("invalid linear predictor in streaming pass");

        ArrayXd mu;
        fam.linkinv(eta, mu);
        if (!fam.validmu(mu))
            throw std::domain_error("invalid mean in streaming pass");

        ArrayXd ya = c.y.array();
        ArrayXd wa = c.w.array();
        dev_sum += fam.dev_resids_sum(ya, mu, wa);

        if (accumulate) {
            ArrayXd dmu, va;
            fam.mu_eta(eta, dmu);
            fam.variance(mu, va);

            // w2 = w_prior * dmu^2 / V(mu)
            ArrayXd w2 = wa * dmu.square() / va;
            // z = eta - offset + (y - mu) / dmu
            ArrayXd z = eta - c.off.array() + (ya - mu) / dmu;

            // XtWX += X^T diag(w2) X using Eigen's rank-update.
            // Build sqrt(w2) * X once and rank-update.
            MatrixXd Xw = c.X.array().colwise() * w2.sqrt();
            XtWX_out->selfadjointView<Eigen::Lower>().rankUpdate(Xw.adjoint());

            // Xtwz += X^T (w2 * z)
            VectorXd wz = (w2 * z).matrix();
            (*Xtwz_out).noalias() += c.X.adjoint() * wz;
        }

        if (intercept_mask) {
            for (int j = 0; j < p; ++j) {
                if (!(*intercept_mask)[j]) continue;
                double v0 = c.X(0, j);
                bool ok = true;
                for (int i = 1; i < n_k; ++i) {
                    if (c.X(i, j) != v0) { ok = false; break; }
                }
                if (!ok) (*intercept_mask)[j] = false;
            }
        }
    }

    if (accumulate) {
        // Symmetrize the lower-triangular accumulator.
        XtWX_out->triangularView<Eigen::Upper>() = XtWX_out->adjoint();
    }

    if (n_total_out) *n_total_out = n_total;
    return dev_sum;
}

// Firth-aware streaming pass.  Computes XtWX, Xtwz*, and deviance in a
// single sweep.  When A_prev is non-null, leverages from the previous
// iteration are used to compute the AS_mean adjustment xi.
// phi_prev: dispersion estimate from the previous iteration.
//           For unit-dispersion families (binomial, Poisson) this is always 1.
//           For the first Firth iteration (A_prev == nullptr) phi_prev is unused.
static double stream_pass_firth(Function& callback,
                                int n_chunks, int p,
                                const VectorXd& beta,
                                const FamilyOps& fam,
                                const MatrixXd* A_prev,
                                double phi_prev,
                                MatrixXd* XtWX_out,
                                VectorXd* Xtwz_out,
                                int* n_total_out, double* phi_out,
                                bool* constant_information,
                                VectorXd* gamma_adjustment,
                                bool current_information = false)
{
    double dev_sum = 0.0;
    int n_total = 0, n_positive = 0;
    fglm::GammaDispersionTerms dispersion_terms;
    const bool reuse_information = (*constant_information || current_information) && A_prev != nullptr;
    const bool gamma_log = fam.code == fglm::FAM_GAMMA_LOG;
    const bool build_gamma_adjustment = gamma_log && A_prev != nullptr &&
        gamma_adjustment->size() == 0;
    if (build_gamma_adjustment) gamma_adjustment->setZero(p);

    if (!reuse_information) XtWX_out->setZero();
    Xtwz_out->setZero();

    for (int k = 1; k <= n_chunks; ++k) {
        Chunk c = pull_chunk(callback, k, p);
        const int n_k = c.y.size();
        n_total += n_k;

        ArrayXd eta = (c.X * beta).array() + c.off.array();
        if (!fam.valideta(eta))
            throw std::domain_error("invalid linear predictor in streaming pass");

        ArrayXd mu_arr;
        fam.linkinv(eta, mu_arr);
        if (!fam.validmu(mu_arr))
            throw std::domain_error("invalid mean in streaming pass");

        ArrayXd ya = c.y.array();
        ArrayXd wa = c.w.array();
        if (fam.code >= fglm::FAM_GAMMA_LOG && fam.code <= fglm::FAM_GAMMA_IDENTITY) {
            // The groups live only for this chunk. Repeated prior weights
            // need one normalizer evaluation, independent of its row count.
            const auto groups = fglm::gamma_weight_groups(c.w);
            for (const auto& group : groups) n_positive += group.second;
            dispersion_terms.add(groups, phi_prev);
        } else n_positive += (wa > 0).count();
        dev_sum += fam.dev_resids_sum(ya, mu_arr, wa);

        ArrayXd dmu, va;
        fam.mu_eta(eta, dmu);
        fam.variance(mu_arr, va);

        ArrayXd w2;
        if (fam.code == fglm::FAM_GAUSSIAN_IDENTITY || gamma_log)
            w2 = wa;
        else if (fam.code == fglm::FAM_POISSON_SQRT) {
            w2 = 4 * wa;
            for (int i = 0; i < n_k; ++i) {
                if (eta[i] >= fglm::thresh_eps()/2) continue;
                // Preserve the implemented derivative floor. If this occurs
                // after caching, amend the retained information for these rows
                // and use the ordinary changing-weight path from now on.
                w2[i] = wa[i] * dmu[i]*dmu[i] / va[i];
                *constant_information = false;
                if (reuse_information && wa[i] > 0)
                    XtWX_out->selfadjointView<Eigen::Lower>().rankUpdate(
                        c.X.row(i).transpose(), w2[i] - 4*wa[i]);
            }
        } else w2 = wa * dmu.square() / va;
        ArrayXd z_arr = eta - c.off.array() + (ya - mu_arr) / dmu;

        if (!reuse_information) {
            MatrixXd Xw = c.X.array().colwise() * w2.sqrt();
            XtWX_out->selfadjointView<Eigen::Lower>().rankUpdate(Xw.adjoint());
        }

        ArrayXd zstar = z_arr;
        if (A_prev != nullptr && fam.code != fglm::FAM_GAUSSIAN_IDENTITY &&
            (!gamma_log || build_gamma_adjustment)) {
            MatrixXd XA = c.X * (*A_prev);
            ArrayXd h_k = (XA.array() * c.X.array()).rowwise().sum() * w2;
            if (gamma_log) {
                // For a log link, d2mu/dmu = 1. The complete adjustment is
                // phi * X'h/2, so only this p-vector needs to survive a pass.
                (*gamma_adjustment).noalias() += c.X.adjoint() * (0.5*h_k).matrix();
            } else {
                ArrayXd d2mu;
                fam.d2mu_deta2(eta, mu_arr, dmu, d2mu);
                for (int i = 0; i < n_k; ++i)
                    if (wa[i] > 0 && w2[i] != 0 && dmu[i] != 0)
                        zstar[i] += phi_prev * h_k[i] * (d2mu[i]/dmu[i]) / (2*w2[i]);
            }
        }

        VectorXd wz = (w2 * zstar).matrix();
        (*Xtwz_out).noalias() += c.X.adjoint() * wz;
    }

    if (gamma_log && A_prev != nullptr)
        (*Xtwz_out).noalias() += phi_prev * (*gamma_adjustment);

    XtWX_out->triangularView<Eigen::Upper>() = XtWX_out->adjoint();

    if (n_total_out) *n_total_out = n_total;
    *phi_out = 1;
    if (!fam.is_unit_dispersion()) {
        if (n_positive <= p) Rcpp::stop("bias reduction requires positive residual degrees of freedom");
        if (fam.code >= 10 && fam.code <= 12) {
            double adjusted = dispersion_terms.numerator(dev_sum,p,phi_prev);
            double step = adjusted * phi_prev / dispersion_terms.d2;
            *phi_out = phi_prev * std::exp(std::max(-0.69314718056, std::min(0.69314718056, step)));
        } else *phi_out = dev_sum / (n_positive-p);
    }
    return dev_sum;
}

// Solve the SPD system A x = rhs with LLT (type 2) or LDLT (type 3),
// factorizing into the supplied decomposition objects so the factor can be
// reused afterwards (log|A|, A^{-1}).  Throws if A is rank-deficient.
static VectorXd spd_factorize_and_solve(int type, const MatrixXd& A,
                                        const VectorXd& rhs,
                                        Eigen::LLT<MatrixXd>& llt,
                                        Eigen::LDLT<MatrixXd>& ldlt)
{
    if (type == 2) {
        llt.compute(A);
        if (llt.info() != Eigen::Success || llt.rcond() < 1e-12)
            Rcpp::stop("Cholesky factorization (LLT) failed; "
                       "design may be rank-deficient. "
                       "Streaming mode requires full column rank.");
        return llt.solve(rhs);
    }
    ldlt.compute(A);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive() || ldlt.rcond() < 1e-12)
        Rcpp::stop("Cholesky factorization (LDLT) failed; "
                   "design may be rank-deficient. "
                   "Streaming mode requires full column rank.");
    return ldlt.solve(rhs);
}

// A^{-1} via a fresh SPD factorization (LLT for type 2, LDLT for type 3).
// Throws if A is rank-deficient.
static MatrixXd spd_inverse(int type, const MatrixXd& A)
{
    const Eigen::Index p = A.rows();
    if (type == 2) {
        Eigen::LLT<MatrixXd> chol(A);
        if (chol.info() != Eigen::Success)
            Rcpp::stop("Cholesky factorization (LLT) failed; "
                       "design may be rank-deficient. "
                       "Streaming mode requires full column rank.");
        return chol.solve(MatrixXd::Identity(p, p));
    }
    Eigen::LDLT<MatrixXd> chol(A);
    if (chol.info() != Eigen::Success)
        Rcpp::stop("Cholesky factorization (LDLT) failed; "
                   "design may be rank-deficient. "
                   "Streaming mode requires full column rank.");
    return chol.solve(MatrixXd::Identity(p, p));
}

// A pass through an incrementally factorized weighted design. Lagged
// leverages use the previous QR factor and the previous working weights,
// as in Zietkiewicz and Kosmidis (2024), section 3.5.
static double stream_qr_pass(Function& callback,int n_chunks,int p,
    const VectorXd& beta,const FamilyOps& fam,fglm::StreamingQR& current,
    const fglm::StreamingQR* previous,const VectorXd* previous_beta,
    bool firth,double phi,double* phi_out,VectorXd* gamma_adjustment,
    std::vector<bool>* intercept_mask=nullptr,int* n_total_out=nullptr)
{
    const bool gamma=fam.code==fglm::FAM_GAMMA_LOG;
    const bool constant=gamma || fam.code==fglm::FAM_GAUSSIAN_IDENTITY;
    const bool cached=previous!=nullptr && (constant ||
        (firth && fglm::bounded_firth_predictors(fam.code) && previous_beta==nullptr));
    if(cached) current.project_with(*previous); else current.reset(p);
    const bool build_gamma=firth && gamma && previous && gamma_adjustment->size()==0;
    if(build_gamma) gamma_adjustment->setZero(p);
    double dev=0; int n_total=0,n_positive=0;
    fglm::GammaDispersionTerms dispersion;
    for(int k=1;k<=n_chunks;++k) {
        Chunk c=pull_chunk(callback,k,p);
        n_total+=c.y.size(); n_positive+=(c.w.array()>0).count();
        ArrayXd eta=fglm::qr_linear_predictor(c.X,beta,c.off).array(),mu,dmu,variance;
        if(!fam.valideta(eta)) throw std::domain_error("invalid linear predictor in streaming QR pass");
        fam.linkinv(eta,mu);
        if(!fam.validmu(mu)) throw std::domain_error("invalid mean in streaming QR pass");
        fam.mu_eta(eta,dmu); fam.variance(mu,variance);
        dev+=fam.dev_resids_sum(c.y.array(),mu,c.w.array());
        VectorXd working=constant ? c.w : (c.w.array()*dmu.square()/variance).matrix();
        VectorXd z=(eta-c.off.array()+(c.y.array()-mu)/dmu).matrix();
        for(int i=0;i<c.y.size();++i) if(c.w[i]==0) { working[i]=0; z[i]=0; }
        if(!working.allFinite() || (working.array()<0).any() || !z.allFinite())
            throw std::domain_error("non-finite working values in streaming QR pass");
        if(firth && previous && fam.code!=fglm::FAM_GAUSSIAN_IDENTITY && (!gamma || build_gamma)) {
            MatrixXd basis=previous->row_basis(c.X);
            VectorXd leverage_weight=working;
            if(!constant && previous_beta) {
                ArrayXd eta0=fglm::qr_linear_predictor(c.X,*previous_beta,c.off).array(),mu0,dm0,var0;
                fam.linkinv(eta0,mu0); fam.mu_eta(eta0,dm0); fam.variance(mu0,var0);
                leverage_weight=(c.w.array()*dm0.square()/var0).matrix();
                for(int i=0;i<c.y.size();++i) if(c.w[i]==0) leverage_weight[i]=0;
            }
            VectorXd h=(basis.array().square().rowwise().sum()*leverage_weight.array()).matrix();
            if(gamma) gamma_adjustment->noalias()+=basis.adjoint()*(0.5*h);
            else {
                ArrayXd second; fam.d2mu_deta2(eta,mu,dmu,second);
                for(int i=0;i<c.y.size();++i) if(working[i]>0 && dmu[i]!=0)
                    z[i]+=phi*h[i]*second[i]/(2*dmu[i]*working[i]);
            }
        }
        current.add(c.X,z,working);
        if(firth && fam.code>=10 && fam.code<=12)
            dispersion.add(fglm::gamma_weight_groups(c.w),phi);
        if(intercept_mask) for(int j=0;j<p;++j)
            if((c.X.col(j).array()!=c.X(0,j)).any()) (*intercept_mask)[j]=false;
    }
    current.finish();
    if(firth && gamma && previous) current.qty.noalias()+=phi*(*gamma_adjustment);
    if(n_total_out) *n_total_out=n_total;
    if(phi_out) {
        *phi_out=1;
        if(firth && !fam.is_unit_dispersion()) {
            if(n_positive<=p) Rcpp::stop("bias reduction requires positive residual degrees of freedom");
            if(fam.code>=10 && fam.code<=12) {
                const double step=dispersion.numerator(dev,p,phi)*phi/dispersion.d2;
                *phi_out=phi*std::exp(std::max(-std::log(2.0),std::min(std::log(2.0),step)));
            } else *phi_out=dev/(n_positive-p);
        }
    }
    return dev;
}

struct QRStationarity {
    double predictor_step=NA_REAL, score_rms=NA_REAL;
    double dispersion_step=NA_REAL, dispersion_score_rms=NA_REAL;
    bool fixed_dispersion=true;
    bool refined=false;
    bool satisfied(double tol) const {
        return std::isfinite(predictor_step) && predictor_step<=tol &&
            std::isfinite(score_rms) && score_rms<=tol &&
            (fixed_dispersion || (std::isfinite(dispersion_step) && dispersion_step<=tol &&
                std::isfinite(dispersion_score_rms) && dispersion_score_rms<=tol));
    }
};

// Verify the remaining adjusted score at the reported coefficients, using the
// current information factor and current leverages (never the lagged update).
// If sqrt(W) X D^-1 P = Q R, q = R^-T P' D^-1 phi U* is accumulated directly
// as (X D^-1 P R^-1)' phi u*. Its information step is D^-1 P R^-1 q.
// Both the fitted-link step and ||q||/sqrt(n phi) are invariant to nonsingular
// changes of coefficient coordinates. Marginal score standardization is not.
static QRStationarity stream_qr_stationarity(Function& callback,int n_chunks,int p,
    const VectorXd& beta,const FamilyOps& fam,const fglm::StreamingQR& qr,double phi,double tol)
{
    QRStationarity result;
    result.fixed_dispersion=fam.is_unit_dispersion();
    result.refined=std::numeric_limits<double>::epsilon()*qr.condition_bound()>tol;
    fglm::StreamingQR refined;
    if(result.refined) {
        // The original QR is backward stable, but X R^-1 can amplify its small
        // backward error in near-dependent directions. Re-factor this already
        // well-conditioned basis for verification, retaining only a p by p
        // factor. Accurate triangular arithmetic preserves the original design.
        refined.reset(p);
        for(int k=1;k<=n_chunks;++k) {
            Chunk c=pull_chunk(callback,k,p);
            ArrayXd eta=fglm::qr_linear_predictor(c.X,beta,c.off).array(),mu,dmu,variance;
            fam.linkinv(eta,mu); fam.mu_eta(eta,dmu); fam.variance(mu,variance);
            VectorXd working=(c.w.array()*dmu.square()/variance).matrix();
            if(fam.code==fglm::FAM_GAMMA_LOG || fam.code==fglm::FAM_GAUSSIAN_IDENTITY) working=c.w;
            for(int i=0;i<c.y.size();++i) if(c.w[i]==0) working[i]=0;
            refined.add(qr.accurate_row_basis(c.X),VectorXd::Zero(c.y.size()),working);
        }
        refined.finish();
    }
    VectorXd q=VectorXd::Zero(p);
    double dev=0; int n_positive=0;
    fglm::GammaDispersionTerms dispersion;
    for(int k=1;k<=n_chunks;++k) {
        Chunk c=pull_chunk(callback,k,p);
        ArrayXd eta=fglm::qr_linear_predictor(c.X,beta,c.off).array(),mu,dmu,variance,second;
        fam.linkinv(eta,mu); fam.mu_eta(eta,dmu); fam.variance(mu,variance);
        if(!fam.valideta(eta) || !fam.validmu(mu)) return result;
        MatrixXd basis=result.refined ? refined.row_basis(qr.accurate_row_basis(c.X)) : qr.row_basis(c.X);
        ArrayXd contribution=c.w.array()*dmu*(c.y.array()-mu)/variance;
        if(fam.code!=fglm::FAM_GAUSSIAN_IDENTITY) {
            fam.d2mu_deta2(eta,mu,dmu,second);
            ArrayXd working=c.w.array()*dmu.square()/variance;
            if(fam.code==fglm::FAM_GAMMA_LOG) working=c.w.array();
            ArrayXd h=basis.array().square().rowwise().sum()*working;
            for(int i=0;i<c.y.size();++i) if(c.w[i]>0 && dmu[i]!=0)
                contribution[i]+=phi*h[i]*second[i]/(2*dmu[i]);
        }
        for(int i=0;i<c.y.size();++i) if(c.w[i]==0) contribution[i]=0;
        q.noalias()+=basis.adjoint()*contribution.matrix();
        n_positive+=(c.w.array()>0).count();
        dev+=fam.dev_resids_sum(c.y.array(),mu,c.w.array());
        if(fam.code>=10 && fam.code<=12)
            dispersion.add(fglm::gamma_weight_groups(c.w),phi);
    }
    if(n_positive<=0 || (!result.fixed_dispersion && n_positive<=p) || !(phi>0) || !q.allFinite()) return result;
    result.score_rms=q.stableNorm()/std::sqrt(n_positive*phi);
    if(!result.fixed_dispersion) {
        const bool gamma=fam.code>=10 && fam.code<=12;
        const double numerator=gamma ? dispersion.numerator(dev,p,phi) : dev-(n_positive-p)*phi;
        const double d2=gamma ? dispersion.d2 : n_positive*phi*phi;
        result.dispersion_step=std::abs(numerator*phi/d2);
        result.dispersion_score_rms=std::abs(numerator)/std::sqrt(2*d2*n_positive);
    }
    // A second, linear-cost pass evaluates the step rather than an upper bound.
    // Compensated products protect cancelling step coefficients as well as beta.
    VectorXd step=result.refined ? refined.solve_transformed(q) : qr.solve_transformed(q);
    if(!step.allFinite()) return result;
    result.predictor_step=0;
    for(int k=1;k<=n_chunks;++k) {
        Chunk c=pull_chunk(callback,k,p);
        MatrixXd design=result.refined ? qr.accurate_row_basis(c.X) : c.X;
        VectorXd change=fglm::qr_linear_predictor(design,step,VectorXd::Zero(c.y.size()));
        for(int i=0;i<c.y.size();++i) if(c.w[i]>0) {
            if(!std::isfinite(change[i])) {result.predictor_step=NA_REAL;return result;}
            result.predictor_step=std::max(result.predictor_step,std::abs(change[i]));
        }
    }
    return result;
}

}  // anonymous namespace


// [[Rcpp::export]]
NumericVector streaming_qr_eta(NumericMatrix x,NumericVector beta,NumericVector offset)
{
    if(x.ncol()!=beta.size() || x.nrow()!=offset.size())
        Rcpp::stop("incompatible streaming QR predictor dimensions");
    return wrap(fglm::qr_linear_predictor(as<MatrixXd>(x),as<VectorXd>(beta),as<VectorXd>(offset)));
}

// [[Rcpp::export]]
List streaming_qr_update(NumericMatrix x,NumericVector z,NumericVector weights,
                         Nullable<List> state=R_NilValue)
{
    if(x.ncol()<1 || z.size()!=x.nrow() || weights.size()!=x.nrow())
        Rcpp::stop("incompatible streaming QR block dimensions");
    fglm::StreamingQR qr;
    if(state.isNotNull()) {
        List old(state.get());
        qr.raw_R=as<MatrixXd>(old["R"]);qr.raw_qty=as<VectorXd>(old["qty"]);
    } else qr.reset(x.ncol());
    if(qr.raw_R.rows()!=x.ncol() || qr.raw_R.cols()!=x.ncol() || qr.raw_qty.size()!=x.ncol())
        Rcpp::stop("incompatible streaming QR state dimensions");
    qr.add(as<MatrixXd>(x),as<VectorXd>(z),as<VectorXd>(weights));
    return List::create(_["R"]=qr.raw_R,_["qty"]=qr.raw_qty);
}

// [[Rcpp::export]]
List streaming_qr_finish(List state)
{
    fglm::StreamingQR qr;
    qr.raw_R=as<MatrixXd>(state["R"]);qr.raw_qty=as<VectorXd>(state["qty"]);
    qr.finish();
    return List::create(_["coefficients"]=qr.solve(),_["qr"]=qr.factor());
}

// [[Rcpp::export]]
List fit_streaming_glm(Function chunk_callback,
                       int      n_chunks,
                       int      p,
                       int      type,
                       double   tol,
                       int      maxit,
                       int      fam_code,
                       Function var,
                       Function mu_eta,
                       Function linkinv,
                       Function dev_resids,
                       Function valideta,
                       Function validmu,
                       Nullable<NumericVector> start = R_NilValue,
                       Nullable<NumericVector> fam_params = R_NilValue,
                       bool     firth = false,
                       Nullable<NumericVector> column_maxabs = R_NilValue)
{
    if (n_chunks < 1) Rcpp::stop("n_chunks must be >= 1");
    if (type != 0 && type != 2 && type != 3)
        Rcpp::stop("for streaming fits, 'method' must be 0 (QR), 2 (LLT), or 3 (LDLT).");

    fglm::FamilyParams fp;
    if (fam_params.isNotNull()) {
        NumericVector fpv(fam_params.get());
        if (fpv.size() >= 1) fp.theta      = fpv[0];
        if (fpv.size() >= 2) fp.var_power  = fpv[1];
        if (fpv.size() >= 3) fp.link_power = fpv[2];
    }

    FamilyOps fam(fam_code, fp, var, mu_eta, linkinv, dev_resids, valideta, validmu);

    VectorXd beta(p);
    if (start.isNotNull()) {
        NumericVector s(start.get());
        if ((int)s.size() != p)
            Rcpp::stop("length(start) does not match number of columns");
        beta = Map<VectorXd>(s.begin(), p);
    } else {
        beta.setZero();
    }

    MatrixXd XtWX(p, p);
    VectorXd Xtwz(p);
    std::vector<bool> intercept_mask(p, true);

    fglm::StreamingQR qr;
    VectorXd initial_adjustment;
    int n_total = 0;
    double dev_curr = type==0 ? stream_qr_pass(chunk_callback,n_chunks,p,beta,fam,qr,nullptr,nullptr,
        false,1,nullptr,&initial_adjustment,&intercept_mask,&n_total) : stream_pass(chunk_callback, n_chunks, p, beta, fam,
                                  /*accumulate=*/!firth,
                                  /*XtWX_out=*/&XtWX, /*Xtwz_out=*/&Xtwz,
                                  &intercept_mask, &n_total);

    bool converged = false;
    int  iter      = 0;
    Eigen::LLT<MatrixXd>  llt;
    Eigen::LDLT<MatrixXd> ldlt;

    MatrixXd A_prev(p, p);
    bool A_prev_available = false;
    double log_det_XtWX = 0.0;
    double phi_prev = 1.0;
    bool constant_information = fam.code == fglm::FAM_GAUSSIAN_IDENTITY ||
        fam.code == fglm::FAM_POISSON_SQRT || fam.code == fglm::FAM_GAMMA_LOG;
    VectorXd gamma_adjustment;

    bool gamma_step_failed = false;
    bool firth_step_failed = false;
    int firth_step_reductions = 0;
    VectorXd predictor_bound;
    if(column_maxabs.isNotNull()) {
        predictor_bound=as<VectorXd>(column_maxabs.get());
        if(predictor_bound.size()!=p || !predictor_bound.allFinite() || (predictor_bound.array()<0).any())
            stop("invalid predictor bounds for streaming step control");
    }
    auto limit_firth_step = [&](VectorXd& proposed) {
        if(!firth || !fglm::bounded_firth_predictors(fam.code)) return false;
        const VectorXd delta=proposed-beta;
        if(!delta.allFinite()) {firth_step_failed=true;return false;}
        // Column bounds, already available from source validation, avoid a
        // source pass for small updates. An actual predictor check is required
        // only when that conservative bound exceeds one link-scale unit.
        if(predictor_bound.size()==p && predictor_bound.dot(delta.cwiseAbs())<=1) return false;
        double fraction=1;
        for(int k=1;k<=n_chunks;++k) {
            Chunk c=pull_chunk(chunk_callback,k,p);
            const VectorXd change=fglm::qr_linear_predictor(c.X,delta,VectorXd::Zero(c.y.size()));
            const double value=fglm::firth_predictor_step_fraction(change,c.w);
            if(!std::isfinite(value)) {firth_step_failed=true;return false;}
            fraction=std::min(fraction,value);
        }
        if(fraction<1) {proposed=beta+fraction*delta;++firth_step_reductions;return true;}
        return false;
    };
    QRStationarity qr_stationarity;
    bool qr_stationarity_current=false;
    const bool safeguarded_gamma = firth && fam.code == fglm::FAM_GAMMA_LOG;
    if(type==0) {
        fglm::StreamingQR previous=qr;
        VectorXd leverage_beta=beta;
        if(safeguarded_gamma) {
            double phi_next=phi_prev;
            dev_curr=stream_qr_pass(chunk_callback,n_chunks,p,beta,fam,qr,&previous,&leverage_beta,
                true,phi_prev,&phi_next,&gamma_adjustment);
            for(iter=1;iter<=maxit;++iter) {
                VectorXd proposed=qr.solve(),delta=proposed-beta;
                const VectorXd direction=previous.coefficient_coordinates(delta);
                // This inexpensive update check schedules exact verification
                // only; neither it nor a small observed step declares convergence.
                if(direction.stableNorm()/std::sqrt(n_total*phi_prev)<=tol &&
                    std::abs(phi_next-phi_prev)/phi_prev<=tol) {
                    qr_stationarity=stream_qr_stationarity(chunk_callback,n_chunks,p,beta,fam,qr,phi_prev,tol);
                    qr_stationarity_current=true;
                    if(qr_stationarity.satisfied(tol)) {converged=true;break;}
                }
                const double quadratic=direction.squaredNorm(),phi_proposed=phi_next;
                double fraction=1,dev_new=0,phi_after=phi_proposed; int halves=0;
                while(true) {
                    bool valid=true;
                    try {
                        dev_new=stream_qr_pass(chunk_callback,n_chunks,p,proposed,fam,qr,&previous,&leverage_beta,
                            true,phi_proposed,&phi_after,&gamma_adjustment);
                    } catch(const std::domain_error&) { valid=false;dev_new=std::numeric_limits<double>::infinity(); }
                    const double change=dev_new-dev_curr-2*phi_prev*gamma_adjustment.dot(
                        previous.coefficient_coordinates(proposed-beta));
                    const double rounding=64*std::numeric_limits<double>::epsilon()*(1+std::abs(dev_new)+std::abs(dev_curr));
                    if(valid && std::isfinite(dev_new) && change<=-0.2*fraction*quadratic+rounding) break;
                    if(++halves>50) {gamma_step_failed=true;break;}
                    fraction*=0.5; proposed=beta+fraction*delta;
                }
                if(gamma_step_failed) break;
                beta=proposed; phi_prev=phi_proposed; phi_next=phi_after; dev_curr=dev_new;
                qr_stationarity_current=false;
                Rcpp::checkUserInterrupt();
            }
        } else for(iter=1;iter<=maxit;++iter) {
            double phi_new=phi_prev;
            if(firth) {
                const bool current_information=fglm::bounded_firth_predictors(fam.code);
                if(current_information) {
                    // Build current information before evaluating the adjustment.
                    // Reusing this factor for projection avoids a second QR.
                    stream_qr_pass(chunk_callback,n_chunks,p,beta,fam,qr,nullptr,nullptr,
                        false,1,nullptr,&gamma_adjustment);
                    previous=qr;
                }
                dev_curr=stream_qr_pass(chunk_callback,n_chunks,p,beta,fam,qr,&previous,
                    current_information ? nullptr : &leverage_beta,true,phi_prev,&phi_new,&gamma_adjustment);
            }
            VectorXd proposed=qr.solve();
            if(firth) {
                limit_firth_step(proposed);
                if(firth_step_failed) break;
                const VectorXd direction=qr.coefficient_coordinates(proposed-beta);
                if(direction.stableNorm()/std::sqrt(n_total*phi_prev)<=tol &&
                    std::abs(phi_new-phi_prev)/phi_prev<=tol) {
                    qr_stationarity=stream_qr_stationarity(chunk_callback,n_chunks,p,beta,fam,qr,phi_prev,tol);
                    qr_stationarity_current=true;
                    if(qr_stationarity.satisfied(tol)) {converged=true;break;}
                }
                previous=qr; leverage_beta=beta;
                beta=proposed;phi_prev=phi_new;
                qr_stationarity_current=false;
            } else {
                double dev_new=0; bool valid=false; int halves=0;
                do {
                    try {
                        dev_new=stream_qr_pass(chunk_callback,n_chunks,p,proposed,fam,qr,nullptr,nullptr,
                            false,1,nullptr,&gamma_adjustment);
                        valid=std::isfinite(dev_new);
                    } catch(const std::domain_error&) {valid=false;dev_new=std::numeric_limits<double>::infinity();}
                    if(valid && dev_new<=dev_curr*(1+tol)) break;
                    if(++halves>25) break;
                    proposed=0.5*(proposed+beta);
                } while(true);
                if(!valid || dev_new>dev_curr*(1+tol)) Rcpp::stop("streaming QR step halving failed");
                const double change=std::abs(dev_new-dev_curr)/(0.1+std::abs(dev_new));
                beta=proposed;dev_curr=dev_new;
                if(change<tol) {converged=true;break;}
            }
            Rcpp::checkUserInterrupt();
        }
        const bool qr_constant=fam.code==fglm::FAM_GAMMA_LOG || fam.code==fglm::FAM_GAUSSIAN_IDENTITY;
        if(firth && !qr_constant && !qr_stationarity_current)
            stream_qr_pass(chunk_callback,n_chunks,p,beta,fam,qr,nullptr,nullptr,false,1,nullptr,&gamma_adjustment);
        if(firth && !qr_stationarity_current)
            qr_stationarity=stream_qr_stationarity(chunk_callback,n_chunks,p,beta,fam,qr,phi_prev,tol);
        // The scheduling gate is not part of the convergence definition. A
        // final exact check can certify the last allowed accepted iterate.
        if(firth && !gamma_step_failed && !firth_step_failed && qr_stationarity.satisfied(tol)) converged=true;
        log_det_XtWX=qr.logdet();
    }
    if (type!=0 && safeguarded_gamma) {
        double phi_next = phi_prev;
        dev_curr = stream_pass_firth(chunk_callback,n_chunks,p,beta,fam,nullptr,
            phi_prev,&XtWX,&Xtwz,nullptr,&phi_next,&constant_information,&gamma_adjustment);
        spd_factorize_and_solve(type,XtWX,Xtwz,llt,ldlt);
        MatrixXd identity = MatrixXd::Identity(p,p);
        if (type == 2) {
            A_prev = llt.solve(identity);
            log_det_XtWX = 2*llt.matrixLLT().diagonal().array().log().sum();
        } else {
            A_prev = ldlt.solve(identity);
            log_det_XtWX = ldlt.vectorD().array().log().sum();
        }
        A_prev_available = true;
        // Build the constant leverage adjustment once. Each accepted trial
        // also supplies the next iteration's score, so line search adds a
        // source pass only when a proposed step is rejected.
        dev_curr = stream_pass_firth(chunk_callback,n_chunks,p,beta,fam,&A_prev,
            phi_prev,&XtWX,&Xtwz,nullptr,&phi_next,&constant_information,&gamma_adjustment);
        for (iter=1;iter<=maxit;++iter) {
            VectorXd proposed;
            if (type == 2) proposed = llt.solve(Xtwz);
            else proposed = ldlt.solve(Xtwz);
            const VectorXd delta = proposed-beta;
            const double quadratic = delta.dot(XtWX*delta);
            const double phi_proposed = phi_next;
            double fraction = 1, dev_new = 0, phi_after = phi_proposed;
            int halves = 0;
            while (true) {
                bool valid = true;
                try {
                    dev_new = stream_pass_firth(chunk_callback,n_chunks,p,proposed,fam,&A_prev,
                        phi_proposed,&XtWX,&Xtwz,nullptr,&phi_after,&constant_information,&gamma_adjustment);
                } catch (const std::domain_error&) {
                    valid = false; dev_new = std::numeric_limits<double>::infinity();
                }
                const double change = dev_new-dev_curr -
                    2*phi_prev*gamma_adjustment.dot(proposed-beta);
                const double rounding = 64*std::numeric_limits<double>::epsilon()*
                    (1+std::abs(dev_new)+std::abs(dev_curr));
                if (valid && std::isfinite(dev_new) &&
                    change <= -0.2*fraction*quadratic + rounding) break;
                if (++halves > 50) { gamma_step_failed = true; break; }
                fraction *= 0.5;
                proposed = beta + fraction*delta;
            }
            if (gamma_step_failed) break;
            beta = proposed; phi_prev = phi_proposed; dev_curr = dev_new; phi_next = phi_after;
            VectorXd score_step;
            if (type == 2) score_step = llt.solve(Xtwz)-beta;
            else score_step = ldlt.solve(Xtwz)-beta;
            if (score_step.cwiseAbs().maxCoeff() < tol &&
                std::abs(phi_next-phi_prev)/phi_prev < tol) {
                converged = true; break;
            }
            Rcpp::checkUserInterrupt();
        }
    }

    if (type!=0 && !safeguarded_gamma) for (iter = 1; iter <= maxit; ++iter) {
        double std_dev, phi_new = phi_prev;
        const bool current_information=firth && fglm::bounded_firth_predictors(fam.code);
        if (firth) {
            if(current_information) {
                stream_pass(chunk_callback,n_chunks,p,beta,fam,true,&XtWX,&Xtwz,nullptr,nullptr);
                spd_factorize_and_solve(type,XtWX,Xtwz,llt,ldlt);
                const MatrixXd identity=MatrixXd::Identity(p,p);
                if(type==2) A_prev=llt.solve(identity); else A_prev=ldlt.solve(identity);
                A_prev_available=true;
            }
            std_dev = stream_pass_firth(chunk_callback, n_chunks, p, beta, fam,
                                        A_prev_available ? &A_prev : nullptr,
                                        phi_prev,
                                        &XtWX, &Xtwz, nullptr, &phi_new,
                                        &constant_information, &gamma_adjustment,current_information);
        } else {
            // The accepted proposal already accumulated the next WLS system.
            std_dev = dev_curr;
        }
        dev_curr = std_dev;

        const bool reuse_factor = firth && (constant_information || current_information) && A_prev_available;
        VectorXd beta_new;
        if (reuse_factor) {
            if (type == 2) beta_new = llt.solve(Xtwz);
            else beta_new = ldlt.solve(Xtwz);
        } else beta_new = spd_factorize_and_solve(type, XtWX, Xtwz, llt, ldlt);

        if (firth) {
            if (!reuse_factor || current_information) {
                // Compute log|X'WX| for penalized deviance
                double ld = 0.0;
                if (type == 2) {
                    const MatrixXd &Lm = llt.matrixLLT();
                    for (int j = 0; j < p; ++j) ld += std::log(Lm(j, j));
                    log_det_XtWX = 2.0 * ld;
                } else {
                    VectorXd D = ldlt.vectorD();
                    for (int j = 0; j < p; ++j) ld += std::log(std::abs(D[j]));
                    log_det_XtWX = ld;
                }
                // Update A_prev for next iteration
                MatrixXd I_p = MatrixXd::Identity(p, p);
                if (type == 2) A_prev = llt.solve(I_p);
                else           A_prev = ldlt.solve(I_p);
                A_prev_available = true;
            }

            double phi_change = std::abs(phi_new - phi_prev) / (1 + std::abs(phi_prev));
            phi_prev = phi_new;

            // Firth convergence: sup-norm of coefficient change
            const bool limited=limit_firth_step(beta_new);
            if(firth_step_failed) break;
            double beta_change = (beta_new - beta).cwiseAbs().maxCoeff();
            beta = beta_new;
            if (!limited && beta_change < tol && phi_change < tol) { converged = true; break; }
        } else {
            // Deviance at proposed beta (no accumulation).
            double dev_new;
            bool   pass_ok;
            try {
                dev_new = stream_pass(chunk_callback, n_chunks, p, beta_new, fam,
                                      true, &XtWX, &Xtwz, nullptr, nullptr);
                pass_ok = true;
            } catch (const std::domain_error&) {
                dev_new = std::numeric_limits<double>::infinity();
                pass_ok = false;
            }

            // Step-halving (Marschner 2011).
            int halv = 0;
            while ((!pass_ok || !std::isfinite(dev_new) ||
                    dev_new > dev_curr * (1.0 + tol)) && halv < 25) {
                beta_new = 0.5 * (beta_new + beta);
                try {
                    dev_new = stream_pass(chunk_callback, n_chunks, p, beta_new, fam,
                                          true, &XtWX, &Xtwz, nullptr, nullptr);
                    pass_ok = true;
                } catch (const std::domain_error&) {
                    dev_new = std::numeric_limits<double>::infinity();
                    pass_ok = false;
                }
                ++halv;
            }

            if (!pass_ok || !std::isfinite(dev_new)) Rcpp::stop("streaming step halving failed");
            double rel = std::abs(dev_new - dev_curr) / (0.1 + std::abs(dev_new));
            beta     = beta_new;
            dev_curr = dev_new;
            if (rel < tol) { converged = true; break; }
        }
    }
    if (iter > maxit) iter = maxit;

    // Changing working weights require information at the reported beta.
    // Constant information is already exact there; reuse it without a pass.
    if (type!=0 && firth && !constant_information) stream_pass(chunk_callback, n_chunks, p, beta, fam,
                /*accumulate=*/true, &XtWX, &Xtwz, nullptr, nullptr);

    // cov.unscaled = (X'WX)^{-1}.
    MatrixXd cov_unscaled = type==0 ? qr.covariance() : spd_inverse(type, XtWX);

    // Final pass: total weighted y, total weights, deviance, null deviance,
    // Pearson chi-square (for dispersion).
    bool has_intercept = false;
    for (int j = 0; j < p; ++j) if (intercept_mask[j]) { has_intercept = true; break; }

    double sum_wy = 0.0, sum_w = 0.0;
    int n_ok = 0;
    {
        for (int k = 1; k <= n_chunks; ++k) {
            Chunk c = pull_chunk(chunk_callback, k, p);
            const int n_k = c.y.size();
            sum_wy += (c.w.array() * c.y.array()).sum();
            sum_w  += c.w.sum();
            for (int i = 0; i < n_k; ++i) if (c.w(i) != 0.0) ++n_ok;
        }
    }
    double wtdmu = has_intercept && sum_w > 0.0 ? sum_wy / sum_w : NA_REAL;

    double dev_final  = 0.0;
    double null_dev   = 0.0;
    double pearson    = 0.0;
    VectorXd adjusted_score = VectorXd::Zero(p);
    fglm::GammaDispersionTerms final_dispersion_terms;
    {
        for (int k = 1; k <= n_chunks; ++k) {
            Chunk c = pull_chunk(chunk_callback, k, p);
            const int n_k = c.y.size();

            ArrayXd eta = type==0 ? fglm::qr_linear_predictor(c.X,beta,c.off).array() :
                ((c.X * beta).array() + c.off.array()).eval();
            ArrayXd mu;
            fam.linkinv(eta, mu);
            ArrayXd ya = c.y.array();
            ArrayXd wa = c.w.array();
            dev_final += fam.dev_resids_sum(ya, mu, wa);

            ArrayXd va;
            fam.variance(mu, va);
            pearson += (wa * (ya - mu).square() / va).sum();
            if (firth) {
                ArrayXd dmu; fam.mu_eta(eta,dmu);
                ArrayXd contribution = wa*dmu*(ya-mu)/(phi_prev*va);
                for (int i=0;i<n_k;++i) if (wa[i] == 0) contribution[i] = 0;
                if (fam.code != fglm::FAM_GAUSSIAN_IDENTITY &&
                    fam.code != fglm::FAM_GAMMA_LOG) {
                    ArrayXd d2mu; fam.d2mu_deta2(eta,mu,dmu,d2mu);
                    MatrixXd xa = type==0 ? qr.row_basis(c.X) : MatrixXd(c.X*cov_unscaled);
                    ArrayXd leverage = type==0 ? xa.array().square().rowwise().sum().eval() :
                        (xa.array()*c.X.array()).rowwise().sum().eval();
                    ArrayXd h = leverage*wa*dmu.square()/va;
                    for (int i=0;i<n_k;++i) if (wa[i]>0 && dmu[i]!=0)
                        contribution[i] += h[i]*d2mu[i]/(2*dmu[i]);
                }
                adjusted_score.noalias() += c.X.adjoint()*contribution.matrix();
                if (fam.code >= 10 && fam.code <= 12)
                    final_dispersion_terms.add(fglm::gamma_weight_groups(c.w),phi_prev);
            }

            ArrayXd mu0(n_k);
            if (has_intercept) {
                mu0.setConstant(wtdmu);
            } else {
                ArrayXd off_arr = c.off.array();
                fam.linkinv(off_arr, mu0);
            }
            null_dev += fam.dev_resids_sum(ya, mu0, wa);
        }
    }

    int    df_residual = n_ok - p;
    // Always return Pearson-based dispersion; the R wrapper overrides this to
    // 1 for fixed-dispersion families (binomial / poisson / negative binomial)
    // so that quasi-binomial / quasi-poisson (which share C++ family codes
    // with binomial / poisson) still get an estimated dispersion.
    double dispersion = (df_residual > 0) ? pearson / (double)df_residual
                                          : std::numeric_limits<double>::quiet_NaN();

    int df_null = n_ok - (has_intercept ? 1 : 0);

    // Standard errors = sqrt(diag(cov.unscaled)) * sqrt(dispersion) (when defined)
    VectorXd se(p);
    for (int j = 0; j < p; ++j)
        se(j) = std::sqrt(cov_unscaled(j, j));
    if (std::isfinite(dispersion))
        se *= std::sqrt(dispersion);

    LogicalVector intercept_lgl(p);
    for (int j = 0; j < p; ++j) intercept_lgl[j] = intercept_mask[j];

    List out = List::create(
        _["coefficients"] = beta,
        _["se"]           = se,
        _["cov.unscaled"] = cov_unscaled,
        _["deviance"]     = dev_final,
        _["null.deviance"]= null_dev,
        _["pearson"]      = pearson,
        _["dispersion"]   = dispersion,
        _["df.residual"]  = df_residual,
        _["df.null"]      = df_null,
        _["n"]            = n_total,
        _["rank"]         = p,
        _["iter"]         = iter,
        _["converged"]    = converged,
        _["intercept_mask"] = intercept_lgl,
        _["has_intercept"] = has_intercept
    );

    if(type==0) out["qr"]=qr.factor();
    if (firth) {
        out["penalized.deviance"] = dev_final - log_det_XtWX;
        out["log.det.XtWX"]       = log_det_XtWX;
        out["bias.dispersion"]    = phi_prev;
        out["firth"]              = true;
        if (fam.code == fglm::FAM_GAMMA_LOG)
            adjusted_score += type==0 ? qr.original_score(gamma_adjustment) : gamma_adjustment;
        VectorXd information = type==0 ? qr.information_diagonal()/phi_prev : VectorXd(XtWX.diagonal()/phi_prev);
        double ds, ds_scaled;
        fglm::dispersion_diagnostics(fam.code,dev_final,n_ok,p,phi_prev,final_dispersion_terms,ds,ds_scaled);
        std::string reason = firth_step_failed ? "nonfinite_scoring_step" : gamma_step_failed ? "step_halving_failed" :
            (converged ? "converged" : "iteration_limit");
        fglm::append_adjusted_diagnostics(out,adjusted_score,information,ds,ds_scaled,reason);
        if(fglm::bounded_firth_predictors(fam.code)) out["step.reductions"]=firth_step_reductions;
        if(type==0) {
            out["remaining.predictor.step"]=qr_stationarity.predictor_step;
            out["adjusted.score.rms"]=qr_stationarity.score_rms;
            out["remaining.dispersion.step"]=qr_stationarity.dispersion_step;
            out["adjusted.dispersion.score.rms"]=qr_stationarity.dispersion_score_rms;
            out["refined.convergence.verification"]=qr_stationarity.refined;
        }
    }

    return out;
}
