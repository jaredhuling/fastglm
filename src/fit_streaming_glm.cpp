#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include <RcppEigen.h>
#include "../inst/include/families.h"
#include "../inst/include/chunk_source.h"
#include "../inst/include/bias_reduction.h"

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
                                VectorXd* gamma_adjustment)
{
    double dev_sum = 0.0;
    int n_total = 0, n_positive = 0;
    fglm::GammaDispersionTerms dispersion_terms;
    const bool reuse_information = *constant_information && A_prev != nullptr;
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

}  // anonymous namespace


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
                       bool     firth = false)
{
    if (n_chunks < 1) Rcpp::stop("n_chunks must be >= 1");
    if (type != 2 && type != 3)
        Rcpp::stop("for streaming fits, 'method' must be 2 (LLT) or 3 (LDLT).");

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

    int n_total = 0;
    double dev_curr = stream_pass(chunk_callback, n_chunks, p, beta, fam,
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
    const bool safeguarded_gamma = firth && fam.code == fglm::FAM_GAMMA_LOG;
    if (safeguarded_gamma) {
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

    if (!safeguarded_gamma)     for (iter = 1; iter <= maxit; ++iter) {
        double std_dev, phi_new = phi_prev;
        if (firth) {
            std_dev = stream_pass_firth(chunk_callback, n_chunks, p, beta, fam,
                                        A_prev_available ? &A_prev : nullptr,
                                        phi_prev,
                                        &XtWX, &Xtwz, nullptr, &phi_new,
                                        &constant_information, &gamma_adjustment);
        } else {
            // The accepted proposal already accumulated the next WLS system.
            std_dev = dev_curr;
        }
        dev_curr = std_dev;

        const bool reuse_factor = firth && constant_information && A_prev_available;
        VectorXd beta_new;
        if (reuse_factor) {
            if (type == 2) beta_new = llt.solve(Xtwz);
            else beta_new = ldlt.solve(Xtwz);
        } else beta_new = spd_factorize_and_solve(type, XtWX, Xtwz, llt, ldlt);

        if (firth) {
            if (!reuse_factor) {
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
            double beta_change = (beta_new - beta).cwiseAbs().maxCoeff();
            beta = beta_new;
            if (beta_change < tol && phi_change < tol) { converged = true; break; }
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
    if (firth && !constant_information) stream_pass(chunk_callback, n_chunks, p, beta, fam,
                /*accumulate=*/true, &XtWX, &Xtwz, nullptr, nullptr);

    // cov.unscaled = (X'WX)^{-1}.
    MatrixXd cov_unscaled = spd_inverse(type, XtWX);

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

            ArrayXd eta = (c.X * beta).array() + c.off.array();
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
                    MatrixXd xa = c.X*cov_unscaled;
                    ArrayXd h = (xa.array()*c.X.array()).rowwise().sum()*wa*dmu.square()/va;
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

    if (firth) {
        out["penalized.deviance"] = dev_final - log_det_XtWX;
        out["log.det.XtWX"]       = log_det_XtWX;
        out["bias.dispersion"]    = phi_prev;
        out["firth"]              = true;
        if (fam.code == fglm::FAM_GAMMA_LOG) adjusted_score += gamma_adjustment;
        VectorXd information = XtWX.diagonal()/phi_prev;
        double ds, ds_scaled;
        fglm::dispersion_diagnostics(fam.code,dev_final,n_ok,p,phi_prev,final_dispersion_terms,ds,ds_scaled);
        std::string reason = gamma_step_failed ? "step_halving_failed" :
            (converged ? "converged" : "iteration_limit");
        fglm::append_adjusted_diagnostics(out,adjusted_score,information,ds,ds_scaled,reason);
    }

    return out;
}
