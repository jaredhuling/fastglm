#define EIGEN_DONT_PARALLELIZE

#include <Rcpp.h>
#include <RcppEigen.h>
#include "../inst/include/families.h"

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
            Rcpp::stop("invalid linear predictor in streaming pass");

        ArrayXd mu;
        fam.linkinv(eta, mu);
        if (!fam.validmu(mu))
            Rcpp::stop("invalid mean in streaming pass");

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
                       Nullable<NumericVector> fam_params = R_NilValue)
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
                                  /*accumulate=*/false,
                                  /*XtWX_out=*/nullptr, /*Xtwz_out=*/nullptr,
                                  &intercept_mask, &n_total);

    bool converged = false;
    int  iter      = 0;
    Eigen::LLT<MatrixXd>  llt;
    Eigen::LDLT<MatrixXd> ldlt;

    for (iter = 1; iter <= maxit; ++iter) {
        // Build XtWX, Xtwz at the current beta and refresh dev_curr.
        dev_curr = stream_pass(chunk_callback, n_chunks, p, beta, fam,
                               /*accumulate=*/true,
                               &XtWX, &Xtwz, nullptr, nullptr);

        VectorXd beta_new(p);
        if (type == 2) {
            llt.compute(XtWX);
            if (llt.info() != Eigen::Success)
                Rcpp::stop("Cholesky factorisation (LLT) failed; "
                                         "design may be rank-deficient. "
                                         "Streaming mode requires full column rank.");
            beta_new = llt.solve(Xtwz);
        } else {
            ldlt.compute(XtWX);
            if (ldlt.info() != Eigen::Success)
                Rcpp::stop("Cholesky factorisation (LDLT) failed; "
                                         "design may be rank-deficient. "
                                         "Streaming mode requires full column rank.");
            beta_new = ldlt.solve(Xtwz);
        }

        // Deviance at proposed beta (no accumulation).
        double dev_new;
        bool   pass_ok;
        try {
            dev_new = stream_pass(chunk_callback, n_chunks, p, beta_new, fam,
                                  false, nullptr, nullptr, nullptr, nullptr);
            pass_ok = true;
        } catch (...) {
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
                                      false, nullptr, nullptr, nullptr, nullptr);
                pass_ok = true;
            } catch (...) {
                dev_new = std::numeric_limits<double>::infinity();
                pass_ok = false;
            }
            ++halv;
        }

        double rel = std::abs(dev_new - dev_curr) / (0.1 + std::abs(dev_new));
        beta     = beta_new;
        dev_curr = dev_new;
        if (rel < tol) { converged = true; break; }
    }
    if (iter > maxit) iter = maxit;

    // Final XtWX/Xtwz at the converged beta to get cov.unscaled.
    stream_pass(chunk_callback, n_chunks, p, beta, fam,
                /*accumulate=*/true, &XtWX, &Xtwz, nullptr, nullptr);

    // cov.unscaled = (X'WX)^{-1}.
    MatrixXd cov_unscaled(p, p);
    if (type == 2) {
        Eigen::LLT<MatrixXd> chol(XtWX);
        if (chol.info() != Eigen::Success)
            Rcpp::stop("Cholesky factorisation (LLT) failed; "
                                     "design may be rank-deficient. "
                                     "Streaming mode requires full column rank.");
        cov_unscaled = chol.solve(MatrixXd::Identity(p, p));
    } else {
        Eigen::LDLT<MatrixXd> chol(XtWX);
        if (chol.info() != Eigen::Success)
            Rcpp::stop("Cholesky factorisation (LDLT) failed; "
                                     "design may be rank-deficient. "
                                     "Streaming mode requires full column rank.");
        cov_unscaled = chol.solve(MatrixXd::Identity(p, p));
    }

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

    int    df_residual = n_total - p;
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

    return List::create(
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
}
