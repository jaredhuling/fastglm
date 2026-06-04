#ifndef GLM_SPARSE_H
#define GLM_SPARSE_H

#include "glm_base.h"
#include "families.h"
#include "chunk_source.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

class glm_sparse : public GlmBase<Eigen::VectorXd, Eigen::MatrixXd>
{
protected:
    typedef Eigen::SparseMatrix<double>            SpMat;
    typedef Eigen::Map<const SpMat>                MapSpMat;
    typedef Eigen::SimplicialLLT<SpMat>            SpLLT;
    typedef Eigen::SimplicialLDLT<SpMat>           SpLDLT;
    typedef Eigen::Map<Eigen::MatrixXd>            MapMat;
    typedef Eigen::Map<Eigen::VectorXd>            MapVec;

    const MapSpMat                X;        // sparse model matrix
    const Eigen::Map<Eigen::VectorXd> Y;
    const Eigen::Map<Eigen::VectorXd> weights;
    const Eigen::Map<Eigen::VectorXd> offset;

    Rcpp::Function variance_fun;
    Rcpp::Function mu_eta_fun;
    Rcpp::Function linkinv;
    Rcpp::Function dev_resids_fun;
    Rcpp::Function valideta;
    Rcpp::Function validmu;

    double tol;
    int maxit;
    int type;       // 2 = LLT, 3 = LDLT (only)
    int rank;
    int fam_code;
    fglm::FamilyParams fam_params;
    bool firth_;
    double log_det_XtWX_;
    Eigen::MatrixXd A_prev;
    bool A_prev_available;

    // Sparse decompositions (analyzePattern done once, factorize per iter)
    SpLLT  Ch;
    SpLDLT ChD;
    bool   pattern_analyzed;

    // X' W^2 X is sparse (with the sparsity pattern of X^T X).
    SpMat  XtWX;
    Eigen::VectorXd Xtwz;

    // ------------------------------------------------------------------
    // family helpers (mirror glm::update_*)
    // ------------------------------------------------------------------
    virtual void update_mu_eta()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            Eigen::Map<Eigen::ArrayXd>       d(mu_eta.data(), mu_eta.size());
            fglm::mu_eta(fam_code, fam_params, e, d);
            return;
        }
        Rcpp::NumericVector v = mu_eta_fun(eta);
        std::copy(v.begin(), v.end(), mu_eta.data());
    }

    virtual void update_var_mu()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<Eigen::ArrayXd>       v(var_mu.data(), var_mu.size());
            fglm::variance(fam_code, fam_params, m, v);
            return;
        }
        Rcpp::NumericVector v = variance_fun(mu);
        std::copy(v.begin(), v.end(), var_mu.data());
    }

    virtual void update_mu()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            Eigen::Map<Eigen::ArrayXd>       m(mu.data(), mu.size());
            fglm::linkinv(fam_code, fam_params, e, m);
            return;
        }
        Rcpp::NumericVector v = linkinv(eta);
        std::copy(v.begin(), v.end(), mu.data());
    }

    virtual void update_eta()
    {
        eta.noalias() = X * beta;
        eta += offset;
    }

    virtual void update_z()
    {
        z = (eta - offset).array() + (Y - mu).array() / mu_eta.array();
    }

    virtual void update_w()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> me(mu_eta.data(), mu_eta.size());
            Eigen::Map<const Eigen::ArrayXd> pw(weights.data(), weights.size());
            Eigen::Map<Eigen::ArrayXd>       ww(w.data(), w.size());
            if (fglm::stable_weights(fam_code, fam_params, m, me, pw, ww))
                return;
        }
        w = (weights.array() * mu_eta.array().square() / var_mu.array()).array().sqrt();
    }

    virtual void update_dev_resids()
    {
        devold = dev;
        double std_dev;
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> y(Y.data(), Y.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> wt(weights.data(), weights.size());
            std_dev = fglm::dev_resids_sum(fam_code, fam_params, y, m, wt);
        } else {
            Rcpp::NumericVector dr = dev_resids_fun(Y, mu, weights);
            std_dev = Rcpp::sum(dr);
        }
        dev = firth_ ? (std_dev - log_det_XtWX_) : std_dev;
    }

    virtual void update_dev_resids_dont_update_old()
    {
        double std_dev;
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> y(Y.data(), Y.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> wt(weights.data(), weights.size());
            std_dev = fglm::dev_resids_sum(fam_code, fam_params, y, m, wt);
        } else {
            Rcpp::NumericVector dr = dev_resids_fun(Y, mu, weights);
            std_dev = Rcpp::sum(dr);
        }
        dev = firth_ ? (std_dev - log_det_XtWX_) : std_dev;
    }

    virtual void step_halve()
    {
        beta = 0.5 * (beta.array() + beta_prev.array());
        update_eta();
        update_mu();
    }

    bool valideta_check()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            return fglm::valideta(fam_code, fam_params, e);
        }
        return Rcpp::as<bool>(valideta(eta));
    }

    bool validmu_check()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            return fglm::validmu(fam_code, fam_params, m);
        }
        return Rcpp::as<bool>(validmu(mu));
    }

    bool converged() override
    {
        if (firth_)
            return (beta - beta_prev).cwiseAbs().maxCoeff() < tol;
        return std::abs(dev - devold) / (0.1 + std::abs(dev)) < tol;
    }

    virtual void run_step_halving(int &iterr)
    {
        const bool firth_skip = firth_;
        if (!std::isfinite(dev)) {
            int itrr = 0;
            while (!std::isfinite(dev)) {
                if (++itrr > maxit) break;
                step_halve();
                update_dev_resids_dont_update_old();
            }
        }
        if (!(valideta_check() && validmu_check())) {
            int itrr = 0;
            while (!(valideta_check() && validmu_check())) {
                if (++itrr > maxit) break;
                step_halve();
            }
            update_dev_resids_dont_update_old();
        }
        if (!firth_skip && (dev - devold) / (0.1 + std::abs(dev)) >= tol && iterr > 0) {
            int itrr = 0;
            while ((dev - devold) / (0.1 + std::abs(dev)) >= -tol) {
                if (++itrr > maxit) break;
                step_halve();
                update_dev_resids_dont_update_old();
            }
        }
    }

    // ------------------------------------------------------------------
    // Sparse linear-algebra helpers shared by the plain and Firth solves.
    // ------------------------------------------------------------------

    // Factorize A with the active sparse Cholesky (LLT for type 2, LDLT for
    // type 3) and solve A x = rhs.  analyzePattern is done once; the symbolic
    // factor and the numeric factor are retained for the post-solve readouts
    // (log|X'WX|, (X'WX)^{-1}).
    Eigen::VectorXd factorize_and_solve(const SpMat &A, const Eigen::VectorXd &rhs)
    {
        if (type == 2) {
            if (!pattern_analyzed) { Ch.analyzePattern(A); pattern_analyzed = true; }
            Ch.factorize(A);
            if (Ch.info() != Eigen::Success)
                Rcpp::stop("Sparse Cholesky factorization failed; X'WX may be singular or not positive definite");
            return Ch.solve(rhs);
        }
        if (!pattern_analyzed) { ChD.analyzePattern(A); pattern_analyzed = true; }
        ChD.factorize(A);
        if (ChD.info() != Eigen::Success)
            Rcpp::stop("Sparse Cholesky factorization failed; X'WX may be singular or not positive definite");
        return ChD.solve(rhs);
    }

    // (X' W^2 X)^{-1} from the retained factorization.
    Eigen::MatrixXd inverse_XtWX()
    {
        Eigen::MatrixXd I_p = Eigen::MatrixXd::Identity(nvars, nvars);
        if (type == 2) return Ch.solve(I_p);
        return ChD.solve(I_p);
    }

    // ------------------------------------------------------------------
    // Firth helpers (sparse path uses lagged leverages from A_prev)
    // ------------------------------------------------------------------
    void compute_d2mu_sparse(Eigen::ArrayXd &d2mu)
    {
        const double eps = std::numeric_limits<double>::epsilon();
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> d(mu_eta.data(), mu_eta.size());
            fglm::d2mu_deta2(fam_code, fam_params, e, m, d, d2mu);
        } else {
            const double h = std::pow(eps, 1.0 / 3.0);
            Eigen::VectorXd eta_p = eta.array() + h;
            Eigen::VectorXd eta_m = eta.array() - h;
            Rcpp::NumericVector dmu_p = mu_eta_fun(eta_p);
            Rcpp::NumericVector dmu_m = mu_eta_fun(eta_m);
            for (int i = 0; i < nobs; ++i)
                d2mu[i] = (dmu_p[i] - dmu_m[i]) / (2.0 * h);
        }
    }

    double compute_firth_phi_sparse()
    {
        const double eps = std::numeric_limits<double>::epsilon();
        bool unit_disp = (fam_code >= fglm::FAM_BINOMIAL_LOGIT &&
                          fam_code <= fglm::FAM_POISSON_SQRT);
        if (fam_code == fglm::FAM_UNKNOWN) unit_disp = true;
        if (unit_disp || nobs <= nvars) return 1.0;
        Eigen::Map<const Eigen::ArrayXd> y_a(Y.data(), Y.size());
        Eigen::Map<const Eigen::ArrayXd> m_a(mu.data(), mu.size());
        Eigen::Map<const Eigen::ArrayXd> w_a(weights.data(), weights.size());
        double phi = fglm::dev_resids_sum(fam_code, fam_params, y_a, m_a, w_a)
                     / (nobs - nvars);
        return (phi < eps) ? 1.0 : phi;
    }

    void compute_leverages_lagged(Eigen::VectorXd &h_out)
    {
        h_out.resize(nobs);
        int chunk = fglm::default_chunk_rows();
        for (Eigen::Index r = 0; r < nobs; r += chunk) {
            Eigen::Index k = std::min<Eigen::Index>(chunk, nobs - r);
            Eigen::MatrixXd X_k = X.middleRows(r, k);
            Eigen::MatrixXd XA_k = X_k * A_prev;
            h_out.segment(r, k) =
                ((XA_k.array() * X_k.array()).rowwise().sum()
                 * w.segment(r, k).array().square()).matrix();
        }
    }

    void compute_log_det_XtWX_sparse()
    {
        double ld = 0.0;
        if (type == 2) {
            SpMat Lsp = Ch.matrixL();
            for (int j = 0; j < nvars; ++j)
                ld += std::log(Lsp.coeff(j, j));
            log_det_XtWX_ = 2.0 * ld;
        } else {
            Eigen::VectorXd D = ChD.vectorD();
            for (int j = 0; j < D.size(); ++j)
                ld += std::log(std::abs(D[j]));
            log_det_XtWX_ = ld;
        }
    }

    void update_A_prev_sparse()
    {
        A_prev = inverse_XtWX();
        A_prev_available = true;
    }

    void solve_wls_firth()
    {
        beta_prev = beta;
        const double eps = std::numeric_limits<double>::epsilon();

        Eigen::ArrayXd d2mu(nobs);
        compute_d2mu_sparse(d2mu);
        double phi = compute_firth_phi_sparse();

        Eigen::VectorXd w2 = w.array().square();
        XtWX = SpMat(X.adjoint()) * w2.asDiagonal() * X;

        if (!A_prev_available) {
            Xtwz.noalias() = X.adjoint() * (w2.array() * z.array()).matrix();
        } else {
            Eigen::VectorXd h_lev;
            compute_leverages_lagged(h_lev);

            Eigen::ArrayXd dmu3 = mu_eta.array().cube();
            for (int i = 0; i < nobs; ++i)
                if (std::abs(dmu3[i]) < eps)
                    dmu3[i] = (dmu3[i] >= 0) ? eps : -eps;

            Eigen::ArrayXd xi = phi * h_lev.array() * d2mu * var_mu.array()
                              / (2.0 * weights.array().max(eps) * dmu3);
            Eigen::VectorXd zstar = z + xi.matrix();
            Xtwz.noalias() = X.adjoint() * (w2.array() * zstar.array()).matrix();
        }

        beta = factorize_and_solve(XtWX, Xtwz);
        rank = nvars;

        compute_log_det_XtWX_sparse();
        update_A_prev_sparse();
    }

    // ------------------------------------------------------------------
    // Core sparse WLS solve: X' diag(w^2) X beta = X' diag(w^2) z.
    // ------------------------------------------------------------------
    virtual void solve_wls(int /*iter*/)
    {
        if (firth_) { solve_wls_firth(); return; }

        beta_prev = beta;

        Eigen::VectorXd w2 = w.array().square();
        // X' W^2 X.  selfadjoint product would be ideal but Eigen's sparse
        // path here is already efficient.
        XtWX = SpMat(X.adjoint()) * w2.asDiagonal() * X;
        Xtwz.noalias() = X.adjoint() * (w2.array() * z.array()).matrix();

        beta = factorize_and_solve(XtWX, Xtwz);
        rank = nvars;  // assume full rank for sparse Cholesky (factorize will fail otherwise)
    }

    virtual void save_se()
    {
        // SE = sqrt(diag((X' W^2 X)^{-1})).
        se = inverse_XtWX().diagonal().array().sqrt();
    }

    virtual void save_vcov()
    {
        vcov = inverse_XtWX();
    }

public:
    glm_sparse(const MapSpMat &X_,
               const Eigen::Map<Eigen::VectorXd> &Y_,
               const Eigen::Map<Eigen::VectorXd> &weights_,
               const Eigen::Map<Eigen::VectorXd> &offset_,
               Rcpp::Function &variance_fun_,
               Rcpp::Function &mu_eta_fun_,
               Rcpp::Function &linkinv_,
               Rcpp::Function &dev_resids_fun_,
               Rcpp::Function &valideta_,
               Rcpp::Function &validmu_,
               double tol_, int maxit_, int type_, int fam_code_,
               const fglm::FamilyParams &fam_params_ = fglm::FamilyParams(),
               bool firth_flag = false) :
        GlmBase<Eigen::VectorXd, Eigen::MatrixXd>(X_.rows(), X_.cols(), tol_, maxit_),
        X(X_), Y(Y_), weights(weights_), offset(offset_),
        variance_fun(variance_fun_), mu_eta_fun(mu_eta_fun_),
        linkinv(linkinv_), dev_resids_fun(dev_resids_fun_),
        valideta(valideta_), validmu(validmu_),
        tol(tol_), maxit(maxit_), type(type_), rank(X_.cols()), fam_code(fam_code_),
        fam_params(fam_params_),
        firth_(firth_flag), log_det_XtWX_(0.0),
        A_prev(X_.cols(), X_.cols()), A_prev_available(false),
        pattern_analyzed(false), Xtwz(X_.cols())
    {}

    void init_parms(const Eigen::Map<Eigen::VectorXd> &start_,
                    const Eigen::Map<Eigen::VectorXd> &mu_,
                    const Eigen::Map<Eigen::VectorXd> &eta_)
    {
        beta = start_;
        eta  = eta_;
        mu   = mu_;
        update_dev_resids();
        rank = nvars;
    }

    virtual int get_rank() { return rank; }
    virtual Eigen::VectorXd get_weights() { return weights; }
    double get_log_det_XtWX() const { return log_det_XtWX_; }
    bool   get_firth()        const { return firth_; }
};

#endif // GLM_SPARSE_H
