#ifndef GLM_SPARSE_H
#define GLM_SPARSE_H

#include "glm_base.h"
#include "families.h"
#include "bias_reduction.h"
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
    fglm::GammaWeightGroups dispersion_weight_groups;
    bool constant_weights;
    bool factor_ready;
    Eigen::VectorXd constant_leverages;
    Eigen::VectorXd gamma_adjustment;
    double gamma_step_phi = 1;
    bool gamma_step_failed = false;
    bool gamma_initial_consistent = false;
    bool firth_step_failed = false;
    bool firth_step_limited = false;
    int firth_step_reductions = 0;

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
    void update_mu_eta() override
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

    void update_var_mu() override
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

    void update_mu() override
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

    void update_eta() override
    {
        eta.noalias() = X * beta;
        eta += offset;
    }

    void update_z() override
    {
        z = (eta - offset).array() + (Y - mu).array() / mu_eta.array();
    }

    void update_w() override
    {
        // The sqrt-link derivative is floored near zero. In that exceptional
        // region its implemented working weights are no longer constant.
        if (fam_code == fglm::FAM_POISSON_SQRT &&
            (eta.array() < fglm::thresh_eps() / 2).any()) {
            constant_weights = false;
            constant_leverages.resize(0);
        }
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

    void update_dev_resids() override
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

    void update_dev_resids_dont_update_old() override
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

    void step_halve() override
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

    bool failed() const override { return gamma_step_failed || firth_step_failed; }

    bool converged() override
    {
        if (firth_) {
            if (failed() || firth_step_limited || (beta-beta_prev).cwiseAbs().maxCoeff() >= tol) return false;
            if (fam_code == fglm::FAM_GAMMA_LOG && gamma_adjustment.size() == nvars) {
                double phi = compute_firth_phi_sparse();
                Eigen::VectorXd score = X.adjoint()*
                    (weights.array()*(Y.array()/mu.array()-1)).matrix() + phi*gamma_adjustment;
                return solve_factored(score).cwiseAbs().maxCoeff() < tol;
            }
            return true;
        }
        return std::abs(dev - devold) / (0.1 + std::abs(dev)) < tol;
    }

    void run_step_halving(int &iterr) override
    {
        firth_step_limited=false;
        if(firth_ && fglm::bounded_firth_predictors(fam_code)) {
            const Eigen::VectorXd delta=beta-beta_prev;
            const Eigen::VectorXd change=X*delta;
            const double fraction=fglm::firth_predictor_step_fraction(change,weights);
            if(!delta.allFinite() || !std::isfinite(fraction)) {
                firth_step_failed=true;beta=beta_prev;
                update_eta();update_mu();update_dev_resids_dont_update_old();return;
            }
            if(fraction<1) {
                beta=beta_prev+fraction*delta;
                firth_step_limited=true;++firth_step_reductions;
                update_eta();update_mu();update_dev_resids_dont_update_old();
            }
        }
        if (firth_ && fam_code == fglm::FAM_GAMMA_LOG && (iterr > 0 || gamma_initial_consistent) &&
            gamma_adjustment.size() == nvars) {
            const Eigen::VectorXd delta = beta-beta_prev;
            const double quadratic = delta.dot(XtWX*delta);
            double fraction = 1;
            int halves = 0;
            while (true) {
                const double change = dev-devold -
                    2*gamma_step_phi*gamma_adjustment.dot(beta-beta_prev);
                const double rounding = 64*std::numeric_limits<double>::epsilon()*
                    (1+std::abs(dev)+std::abs(devold));
                if (std::isfinite(dev) && valideta_check() && validmu_check() &&
                    change <= -0.2*fraction*quadratic + rounding) break;
                if (++halves > 50) {
                    gamma_step_failed = true;
                    beta = beta_prev; update_eta(); update_mu();
                    update_dev_resids_dont_update_old();
                    break;
                }
                step_halve(); fraction *= 0.5;
                update_dev_resids_dont_update_old();
            }
            return;
        }
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
    // type 3). analyzePattern is done once; the symbolic
    // factor and the numeric factor are retained for the post-solve readouts
    // (log|X'WX|, (X'WX)^{-1}).
    void factorize_information(const SpMat &A)
    {
        if (type == 2) {
            if (!pattern_analyzed) { Ch.analyzePattern(A); pattern_analyzed = true; }
            Ch.factorize(A);
            if (Ch.info() != Eigen::Success)
                Rcpp::stop("Sparse Cholesky factorization failed; X'WX may be singular or not positive definite");
            const SpMat &factor = Ch.matrixL().nestedExpression();
            double smallest = std::numeric_limits<double>::infinity(), largest = 0;
            for (int j = 0; j < nvars; ++j) {
                double d = factor.coeff(j, j);
                smallest = std::min(smallest, d); largest = std::max(largest, d);
            }
            if (smallest <= 1e-6 * largest)
                Rcpp::stop("Sparse Cholesky factorization is ill-conditioned; rescale or reduce the design");
            return;
        }
        if (!pattern_analyzed) { ChD.analyzePattern(A); pattern_analyzed = true; }
        ChD.factorize(A);
        if (ChD.info() != Eigen::Success || (ChD.vectorD().array() <= 1e-12 * ChD.vectorD().maxCoeff()).any())
            Rcpp::stop("Sparse Cholesky factorization failed; X'WX may be singular or not positive definite");
    }

    Eigen::VectorXd solve_factored(const Eigen::VectorXd &rhs)
    {
        if (type == 2) return Ch.solve(rhs);
        return ChD.solve(rhs);
    }

    Eigen::VectorXd factorize_and_solve(const SpMat &A, const Eigen::VectorXd &rhs)
    {
        factorize_information(A);
        factor_ready = true;
        return solve_factored(rhs);
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
        Eigen::Map<const Eigen::ArrayXd> yy(Y.data(), Y.size());
        Eigen::Map<const Eigen::ArrayXd> mm(mu.data(), mu.size());
        Eigen::Map<const Eigen::ArrayXd> ww(weights.data(), weights.size());
        double d = fam_code >= 0 ? fglm::dev_resids_sum(fam_code, fam_params, yy, mm, ww)
            : Rcpp::sum(Rcpp::as<Rcpp::NumericVector>(dev_resids_fun(Y, mu, weights)));
        if (fam_code >= 10 && fam_code <= 12 && dispersion_weight_groups.empty())
            dispersion_weight_groups = fglm::gamma_weight_groups(weights);
        return fglm::adjusted_dispersion(fam_code, d, weights, rank,
                                         &dispersion_weight_groups);
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
        bool reuse_factor = constant_weights && factor_ready;
        Eigen::VectorXd w2 = w.array().square();
        if (!reuse_factor)
            XtWX = SpMat(X.adjoint()) * w2.asDiagonal() * X;

        if ((fam_code == fglm::FAM_GAMMA_LOG && !A_prev_available) ||
            fglm::bounded_firth_predictors(fam_code)) {
            factorize_information(XtWX);
            factor_ready = true; reuse_factor = true;
            compute_log_det_XtWX_sparse(); update_A_prev_sparse();
        }
        const bool needs_adjustment = A_prev_available &&
            fam_code != fglm::FAM_GAUSSIAN_IDENTITY;
        Eigen::VectorXd zstar = z;
        if (needs_adjustment) {
            Eigen::ArrayXd d2mu(nobs);
            compute_d2mu_sparse(d2mu);
            double phi = compute_firth_phi_sparse();
            gamma_step_phi = phi;
            Eigen::VectorXd h_lev;
            if (constant_weights && constant_leverages.size() == nobs)
                h_lev = constant_leverages;
            else {
                compute_leverages_lagged(h_lev);
                if (constant_weights) constant_leverages = h_lev;
            }
            if (fam_code == fglm::FAM_GAMMA_LOG && gamma_adjustment.size() == 0)
                gamma_adjustment.noalias() = 0.5 * X.adjoint() * h_lev;
            for (int i = 0; i < nobs; ++i)
                if (weights[i] > 0 && w2[i] != 0 && mu_eta[i] != 0)
                    zstar[i] += phi * h_lev[i] * (d2mu[i] / mu_eta[i]) / (2 * w2[i]);
        }
        Xtwz.noalias() = X.adjoint() * (w2.array() * zstar.array()).matrix();
        beta = reuse_factor ? solve_factored(Xtwz) : factorize_and_solve(XtWX, Xtwz);
        rank = nvars;
        if (!reuse_factor) {
            compute_log_det_XtWX_sparse();
            update_A_prev_sparse();
        }
    }

    // ------------------------------------------------------------------
    // Core sparse WLS solve: X' diag(w^2) X beta = X' diag(w^2) z.
    // ------------------------------------------------------------------
    void solve_wls(int /*iter*/) override
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

    void finalize_inference(int /*iter*/) override
    {
        // Rebuild information at the reported mean, without computing a
        // coefficient update or bias adjustment that would be discarded.
        update_var_mu();
        update_mu_eta();
        update_w();
        if (!(firth_ && constant_weights && factor_ready)) {
            Eigen::VectorXd w2 = w.array().square();
            XtWX = SpMat(X.adjoint()) * w2.asDiagonal() * X;
            factorize_information(XtWX);
            factor_ready = true;
        }
        if (firth_) {
            const double previous_log_det = log_det_XtWX_;
            compute_log_det_XtWX_sparse();
            dev += previous_log_det - log_det_XtWX_;
        }
        save_vcov();
    }

    void save_se() override
    {
        se = inverse_XtWX().diagonal().array().sqrt();
    }

    void save_vcov() override
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
        constant_weights(fam_code_ == fglm::FAM_GAUSSIAN_IDENTITY ||
                         fam_code_ == fglm::FAM_POISSON_SQRT ||
                         fam_code_ == fglm::FAM_GAMMA_LOG),
        factor_ready(false), pattern_analyzed(false), Xtwz(X_.cols())
    {}

    void init_parms(const Eigen::Map<Eigen::VectorXd> &start_,
                    const Eigen::Map<Eigen::VectorXd> &mu_,
                    const Eigen::Map<Eigen::VectorXd> &eta_) override
    {
        beta = start_;
        eta  = eta_;
        mu   = mu_;
        if (firth_ && fam_code == fglm::FAM_GAMMA_LOG) {
            const Eigen::VectorXd mapped = X*beta + offset;
            gamma_initial_consistent = (mapped-eta).cwiseAbs().maxCoeff() <=
                16*std::numeric_limits<double>::epsilon()*(1+eta.cwiseAbs().maxCoeff());
        }
        update_dev_resids();
        rank = nvars;
    }

    int get_rank() override { return rank; }
    Eigen::VectorXd get_weights() override { return weights; }
    double get_log_det_XtWX() const { return log_det_XtWX_; }
    double get_bias_dispersion() { return compute_firth_phi_sparse(); }
    void append_bias_diagnostics(Rcpp::List& out, double phi, bool inference)
    {
        const std::string reason = firth_step_failed ? "nonfinite_scoring_step" : gamma_step_failed ? "step_halving_failed" :
            (conv ? "converged" : "iteration_limit");
        out["convergence.reason"] = reason;
        if(fglm::bounded_firth_predictors(fam_code)) out["step.reductions"]=firth_step_reductions;
        if (!inference && !constant_weights) return;
        Eigen::VectorXd score;
        if (fam_code == fglm::FAM_GAMMA_LOG && gamma_adjustment.size() == nvars) {
            score = X.adjoint()*(weights.array()*(Y.array()/mu.array()-1)/phi).matrix() + gamma_adjustment;
        } else if (fam_code == fglm::FAM_GAUSSIAN_IDENTITY) {
            score = X.adjoint()*(weights.array()*(Y.array()-mu.array())/phi).matrix();
        } else {
            if (!inference) { update_var_mu(); update_mu_eta(); update_w(); }
            Eigen::VectorXd h;
            if (constant_leverages.size() == nobs) h = constant_leverages;
            else { update_A_prev_sparse(); compute_leverages_lagged(h); }
            Eigen::ArrayXd d2mu(nobs); compute_d2mu_sparse(d2mu);
            Eigen::VectorXd contribution = Eigen::VectorXd::Zero(nobs);
            for (int i=0;i<nobs;++i) if (weights[i] > 0 && mu_eta[i] != 0)
                contribution[i] = weights[i]*mu_eta[i]*(Y[i]-mu[i])/(phi*var_mu[i]) +
                    h[i]*d2mu[i]/(2*mu_eta[i]);
            score = X.adjoint()*contribution;
        }
        Eigen::VectorXd information = XtWX.diagonal()/phi;
        const double dev = Rcpp::as<double>(out["deviance"]);
        fglm::GammaDispersionTerms terms;
        if (fam_code >= 10 && fam_code <= 12) {
            if (dispersion_weight_groups.empty()) dispersion_weight_groups = fglm::gamma_weight_groups(weights);
            terms.add(dispersion_weight_groups,phi);
        }
        double ds, ds_scaled;
        fglm::dispersion_diagnostics(fam_code,dev,(weights.array()>0).count(),rank,phi,terms,ds,ds_scaled);
        fglm::append_adjusted_diagnostics(out,score,information,ds,ds_scaled,reason);
    }
    bool   get_firth()        const { return firth_; }
};

#endif // GLM_SPARSE_H
