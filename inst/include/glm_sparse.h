#ifndef GLM_SPARSE_H
#define GLM_SPARSE_H

#include "glm_base.h"
#include "families.h"

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
        w = (weights.array() * mu_eta.array().square() / var_mu.array()).array().sqrt();
    }

    virtual void update_dev_resids()
    {
        devold = dev;
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> y(Y.data(), Y.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> wt(weights.data(), weights.size());
            dev = fglm::dev_resids_sum(fam_code, fam_params, y, m, wt);
            return;
        }
        Rcpp::NumericVector dr = dev_resids_fun(Y, mu, weights);
        dev = Rcpp::sum(dr);
    }

    virtual void update_dev_resids_dont_update_old()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> y(Y.data(), Y.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> wt(weights.data(), weights.size());
            dev = fglm::dev_resids_sum(fam_code, fam_params, y, m, wt);
            return;
        }
        Rcpp::NumericVector dr = dev_resids_fun(Y, mu, weights);
        dev = Rcpp::sum(dr);
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

    virtual void run_step_halving(int &iterr)
    {
        if (std::isinf(dev)) {
            int itrr = 0;
            while (std::isinf(dev)) {
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
        if ((dev - devold) / (0.1 + std::abs(dev)) >= tol && iterr > 0) {
            int itrr = 0;
            while ((dev - devold) / (0.1 + std::abs(dev)) >= -tol) {
                if (++itrr > maxit) break;
                step_halve();
                update_dev_resids_dont_update_old();
            }
        }
    }

    // ------------------------------------------------------------------
    // Core sparse WLS solve: X' diag(w^2) X beta = X' diag(w^2) z.
    // ------------------------------------------------------------------
    virtual void solve_wls(int /*iter*/)
    {
        beta_prev = beta;

        Eigen::VectorXd w2 = w.array().square();
        // X' W^2 X.  selfadjoint product would be ideal but Eigen's sparse
        // path here is already efficient.
        XtWX = SpMat(X.adjoint()) * w2.asDiagonal() * X;
        Xtwz.noalias() = X.adjoint() * (w2.array() * z.array()).matrix();

        if (type == 2) {
            if (!pattern_analyzed) { Ch.analyzePattern(XtWX); pattern_analyzed = true; }
            Ch.factorize(XtWX);
            beta = Ch.solve(Xtwz);
        } else { // type == 3
            if (!pattern_analyzed) { ChD.analyzePattern(XtWX); pattern_analyzed = true; }
            ChD.factorize(XtWX);
            beta = ChD.solve(Xtwz);
        }
        rank = nvars;  // assume full rank for sparse Cholesky (factorize will fail otherwise)
    }

    virtual void save_se()
    {
        // SE = sqrt(diag((X' W^2 X)^{-1})).  Solve against I_p and take diag.
        Eigen::MatrixXd I_p = Eigen::MatrixXd::Identity(nvars, nvars);
        Eigen::MatrixXd inv;
        if (type == 2) inv = Ch.solve(I_p);
        else           inv = ChD.solve(I_p);
        se = inv.diagonal().array().sqrt();
    }

    virtual void save_vcov()
    {
        Eigen::MatrixXd I_p = Eigen::MatrixXd::Identity(nvars, nvars);
        if (type == 2) vcov = Ch.solve(I_p);
        else           vcov = ChD.solve(I_p);
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
               const fglm::FamilyParams &fam_params_ = fglm::FamilyParams()) :
        GlmBase<Eigen::VectorXd, Eigen::MatrixXd>(X_.rows(), X_.cols(), tol_, maxit_),
        X(X_), Y(Y_), weights(weights_), offset(offset_),
        variance_fun(variance_fun_), mu_eta_fun(mu_eta_fun_),
        linkinv(linkinv_), dev_resids_fun(dev_resids_fun_),
        valideta(valideta_), validmu(validmu_),
        tol(tol_), maxit(maxit_), type(type_), rank(X_.cols()), fam_code(fam_code_),
        fam_params(fam_params_),
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
};

#endif // GLM_SPARSE_H
