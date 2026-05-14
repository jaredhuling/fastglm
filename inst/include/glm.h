#ifndef GLM_H
#define GLM_H

#include "glm_base.h"
#include "families.h"
#include "chunk_source.h"

using Eigen::ArrayXd;
using Eigen::FullPivHouseholderQR;
using Eigen::ColPivHouseholderQR;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::HouseholderQR;
using Eigen::JacobiSVD;
using Eigen::BDCSVD;
using Eigen::LDLT;
using Eigen::LLT;
using Eigen::Lower;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::SelfAdjointEigenSolver;
using Eigen::SelfAdjointView;
using Eigen::TriangularView;
using Eigen::VectorXd;
using Eigen::Upper;
using Eigen::EigenBase;


class glm: public GlmBase<Eigen::VectorXd, Eigen::MatrixXd> //Eigen::SparseVector<double>
{
protected:


    
    typedef double Double;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseMatrix<double> SpMat;
    typedef Eigen::SparseVector<double> SparseVector;
    
    typedef MatrixXd::Index                                 Index;
    typedef MatrixXd::Scalar                                Scalar;
    typedef MatrixXd::RealScalar                            RealScalar;
    typedef ColPivHouseholderQR<MatrixXd>::PermutationType  Permutation;
    typedef Permutation::IndicesType                        Indices;
    
    const Map<MatrixXd> X;
    const Map<VectorXd> Y;
    const Map<VectorXd> weights;
    const Map<VectorXd> offset;
    
    Function variance_fun;
    Function mu_eta_fun;
    Function linkinv;
    Function dev_resids_fun;
    Function valideta;
    Function validmu;
    
    double tol;
    int maxit;
    int type;
    bool is_big_matrix;
    int rank;
    int fam_code;   // fglm::FamilyCode; -1 = unknown (use R callbacks)
    fglm::FamilyParams fam_params;  // theta / var_power / link_power for params-aware families
    bool firth_;                    // true = Firth bias-reduced score
    double log_det_XtWX_;           // log|X'WX| from most recent solve_wls (used by Firth)
    MatrixXd A_prev;                // (X'WX)^{-1} from previous IRLS iteration (lagged leverages)
    bool     A_prev_available;
    
    
    FullPivHouseholderQR<MatrixXd> FPQR;
    ColPivHouseholderQR<MatrixXd> PQR;
    BDCSVD<MatrixXd> bSVD;
    HouseholderQR<MatrixXd> QR;
    LLT<MatrixXd>  Ch;
    LDLT<MatrixXd> ChD;
    JacobiSVD<MatrixXd>  UDV;
    
    SelfAdjointEigenSolver<MatrixXd> eig;
    
    Permutation                  Pmat;
    MatrixXd                     Rinv;
    VectorXd                     effects;

    // Pre-allocated solver workspace, reused across IRLS iterations.
    // Avoids one n*p alloc and one n alloc per iteration in solve_wls().
    // For big.matrix mode with method 2/3, WX is left empty -- we accumulate
    // X'WX and X'Wz directly via row-block streaming kernels and never
    // materialize the n*p product.
    MatrixXd                     WX;     // w.asDiagonal() * X (dense path only)
    VectorXd                     wz;     // w .* z
    MatrixXd                     XtWX_buf;
    VectorXd                     Xtwz_buf; // for streaming accumulation
    bool                         use_streaming;
    
    RealScalar threshold() const 
    {
        //return m_usePrescribedThreshold ? m_prescribedThreshold
        //: numeric_limits<double>::epsilon() * nvars; 
        return numeric_limits<double>::epsilon() * nvars; 
    }
    
    // from RcppEigen
    inline ArrayXd Dplus(const ArrayXd& d) 
    {
        ArrayXd   di(d.size());
        double  comp(d.maxCoeff() * threshold());
        for (int j = 0; j < d.size(); ++j) di[j] = (d[j] < comp) ? 0. : 1./d[j];
        rank          = (di != 0.).count();
        return di;
    }
    
    // X' W X using the rank-1 update on the pre-multiplied WX = w * X.
    // Reuses the XtWX_buf member to avoid per-iteration allocation.
    void update_XtWX()
    {
        XtWX_buf.setZero();
        XtWX_buf.selfadjointView<Lower>().rankUpdate(WX.adjoint());
    }

    virtual void update_mu_eta() override
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            Eigen::Map<Eigen::ArrayXd>       d(mu_eta.data(), mu_eta.size());
            fglm::mu_eta(fam_code, fam_params, e, d);
            return;
        }
        NumericVector mu_eta_nv = mu_eta_fun(eta);
        std::copy(mu_eta_nv.begin(), mu_eta_nv.end(), mu_eta.data());
    }

    virtual void update_var_mu() override
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<Eigen::ArrayXd>       v(var_mu.data(), var_mu.size());
            fglm::variance(fam_code, fam_params, m, v);
            return;
        }
        NumericVector var_mu_nv = variance_fun(mu);
        std::copy(var_mu_nv.begin(), var_mu_nv.end(), var_mu.data());
    }

    virtual void update_mu() override
    {
        // mu <- linkinv(eta <- eta + offset)
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            Eigen::Map<Eigen::ArrayXd>       m(mu.data(), mu.size());
            fglm::linkinv(fam_code, fam_params, e, m);
            return;
        }
        NumericVector mu_nv = linkinv(eta);
        std::copy(mu_nv.begin(), mu_nv.end(), mu.data());
    }
    
    virtual void update_eta() override
    {
        // eta <- drop(x %*% beta) + offset
        if (use_streaming) {
            fglm::apply_X_streamed(X, beta, offset, eta);
        } else {
            eta.noalias() = X * beta;
            eta += offset;
        }
    }
    
    virtual void update_z() override
    {
        // z <- (eta - offset)[good] + (y - mu)[good]/mu.eta.val[good]
        z = (eta - offset).array() + (Y - mu).array() / mu_eta.array();
    }
    
    virtual void update_w() override
    {
        // w <- sqrt((weights[good] * mu.eta.val[good]^2)/variance(mu)[good])
        w = (weights.array() * mu_eta.array().square() / var_mu.array()).array().sqrt();
    }
    
    virtual void update_dev_resids() override
    {
        devold = dev;
        double std_dev;
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> y(Y.data(), Y.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> w(weights.data(), weights.size());
            std_dev = fglm::dev_resids_sum(fam_code, fam_params, y, m, w);
        } else {
            NumericVector dev_resids = dev_resids_fun(Y, mu, weights);
            std_dev = sum(dev_resids);
        }
        // Firth penalized deviance: dev* = -2 * (l(beta) + 0.5 * log|I(beta)|)
        //                                = std_dev - log|X'WX|.
        dev = firth_ ? (std_dev - log_det_XtWX_) : std_dev;
    }

    virtual void update_dev_resids_dont_update_old() override
    {
        double std_dev;
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> y(Y.data(), Y.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> w(weights.data(), weights.size());
            std_dev = fglm::dev_resids_sum(fam_code, fam_params, y, m, w);
        } else {
            NumericVector dev_resids = dev_resids_fun(Y, mu, weights);
            std_dev = sum(dev_resids);
        }
        dev = firth_ ? (std_dev - log_det_XtWX_) : std_dev;
    }

    virtual void step_halve() override
    {
        // take half step
        beta = 0.5 * (beta.array() + beta_prev.array());

        update_eta();

        update_mu();
    }

    // For Firth, the convergence check on penalized deviance has a one-step
    // lag in the log|X'WX| term (we form log|X'W_{k-1} X| inside solve_wls
    // before updating beta to beta_k).  That lag can satisfy |Δdev| < tol
    // before the penalized score reaches zero.  Override converged() to test
    // the coefficient change ||Δβ||_∞ instead, which is the convergence
    // criterion every reference Firth implementation (logistf, brglm) uses.
    bool converged() override
    {
        if (firth_) {
            return (beta - beta_prev).cwiseAbs().maxCoeff() < tol;
        }
        return std::abs(dev - devold) / (0.1 + std::abs(dev)) < tol;
    }

    bool valideta_check()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            return fglm::valideta(fam_code, fam_params, e);
        }
        return as<bool>(valideta(eta));
    }

    bool validmu_check()
    {
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            return fglm::validmu(fam_code, fam_params, m);
        }
        return as<bool>(validmu(mu));
    }

    virtual void run_step_halving(int &iterr) override
    {
        // Firth: penalized dev is std_dev - log|X'WX| where the log-det term
        // is computed at the *previous* beta inside solve_wls_firth.  That
        // one-step lag can briefly inflate the penalized dev and trigger
        // spurious step halving that drags the iteration off the true MLE.
        // Only halve to recover from infinite dev / invalid (eta, mu); skip
        // the "increasing deviance" branch.
        const bool firth_skip_dev_halving = firth_;
        // check for infinite deviance
        if (std::isinf(dev))
        {
            int itrr = 0;
            while(std::isinf(dev))
            {
                ++itrr;
                if (itrr > maxit)
                {
                    break;
                }

                step_halve();
                update_dev_resids_dont_update_old();
            }
        }

        // check for boundary violations
        if (!(valideta_check() && validmu_check()))
        {
            int itrr = 0;
            while(!(valideta_check() && validmu_check()))
            {
                ++itrr;
                if (itrr > maxit)
                {
                    break;
                }
                
                //std::cout << "half step (boundary)!" << itrr << std::endl;
                
                step_halve();
                
            }
            
            update_dev_resids_dont_update_old();
        }
        
        
        // check for increasing deviance
        //std::abs(deviance - deviance_prev) / (0.1 + std::abs(deviance)) < tol_irls
        if (!firth_skip_dev_halving &&
            (dev - devold) / (0.1 + std::abs(dev)) >= tol && iterr > 0)
        {
            int itrr = 0;
            
            //std::cout << "dev:" << deviance << "dev prev:" << deviance_prev << std::endl;
            
            while((dev - devold) / (0.1 + std::abs(dev)) >= -tol)
            {
                ++itrr;
                if (itrr > maxit)
                {
                    break;
                }
                
                //std::cout << "half step (increasing dev)!" << itrr << std::endl;
                
                step_halve();
                
                
                update_dev_resids_dont_update_old();
            }
        }
    }
    
    // Generalized Firth / AS_mean bias-reducing penalty for arbitrary GLM
    // families.  Supports all decomposition types on the dense path and
    // streaming (big.matrix) via lagged leverages.
    //
    // The AS_mean adjustment shifts the working response by
    //   ξ_i = φ · h_i · (d²μ/dη²)_i · V(μ_i) / (2 · pw_i · (dμ/dη)³_i)
    // where h_i is the leverage diagonal of the hat matrix.
    //
    // Dense path: exact leverages from the current iteration's decomposition.
    // Streaming path: lagged leverages from A_prev = (X'WX)^{-1} of the
    //   previous iteration.  The fixed point is identical.

    // Compute d²μ/dη² for all observations.
    void compute_d2mu(Eigen::ArrayXd &d2mu)
    {
        const double eps = std::numeric_limits<double>::epsilon();
        if (fam_code >= 0) {
            Eigen::Map<const Eigen::ArrayXd> e(eta.data(), eta.size());
            Eigen::Map<const Eigen::ArrayXd> m(mu.data(), mu.size());
            Eigen::Map<const Eigen::ArrayXd> d(mu_eta.data(), mu_eta.size());
            fglm::d2mu_deta2(fam_code, fam_params, e, m, d, d2mu);
        } else {
            const double h = std::pow(eps, 1.0 / 3.0);
            VectorXd eta_p = eta.array() + h;
            VectorXd eta_m = eta.array() - h;
            NumericVector dmu_p = mu_eta_fun(eta_p);
            NumericVector dmu_m = mu_eta_fun(eta_m);
            for (int i = 0; i < nobs; ++i)
                d2mu[i] = (dmu_p[i] - dmu_m[i]) / (2.0 * h);
        }
    }

    // Estimate the dispersion φ (1 for binomial/Poisson).
    double compute_firth_phi()
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

    // Compute ξ from h, d2mu, var_mu, mu_eta, weights, phi and form z* = z + ξ.
    void apply_firth_xi(const VectorXd &h_lev, const Eigen::ArrayXd &d2mu,
                        double phi, VectorXd &zstar)
    {
        const double eps = std::numeric_limits<double>::epsilon();
        Eigen::ArrayXd dmu3 = mu_eta.array().cube();
        for (int i = 0; i < nobs; ++i)
            if (std::abs(dmu3[i]) < eps) dmu3[i] = (dmu3[i] >= 0) ? eps : -eps;

        Eigen::ArrayXd xi = phi * h_lev.array() * d2mu * var_mu.array()
                          / (2.0 * weights.array().max(eps) * dmu3);
        zstar = z + xi.matrix();
    }

    // Compute leverages from the current decomposition (dense path).
    void compute_leverages_dense(VectorXd &h_out)
    {
        if (type == 2) {
            // LLT: h_i = ||L^{-1} (WX_i)'||^2
            MatrixXd V = WX.transpose();
            Ch.matrixL().solveInPlace(V);
            h_out = V.colwise().squaredNorm();
        } else if (type == 3) {
            // LDLT: h_i from (LD^{1/2})^{-1}
            MatrixXd inv = ChD.solve(MatrixXd::Identity(nvars, nvars));
            h_out.resize(nobs);
            for (int i = 0; i < nobs; ++i) {
                auto wx_i = WX.row(i);
                h_out[i] = wx_i * inv * wx_i.transpose();
            }
        } else if (type == 5) {
            // SVD of WX = USV': h_i = ||U_i||^2
            h_out = bSVD.matrixU().rowwise().squaredNorm();
        } else {
            // QR methods (0, 1, 4): h_i from Q.
            // Compute Q explicitly as Q * I_rank (n x rank).
            MatrixXd Q;
            int r = nvars;
            if (type == 0) {
                r = PQR.rank();
                Q = PQR.householderQ() * MatrixXd::Identity(nobs, r);
            } else if (type == 1) {
                Q = QR.householderQ() * MatrixXd::Identity(nobs, r);
            } else { // type == 4
                r = FPQR.rank();
                Q = FPQR.matrixQ() * MatrixXd::Identity(nobs, r);
            }
            h_out = Q.rowwise().squaredNorm();
        }
    }

    // Compute log|X'WX| from the current decomposition.
    void compute_log_det_XtWX()
    {
        double ld = 0.0;
        if (type == 2) {
            const MatrixXd &Lstore = Ch.matrixLLT();
            for (int j = 0; j < Lstore.cols(); ++j) ld += std::log(Lstore(j, j));
            log_det_XtWX_ = 2.0 * ld;
        } else if (type == 3) {
            Eigen::VectorXd D = ChD.vectorD();
            for (int j = 0; j < D.size(); ++j) ld += std::log(std::abs(D[j]));
            log_det_XtWX_ = ld;
        } else if (type == 5) {
            Eigen::ArrayXd s = bSVD.singularValues().array();
            for (int j = 0; j < s.size(); ++j) ld += std::log(s[j]);
            log_det_XtWX_ = 2.0 * ld;
        } else {
            // QR methods: |X'WX| = |R'R| = |R|^2, log|R| = sum(log|R_jj|)
            MatrixXd R;
            if (type == 0)      R = PQR.matrixQR().topRows(nvars);
            else if (type == 1) R = QR.matrixQR().topRows(nvars);
            else                R = FPQR.matrixQR().topRows(nvars);
            for (int j = 0; j < nvars; ++j) ld += std::log(std::abs(R(j, j)));
            log_det_XtWX_ = 2.0 * ld;
        }
    }

    // Update A_prev = (X'WX)^{-1} from current decomposition.
    void update_A_prev()
    {
        MatrixXd I_p = MatrixXd::Identity(nvars, nvars);
        if (type == 2)      A_prev = Ch.solve(I_p);
        else if (type == 3) A_prev = ChD.solve(I_p);
        else if (type == 5) {
            const MatrixXd &V = bSVD.matrixV();
            Eigen::ArrayXd s = bSVD.singularValues().array();
            Eigen::ArrayXd dinv2(s.size());
            for (int j = 0; j < s.size(); ++j) dinv2[j] = 1.0 / (s[j] * s[j]);
            A_prev.noalias() = V * dinv2.matrix().asDiagonal() * V.adjoint();
        } else {
            // QR methods: (X'WX)^{-1} = R^{-1} R'^{-1} (or with permutation)
            if (type == 0) {
                MatrixXd Ri = PQR.matrixQR().topRows(nvars)
                    .triangularView<Upper>().solve(I_p);
                MatrixXd PRi = Pmat * Ri;
                A_prev.noalias() = PRi * PRi.transpose();
            } else if (type == 1) {
                MatrixXd Ri = QR.matrixQR().topRows(nvars)
                    .triangularView<Upper>().solve(I_p);
                A_prev.noalias() = Ri * Ri.transpose();
            } else { // type == 4
                MatrixXd Ri = FPQR.matrixQR().topRows(nvars)
                    .triangularView<Upper>().solve(I_p);
                MatrixXd PRi = Pmat * Ri;
                A_prev.noalias() = PRi * PRi.transpose();
            }
        }
        A_prev_available = true;
    }

    void solve_wls_firth()
    {
        beta_prev = beta;

        Eigen::ArrayXd d2mu(nobs);
        compute_d2mu(d2mu);
        double phi = compute_firth_phi();

        // --- Streaming path (big.matrix + Cholesky) ---
        if (use_streaming) {
            if (!A_prev_available) {
                // First iteration: standard solve, no Firth adjustment
                fglm::accumulate_xtwx_streamed(X, w, XtWX_buf);
                fglm::accumulate_xtwz_streamed(X, w, z, Xtwz_buf);
            } else {
                fglm::accumulate_firth_streamed(
                    X, w, z, Eigen::Map<const Eigen::VectorXd>(d2mu.data(), nobs),
                    var_mu, mu_eta, weights, phi, A_prev,
                    XtWX_buf, Xtwz_buf);
            }
            if (type == 2) {
                Ch.compute(XtWX_buf.selfadjointView<Lower>());
                beta = Ch.solve(Xtwz_buf);
            } else {
                ChD.compute(XtWX_buf.selfadjointView<Lower>());
                Dplus(ChD.vectorD());
                beta = ChD.solve(Xtwz_buf);
            }
            compute_log_det_XtWX();
            update_A_prev();
            rank = nvars;
            return;
        }

        // --- Dense path: exact leverages from current decomposition ---
        WX.noalias() = w.asDiagonal() * X;
        wz.noalias() = w.cwiseProduct(z);

        // Decompose
        if (type == 0) {
            PQR.compute(WX);
            Pmat = PQR.colsPermutation();
            rank = PQR.rank();
        } else if (type == 1) {
            QR.compute(WX);
            rank = nvars;
        } else if (type == 2) {
            update_XtWX();
            Ch.compute(XtWX_buf.selfadjointView<Lower>());
            rank = nvars;
        } else if (type == 3) {
            update_XtWX();
            ChD.compute(XtWX_buf.selfadjointView<Lower>());
            Dplus(ChD.vectorD());
            rank = nvars;
        } else if (type == 4) {
            FPQR.compute(WX);
            Pmat = FPQR.colsPermutation();
            rank = FPQR.rank();
        } else if (type == 5) {
            bSVD.compute(WX, ComputeThinU | ComputeThinV);
            rank = bSVD.rank();
        }

        // Compute leverages and Firth adjustment
        VectorXd h_lev;
        compute_leverages_dense(h_lev);

        VectorXd zstar;
        apply_firth_xi(h_lev, d2mu, phi, zstar);

        // Solve with modified response
        wz.noalias() = w.cwiseProduct(zstar);
        if (type == 0) {
            if (rank == nvars) {
                beta = PQR.solve(wz);
            } else {
                effects = PQR.householderQ().adjoint() * wz;
                beta.head(rank) = PQR.matrixQR().topLeftCorner(rank, rank)
                    .triangularView<Upper>().solve(effects.head(rank));
                beta = Pmat * beta;
            }
        } else if (type == 1) {
            beta = QR.solve(wz);
        } else if (type == 2) {
            beta = Ch.solve(WX.adjoint() * wz);
        } else if (type == 3) {
            beta = ChD.solve(WX.adjoint() * wz);
        } else if (type == 4) {
            if (rank == nvars) {
                beta = FPQR.solve(wz);
            } else {
                effects = FPQR.matrixQ().adjoint() * wz;
                beta.head(rank) = FPQR.matrixQR().topLeftCorner(rank, rank)
                    .triangularView<Upper>().solve(effects.head(rank));
                beta = Pmat * beta;
            }
        } else if (type == 5) {
            beta = bSVD.solve(wz);
        }

        compute_log_det_XtWX();
        update_A_prev();
    }

    // much of solve_wls() comes directly
    // from the source code of the RcppEigen package
    virtual void solve_wls(int iter) override
    {
        if (firth_) { solve_wls_firth(); return; }
        //enum {ColPivQR_t = 0, QR_t, LLT_t, LDLT_t, SVD_t, SymmEigen_t, GESDD_t};

        beta_prev = beta;

        // Streaming path for big.matrix + Cholesky-based methods (2 or 3):
        // never materialize the n*p product W*X.  Instead accumulate
        // X'W^2 X and X'W^2 z directly in row blocks.
        if (use_streaming) {
            fglm::accumulate_xtwx_streamed(X, w, XtWX_buf);
            fglm::accumulate_xtwz_streamed(X, w, z, Xtwz_buf);
            if (type == 2) {
                Ch.compute(XtWX_buf.selfadjointView<Lower>());
                beta = Ch.solve(Xtwz_buf);
            } else if (type == 3) {
                ChD.compute(XtWX_buf.selfadjointView<Lower>());
                Dplus(ChD.vectorD());  // sets rank
                beta = ChD.solve(Xtwz_buf);
            }
            return;
        }

        // Pre-multiply once per iteration into pre-allocated buffers,
        // then feed them to every decomposition.  noalias() avoids a
        // temporary; both buffers are sized in the constructor.
        WX.noalias() = w.asDiagonal() * X;
        wz.noalias() = w.cwiseProduct(z);

        if (type == 0)
        {
            PQR.compute(WX); // decompose the model matrix
            Pmat = (PQR.colsPermutation());
            rank                               = PQR.rank();
            if (rank == nvars)
            {	// full rank case
                beta     = PQR.solve(wz);
            } else
            {
                Rinv = (PQR.matrixQR().topLeftCorner(rank, rank).
                                                      triangularView<Upper>().
                                                      solve(MatrixXd::Identity(rank, rank)));
                effects = PQR.householderQ().adjoint() * wz;
                beta.head(rank)                 = Rinv * effects.head(rank);
                beta                            = Pmat * beta;

                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                effects.tail(nobs - rank).setZero();
            }
        } else if (type == 1)
        {
            QR.compute(WX);
            beta                     = QR.solve(wz);
        } else if (type == 2)
        {
            update_XtWX();
            Ch.compute(XtWX_buf.selfadjointView<Lower>());
            beta            = Ch.solve(WX.adjoint() * wz);
        } else if (type == 3)
        {
            update_XtWX();
            ChD.compute(XtWX_buf.selfadjointView<Lower>());
            Dplus(ChD.vectorD());	// to set the rank
            //FIXME: Check on the permutation in the LDLT and incorporate it in
            //the coefficients and the standard error computation.
            beta            = ChD.solve(WX.adjoint() * wz);
        } else if (type == 4)
        {
            FPQR.compute(WX); // decompose the model matrix
            Pmat = (FPQR.colsPermutation());
            rank                               = FPQR.rank();
            if (rank == nvars)
            {	// full rank case
                beta     = FPQR.solve(wz);
            } else
            {
                Rinv = (FPQR.matrixQR().topLeftCorner(rank, rank).
                            triangularView<Upper>().
                            solve(MatrixXd::Identity(rank, rank)));
                effects = FPQR.matrixQ().adjoint() * wz;
                beta.head(rank)                 = Rinv * effects.head(rank);
                beta                            = Pmat * beta;

                effects.tail(nobs - rank).setZero();
            }
        } else if (type == 5)
        {
            bSVD.compute(WX, ComputeThinU | ComputeThinV);
            rank                               = bSVD.rank();
            beta                               = bSVD.solve(wz);
            //FIXME: Check on the permutation in the LDLT and incorporate it in
            //the coefficients and the standard error computation.
        }
        // } else if (type == 4)
        // {
        // //     UDV.compute((w.asDiagonal() * X).jacobiSvd(ComputeThinU|ComputeThinV));
        // //     MatrixXd             VDi(UDV.matrixV() *
        // //         Dplus(UDV.singularValues().array()).matrix().asDiagonal());
        // //     beta                   = VDi * UDV.matrixU().adjoint() * (z.array() * w.array()).matrix();
        // //     //m_fitted                 = X * m_coef;
        // //     //m_se                     = VDi.rowwise().norm();
        // // } else if (type == 5)
        // // {
        //     eig.compute(XtWX().selfadjointView<Lower>());
        //     MatrixXd   VDi(eig.eigenvectors() *
        //         Dplus(eig.eigenvalues().array()).sqrt().matrix().asDiagonal());
        //     beta         = VDi * VDi.adjoint() * X.adjoint() * (z.array() * w.array()).matrix();
        //     //m_fitted       = X * m_coef;
        //     //m_se           = VDi.rowwise().norm();
        // }
        
    }
    
    virtual void save_vcov() override
    {
        const double NaN = std::numeric_limits<double>::quiet_NaN();
        MatrixXd I_p = MatrixXd::Identity(nvars, nvars);

        if (type == 0)
        {
            if (rank == nvars)
            {
                MatrixXd RfullInv = PQR.matrixQR().topRows(nvars).
                    triangularView<Upper>().solve(I_p);
                MatrixXd PRinv = Pmat * RfullInv;
                vcov.noalias() = PRinv * PRinv.transpose();
            } else
            {
                MatrixXd cov_red = Rinv * Rinv.transpose();
                MatrixXd cov_padded = MatrixXd::Zero(nvars, nvars);
                cov_padded.topLeftCorner(rank, rank) = cov_red;
                vcov.noalias() = Pmat * cov_padded * Pmat.transpose();
                for (int j = rank; j < nvars; ++j)
                {
                    int aliased = Pmat.indices()[j];
                    vcov.row(aliased).setConstant(NaN);
                    vcov.col(aliased).setConstant(NaN);
                }
            }
        } else if (type == 1)
        {
            MatrixXd RfullInv = QR.matrixQR().topRows(nvars).
                triangularView<Upper>().solve(I_p);
            vcov.noalias() = RfullInv * RfullInv.transpose();
        } else if (type == 2)
        {
            vcov = Ch.solve(I_p);
        } else if (type == 3)
        {
            vcov = ChD.solve(I_p);
        } else if (type == 4)
        {
            if (rank == nvars)
            {
                MatrixXd RfullInv = FPQR.matrixQR().topRows(nvars).
                    triangularView<Upper>().solve(I_p);
                MatrixXd PRinv = Pmat * RfullInv;
                vcov.noalias() = PRinv * PRinv.transpose();
            } else
            {
                MatrixXd cov_red = Rinv * Rinv.transpose();
                MatrixXd cov_padded = MatrixXd::Zero(nvars, nvars);
                cov_padded.topLeftCorner(rank, rank) = cov_red;
                vcov.noalias() = Pmat * cov_padded * Pmat.transpose();
                for (int j = rank; j < nvars; ++j)
                {
                    int aliased = Pmat.indices()[j];
                    vcov.row(aliased).setConstant(NaN);
                    vcov.col(aliased).setConstant(NaN);
                }
            }
        } else if (type == 5)
        {
            // (X' W^2 X)^{-1} = V D^{-2} V'  where WX = U D V'
            const MatrixXd &V = bSVD.matrixV();
            ArrayXd s = bSVD.singularValues().array();
            ArrayXd dinv2 = ArrayXd::Zero(s.size());
            double comp = (s.size() > 0 ? s.maxCoeff() : 0.0) * threshold();
            for (int j = 0; j < s.size(); ++j) dinv2[j] = (s[j] < comp) ? 0.0 : 1.0/(s[j]*s[j]);
            vcov.noalias() = V * dinv2.matrix().asDiagonal() * V.adjoint();
        }
    }

    virtual void save_se() override
    {

        if (type == 0)
        {
            if (rank == nvars) 
            {	// full rank case
                se       = Pmat * PQR.matrixQR().topRows(nvars).
                    triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
                return;
            } else 
            {
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                se.head(rank)                    = Rinv.rowwise().norm();
                se                               = Pmat * se;
            }
        } else if (type == 1)
        {
            se                       = QR.matrixQR().topRows(nvars).
                triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
        } else if (type == 2)
        {
            se              = Ch.matrixL().solve(MatrixXd::Identity(nvars, nvars)).colwise().norm();
        } else if (type == 3)
        {
            se              = ChD.solve(MatrixXd::Identity(nvars, nvars)).diagonal().array().sqrt();
        } else if (type == 4)
        {
            if (rank == nvars) 
            {	// full rank case
                se       = Pmat * FPQR.matrixQR().topRows(nvars).
                triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
                return;
            } else 
            {
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                se.head(rank)                    = Rinv.rowwise().norm();
                se                               = Pmat * se;
            }
        } else if (type == 5)
        {
            // SE_i = sqrt((X' W^2 X)^{-1}_{ii}) = ||V_i row||_{D^{-1}}
            const MatrixXd &V = bSVD.matrixV();
            ArrayXd s = bSVD.singularValues().array();
            ArrayXd dinv = ArrayXd::Zero(s.size());
            double comp = (s.size() > 0 ? s.maxCoeff() : 0.0) * threshold();
            for (int j = 0; j < s.size(); ++j) dinv[j] = (s[j] < comp) ? 0.0 : 1.0/s[j];
            MatrixXd VDinv = V * dinv.matrix().asDiagonal();
            se = VDinv.rowwise().norm();
        }

    }
    


public:
    glm(const Map<MatrixXd> &X_,
        const Map<VectorXd> &Y_,
        const Map<VectorXd> &weights_,
        const Map<VectorXd> &offset_,
        Function &variance_fun_,
        Function &mu_eta_fun_,
        Function &linkinv_,
        Function &dev_resids_fun_,
        Function &valideta_,
        Function &validmu_,
        double tol_ = 1e-6,
        int maxit_ = 100,
        int type_ = 1,
        bool is_big_matrix_ = false,
        int fam_code_ = -1,
        const fglm::FamilyParams &fam_params_ = fglm::FamilyParams(),
        bool firth_flag = false) :
    GlmBase<Eigen::VectorXd, Eigen::MatrixXd>(X_.rows(), X_.cols(),
                                                     tol_, maxit_),
                                                     X(X_),
                                                     Y(Y_),
                                                     weights(weights_),
                                                     offset(offset_),
                                                     variance_fun(variance_fun_),
                                                     mu_eta_fun(mu_eta_fun_),
                                                     linkinv(linkinv_),
                                                     dev_resids_fun(dev_resids_fun_),
                                                     valideta(valideta_),
                                                     validmu(validmu_),
                                                     tol(tol_),
                                                     maxit(maxit_),
                                                     type(type_),
                                                     is_big_matrix(is_big_matrix_),
                                                     fam_code(fam_code_),
                                                     fam_params(fam_params_),
                                                     firth_(firth_flag),
                                                     log_det_XtWX_(0.0),
                                                     A_prev(X_.cols(), X_.cols()),
                                                     A_prev_available(false),
                                                     WX( (is_big_matrix_ && (type_ == 2 || type_ == 3)) ? 0 : X_.rows(),
                                                         (is_big_matrix_ && (type_ == 2 || type_ == 3)) ? 0 : X_.cols()),
                                                     wz( (is_big_matrix_ && (type_ == 2 || type_ == 3)) ? 0 : X_.rows()),
                                                     XtWX_buf(X_.cols(), X_.cols()),
                                                     Xtwz_buf(X_.cols()),
                                                     use_streaming(is_big_matrix_ && (type_ == 2 || type_ == 3))
                                                     {}
    
    
    // must set params to starting vals
    void init_parms(const Map<VectorXd> & start_,
                    const Map<VectorXd> & mu_,
                    const Map<VectorXd> & eta_) override
    {
        beta = start_;
        eta = eta_;
        mu = mu_;
        //eta.array() += offset.array();
        
        //update_var_mu();
        
        //update_mu_eta();
        
        //update_mu();
        
        update_dev_resids();
        
        //update_z();
        
        //update_w();
        
        rank = nvars;
    }
    
    virtual VectorXd get_beta() override
    {
        if (type == 0 || type == 4)
        {
            if (rank != nvars)
            {
                //beta.head(rank)                 = Rinv * effects.head(rank);
                //beta = Pmat * beta;
            }
        }
        
        return beta;
    }
    
    virtual VectorXd get_weights()  override { return weights; }
    virtual int get_rank()          override { return rank; }

    // Allow the NB / hurdle / ZI drivers to update theta (and other
    // params-aware fields) between successive IRLS passes, without
    // reconstructing the solver.
    void set_fam_params(const fglm::FamilyParams &p) { fam_params = p; }

    // Penalized log-likelihood pieces, exposed so the Firth driver can
    // report them back to R.
    double get_log_det_XtWX() const { return log_det_XtWX_; }
    bool   get_firth()        const { return firth_; }

};




#endif // GLM_H