#ifndef GLM_H
#define GLM_H

#include "glm_base.h"

using Eigen::ArrayXd;
using Eigen::ColPivHouseholderQR;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::HouseholderQR;
using Eigen::JacobiSVD;
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
    
    double tol;
    int maxit;
    int type;
    int rank;
    
    
    ColPivHouseholderQR<MatrixXd> PQR;
    HouseholderQR<MatrixXd> QR;
    LLT<MatrixXd>  Ch;
    LDLT<MatrixXd> ChD;
    JacobiSVD<MatrixXd>  UDV;
    
    SelfAdjointEigenSolver<MatrixXd> eig;
    
    Permutation                  Pmat;
    MatrixXd                     Rinv;
    
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
    
    MatrixXd XtWX() const 
    {
        return MatrixXd(nvars, nvars).setZero().selfadjointView<Lower>().
        rankUpdate( (w.asDiagonal() * X).adjoint());
    }

    virtual void update_mu_eta()
    {
        NumericVector mu_eta_nv = mu_eta_fun(eta);
        
        std::copy(mu_eta_nv.begin(), mu_eta_nv.end(), mu_eta.data());
    }
    
    virtual void update_var_mu()
    {
        NumericVector var_mu_nv = variance_fun(mu);
        
        std::copy(var_mu_nv.begin(), var_mu_nv.end(), var_mu.data());
    }
    
    virtual void update_mu()
    {
        // mu <- linkinv(eta <- eta + offset)
        NumericVector mu_nv = linkinv(eta);
        
        std::copy(mu_nv.begin(), mu_nv.end(), mu.data());
    }
    
    virtual void update_eta()
    {
        // eta <- drop(x %*% start)
        eta = X * beta + offset;
    }
    
    virtual void update_z()
    {
        // z <- (eta - offset)[good] + (y - mu)[good]/mu.eta.val[good]
        z = (eta - offset).array() + (Y - mu).array() / mu_eta.array();
    }
    
    virtual void update_w()
    {
        // w <- sqrt((weights[good] * mu.eta.val[good]^2)/variance(mu)[good])
        w = (weights.array() * mu_eta.array().square() / var_mu.array()).array().sqrt();
    }
    
    virtual void update_dev_resids()
    {
        devold = dev;
        NumericVector dev_resids = dev_resids_fun(Y, mu, weights);
        dev = sum(dev_resids);
    }
    
    // much of solve_wls() comes directly
    // from the source code of the RcppEigen package
    virtual void solve_wls()
    {
        //lm ans(do_lm(X, Y, w, type));
        //wls ans(ColPivQR(X, z, w));
        
        //enum {ColPivQR_t = 0, QR_t, LLT_t, LDLT_t, SVD_t, SymmEigen_t, GESDD_t};
        
        
        if (type == 0)
        {
            PQR.compute(w.asDiagonal() * X); // decompose the model matrix
            Pmat = (PQR.colsPermutation());
            rank                               = PQR.rank();
            if (rank == nvars) 
            {	// full rank case
                beta     = PQR.solve( (z.array() * w.array()).matrix() );
                // m_fitted   = X * m_coef;
                //m_se       = Pmat * PQR.matrixQR().topRows(m_p).
                //triangularView<Upper>().solve(MatrixXd::Identity(nvars, nvars)).rowwise().norm();
            } else 
            {
                Rinv = (PQR.matrixQR().topLeftCorner(rank, rank).
                                                      triangularView<Upper>().
                                                      solve(MatrixXd::Identity(rank, rank)));
                VectorXd                  effects(PQR.householderQ().adjoint() * (z.array() * w.array()).matrix());
                beta.head(rank)                 = Rinv * effects.head(rank);
                beta                            = Pmat * beta;
                
                // create fitted values from effects
                // (can't use X*m_coef if X is rank-deficient)
                //effects.tail(m_n - rank).setZero();
                //m_fitted                          = PQR.householderQ() * effects;
                //m_se.head(m_r)                    = Rinv.rowwise().norm();
                //m_se                              = Pmat * m_se;
            }
        } else if (type == 1)
        {
            QR.compute(w.asDiagonal() * X);
            beta                     = QR.solve((z.array() * w.array()).matrix());
            //m_fitted                   = X * m_coef;
            //m_se                       = QR.matrixQR().topRows(m_p).
            //triangularView<Upper>().solve(I_p()).rowwise().norm();
        } else if (type == 2)
        {
            Ch.compute(XtWX().selfadjointView<Lower>());
            beta            = Ch.solve((w.asDiagonal() * X).adjoint() * (z.array() * w.array()).matrix());
            //m_fitted          = X * m_coef;
            //m_se              = Ch.matrixL().solve(I_p()).colwise().norm();
        } else if (type == 3)
        {
            ChD.compute(XtWX().selfadjointView<Lower>());
            Dplus(ChD.vectorD());	// to set the rank
            //FIXME: Check on the permutation in the LDLT and incorporate it in
            //the coefficients and the standard error computation.
            //	m_coef            = Ch.matrixL().adjoint().
            //	    solve(Dplus(D) * Ch.matrixL().solve(X.adjoint() * y));
            beta            = ChD.solve((w.asDiagonal() * X).adjoint() * (z.array() * w.array()).matrix());
            //m_fitted          = X * m_coef;
            //m_se              = Ch.solve(I_p()).diagonal().array().sqrt();
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
    
    virtual void save_se()
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
        double tol_ = 1e-6,
        int maxit_ = 100,
        int type_ = 1) :
    GlmBase<Eigen::VectorXd, Eigen::MatrixXd>(X_.rows(), X_.cols(),
                                                     tol_, maxit_),
                                                     X(X_),
                                                     Y(Y_),
                                                     weights(weights_),
                                                     offset(offset_),
                                                     //X(X_.data(), X_.rows(), X_.cols()),
                                                     //Y(Y_.data(), Y_.size()),
                                                     variance_fun(variance_fun_),
                                                     mu_eta_fun(mu_eta_fun_),
                                                     linkinv(linkinv_),
                                                     dev_resids_fun(dev_resids_fun_),
                                                     tol(tol_),
                                                     maxit(maxit_),
                                                     type(type_)
                                                     {}
    
    
    // must set params to starting vals
    void init_parms()
    {
        beta.setZero();
        eta.setZero();
        eta.array() += offset.array();
        
        update_var_mu();
        
        update_mu_eta();
        
        update_mu();
        
        NumericVector dev_resids = dev_resids_fun(Y, mu, weights);
        dev = sum(dev_resids);
        
        update_z();
        
        update_w();
    }
    
    
    virtual VectorXd get_weights() { return weights; }

};




#endif // GLM_H