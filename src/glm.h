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
            se              = Ch.solve(MatrixXd::Identity(nvars, nvars)).diagonal().array().sqrt();
        }
        
    }
    
    /*
    ColPivQR::ColPivQR(const Map<MatrixXd> &X, const Map<VectorXd> &y, const Map<VectorXd> &wts)
        : wls(X, y, wts)
    {
        ColPivHouseholderQR<MatrixXd> PQR(wts.asDiagonal() * X); // decompose the model matrix
        Permutation                  Pmat(PQR.colsPermutation());
        m_r                               = PQR.rank();
        if (rank == nvars) 
        {	// full rank case
            m_coef     = PQR.solve( (y.array() * wts.array()).matrix() );
            // m_fitted   = X * m_coef;
            m_se       = Pmat * PQR.matrixQR().topRows(m_p).
            triangularView<Upper>().solve(I_p()).rowwise().norm();
            return;
        } 
        MatrixXd                     Rinv(PQR.matrixQR().topLeftCorner(m_r, m_r).
                                              triangularView<Upper>().
                                              solve(MatrixXd::Identity(m_r, m_r)));
        VectorXd                  effects(PQR.householderQ().adjoint() * (y.array() * wts.array()).matrix());
        m_coef.head(rank)                 = Rinv * effects.head(m_r);
        m_coef                            = Pmat * m_coef;
        // create fitted values from effects
        // (can't use X*m_coef if X is rank-deficient)
        effects.tail(m_n - m_r).setZero();
        m_fitted                          = PQR.householderQ() * effects;
        m_se.head(m_r)                    = Rinv.rowwise().norm();
        m_se                              = Pmat * m_se;
    }
     */
    
    /*
    lm::lm(const Map<MatrixXd> &X, const Map<VectorXd> &y)
        : m_X(X),
          m_y(y),
          m_n(X.rows()),
          m_p(X.cols()),
          m_coef(VectorXd::Constant(m_p, ::NA_REAL)),
          m_r(::NA_INTEGER),
          m_fitted(m_n),
          m_se(VectorXd::Constant(m_p, ::NA_REAL)),
          m_usePrescribedThreshold(false) {
    }
    
    lm& lm::setThreshold(const RealScalar& threshold) {
        m_usePrescribedThreshold = true;
        m_prescribedThreshold = threshold;
        return *this;
    }
    
    inline ArrayXd lm::Dplus(const ArrayXd& d) {
        ArrayXd   di(d.size());
        double  comp(d.maxCoeff() * threshold());
        for (int j = 0; j < d.size(); ++j) di[j] = (d[j] < comp) ? 0. : 1./d[j];
        m_r          = (di != 0.).count();
        return di;
    }
    
    MatrixXd lm::XtX() const {
        return MatrixXd(m_p, m_p).setZero().selfadjointView<Lower>().
        rankUpdate(m_X.adjoint());
    }
    
    / Returns the threshold that will be used by certain methods such as rank().
    * 
    *  The default value comes from experimenting (see "LU precision
    *  tuning" thread on the Eigen list) and turns out to be
    *  identical to Higham's formula used already in LDLt. 
    *
    *  @return The user-prescribed threshold or the default.
    
    
    
    RealScalar lm::threshold() const {
        return m_usePrescribedThreshold ? m_prescribedThreshold
        : numeric_limits<double>::epsilon() * m_p; 
    }
    
    
    ColPivQR::ColPivQR(const Map<MatrixXd> &X, const Map<VectorXd> &y)
        : lm(X, y) 
    {
        ColPivHouseholderQR<MatrixXd> PQR(X); // decompose the model matrix
        Permutation                  Pmat(PQR.colsPermutation());
        m_r                               = PQR.rank();
        if (m_r == m_p) {	// full rank case
            m_coef     = PQR.solve(y);
            m_fitted   = X * m_coef;
            m_se       = Pmat * PQR.matrixQR().topRows(m_p).
            triangularView<Upper>().solve(I_p()).rowwise().norm();
            return;
        } 
        MatrixXd                     Rinv(PQR.matrixQR().topLeftCorner(m_r, m_r).
                                              triangularView<Upper>().
                                              solve(MatrixXd::Identity(m_r, m_r)));
        VectorXd                  effects(PQR.householderQ().adjoint() * y);
        m_coef.head(m_r)                  = Rinv * effects.head(m_r);
        m_coef                            = Pmat * m_coef;
        // create fitted values from effects
        // (can't use X*m_coef if X is rank-deficient)
        effects.tail(m_n - m_r).setZero();
        m_fitted                          = PQR.householderQ() * effects;
        m_se.head(m_r)                    = Rinv.rowwise().norm();
        m_se                              = Pmat * m_se;
    }
    
    QR::QR(const Map<MatrixXd> &X, const Map<VectorXd> &y) : lm(X, y) {
        HouseholderQR<MatrixXd> QR(X);
        m_coef                     = QR.solve(y);
        m_fitted                   = X * m_coef;
        m_se                       = QR.matrixQR().topRows(m_p).
        triangularView<Upper>().solve(I_p()).rowwise().norm();
    }
    
    
    Llt::Llt(const Map<MatrixXd> &X, const Map<VectorXd> &y) : lm(X, y) {
        LLT<MatrixXd>  Ch(XtX().selfadjointView<Lower>());
        m_coef            = Ch.solve(X.adjoint() * y);
        m_fitted          = X * m_coef;
        m_se              = Ch.matrixL().solve(I_p()).colwise().norm();
    }
    
    Ldlt::Ldlt(const Map<MatrixXd> &X, const Map<VectorXd> &y) : lm(X, y) {
        LDLT<MatrixXd> Ch(XtX().selfadjointView<Lower>());
        Dplus(Ch.vectorD());	// to set the rank
        //FIXME: Check on the permutation in the LDLT and incorporate it in
        //the coefficients and the standard error computation.
        //	m_coef            = Ch.matrixL().adjoint().
        //	    solve(Dplus(D) * Ch.matrixL().solve(X.adjoint() * y));
        m_coef            = Ch.solve(X.adjoint() * y);
        m_fitted          = X * m_coef;
        m_se              = Ch.solve(I_p()).diagonal().array().sqrt();
    }
    
    int gesdd(MatrixXd& A, ArrayXd& S, MatrixXd& Vt) {
        int info, mone = -1, m = A.rows(), n = A.cols();
        std::vector<int> iwork(8 * n);
        double wrk;
        if (m < n || S.size() != n || Vt.rows() != n || Vt.cols() != n)
            throw std::invalid_argument("dimension mismatch in gesvd");
        F77_CALL(dgesdd)("O", &m, &n, A.data(), &m, S.data(), A.data(),
                 &m, Vt.data(), &n, &wrk, &mone, &iwork[0], &info);
        int lwork(wrk);
        std::vector<double> work(lwork);
        F77_CALL(dgesdd)("O", &m, &n, A.data(), &m, S.data(), A.data(),
                 &m, Vt.data(), &n, &work[0], &lwork, &iwork[0], &info);
        return info;
    }
    
    GESDD::GESDD(const Map<MatrixXd>& X, const Map<VectorXd> &y) : lm(X, y) {
        MatrixXd   U(X), Vt(m_p, m_p);
        ArrayXd   S(m_p);
        if (gesdd(U, S, Vt)) throw std::runtime_error("error in gesdd");
        MatrixXd VDi(Vt.adjoint() * Dplus(S).matrix().asDiagonal());
        m_coef      = VDi * U.adjoint() * y;
        m_fitted    = X * m_coef;
        m_se        = VDi.rowwise().norm();
    }
    
    SVD::SVD(const Map<MatrixXd> &X, const Map<VectorXd> &y) : lm(X, y) {
        JacobiSVD<MatrixXd>  UDV(X.jacobiSvd(ComputeThinU|ComputeThinV));
        MatrixXd             VDi(UDV.matrixV() *
            Dplus(UDV.singularValues().array()).matrix().asDiagonal());
        m_coef                   = VDi * UDV.matrixU().adjoint() * y;
        m_fitted                 = X * m_coef;
        m_se                     = VDi.rowwise().norm();
    }
    
    SymmEigen::SymmEigen(const Map<MatrixXd> &X, const Map<VectorXd> &y)
        : lm(X, y) {
        SelfAdjointEigenSolver<MatrixXd> eig(XtX().selfadjointView<Lower>());
        MatrixXd   VDi(eig.eigenvectors() *
            Dplus(eig.eigenvalues().array()).sqrt().matrix().asDiagonal());
        m_coef         = VDi * VDi.adjoint() * X.adjoint() * y;
        m_fitted       = X * m_coef;
        m_se           = VDi.rowwise().norm();
    }
    
    enum {ColPivQR_t = 0, QR_t, LLT_t, LDLT_t, SVD_t, SymmEigen_t, GESDD_t};
    
    static inline lm do_lm(const Map<MatrixXd> &XX, 
                           const Map<VectorXd> &y, 
                           const Map<VectorXd> &wts, 
                           int type) 
    {
        switch(type) {
        case ColPivQR_t:
            return ColPivQR(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        case QR_t:
            return QR(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        case LLT_t:
            return Llt(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        case LDLT_t:
            return Ldlt(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        case SVD_t:
            return SVD(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        case SymmEigen_t:
            return SymmEigen(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        case GESDD_t:
            return GESDD(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );
        }
        throw invalid_argument("invalid type");
        return ColPivQR(wts.asDiagonal() * XX, (y.array() * wts.array()).matrix() );	// -Wall
    }
    */


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
    
    
    virtual VectorXd get_beta() { return beta; }
    virtual VectorXd get_se()   { return se; }
    virtual MatrixXd get_vcov() { return vcov; }

};




#endif // GLM_H