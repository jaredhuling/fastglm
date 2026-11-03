#ifndef GLM_BASE_H
#define GLM_BASE_H

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;
using namespace std;

using Eigen::Map;


template<typename VecTypeX, typename MatTypeX>
class GlmBase
{
protected:
    
    const int nvars;      // dimension of beta
    const int nobs;       // number of rows
    
    VecTypeX beta;        // parameters to be optimized
    VecTypeX beta_prev;   // auxiliary parameters
    
    VecTypeX eta;
    VecTypeX var_mu;
    VecTypeX mu_eta;
    VecTypeX mu;
    VecTypeX z;
    VecTypeX w;
    MatTypeX vcov;
    VecTypeX se;
    double dev, devold, devnull;
    
    int maxit;            // max iterations
    double tol;           // tolerance for convergence
    bool conv;
    
    
    virtual bool converged()
    {
        if (std::abs(dev - devold)/(0.1 + std::abs(dev)) < tol)
        {
            return true;
        } else 
        {
            return false;
        }
    }
    
    
    virtual void update_eta()
    {
        
    }
    
    virtual void update_var_mu()
    {
        
    }
    
    virtual void update_mu_eta()
    {
        
    }
    
    virtual void update_mu()
    {
        
    }
    
    virtual void update_z()
    {
        
    }
    
    virtual void update_w()
    {
        
    }
    
    virtual void step_halve()
    {
        
    }
    
    virtual void run_step_halving(int &iterr)
    {
        
    }
    
    virtual void update_dev_resids()
    {
        
    }
    
    virtual void update_dev_resids_dont_update_old()
    {
        
    }
    
    virtual void solve_wls(int iter)
    {
        
    }
    
    virtual void save_se()
    {

    }

    virtual void save_vcov()
    {
        // default: leave vcov untouched (caller treats absence as "not available")
    }

    virtual void finalize_inference(int iter)
    {
        // Generic path also updates the estimable coefficient representation
        // when a rank-revealing decomposition changes its pivot ordering.
        VecTypeX fitted_beta = beta;
        update_var_mu(); update_mu_eta(); update_z(); update_w();
        solve_wls(iter);
        if (get_rank() == nvars) beta = fitted_beta;
        else { update_eta(); update_mu(); update_dev_resids_dont_update_old(); }
        save_vcov();
    }


public:
    GlmBase(int n_, int p_,
            double tol_ = 1e-6,
            int maxit_ = 100) :
    nvars(p_), nobs(n_),
    beta(p_), 
    beta_prev(p_), // allocate space but do not set values
    eta(n_),
    var_mu(n_),
    mu_eta(n_),
    mu(n_),
    z(n_),
    w(n_),
    vcov(p_, p_),
    se(p_),
    dev(0.0), devold(0.0), devnull(0.0),
    maxit(maxit_),
    tol(tol_), conv(false)
    {}
    
    virtual ~GlmBase() {}
    
    virtual void init_parms(const Map<VecTypeX> & start_, 
                            const Map<VecTypeX> & mu_,
                            const Map<VecTypeX> & eta_) {}
    
    void update_beta()
    {
        //VecTypeX newbeta(nvars);
        next_beta(beta);
        //beta.swap(newbeta);
    }
    
    int solve(int maxit, bool inference = true)
    {
        int i;
        
        conv = false;
        
        for(i = 0; i < maxit; ++i)
        {

            Rcpp::checkUserInterrupt();
            update_var_mu();

            update_mu_eta();

            update_z();

            update_w();

            solve_wls(i);

            update_eta();

            update_mu();

            update_dev_resids();
            
            run_step_halving(i);
            
            if (!std::isfinite(dev) && i == 0)
            {
                stop("cannot find valid starting values: please specify some");
            }
            
            if(converged())
            {
                conv = true;
                break;
            }

            
        }
        
        if (inference) {
            // Evaluate information at the reported estimate. The last IRLS
            // factor otherwise belongs to the preceding coefficient vector.
            finalize_inference(std::min(i + 1, maxit));
            se = vcov.diagonal().array().sqrt();
        } else { vcov.resize(0, 0); se.resize(0); }

        return std::min(i + 1, maxit);
    }
    
    virtual VecTypeX get_beta()     { return beta; }
    virtual VecTypeX get_eta()      { return eta; }
    virtual VecTypeX get_se()       { return se; }
    virtual VecTypeX get_mu()       { return mu; }
    virtual VecTypeX get_weights()  { return w; }
    virtual VecTypeX get_w()        { return w.array().square(); }
    virtual double get_dev()        { return dev; }
    virtual int get_rank()          { return nvars; }
    virtual MatTypeX get_vcov()     { return vcov; }
    virtual bool get_converged()    { return conv; }

    // R wrappers copy these vectors into R-owned storage. Returning a view
    // avoids an intermediate observation-sized Eigen allocation and copy.
    const VecTypeX& get_eta_ref() const { return eta; }
    const VecTypeX& get_mu_ref() const { return mu; }
    
};



#endif // GLM_BASE_H
