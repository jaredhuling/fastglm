#ifndef GLM_BASE_H
#define GLM_BASE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>

using namespace Rcpp;
using namespace std;

using Eigen::Map;


// Process-wide default convergence rule for new solvers (see
// GlmBase::solve). 1 is the package default; set_conv_rule() in
// src/conv_rule.cpp changes it for experiments.
inline int& fastglm_conv_rule_default() { static int rule = 1; return rule; }

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
    
    
    virtual bool failed() const { return false; }

    // Hooks for the decrement-based convergence check (see solve()). The
    // defaults keep the original behavior for solvers that do not opt in.
    int conv_rule = fastglm_conv_rule_default();
    bool info_current = false;      // last factorization is at the reported estimate's weights
    std::string stop_reason = "iteration_limit";
    virtual bool supports_decrement() const { return false; }
    virtual bool halving_was_exhausted() const { return false; }
    virtual bool fit_is_valid() { return true; }
    virtual bool can_reuse_factorization() { return false; }
    virtual void restore_previous()
    {
        beta = beta_prev;
        update_eta();
        update_mu();
        update_dev_resids_dont_update_old();
    }

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
    
    // Convergence rule.
    //   0  deviance change only (behavior of fastglm 0.1.4 and glm.fit)
    //   1  deviance change, confirmed by the Fisher decrement before
    //      convergence is reported (default)
    //   2  Fisher decrement, followed by one final step taken with the
    //      factorization already in hand (experimental)
    // The decrement at the current estimate is
    //   lambda2 = g' H^{-1} g = sum_i (w_i * (x_i' delta))^2,
    // where delta is the full Fisher step from the next weighted least
    // squares solve and w are the square-root working weights. It equals the
    // deviance decrease predicted by that step and is zero exactly when the
    // score is zero, so it separates a converged fit from one whose steps
    // were halved to nothing.
    void set_conv_rule(int rule) { conv_rule = rule; }
    std::string get_stop_reason() const { return stop_reason; }

    int solve(int maxit, bool inference = true)
    {
        conv = false; info_current = false; stop_reason = "iteration_limit";
        const bool use_decrement = conv_rule > 0 && supports_decrement();
        if (!use_decrement) return solve_legacy(maxit, inference);

        bool pending = false;     // deviance rule fired; confirm with the decrement
        int updates = 0;          // accepted coefficient updates (reported iterations)
        int i;
        VecTypeX eta_before;
        // One confirmation pass is allowed beyond maxit when the deviance
        // rule fires on the last iteration.
        for (i = 0; i < maxit || (pending && i == maxit); ++i)
        {
            Rcpp::checkUserInterrupt();
            update_var_mu();
            update_mu_eta();
            update_z();
            update_w();
            solve_wls(i);
            eta_before = eta;
            update_eta();
            const double lambda2 = (w.array() * (eta - eta_before).array()).square().sum();
            const bool small = i > 0 && std::isfinite(lambda2) &&
                lambda2 / (0.1 + std::abs(dev)) < tol;

            if (small && (pending || conv_rule == 2)) {
                if (conv_rule == 2) {
                    // Final step from the converged estimate; keep it only if
                    // the fit stays valid and the deviance does not increase.
                    update_mu();
                    update_dev_resids();
                    if (std::isfinite(dev) && fit_is_valid() &&
                        dev <= devold * (1 + 1e-12) + 1e-12) ++updates;
                    else restore_previous();
                } else {
                    // Only eta has moved in this pass; mu and the deviance
                    // are still those of the estimate being confirmed.
                    beta = beta_prev;
                    eta.swap(eta_before);
                }
                conv = true; info_current = true; stop_reason = "converged";
                break;
            }
            pending = false;
            if (i == maxit) { beta = beta_prev; eta.swap(eta_before); break; }   // confirmation failed at the limit

            update_mu();
            update_dev_resids();
            run_step_halving(i);
            if (failed()) { stop_reason = "step_halving_failed"; break; }
            if (!std::isfinite(dev) && i == 0)
                stop("cannot find valid starting values: please specify some");

            if (halving_was_exhausted()) {
                // No acceptable step along the Fisher direction. The estimate
                // is converged only if the score at it is already small.
                restore_previous();
                if (i == 0)
                    stop("no valid set of coefficients has been found: please supply starting values");
                if (small) { conv = true; info_current = true; stop_reason = "converged"; }
                else stop_reason = "step_halving_failed";
                break;
            }
            ++updates;

            if (conv_rule == 1 && converged()) pending = true;
        }

        if (inference) {
            if (info_current && can_reuse_factorization()) save_vcov();
            else finalize_inference(std::min(updates + 1, maxit));
            se = vcov.diagonal().array().sqrt();
        } else { vcov.resize(0, 0); se.resize(0); }

        return conv ? std::max(updates, 1) : std::min(i + 1, maxit);
    }

    int solve_legacy(int maxit, bool inference)
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
            if (failed()) break;

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
        stop_reason = conv ? "converged" : (failed() ? "step_halving_failed" : "iteration_limit");

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
