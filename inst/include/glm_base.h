#ifndef GLM_BASE_H
#define GLM_BASE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <string>
#include <limits>

using namespace Rcpp;
using namespace std;

using Eigen::Map;


#include "conv_rule.h"
#include "trace.h"

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

    // Progress trace (trace.h): off unless the caller enables it with
    // set_trace(). halvings_ counts the step halvings of the fit; solvers
    // that halve steps increment it in step_halve().
    bool trace_ = false;
    int halvings_ = 0;

    // Hooks for the decrement-based convergence check (see solve()). The
    // defaults keep the original behavior for solvers that do not opt in.
    int conv_rule = fastglm_conv_rule_default();
    bool info_current = false;      // last factorization is at the reported estimate's weights
    std::string stop_reason = "iteration_limit";
    virtual bool supports_decrement() const { return false; }
    virtual bool halving_was_exhausted() const { return false; }
    virtual bool fit_is_valid() { return true; }
    virtual bool can_reuse_factorization() { return false; }
    // delta' X'WX delta for delta = beta - beta_prev, when the solver holds
    // X'WX; NaN otherwise.
    virtual double fisher_decrement() { return std::numeric_limits<double>::quiet_NaN(); }
    // Boundary-aware step (boundary_step.h): called after the weighted least
    // squares solve; may replace beta by a constrained update when the full
    // update leaves the valid domain. Sets boundary_held to the number of rows
    // held near their bound (0 when the step was not used).
    virtual void apply_boundary_step(int iter) { boundary_held = 0; }
    int boundary_held = 0;
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
    // One trace line per iteration of solve() when on (trace.h). Count
    // models that run solve() inside an outer loop trace the outer loop
    // instead and leave this off.
    void set_trace(bool on) { trace_ = on; }
    // Step halvings of all solve() calls so far (for outer-loop traces).
    int get_halvings() const { return halvings_; }
    std::string get_stop_reason() const { return stop_reason; }

    int solve(int maxit, bool inference = true)
    {
        conv = false; info_current = false; stop_reason = "iteration_limit";
        const bool use_decrement = conv_rule > 0 && supports_decrement();
        if (!use_decrement) return solve_legacy(maxit, inference);

        bool pending = false;     // deviance rule fired; confirm with the decrement
        int updates = 0;          // accepted coefficient updates (reported iterations)
        int i;
        // Trace: passes are evaluations of the deviance over all rows (one
        // per iteration and one per halving); one line per accepted update.
        const fglm_trace::Tracer tracer(trace_);
        int passes = 0;
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
            apply_boundary_step(i);
            // The decrement from the factorization (p x p work) when the
            // solver holds X'WX; otherwise from the change in eta.
            double lambda2 = fisher_decrement();
            const bool from_factor = std::isfinite(lambda2);
            if (!from_factor) {
                eta_before = eta;
                update_eta();
                lambda2 = (w.array() * (eta - eta_before).array()).square().sum();
            }
            const bool small = i > 0 && std::isfinite(lambda2) &&
                lambda2 / (0.1 + std::abs(dev)) < tol;

            if (small && pending) {
                // Rule 1: the decrement confirms the deviance rule. mu and the
                // deviance are still those of the estimate being confirmed.
                beta = beta_prev;
                if (!from_factor) eta.swap(eta_before);
                conv = true; info_current = true; stop_reason = "converged";
                break;
            }
            if (from_factor) { eta_before = eta; update_eta(); }
            if (small && conv_rule == 2) {
                // Rule 2: the decrement is small, so take this step and stop
                // if its deviance change is below tol, which is where glm.fit
                // stops; the information from this pass (at the preceding
                // iterate, as in glm.fit) gives the covariance without another
                // pass. Otherwise the step is an ordinary accepted step.
                update_mu();
                update_dev_resids();
                // Accepted as in run_step_halving: valid, and the deviance
                // does not rise by tol or more.
                const bool ok = std::isfinite(dev) && fit_is_valid() &&
                    (dev - devold) / (0.1 + std::abs(dev)) < tol;
                if (!ok) {
                    restore_previous();
                    conv = true; info_current = true; stop_reason = "converged";
                    break;
                }
                ++updates;
                tracer.iteration(updates, ++passes, dev, 0, "deviance");
                if (std::abs(dev - devold) / (0.1 + std::abs(dev)) < tol) {
                    conv = true; info_current = true; stop_reason = "converged";
                    break;
                }
                continue;
            }
            pending = false;
            if (i == maxit) { beta = beta_prev; eta.swap(eta_before); break; }   // confirmation failed at the limit

            update_mu();
            update_dev_resids();
            const int halvings_before = halvings_;
            run_step_halving(i);
            const int halved = halvings_ - halvings_before;
            passes += 1 + halved;
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
            tracer.iteration(updates, passes, dev, halved, "deviance");

            if (conv_rule == 1 && converged()) pending = true;
        }

        // Converged with rows held near a bound of the valid domain: the score
        // is not zero there, and the information-based covariance does not
        // have its usual interpretation.
        if (conv && boundary_held > 0) stop_reason = "converged_boundary";

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
        const fglm_trace::Tracer tracer(trace_);
        int passes = 0;

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

            const int halvings_before = halvings_;
            run_step_halving(i);
            const int halved = halvings_ - halvings_before;
            passes += 1 + halved;
            if (failed()) break;

            if (!std::isfinite(dev) && i == 0)
            {
                stop("cannot find valid starting values: please specify some");
            }
            tracer.iteration(i + 1, passes, dev, halved, "deviance");

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
