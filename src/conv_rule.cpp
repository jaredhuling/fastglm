#include <RcppEigen.h>
#include "glm_base.h"

// Set the convergence rule used by new IRLS solvers and return the previous
// value. 0: deviance change only (fastglm 0.1.4); 1: deviance change
// confirmed by the Fisher decrement (default); 2: Fisher decrement with a
// final step (experimental). Used through the internal R function
// .fastglm_conv_rule().
// [[Rcpp::export]]
int set_conv_rule(int rule)
{
    int old = fastglm_conv_rule_default();
    if (rule < 0 || rule > 2) Rcpp::stop("rule must be 0, 1, or 2");
    fastglm_conv_rule_default() = rule;
    return old;
}
