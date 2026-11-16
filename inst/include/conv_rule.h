#ifndef FASTGLM_CONV_RULE_H
#define FASTGLM_CONV_RULE_H

// Process-wide convergence rule for IRLS fits (dense, big.matrix, sparse
// through GlmBase::solve, and the streaming fitter):
//   0  deviance change only (fastglm 0.1.4 and glm.fit)
//   1  deviance change confirmed by the Fisher decrement
//   2  Fisher decrement, then one final step with the factorization in hand
// set_conv_rule() in src/conv_rule.cpp changes it.
inline int& fastglm_conv_rule_default() { static int rule = 1; return rule; }

#endif
