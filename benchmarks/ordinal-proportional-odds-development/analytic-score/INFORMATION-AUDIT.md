# Expected versus observed information: post-study diagnosis

The analytic candidate was frozen with expected Fisher information. After the
main and stress studies, an independent check used the Cheese example in the
SAS/STAT 12.1 LOGISTIC documentation (Example 54.3, Output 54.3.1). The published
statistic is 17.2866 with 21 df; the expected-information calculation gives
20.9379 with the same df and matching null coefficients. This difference is
retained, not hidden as numerical agreement.

SAS's general score-statistic definition allows either observed or expected
information (Score Statistics and Tests, pp. 4246-4248). A direct derivation of
the observed negative Hessian, checked against numerical differentiation,
produces a singular matrix on this example. Its generalized-inverse quadratic
form agrees with the published statistic within 0.001, while the expected
information is nonsingular. This is evidence consistent with an observed-
information computation in this example; it is not an audit of proprietary
SAS internals or a claim of bitwise equivalence.

For cumulative predictors eta_j linear in the full local parameters, write
z_j=d eta_j/d parameter, f_j=F_j(1-F_j), and
t=1/expm1(eta_k-eta_{k-1}). An interior outcome k contributes observed information

    f_k z_k z_k' + f_{k-1} z_{k-1} z_{k-1}'
      + t(1+t)(z_k-z_{k-1})(z_k-z_{k-1})'.

An endpoint contributes its single f_j z_j z_j' term. Expectation under the
ordinal model recovers the candidate's Fisher information. Both choices have
the same regular asymptotic limit but need not have the same finite-sample law.
The generalized inverse is used only to diagnose the published example; it is
not introduced into the candidate or the replay comparison.

Before computing the replay comparison, fix four n=80 cells: main regular5,
main rare_lower3, stress many_predictors5 and stress leverage3, with all 1,000
original dataset seeds in each cell. Refit the same ML null and verify the
stored expected-information statistic. Compute the observed-information
projected score only when Cholesky factorization succeeds. Compare 5% and 1%
rejection on shared available datasets; preserve every unavailable record.
This is a targeted diagnosis using already inspected development data, not
fresh confirmation and not a fitted correction to the expected-information test.

Primary sources read November 3, 2026:

* [SAS/STAT 12.3 LOGISTIC manual](https://support.sas.com/documentation/onlinedoc/stat/123/logistic.pdf),
  Score Statistics and Tests and Testing the Parallel Lines Assumption,
  pp. 4246-4248.
* [SAS/STAT 12.1, Example 54.3](https://support.sas.com/documentation/cdl/en/statug/65328/HTML/default/statug_logistic_examples03.htm),
  full data table, model definition and score output. Only its numerical data
  and reported statistic are retained; original source programs are not copied.

The expected-information Rao score is established methodology, not a newly
invented correction. The distinction matters for external validation and for
interpreting earlier reports of liberal score tests. It does not by itself
explain every published rejection-rate result.
