# Next development candidate: fixed-metric null simulation

This is a specification for investigation, not an implemented or validated
public test. It follows the mechanism study; it is not a retrospective claim
that the oracle calculations already validate a fitted-null bootstrap.

## Statistic and calibration

1. Fit the proportional-odds null by mean bias reduction, obtaining coherent
   category probabilities at the original fixed predictor design.
2. At that null parameter, compute the joint binary score variance B0 and
   block-diagonal ordinary binary sensitivity A0. Set
   M0 = D A0^{-1} B0 A0^{-T} D'. This is the true-linear covariance in the
   mechanism study with the unknown parameter replaced by its null estimate.
   Require a justified, full-rank metric for the original declared contrasts.
3. Compute the separate binary meanBR slope contrast Delta on the observed
   response and T = Delta' M0^{-1} Delta.
4. Simulate new ordinal responses under the fitted null at the same design.
   Refit the binary meanBR models with the same declared threshold set and
   compute T_b = Delta_b' M0^{-1} Delta_b. Hold M0 fixed for the observed and
   simulated statistics. There is no per-resample covariance estimate to
   invert, no dimension reduction, and no need to refit an ordinal null solely
   to compute T_b. Binary fits can reuse design setup.
5. Use (1 + number of T_b >= T)/(B+1). The plus-one formula handles Monte Carlo
   discreteness; it does not repair nuisance-parameter estimation error.

This discrepancy deliberately uses a fixed scaling, rather than treating a
local covariance formula as an exact small-sample sampling variance. If the
null parameter were known, any fixed positive-definite metric would permit
the exchangeability argument in THEORY.md. Using M0 is motivated by regular
asymptotic scale and computational convenience, not by its rejection rate in
the development grid. Both calibration and power must be evaluated anew.

## Scope of an asymptotic argument

For independent outcomes conditional on fixed designs, fixed predictor and
category dimensions, correctly specified proportional odds, well-separated
finite thresholds, probabilities uniformly away from zero, controlled design
leverage, and nonsingular limiting information and contrast covariance:

* The binary and null-model meanBR estimates must be consistent local roots.
  The score adjustments are O(1), while ordinary scores are O_p(sqrt(n));
  their contribution to the first-order estimator expansion vanishes.
* The stacked binary slopes then admit a joint asymptotic linear expansion.
  With M0 consistently estimating their contrast covariance, the observed
  quadratic statistic has a chi-square limit with p(K-2) degrees of freedom.
* Smoothness and corresponding conditional expansions for simulated roots
  can establish consistency of the fitted-null bootstrap distribution under
  those regular assumptions. With a finite simulation budget, p-values also
  retain Monte Carlo discreteness and randomness.

These are proof obligations and a route to an argument, not a completed
uniform finite-sample theorem. In particular, probabilities approaching zero,
coincident cutpoints, separation-like data, or weak contrast information fall
outside these regular conditions. Those are the difficult cases motivating
this work. The complete hypotheses of an applicable bootstrap theorem must
be checked before asserting a formal result.

## Decisions still needed before a frozen calibration study

* Original empty interior categories: the current ordinal fitter collapses
  them, retains their labels with zero probability, and has coincident
  cutpoints. M0 can consequently be singular. This needs an explicitly
  justified inferential scope or a different nuisance-estimation construction;
  arbitrary smoothing or numerical rank reduction would change the method.
* Selection: if original datasets with empty categories receive no test,
  distinguish unconditional rejection, result availability, and rejection
  conditional on reporting. Simulating all outcomes and silently interpreting
  that calibration as conditional on observed categories is unjustified.
  A matched conditional simulation scheme would have its own nuisance and
  computational questions. Empty simulated categories alone do not require
  dropping a replicate: the fixed-dimensional binary contrast remains defined.
* Genuine binary-fit failures: preserve them with an explicit reporting rule.
  A conservative upper-tail bound can count failed draws as exceedances when
  an underlying well-defined statistic exists; one must not delete failures
  or claim unconditional success from only successful draws. The observed
  statistic still requires a valid fit.
* Sparse nuisance uncertainty: fitting the null from one or two endpoint
  observations can change its simulated tail materially. Examine conditional
  frequency patterns, nuisance estimation error, and freshly fixed alternatives
  in the next study. Report 1% as well as 5% and Monte Carlo uncertainty.
* Exact finite-sample control, if required: known-parameter rank validity does
  not transfer to a plug-in null. A supremum over the composite null or a
  confidence-set construction with a coverage-error allowance would need a
  certified bound, not a finite grid presented as exhaustive. Its feasibility
  remains unresolved. See Dufour in THEORY.md for the general distinction.

The immediate aim is a defensible, computationally practical calibration
procedure with stated limits. A new public test should await a completed
nuisance-handling specification and fresh null/power evidence.
