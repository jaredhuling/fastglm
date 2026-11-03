# Separating covariance and distributional approximations

The inferential target is equality of the slopes in the cumulative binary
logistic models. Estimation is performed separately at each threshold using
mean bias reduction. The contrast vector Delta subtracts each first-threshold
slope from its counterpart at later thresholds. Under the null, the population
contrast is zero; the finite-sample mean of its estimator need not be zero.

## Exact binary score derivative

For a fixed individual-observation design Z (including its intercept), write
p=logistic(Z theta), w=p(1-p), A=Z' diag(w) Z,
v_i=z_i' A^{-1} z_i, and h_i=w_i v_i. The adjusted binary score is

    U*(theta) = Z'(y-p) + a(theta),
    a(theta) = Z'{h (1/2-p)}.

This is the logistic specialization of the binomial adjustment in Kosmidis
(2014), Section 4.4, equation (11), and agrees with the package's binomial
Jeffreys-penalty gradient. For coefficient k,

    dp/dtheta_k = w z_k,
    dw/dtheta_k = w (1-2p) z_k,
    dA/dtheta_k = Z' diag(dw/dtheta_k) Z,
    dh_i/dtheta_k = (dw_i/dtheta_k) v_i
                    - w_i z_i' A^{-1} (dA/dtheta_k) A^{-1} z_i,
    da/dtheta_k = Z'{(dh/dtheta_k)(1/2-p) - h w z_k}.

Consequently H=-dU*/dtheta'=A-da/dtheta' is the exact adjusted sensitivity.
algebra.R computes these expressions without an observation-by-observation
hat matrix. Storage uses observation-by-parameter matrices and small parameter
matrices, preserving the package's large-data architecture. Numerical-Jacobian
agreement and Hessian symmetry are implementation checks, not proofs of a
finite-sample covariance formula.

Conditional on Z, a(theta) is deterministic at a fixed theta. Therefore the
random score variance at that theta is unchanged by adding a(theta). It would
be incorrect to treat arbitrary per-observation allocations of the adjustment
as independent extra random contributions to the sandwich meat. Replacing A
by H corrects a derivative approximation; H^{-1} B H^{-T} is still a local
linearization of a nonlinear estimator, not its exact sampling covariance.

## Score covariance and nonlinear estimation

For events Y>j and Y>k, let s_ij be the true survival probability. Their covariance
is s_i,max(j,k)-s_ij s_ik. Summing that coefficient times z_i z_i' yields the
true cross-threshold score covariance B_true. The adjustment contributes no
additional score variance under the fixed-design argument above. B_emp instead
uses realized products of ordinary residuals at fitted binary probabilities.
B_null replaces the true survival probabilities by a coherent fitted ordinal
distribution; it estimates the conditional variance of the scores, rather than
their uncentered second moment at arbitrarily misspecified binary parameters.

The study's oracle substitutions diagnose errors in estimating B while keeping
the nonlinear binary estimates and their fitted sensitivity unchanged. The
additional true-parameter linearization fixes both sensitivity and B. None of
these substitutions makes a linearization exact in small samples.

C_MC estimates the actual sampling covariance of Delta from independent outcomes
at the same fixed predictor design. Using it on an independent evaluation bank
removes dataset-specific covariance estimation, up to finite moment-bank error.
Centering by a separate bank mean additionally removes the estimated mean shift.
Residual mismatch with chi-square is evidence about distributional shape, with
remaining moment-bank uncertainty. Replacing only the dataset-specific empirical
covariance by its independent mean examines inversion variability and dependence
between covariance and slope estimates jointly; it does not isolate those two
effects from each other.

## Simulation calibration and nuisance parameters

Given the independent moment bank, the oracle reference and evaluation statistics
use the same fixed function of independent outcomes from the same true model.
For a single evaluation statistic, its upper-tail rank among the reference
statistics and itself is exchangeable. Counting ties conservatively with >=
gives a super-uniform p-value (1 + exceedances)/(B+1). The same argument applies
within a predeclared subset, such as all categories observed, when reference
and evaluation statistics are sampled under the same selection event and we
condition on the selected reference-bank size. This conditional statement does
not account for the frequency with which a user would receive no result.

This argument uses the true model. Replacing its unknown parameters by estimates
from the observed data removes exact exchangeability. A fitted-null bootstrap
requires a separate approximation argument; the successful oracle calculation
cannot justify it by itself. Shared reference-bank randomness also means the
reported evaluation-bank Monte Carlo intervals condition on that reference bank.

A potential finite-sample route is to maximize valid Monte Carlo p-values over
the composite null, or over a confidence set with an explicit coverage-error
allowance. Such a route needs a justified parameter set and certified control
of the maximization; an arbitrary nuisance grid is not a valid substitute.
Its computational feasibility is unresolved for this package. A regular
parametric bootstrap may be more practical, but its sparse-sample accuracy must
be established empirically and its regularity assumptions stated.

## Evidence and assumptions

The score derivative and nested-event covariance above are algebraic identities.
The sandwich interpretation is conditional on local linearization and appropriate
sampling regularity. The Monte Carlo rank argument requires matched distributions
and exchangeability; the study guarantees these only for its oracle construction
through fixed designs and independent deterministic seed blocks. Factorizations,
convergence, and category counts are checked numerically. Independence and correct
model specification would be assumptions in an application.

This study adapts established estimating-equation and Monte Carlo methods. It
makes no claim of a new theorem or a validated public hypothesis test.

## Sources and reading coverage

* Kosmidis (2014), *Improved estimation in cumulative link models*, JRSS B 76,
  169–196, doi:10.1111/rssb.12025. The published version was read in Section 4.4,
  equation (11), for the binomial specialization, supplementing the earlier
  reading recorded in ../METHODS.md. [Published PDF](https://wrap.warwick.ac.uk/id/eprint/98828/7/WRAP-improved-estimation-cumulative-link-models-Kosmidis-2014.pdf).
  SHA-256: dd66691df64cc60ec764460dfff70f426195e69c6a749a6c434a5f206c9776e2.
* Dufour, *Monte Carlo tests with nuisance parameters: a general approach to
  finite-sample inference and nonstandard asymptotics*, author manuscript dated
  August 12, 2005. [Author PDF](https://jeanmariedufour.github.io/Dufour_1995_MCT_W.pdf).
  SHA-256: f230499e333aba4dd837dc62e21aa3ed61e313a7b2534d5049cf8af38f061599.
  The rank construction and the distinction from estimated-nuisance bootstrap
  motivate the investigation. The published article is Journal of Econometrics
  133 (2006), 443–477, doi:10.1016/j.jeconom.2005.06.007; numbering here refers
  to the author manuscript. The full bootstrap theorem has not been applied
  to this ordinal estimator; its hypotheses remain an open obligation.
