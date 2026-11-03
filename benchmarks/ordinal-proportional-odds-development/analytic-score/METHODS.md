# Analytic proportional-odds score comparison

This development prototype follows the user's preference for deterministic
inference. Simulation is used to evaluate the procedures, not to obtain a
p-value when analyzing data. The project's computational requirement is a fast
test based on one proportional-odds null fit and blockwise matrix accumulation.
The earlier fitted-null simulation proposal is retained as historical work;
it is not the active implementation direction.

## Target and exact calculation

For independent observations at fixed predictors x_i, define q=K-1 and

    logit Pr(Y_i <= j) = alpha_j - x_i' beta - x_i' gamma_j,
    gamma_1 = 0.

The null is gamma_2=...=gamma_q=0. Thresholds and common slopes are nuisance
parameters theta=(alpha,beta); gamma has r=p(K-2) entries. This parameterization
defines a valid local alternative around an interior null at a finite design.
For a growing design, bounded covariates and separated thresholds ensure the
needed local neighborhood. It does not assert globally ordered probabilities
for arbitrary unequal slopes on an unbounded predictor domain.

For category k let pi_ik=F_ik-F_i,k-1, with F_i0=0 and F_iK=1. At the null,
f_ij=F_ij(1-F_ij), and the derivative with respect to cumulative predictor eta_j is

    d log(pi_ik)/d eta_j = f_ij {1(k=j)-1(k=j+1)} / pi_ik.

Multiplying by derivatives of eta_j gives each category's full parameter score
s_ik. The ordinary score and expected information, both in sum normalization, are

    U = sum_i s_i,Y_i,
    I = sum_i sum_k pi_ik s_ik s_ik'.

The implementation evaluates stable log-probability derivatives and accumulates
the category contributions by blocks. Partition I and U into theta and gamma.
Then define

    e = U_gamma - I_gamma,theta I_theta,theta^{-1} U_theta,
    J = I_gamma,gamma - I_gamma,theta I_theta,theta^{-1} I_theta,gamma,
    Q = e' J^{-1} e.

The first candidate uses the ML null estimate. Its nuisance score is zero up to
optimization tolerance, reproducing the usual efficient Rao score statistic.
The second uses the mean-bias-reduced null estimate with the same ordinary
likelihood derivatives and the nuisance projection above. Its p-value is
Pr(chi-square_r >= Q). It is called **score at meanBR**, not a Firth-corrected
test with established higher-order accuracy.

At a meanBR fit, U_theta is generally nonzero. Using U'I^{-1}U without subtracting
the nuisance component adds U_theta'I_theta,theta^{-1}U_theta to Q. This identity
is checked explicitly; the extra component does not test proportional odds.
No multiplier, empirical degree of freedom, fitted rejection threshold, or
data-dependent switch between the candidates is introduced.

## What is justified, and what remains open

The category score, information identity, Schur complement and norm decomposition
are algebraic identities at an interior parameter. Numerical likelihood
derivatives, exact enumeration of a tiny outcome space, and production-null
identities independently check their implementation.

Under a correctly specified, regular interior null, independent observations,
fixed p and K, controlled leverage, nonsingular limiting information and
consistent local null roots, the efficient score has a multivariate normal
first-order limit. The meanBR adjustment is order one while the likelihood
score is order sqrt(n). The meanBR and ML null estimates then differ by
order 1/n, and the projected scores are first-order equivalent. Both Q statistics
therefore have a chi-square_r limit under these conditions. This is a conditional
first-order argument, not a second-order correction or a finite-sample result.

For example, bounded full-rank predictor designs with a limiting nondegenerate
empirical distribution, fixed separated thresholds and finite coefficients
provide a regular regime. Sparse categories in small samples are precisely
where that approximation can be poor. A meanBR fit alone does not establish a
better tail approximation. The empirical study is designed to check that issue.

The regular full-dimensional calculation excludes original datasets with empty
categories. No category is dropped and no generalized inverse or reduced df is
substituted. Availability and conditional rejection are reported separately.
Optimization and factorization failures are retained. Positive probabilities,
convergence and factorization are checked; independence and correct model
specification are application assumptions, and limiting regularity is not
certified by those numerical checks.

## Computational scope

With d=q(p+1), the prototype's accumulation costs O(n K d^2) and stores
O(b d+d^2) auxiliary entries for block size b; factorization costs O(d^3).
It does not form an n-by-n matrix or fit an unrestricted ordinal alternative.
Dense individual-observation logistic models are the present scope. Grouped
frequencies, offsets, formula-term tests and other links need their own
implementation and checks before production integration. Benchmark timings
here describe the prototype, not an optimized production implementation.

## Primary literature and attribution

Liu, A., He, H., Tu, X. M., and Tang, W. (2023), *On testing proportional odds
assumptions for proportional odds models*, General Psychiatry 36(3), e101048,
[doi:10.1136/gpsych-2023-101048](https://doi.org/10.1136/gpsych-2023-101048).
The [published PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410795/)
was read in the model/test definitions, Score test, Simulation studies and
Discussion sections on November 3, 2026. The score construction is established;
the article's simulation study also reports small-sample over-rejection.
This implementation derives the derivatives directly rather than copying
software from the article. Bibliographic metadata were checked against the
published full text. No novelty claim is made for the score test or projection.

Peterson and Harrell (1990), *Partial Proportional Odds Models for Ordinal
Response Variables*, Applied Statistics 39(2), 205-217, doi:10.2307/2347760,
is an earlier primary antecedent identified through the publisher record and
references. The full published text was not accessible in this pass, so it is
a bibliographic lead, not evidence that an unread higher-order argument applies.
The specialization to a meanBR null root is labeled an adaptation with only
the first-order argument above; priority and higher-order validity are unresolved.

The bounded search used proportional-odds score tests, small-sample bias reduction
and Bartlett-type correction terms. It does not establish the absence of prior
work. Any later Bartlett-type or saddlepoint repair requires a separate exact
derivation including nuisance estimation; merely replacing moments at a fitted
null is not such a derivation.
