# Production proportional-odds score test

The inferential question is whether selected predictor associations have a
common cumulative log odds ratio across thresholds. This is an established
likelihood score construction. The contribution here is its integration into
fastglm with grouped frequencies, offsets, formula terms, bounded accumulation
blocks, and explicit failures. No novelty or higher-order Firth claim is made.

## Statistic and frequency convention

Let q=K-1 and let z_i contain the s selected columns of the location design x_i.
The local alternative is

    logit Pr(Y_i <= j | x_i) = alpha_j - x_i' beta - offset_i - z_i' gamma_j,
    gamma_1 = 0.

The null fixes every gamma_j to zero. All thresholds and common slopes are
nuisance parameters theta. Other predictors retain common slopes under this
alternative. In particular, this selected-term test is not an unrestricted
partial-proportional-odds fit for all remaining terms.

Let m_ik be the effective integer frequency for category k on row i, including
any supplied frequency weight; m_i=sum_k m_ik and N=sum_i m_i. For a single
individual response only one m_ik is nonzero. At the ML null, pi_ik is the
category probability and s_ik is the full parameter derivative of log(pi_ik).
The implementation calculates

    Ubar = sum_i sum_k m_ik s_ik / N,
    Ibar = sum_i m_i sum_k pi_ik s_ik s_ik' / N.

Expected information averages over all possible categories at the fitted null.
It is not the Hessian evaluated only at the observed outcomes. Partition Ubar
and Ibar into nuisance theta and tested gamma coordinates, and write

    ebar = Ubar_gamma - Ibar_gamma,theta Ibar_theta,theta^{-1} Ubar_theta,
    Jbar = Ibar_gamma,gamma
           - Ibar_gamma,theta Ibar_theta,theta^{-1} Ibar_theta,gamma,
    Q = N ebar' Jbar^{-1} ebar.

The nuisance score is zero at an exact ML root; retaining its projection also
accounts for numerical root tolerance. Diagonal coordinate normalization and
predictor centering/scaling leave this quadratic form invariant. The unit tests
compare numerical likelihood derivatives, frequency expansion, count doubling,
outcome reversal, predictor units, and accumulation block sizes. Twenty further
production/prototype comparisons differed by at most 8.9e-15.

The reference is chi-square with s(K-2) degrees of freedom. A regular null with
independent observations, fixed p and K, separated thresholds, finite slopes,
bounded full-rank predictors with nondegenerate limiting information, and a
consistent interior ML root supplies a regime in which the usual efficient
score central limit theorem gives this limit. Grouping is an exact aggregation
of independent observations at repeated predictor patterns. The total frequency
N, rather than the number of stored rows, governs the sample size.

This is a first-order asymptotic justification. It does not assert exact size,
uniform small-sample accuracy, higher-order bias correction, or validity for
dependent outcomes. For unequal slopes the model need only be ordered in a
neighborhood of the null on the supported predictor domain; no unrestricted
global model on arbitrary unbounded predictors is fitted.

## Implementation and assumption audit

The public assessment function leaves its default unchanged and requests the
test only through `po.test="score"`. It refits an ML null on the assessment
outcomes even when the original model used mean bias reduction. Supplied
evaluation outcomes affect this refit, but they do not alter the original
predictions or the accompanying prediction scores.

| Requirement | Treatment |
| --- | --- |
| Integer effective frequencies | Validated by the assessment response parser. |
| Complete, positive-frequency rows | The same included rows as prediction assessment are used. |
| Original category scale | Retained; any empty declared category makes the test unavailable. |
| Identifiable numerical design | Dense numeric design and rank checked during null fitting. |
| Usable ML root | Convergence checked; no mean-bias-reduced fallback. |
| Positive model probabilities | Stable formulas used; unrepresentable positive probabilities cause failure. |
| Usable expected information | Cholesky and normalized reciprocal-condition checks; no generalized inverse or changed degrees of freedom. |
| Independence and correct model specification | Application assumptions, not established by a numerical diagnostic. |
| Asymptotic regularity and adequate information | Theoretical assumptions; the finite confirmation grid cannot certify them in every application. |

Stored formula assignments map each selected term to all of its model-matrix
columns. This metadata survives prepared designs, serialization, and refits.
Older saved formula fits without those assignments still permit a global test;
selected-term tests request a refit explicitly.

## Computational scope

For d=q+p+s(q-1), accumulation costs O(n K d^2), where n is the number of
stored rows, and factorization costs O(d^3). Auxiliary block storage is
O(b d+d^2); the existing dense null fit also retains O(n p) design storage.
Grouped observations are not expanded to N individuals. There is no n-by-n
matrix and no fitted unrestricted alternative. High-dimensional designs or
many outcome categories can still make the d-by-d information expensive.

`timing.R` measures the full public assessment, which includes prediction
scores, calibration summaries, the ML null refit, and score accumulation.
Those timings are local measurements, not universal speed claims.

## Primary sources

Liu, He, Tu, and Tang (2023), *On testing proportional odds assumptions for
proportional odds models*, General Psychiatry 36(3), e101048,
[published full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410795/),
doi:10.1136/gpsych-2023-101048, describes the established score-test framework
and small-sample limitations. SAS Institute (2013), *SAS/STAT 12.3 User's Guide:
The LOGISTIC Procedure*, pages 4246 to 4248,
[manual](https://support.sas.com/documentation/onlinedoc/stat/123/logistic.pdf),
allows expected or observed information in its score-test definition. The
implementation here consistently uses expected information. These source
sections were read during development; the source hashes and the distinction
from the observed-information Cheese example are retained in the separate
development study's `sources.csv` and `INFORMATION-AUDIT.md`.
