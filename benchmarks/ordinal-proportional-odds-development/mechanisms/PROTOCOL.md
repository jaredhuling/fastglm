# Fixed-design mechanism study

Frozen before simulation, November 3, 2026. This is a development diagnosis of
the failed proportional-odds test, not a comparison for choosing a covariance
formula by its rejection rate. Production fastglm remains at b5e08af. The
original random-design study and its frozen methods remain unchanged.

## Questions

1. Does using the exact derivative of the adjusted binary score materially
   change the covariance or null rejection distribution?
2. Does replacing empirical score products by the true score variance remove
   the discrepancy? Does a fitted proportional-odds score variance behave
   similarly? True quantities are diagnostic inputs only.
3. With an independently estimated sampling covariance of the slope contrasts,
   does the chi-square reference still fail? Does removing their independently
   estimated finite-sample mean change this conclusion?
4. Does replacing a dataset-specific covariance by its independent average
   change rejection? This assesses variability of covariance estimates and
   their dependence on the estimated contrasts, jointly.
5. What changes when empty-category datasets are retained in the fixed-dimensional
   meanBR slope comparison? This distinguishes selection from other mechanisms.

## Design

All data use the proportional-odds null. The covariate distribution and model
parameters match the corresponding scenarios in the first study, but the
covariates are drawn once per cell and held fixed over independent responses.
One-predictor designs use x1 uniform on [-1,1]. The five-category design uses
independent x1,x2 uniform on [-1,1] and x3 Bernoulli(0.4).

| Scenario | n | Thresholds | Slopes |
|---|---|---|---|
| regular3 | 80,1000 | (-0.8,0.8) | 0.7 |
| rare_lower3 | 80,250,1000 | (-3.8,0.4) | 0.7 |
| rare_interior3 | 80,1000 | (-0.08,0.08) | 0.7 |
| regular5 | 80,1000 | (-1.8,-0.6,0.6,1.8) | (0.6,-0.5,0.4) |

There are nine fixed designs. Each has three independent response banks:
1,000 datasets to estimate moments, 1,000 for an oracle null reference, and
2,000 for evaluation. Total: 36,000 response datasets. The separate pilot uses
replication indices 9001:9002 and is excluded. Design seeds are 30361100+cell;
response seeds are 40361100+100000*cell+10000*bank+replication. Banks are numbered
1,2,3. RNG: Mersenne-Twister/Inversion/Rejection. The order of cells is in
config.R. Three processes and one BLAS/OpenMP thread per process are used.

This restricted design set answers the mechanism questions raised by the
original study; it is not an untouched confirmation panel. Using one predictor
design per cell isolates conditional sampling behavior and limits generalization
across predictor configurations. Strong effects and correlated predictors were
not retained because the immediate questions concern the observed endpoint and
joint-test failures. No power study is conducted here.

## Covariance comparisons

Binary meanBR fits use the existing fastglm implementation. Let A be the ordinary
score sensitivity, H the exact adjusted-score sensitivity, B_emp the empirical
individual score product, B_true the true nested-binary score covariance, and
B_null the corresponding covariance under a fitted proportional-odds meanBR
model. All use unnormalized sums. Compare all six pairs of sensitivity
(A,H) and score covariance (B_emp,B_true,B_null), holding the slope estimates
fixed. Also evaluate the ordinary linearization at the true common parameters.
The algebra and limitations are in THEORY.md. A positive-definite factorization
is required to compute a quadratic statistic; no generalized inverse or
data-dependent rank reduction is used.

The independent moment bank supplies the mean mu_MC and covariance C_MC of the
contrast vector. Evaluate its quadratic form using C_MC, both with null center
zero and after subtracting mu_MC. Also evaluate the original statistic with
the independent mean of its empirical ordinary covariance held fixed. These
are oracle diagnostics, not candidate package methods.

For an oracle simulation reference, hold C_MC fixed and use the uncentered
quadratic statistic on the independent reference and evaluation banks. The
upper-tail p-value is (1 + number of reference statistics >= observed)/(B+1).
Because both banks use the true generating parameters, this isolates simulation
calibration from nuisance-parameter estimation. It is not a fitted-null
bootstrap. Shared reference-bank randomness is not included in the conditional
binomial Monte Carlo error for the evaluation bank.

## Selection, failures, and reporting

Report two populations separately: all response datasets with successful binary
fits, and the subset with every category observed. Moment and reference banks
use the same population definition as their evaluation bank. Empty interior
categories give repeated binary fits, but the slope vector still has its fixed
dimension; singular individual covariance statistics remain unavailable.
The full sampling covariance across datasets can still be nonsingular.

No failed dataset is replaced. Record category counts, fitting errors and
warnings, adjusted-score residuals, the relative change from A to H, and
method-specific covariance failures. Report availability, rejection at 5% and
1%, separate Wilson intervals, paired rejection differences on common available
datasets, contrast bias and variability, and covariance scale diagnostics.
No multiplicative correction or numerical eligibility threshold will be selected
from rejection rates. Numerical identities are checked before the study.

The primary interpretation compares mechanisms within a cell. A successful
oracle comparison establishes diagnostic feasibility, not a deployable test.
Any operating proposal must remove oracle inputs and receive a separate
theoretical argument and evaluation with fresh evidence.

## Reproduction

Run validate.R first, then run.R with pilot or run. All paths are relative to the
package root. FASTGLM_STUDY_LIBRARY optionally selects the installed library.
run.R records input hashes, dependency versions, design objects and session
information, and refuses an incompatible resume. audit and summarize actions
inspect saved data without fitting models. Raw bank files retain seed mappings,
all contrasts and method statistics, per-dataset variance diagnostics, and
the accumulated covariance matrices needed for independent averages.
