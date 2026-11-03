# Proportional-odds comparison: frozen development study

## Purpose and methods

Frozen before execution on November 3, 2026. This study evaluates the candidate
in `../prototype.R` at fastglm revision `b5e08af`. It asks whether its
chi-square approximation is usable at the sample sizes below. This is a
development evaluation, not a release qualification or a claim of novelty.
The public diagnostic currently reports descriptive slopes without p-values.

Each generated dataset is used by three methods:

1. The committed empirical-score covariance with ML binary fits (`ML`).
2. The identical covariance with mean bias reduction (`meanBR`).
3. `car::poTest()` applied to `MASS::polr()` (`car`). This is an external
   conventional Brant comparator using its installed defaults. Its p-value is
   computed from the returned chi-square statistic and degrees of freedom.

The prototype's coefficients, covariance, factorization rule, fitting tolerance
and maximum iterations remain unchanged. No rejection threshold or covariance
correction will be tuned to the results. Method source and installed dependency
versions are recorded with input hashes. The study stops if frozen files change.

## Generating distributions

All predictors are bounded and observations independent. For one predictor,
x1 is uniform on [-1,1]. Three-predictor designs use independent x1 and x2
uniform on [-1,1], and x3 Bernoulli(0.4), unless specified otherwise. Three-slope
models use beta=(0.6,-0.5,0.4). Cumulative probabilities are

    Pr(Y <= j | x) = logistic(alpha_j - x'beta - delta_j*x1).

Offsets are zero. Outcomes are drawn from these cumulative probabilities,
retaining every declared category even when unobserved. No dataset is replaced.

| Scenario | Thresholds alpha | Slopes or delta | Scientific role |
|---|---|---|---|
| regular3 | (-0.8,0.8) | beta=0.7 | Ordinary one-predictor null |
| regular5 | (-1.8,-0.6,0.6,1.8) | Three slopes | More thresholds and predictors |
| correlated5 | Same as regular5 | Three slopes | Correlated predictors: x1=u, x2=0.7u+0.3v, x3=0.7u+0.3w, independent uniform u,v,w |
| rare_lower3 | (-3.8,0.4) | beta=0.7 | Rare lowest category |
| rare_upper3 | (-0.4,3.8) | beta=0.7 | Mirrored endpoint distribution |
| rare_interior3 | (-0.08,0.08) | beta=0.7 | Few observations distinguish adjacent binary fits |
| rare_interior5 | (-1.8,-0.08,0.08,1.8) | Three slopes | Sparse interior with more contrasts |
| strong_signal3 | (-0.8,0.8) | beta=4 | Extreme binary probabilities and separation risk |
| alternative3 | (-0.8,0.8) | beta=0.7; delta=(-0.5,0.5) | Nonparallel slope alternative |
| alternative5 | (-1.8,-0.6,0.6,1.8) | Three slopes; delta=(-0.6,-0.2,0.2,0.6) | Nonparallel slope alternative with more contrasts |

All unspecified delta values are zero. Threshold differences exceed the maximum
possible change in delta*x1, so cumulative probabilities remain ordered even
under the alternatives. All nulls have fixed positive category probabilities;
sparse settings approach practical information limits without asserting an
asymptotic violation. Strong-signal and sparse cases are explicit stress cases.

## Replication and execution

Cross each scenario with n=80,250,1000, with 1,000 datasets per cell: 30,000
datasets and 90,000 method records. Seeds are 20261108+10000*cell+replication,
with cells in the order in `config.R`. RNG is Mersenne-Twister, Inversion,
Rejection. Seeds depend on datasets and not scheduling. Three pilot replications
per cell use replication indices 9001:9003 solely for timing and execution
checks; their outcomes do not enter the main results. Parallel execution uses
three processes and one BLAS/OpenMP thread per process. Computations use the
installed fastglm package and the committed research prototype.

The single dispatcher is `run.R`: `pilot`, `run`, `summarize`, or `audit`.
Run from the package root with Rscript and the installed development package on
the R library path. The root and library can be relocated. `run` resumes only
complete cell files under matching source hashes and dependency versions.
Raw method records include unavailable results and warnings. A manifest records
source identity, environment, and expected counts. Derived reports are regenerated
from the records. Numerical results may differ slightly across platforms;
elapsed timings are descriptive and are not a controlled speed benchmark.

## Metrics, failures, and decision

Report availability and rejection at nominal 5% and 1%, conditional on an
available result, with Wilson intervals and Monte Carlo standard errors.
Also report rejected/all datasets, and bounds assigning all unavailable cases
first to nonrejection and then to rejection. These bounds distinguish missing
inference from an apparently conservative test. A finite nonnegative statistic
and positive expected degrees of freedom are required for an external result.
Warnings remain attached to available results; car does not expose all auxiliary
fit convergence statuses. Do not infer successful fitting merely from its return.

For each cell and pair of methods, report the rejection difference and its
paired Monte Carlo standard error on the shared available datasets. Report
failed-fit reasons, empty-category frequency, covariance conditioning,
warnings, median and 90th-percentile runtime. The comparator's time includes
the ordinal fit required by its interface; the candidate time includes only
its binary fits and covariance calculation.

For the candidate methods, also report the bias, empirical SD, root mean squared
error, root mean estimated variance, 95% interval coverage and width for the
x1 slope difference between the second and first binary thresholds. Its truth
is delta_2-delta_1. These are diagnostic contrasts, not multiplicity-adjusted
intervals for all slopes.

Use Bonferroni-adjusted Wilson intervals across all 72 null method/cell
combinations to flag departures from 5%; these flags organize interpretation,
not promotion by a single threshold. The primary decision is whether systematic
inflation, poor availability, or weak power warrants retaining a descriptive
diagnostic. Favorable cells do not authorize a general test. No production
inference is changed by this study. A candidate or calibration change requires
a separate protocol and fresh evidence.
