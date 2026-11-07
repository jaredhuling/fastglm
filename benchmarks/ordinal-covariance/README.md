# Expected and observed information for ordinal standard errors

`run.R` compares slope standard errors from expected and observed information with
the sampling variability of the estimates. Designs use logit, probit, and cloglog
links, n of 500 and 2,000, and 5 categories up to one category per observation, with
1,000 replicates (200 when K is at least 1,000). `summary.csv` holds the results and
`costs.csv` the timing of each step. Raw replicates are on Box; see `STORAGE.md`.

## Findings

For the slopes of correctly specified models, the two choices were nearly equivalent.
Averaged within a design, the ratio of expected to observed standard errors ranged
from 0.993 to 1.000. The largest difference for a single fit was about 12%, at cloglog
with n = 500. Coverage of nominal 95% intervals differed by at most 1.0 percentage point
between the two for any slope. By slope, coverage ranged from 92.5% to 96.5% for expected
information and from 92.0% to 96.5% for observed information. Six of the 90 values for
each lie beyond 1.96 Monte Carlo standard errors of 95%, about the 4.5 expected by chance,
and none lie beyond 3.

The cost of the two choices differs far more than their results at very many categories.
At n = 20,000 and K = 2,000, the expected pass with its inverse took about 0.77 seconds,
while an observed pass took 0.006 seconds and its slope solve 0.001 seconds (`costs.csv`).

The designs ran in two sessions: 19 on one build and 11 on a later build. Those builds
differ only in code that does not change these fits. Costs come from the later build.

## Thresholds and misspecification

`extended.R` compares expected, observed, and robust standard errors for thresholds and
slopes, with the true link and with misspecified ones (logit fitted to probit, cloglog,
heavy-tailed, and skewed errors; probit fitted to logit and heavy-tailed errors). Each
design has 1,000 replicates, except that 9 logit and 6 probit replicates with 200
categories had an empty category and were dropped. Under misspecification, the target
is the fit to 2 million simulated rows. `extended-summary.csv` holds the results.

Expected and observed information were practically interchangeable: their mean standard
errors differed by at most 0.5% under the true link and 1.8% under misspecification.

| 95% interval coverage | Expected | Observed | Robust |
|:--|--:|--:|--:|
| Slopes, true link | 94.9% | 94.9% | 94.8% |
| Thresholds, true link | 95.1% | 95.1% | 95.0% |
| Slopes, misspecified | 94.7% | 94.7% | 94.9% |
| Thresholds, misspecified | 94.8% | 94.8% | 95.0% |

Coverage was within Monte Carlo error of 95% except for model-based thresholds under
misspecification, which ranged from 92% to 97.5%, with 7 expected and 8 observed values of
72 beyond three Monte Carlo standard errors. Robust thresholds had none. Robust covariance
is therefore the safer choice when the link is in doubt.

## Limits

Thresholds were studied up to 200 categories, not at one category per observation. The
designs use three predictors, and misspecification covers the listed error distributions only.
