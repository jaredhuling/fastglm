# Ordinal endpoint bootstrap: regression development simulation

## Question and decision

This simulation follows the exact intercept-only enumeration in
[`RESULTS.md`](RESULTS.md). It checks whether mean-bias-reduced (meanBR)
parametric bootstrap intervals retain useful coverage in regression models
when endpoint-empty refits are allowed, while interior-empty fits remain
excluded. The experiment used the package's `fastglm_ordinal()` and
`fastglm_bootstrap()` implementations from fastglm 0.1.3.

The candidate rule strongly improved availability in the rare-endpoint
setting, and its coverage was generally close to 95%. This is encouraging but
not enough to change the public bootstrap rule: coverage was somewhat low for
some threshold and slope cells, while the sparse-interior setting produced
too few available intervals for a useful calibration assessment. The result
supports a larger, focused validation of endpoint-empty meanBR inference; it
does not support relaxing the safeguard for interior-empty refits.

## Design and audit

The frozen design is documented in
[`REGRESSION-PROTOCOL.md`](REGRESSION-PROTOCOL.md), and the runner is
[`regression-simulation.R`](regression-simulation.R). It includes three
categories, one balanced fixed predictor with `x = -1, 0, 1`, sample sizes
60, 120, and 240, and 250 outer datasets in each of nine cells. Each eligible
meanBR fit received 199 parametric refits. The 2,250 outer datasets were
generated with master seed `20261104`; a separate timing pilot used a different
seed and is not included.

The current policy requires all categories to be present in the original fit
and every refit. The candidate allows endpoint emptiness under meanBR but
requires the interior category to remain represented in the original fit and
every refit. Both use the same fitted datasets and refit outcomes. Candidate
intervals are finite-`R` type-7 95% percentile intervals for each threshold
and the common slope.

The output audit found all nine planned cells with 250 datasets each. Every
original meanBR fit converged. No bootstrap refit or statistic failures were
observed. Results are stored in [`regression-raw.csv`](regression-raw.csv),
with audited cell counts in [`regression-cell-audit.csv`](regression-cell-audit.csv)
and derived metrics in [`regression-summary.csv`](regression-summary.csv).
The summary can be regenerated with
[`summarize-regression.R`](summarize-regression.R).

## Interval availability and coverage

Coverage and Monte Carlo standard errors are conditional on an interval being
available under the candidate rule. The table reports the number of available
outer datasets for each cell. Monte Carlo standard errors are about 0.013 to
0.018 when all 250 datasets contribute. Sparse-interior coverage estimates
have much larger uncertainty because very few intervals are available.

| Scenario | n | Current availability | Candidate availability | alpha1 coverage (n) | alpha2 coverage (n) | slope coverage (n) |
|---|---:|---:|---:|---:|---:|---:|
| Rare lower endpoint | 60 | 1.2% | 100% | 94.8% (250) | 91.2% (250) | 94.8% (250) |
| Rare lower endpoint | 120 | 6.0% | 100% | 95.6% (250) | 93.2% (250) | 92.8% (250) |
| Rare lower endpoint | 240 | 44.8% | 100% | 95.6% (250) | 94.8% (250) | 93.2% (250) |
| Sparse interior | 60 | 0.4% | 0.4% | 100% (1) | 0% (1) | 100% (1) |
| Sparse interior | 120 | 1.6% | 1.6% | 100% (4) | 75% (4) | 75% (4) |
| Sparse interior | 240 | 15.2% | 15.2% | 97.4% (38) | 97.4% (38) | 94.7% (38) |
| Moderate control | 60 | 100% | 100% | 93.6% (250) | 96.0% (250) | 96.0% (250) |
| Moderate control | 120 | 100% | 100% | 92.4% (250) | 95.6% (250) | 92.0% (250) |
| Moderate control | 240 | 100% | 100% | 91.2% (250) | 95.2% (250) | 91.6% (250) |

For rare endpoints, endpoint-empty original or bootstrap fits were common at
small n: the original lower endpoint was empty in 29.2%, 8.8%, and 0.8% of
datasets at `n = 60`, `120`, and `240`. All rare-endpoint original fits
retained the interior category and converged under meanBR. Candidate
availability was 100% in all three rare-endpoint cells. The current
rule withheld intervals for most datasets at `n = 60` and `n = 120`, and for
more than half at `n = 240`.

The rare-endpoint point-estimate biases were modest and generally declined
with n. For example, the bias in the rare first threshold decreased from 0.106
to 0.021 across the sample-size grid; the slope bias was between -0.008 and
0.011. Empirical standard deviations declined with sample size. The moderate
control biases were all within 0.021 in absolute value. Candidate median
interval widths also declined with n; in the rare-endpoint setting, widths
changed from 2.45 to 1.78 for the rare threshold and from 1.45 to 0.73 for the
slope. Directional misses were mostly balanced, although some individual
cells showed mild asymmetry. These summaries and the empirical SDs are in the
CSV output.

The sparse-interior scenario behaves differently. Only 64%, 80.8%, and 97.6%
of original fits retained the interior category as n increased. Even among
those fits, candidate availability was 0.4%, 1.6%, and 15.2%. Bootstrap
refits had an empty interior category at average rates of 26.0%, 19.5%, and
8.5%, respectively. The conditional coverage estimates at n=60 and n=120 are
based on one and four intervals; they do not establish calibration. At n=240,
38 intervals were available, still too few for a strong conclusion.

## Interpretation

The rare-endpoint results support the structural argument: when only an
endpoint is empty, meanBR can keep the full ordinal model and produce
regression bootstrap replicates with finite thresholds and slopes. Here,
allowing those endpoint-empty fits recovered many intervals that the current
uniform empty-category rule suppresses.

The results do not establish nominal coverage across settings. Some slope and
threshold coverage estimates are 2 to 4 percentage points below 95%, with
outer Monte Carlo standard errors around 1.5 to 1.8 percentage points.
Bootstrap quantile Monte Carlo error from `R = 199` adds further variability.
The sparse-interior control confirms that the endpoint exception does not
solve interior-category collapse or its poor bootstrap availability. No
failures were observed, but that only describes this design and optimizer.

These first-wave results alone were not enough to change the public rule. The
higher-precision follow-up in [`wave2/RESULTS.md`](wave2/RESULTS.md) added a
rare upper endpoint, a two-predictor model, 500 outer datasets, and 499 refits
per eligible fit. That follow-up supports a narrow parametric-meanBR endpoint
exception, which is now reflected in the implementation and ordinal vignette.
Interior-empty fits remain ineligible.
