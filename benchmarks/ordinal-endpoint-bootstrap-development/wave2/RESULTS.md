# Wave 2 results: endpoint-empty ordinal regression bootstrap

## Main finding

The higher-precision study confirms that the current empty-category rule
suppresses valid-looking meanBR bootstrap intervals when the only empty
categories are endpoints. Across the rare-lower, rare-upper, and
two-predictor rare-lower settings, the endpoint-allowance rule made intervals
available in all 4,500 outer datasets. The current rule's availability ranged
from 0% to 61.4% in those same cells. The moderate control had 100% candidate
availability and 99.8% to 100% current availability.

Coverage was generally close to 95%, but not uniformly so. A few cells were
about 2 to 3 Monte Carlo standard errors below nominal, including the rare
lower-endpoint second threshold at n=60 and the moderate-control slope at
n=120 and n=240. This is enough evidence to motivate a narrowly scoped
endpoint exception for further package-level review, but it does not justify
claiming calibrated coverage across settings or relaxing the interior-category
guard.

## Design and completion audit

The frozen design is in [`PROTOCOL.md`](PROTOCOL.md); the runner is
[`regression-wave2.R`](regression-wave2.R). It used fastglm 0.1.3, 500 outer
datasets per cell, and 499 parametric refits per original fit. There were four
settings: rare lower endpoint, its category-reversed rare upper endpoint, a
moderate one-predictor control, and a rare lower endpoint model with two
predictors. Sample sizes were 60, 120, and 240 for the one-predictor models,
and 90, 180, and 360 for the two-predictor model.

The 12 expected cells each contain 500 datasets (6,000 total). All original
fits retained the interior category and converged. Across all 2,994,000
parametric refits, there were no interior-empty refits, nonconverged refits,
or statistic failures. Endpoint-empty refits were retained under meanBR. The
raw dataset-level records, session information, cell audit, and derived
summary are in this directory; run `Rscript
benchmarks/ordinal-endpoint-bootstrap-development/wave2/summarize-wave2.R`
from the repository root to recreate the summary and audit.

## Availability and coverage

Coverage is conditional on candidate interval availability. Each coverage
estimate uses 500 datasets, giving Monte Carlo standard errors of about 0.005
to 0.012. Intervals are finite-`R` 95% type-7 percentile intervals for each
threshold and slope. The candidate rule permits empty endpoints but requires
the interior category and a finite, converged statistic in the original fit
and every refit.

| Scenario | n | Current availability | Candidate availability | Original endpoint empty | alpha1 coverage (MCSE) | alpha2 coverage (MCSE) | beta1 coverage (MCSE) | beta2 coverage (MCSE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rare lower endpoint | 60 | 0.0% | 100% | 31.4% | 95.8% (0.9%) | 91.6% (1.2%) | 95.2% (1.0%) | — |
| Rare lower endpoint | 120 | 2.2% | 100% | 10.0% | 95.2% (1.0%) | 94.0% (1.1%) | 94.8% (1.0%) | — |
| Rare lower endpoint | 240 | 30.4% | 100% | 0.6% | 96.4% (0.8%) | 96.2% (0.9%) | 94.2% (1.0%) | — |
| Rare upper endpoint | 60 | 0.4% | 100% | 27.8% | 96.8% (0.8%) | 95.0% (1.0%) | 94.8% (1.0%) | — |
| Rare upper endpoint | 120 | 2.0% | 100% | 9.4% | 96.0% (0.9%) | 95.0% (1.0%) | 93.6% (1.1%) | — |
| Rare upper endpoint | 240 | 28.6% | 100% | 0.8% | 94.2% (1.0%) | 96.0% (0.9%) | 95.6% (0.9%) | — |
| Moderate control | 60 | 99.8% | 100% | 0.0% | 94.8% (1.0%) | 94.4% (1.0%) | 95.8% (0.9%) | — |
| Moderate control | 120 | 100% | 100% | 0.0% | 94.2% (1.0%) | 97.2% (0.7%) | 92.2% (1.2%) | — |
| Moderate control | 240 | 100% | 100% | 0.0% | 93.8% (1.1%) | 95.2% (1.0%) | 92.0% (1.2%) | — |
| Rare lower, two predictors | 90 | 0.2% | 100% | 16.8% | 98.8% (0.5%) | 94.2% (1.0%) | 93.6% (1.1%) | 95.6% (0.9%) |
| Rare lower, two predictors | 180 | 11.0% | 100% | 3.2% | 94.4% (1.0%) | 95.0% (1.0%) | 94.8% (1.0%) | 95.2% (1.0%) |
| Rare lower, two predictors | 360 | 61.4% | 100% | 0.0% | 94.4% (1.0%) | 94.6% (1.0%) | 94.4% (1.0%) | 94.6% (1.0%) |

The current rule's low availability aligns with endpoint-empty original fits
and endpoint-empty refits. In the rare-lower one-predictor case, for example,
the mean fraction of bootstrap refits with an empty endpoint was 28.7%, 13.6%,
and 2.8% at n=60, 120, and 240. The corresponding rates in the rare-upper
setting were 26.5%, 13.5%, and 2.8%; in the two-predictor setting they were
18.4%, 6.9%, and 0.7%. The endpoint allowance therefore recovered intervals
without changing the estimator, bootstrap draws, or quantile calculation.

Coverage is not consistently one-sided in the available results. For example,
the n=60 rare-lower alpha2 interval undercovered through lower-end misses,
whereas some moderate-control slope undercoverage had misses in both
directions. Point-estimate biases were generally modest and empirical standard
deviations declined with sample size. Median interval widths also declined;
full parameter-level bias, empirical SD, interval width, and directional-miss
summaries are in [`wave2-summary.csv`](wave2-summary.csv).

## Relation to the sparse-interior negative control

Wave 2 did not add a new sparse-interior cell. The separate wave-1 negative
control remains decisive for the safeguard: candidate availability there was
only 0.4%, 1.6%, and 15.2% at n=60, 120, and 240, with interior-empty refits
common. The endpoint exception must therefore continue to reject interior
collapse. See the parent [`REGRESSION-RESULTS.md`](../REGRESSION-RESULTS.md).

## Assessment and next decision

The evidence supports distinguishing endpoint emptiness from interior
collapse in meanBR bootstrap availability. It is consistent across lower and
upper endpoints and extends to the balanced two-predictor model. However,
several moderate-control slope coverage estimates are around 92%, while the
rare-endpoint cases are closer to nominal. The coverage findings are finite
sample and scenario-specific; they are not a general bootstrap validity
result.

The public availability rule now allows empty endpoints only for meanBR
parametric bootstraps. It preserves the existing failure and nonconvergence
safeguards and rejects original fits or refits with an empty interior category.
The estimator and percentile calculation are unchanged. The public help and
ordinal vignette explain why endpoint emptiness leaves the threshold
parameters intact, while interior emptiness ties thresholds and changes the
fitted model. The study supports this scoped availability change; it does not
establish bootstrap coverage for other resampling schemes, links, designs, or
data-generating conditions.
