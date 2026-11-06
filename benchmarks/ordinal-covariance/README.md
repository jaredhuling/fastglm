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

## Limits

These results cover slopes only, under the true link, with no misspecification. They
do not cover threshold standard errors, which rest on few observations when there are
very many categories. They do not by themselves justify changing what `vcov()` reports.
