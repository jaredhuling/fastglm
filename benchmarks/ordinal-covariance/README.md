# Expected and observed information for ordinal standard errors

`run.R` compares slope standard errors from expected and observed information with
the sampling variability of the estimates. Designs use logit, probit, and cloglog
links, n of 500 and 2,000, and 5 categories up to one category per observation, with
1,000 replicates (200 when K is at least 1,000). `summary.csv` holds the results and
`costs.csv` the timing of each step. Raw replicates are on Box; see `STORAGE.md`.

## Findings

Pending the corrected run. An earlier run reused seeds across replicates and is invalid.

## Limits

These results cover slopes only, under the true link, with no misspecification. They
do not cover threshold standard errors, which rest on few observations when there are
very many categories. They do not by themselves justify changing what `vcov()` reports.
