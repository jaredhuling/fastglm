# Independent confirmation of the ordinal score test

This protocol freezes the production candidate before generating confirmation
outcomes. It follows the development study in
`../ordinal-proportional-odds-development/analytic-score/`. No confirmation
outcomes choose the information matrix, null estimator, numerical tolerance,
reference distribution, or model implementation.

The candidate is the public `fastglm_ordinal_assess(..., po.test="score")`
implementation. It refits the proportional-odds null by maximum likelihood and
uses expected Fisher information, with nuisance-score projection and an
asymptotic chi-square reference. This is deterministic inference: simulation
evaluates the procedure but is not used to calculate an applied p-value.
The default assessment remains descriptive (`po.test="none"`).

## Questions and design

We assess size at nominal 5% and 1%, availability, and power for a joint test of
all predictors and a test of the first predictor alone. Other predictors retain
common slopes in the selected-predictor alternative. We do not interpret that
test as allowing arbitrary nonproportional slopes for the other predictors.

There are nine scenarios, three sample sizes (100, 400, 1600), and 1000
replications per cell: 27,000 independently generated datasets and 54,000 test
records. `config.R` gives the exact distributions, slopes, thresholds, offsets,
grouped frequencies, and two ordered nonproportional alternatives. These
parameters differ from the development grid. The scenarios cover four and six
categories, correlated predictors, rare endpoints, a narrow interior category,
an uncommon binary predictor, high leverage, and grouped counts with offsets.
The grouped scenario has 20 fixed predictor patterns and increasing frequency
per pattern; its sample size denotes individual frequency, not row count.

RNG: Mersenne-Twister/Inversion/Rejection. Dataset seed is
`110361101 + 10000 * cell + replication`. Cells are ordered by `config.R`.
Replications 9001 and 9002 are timing-only pilot data and are excluded from the
confirmation analysis. Both tests use the same dataset. Up to three processes
run cells concurrently with one numerical-library thread per process.

The original model is ML, with tolerance 1e-10 and maximum 500 iterations.
The public assessment independently refits its ML null with the same controls.
Original fitting failures are retained for both tests, never replaced with new
data. Empty categories, deficient designs, nonconvergence, and unusable
information remain explicit failures. There is no silent switch to bias
reduction, a generalized inverse, fewer categories, or an observed-information
test. Warnings and category totals are retained alongside every result.

## Evidence and interpretation

Report cell-level rejection rates among available tests with 95% Wilson Monte
Carlo intervals, availability, and bounds obtained by counting every unavailable
test as respectively a nonrejection or a rejection. The bounds describe missing
test decisions, not a proposed failure-handling procedure. Include all failures,
degrees of freedom, empirical statistic quantiles, and elapsed times. Do not
pool null and alternative cells or use pooled results to conceal difficult
designs. The 1% and 5% intervals are pointwise Monte Carlo intervals, not
simultaneous inference over the grid.

Promotion requires numerical validation, a completed identity audit, and a
scientific interpretation of every scenario. No finite grid establishes exact
size or uniform validity. Unexplained substantial inflation would require
diagnosis before promotion; the method will not be tuned on this confirmation
bank. Regular first-order inference, rather than a higher-order Firth test, is
the intended claim. Small or sparse samples and lack of a finite ML null remain
limitations even if the original mean-bias-reduced model has finite estimates.

## Execution and audit

Use `FASTGLM_CONFIRM_LIBRARY` to select the frozen installation. Run
`Rscript benchmarks/ordinal-score-confirmation/run.R pilot`, then `run`.
The first action saves hashes of source, installed files, this protocol, the
generator, runner, and summarizer. It also saves the exact source contents and
session information. Frozen identities are checked before and after execution.
Each completed cell is saved by atomic rename. Existing cells are audited
before reuse. No failed dataset is rerun as a new replicate.

`Rscript benchmarks/ordinal-score-confirmation/run.R audit` checks all record
counts, seeds, cell labels, degrees of freedom, finite statistics and p-values,
and frozen identities, then regenerates all summary CSV files from raw records.
An independent relocation check must reproduce those CSV files byte for byte.
`ASSESSMENT.md`, written after the run, records interpretation and the decision.
The final evidence inventory is written after that assessment. Local execution
is followed through to audit and reporting in the same work session.
