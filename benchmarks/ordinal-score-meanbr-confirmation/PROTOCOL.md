# Confirmation of explicit ML and meanBR null score tests

This study evaluates the production implementation of the ordinary efficient
score at either an ML or a mean-bias-reduced proportional-odds null. The latter
is an explicit user choice, not a fallback after inspecting an ML result.
Both retain the same nuisance projection, expected information, and chi-square
reference. No simulated critical values, resampling, multipliers, or adjusted
degrees of freedom are used. The ML default is unchanged.

The implementation and this protocol are frozen before confirmation outcomes.
The earlier meanBR prototype and the diagnosis of the original sparse-predictor
failures are developmental evidence. This evaluation has a separate root and
fresh seeds. It uses the nine previously specified generator scenarios from
`../ordinal-score-confirmation/config.R`, now at n=80, 300, and 1200, with 500
replications per cell. There are 13,500 datasets and 54,000 records: two null
estimators and both joint and selected-first-predictor tests on every dataset.
The grouped scenario has 20 fixed predictor patterns, hence integer frequencies
4, 15, and 60 per pattern at these sample sizes.

Seeds are `133611101 + 10000 * cell + replication` under
Mersenne-Twister/Inversion/Rejection. Pilot replications 9001 and 9002 are
timing and interface checks only. At most three processes run cells concurrently,
with numerical-library threads fixed to one. Every applied test calls the public
`fastglm_po_test()` with explicit null estimation and the same 1e-10 tolerance
and maximum 500 iterations. A fixed, independent reference model supplies only
design-column and category metadata; the public function refits each requested
null to the supplied dataset. No reference coefficients enter the statistic.

Report nominal 5% and 1% rejection, 95% Wilson Monte Carlo intervals, availability,
all failure reasons, and bounds over unavailable decisions. Also report paired
rejection on samples where both methods are available. Keep separated samples
in the unconditional denominator and do not replace failed datasets. Examine
the two nonproportional alternatives for power. Do not interpret conservatism as
better calibration merely because a method rejects less often.

This evaluates a first-order approximation. Modest finite-sample distortion is
possible, and exact size in every cell is not an acceptance criterion. Promotion
requires numerical derivative validation, honest availability handling, completed
raw-record and identity audits, and a scientific interpretation over the full
grid. A material unexplained distortion requires diagnosis rather than tuning.
Neither estimator supplies information for an absent predictor level, and empty
declared categories remain unsupported for this test.

`run.R pilot` freezes method/source and installed-file hashes and records exact
source bytes. `run.R run` generates each cell once by atomic save and then audits
and summarizes it. `run.R audit` regenerates all summaries without generating
datasets. Existing cells are never overwritten. A separate `diagnose.R` replays
all 116 original n=100 sparse-predictor failures, preserving their developmental
status. Final reporting and a relocation check complete the study.
