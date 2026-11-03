# Explicit null-estimator confirmation

`ASSESSMENT.md` reports the completed diagnosis, calibration, power, availability,
interface changes, and decision. `METHODS.md` states the separation certificate,
the first-order argument, primary-source attribution, and remaining limits.

The independent confirmation comprises 13,500 datasets and 54,000 test records.
`PROTOCOL.md` and the source/installation identities were frozen before its
outcomes. `raw/` retains all records, including failures. `rates.csv` contains
nominal 5% and 1% rejection, Wilson Monte Carlo intervals, and missing-decision
bounds; `paired.csv` compares common-availability samples. `diagnosis.csv` is a
separate replay of previously observed failures, not confirmation evidence.

From the repository root, set `FASTGLM_MEANBR_LIBRARY` to the matching installed
package. The original local library was `/private/tmp/fastglm-meanbr-lib`.
For a new environment, reconstruct `source-snapshot.rds` alongside the unchanged
package files, install it into an isolated library, and distinguish changed
installation hashes from source or numerical differences. The saved snapshot
contains the exact R and native sources, DESCRIPTION, and NAMESPACE.

    Rscript benchmarks/ordinal-score-meanbr-confirmation/run.R audit
    Rscript benchmarks/ordinal-score-meanbr-confirmation/interpret.R
    python3 benchmarks/ordinal-score-meanbr-confirmation/check-regeneration.py

The audit never generates datasets. A new run requires a separate study root;
the runner refuses to overwrite existing cells. The pilot had 216 records and
estimated about two minutes of serial computation for the full grid. Three
processes were used with numerical-library thread counts set to one.

`checks/` retains the package tests, check, rendering and run records.
`provenance.csv` inventories final evidence by SHA-256. The regeneration script
verifies the inventory, reconstructs frozen sources in a temporary directory,
audits every record, and reproduces the tables and assessment byte for byte.
The `preflight/` directory preserves a runner parenthesis error and its correction
before any pilot or confirmation dataset was generated; no method changed.
