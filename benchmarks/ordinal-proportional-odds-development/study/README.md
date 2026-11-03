# Proportional-odds test development study

This directory contains the frozen comparison described in `PROTOCOL.md`.
The candidate uses the committed `../prototype.R`; package code remains at
fastglm revision `b5e08af`. `manifest.txt` identifies the numerical inputs and
dependency versions, and `session.txt` records the R environment. The installed
package files used for this run are identified in `installed-package-hashes.csv`.
That file describes the local installation; compiling on another platform can
change binary hashes without changing the source method.

## Commands

Run commands from the fastglm repository root. Install that source revision and
the `car` and `MASS` dependencies first. `FASTGLM_STUDY_LIBRARY` can identify an
additional R library; it is optional when the required package is on the normal
library path. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` to 1 for comparable timing conditions.

```sh
Rscript benchmarks/ordinal-proportional-odds-development/study/run.R audit
Rscript benchmarks/ordinal-proportional-odds-development/study/run.R summarize
Rscript benchmarks/ordinal-proportional-odds-development/study/diagnose.R
Rscript benchmarks/ordinal-proportional-odds-development/study/interpret.R
python3 benchmarks/ordinal-proportional-odds-development/study/check-regeneration.py
```

The audit checks the complete method/replication grid and frozen identity.
Summarization regenerates tables and figures from saved records. Neither audit
nor summarization fits models. They require the versions recorded in the
manifest so an environment change is visible. Diagnosis replays 30 selected
meanBR fits and independently checks their covariance. Interpretation generates
the assessment from the saved results. The Python check regenerates summaries
in a separate temporary directory and compares the numerical tables, report,
and provenance; plot file bytes are excluded because graphics devices can
include timestamps.

The original computation was launched with:

```sh
FASTGLM_STUDY_WORKERS=3 Rscript benchmarks/ordinal-proportional-odds-development/study/run.R run
```

`run` preserves completed cell files and computes missing cells. A fresh run
should use a separate copy of the study directory without `raw/`, preserving
the frozen source files and manifest. Do not mix results from changed candidates
or settings into this directory. Independent reruns use the same dataset seeds;
small numerical differences across platforms are possible. To change a design,
method, dependency version, or seed block, create a new study version.

## Files and interpretation

`raw/cell-*.rds` contains one row per method and dataset, including unavailable
results, errors, warnings, timing, and category-frequency diagnostics.
`pilot.rds` contains the separate timing pilot and is excluded from summaries.
`summary.csv`, `paired.csv`, and `failure-counts.csv` are derived from the raw
records. `RESULTS.md` presents all cells; `ASSESSMENT.md` gives their scientific
interpretation. `null-rejection.pdf` and `null-rejection.png` show false-positive
rates conditional on available tests. The tables also report availability and
bounds accounting for unavailable results.

The report is a development assessment. A successful numerical audit does not
establish calibration of the test or justify changing the public API.
