# Analytic proportional-odds score development

Read [ASSESSMENT.md](ASSESSMENT.md) for the findings and decision,
[METHODS.md](METHODS.md) for the exact expected-information score construction,
and [INFORMATION-AUDIT.md](INFORMATION-AUDIT.md) for its distinction from
observed-information calculations. No production API was changed.

The analysis-time method is deterministic. Simulations evaluate its behavior;
they do not supply its p-values. The main development study has 30,000 datasets,
the additional fixed stress panel 8,000, and the information comparison replays
4,000 of those datasets. None is presented as independent confirmation.

Run from the package root with fastglm built from revision b5e08af (0.1.3), R
and numDeriv installed. `FASTGLM_STUDY_LIBRARY` optionally selects a separate
installed library. The manifests pin actual installed files as well as version
numbers. Do not overwrite a manifest to resume with different code or software.
Cross-platform reproduction should compare numerical values with justified
tolerance rather than require byte-identical compiled package files; this pass
verifies relocated reconstruction on the recorded platform.

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/validate.R
Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/run.R audit
Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/run.R summarize
Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/stress.R summarize
Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/timing-summary.R
Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/interpret.R
python3 benchmarks/ordinal-proportional-odds-development/analytic-score/check-regeneration.py
```

`run.R pilot` runs only the excluded timing sample; `run.R run` and `stress.R run`
generate missing compatible banks using three processes. They preserve existing
completed banks. `information-audit.R` replays the four specified cells and the
published numerical example; unlike summary commands, it fits models.
`benchmark.R` performs the separate descriptive timing exercise. Both write
their own outputs and do not change frozen method records.

`PROTOCOL.md` and `STRESS-PROTOCOL.md` define the two empirical panels. `raw/`
and `stress-raw/` retain every method outcome, seed, error and warning. CSV
summaries report conditional rejection, availability, paired comparisons and
unconditional bounds when tests are unavailable. `provenance.csv` hashes the
complete retained evidence set; regenerated charts and the inventory itself
are excluded from their own hash inventory.

`timing-pairs.csv` includes paired aggregate-time ratios and median individual
ratios from the original main run. Individual ratios omit zero denominators
caused by timer resolution and report their retained counts; they should not
be treated as precise microbenchmarks.

Tests with empty original categories or deficient predictor rank are not
silently redefined. ML and meanBR convergence failures remain visible, and
there is no automatic method switch. The theoretical scope is regular
first-order inference; a finite-sample guarantee is not claimed.
