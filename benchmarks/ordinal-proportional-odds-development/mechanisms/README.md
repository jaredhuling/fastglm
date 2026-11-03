# Proportional-odds test mechanism investigation

Read [ASSESSMENT.md](ASSESSMENT.md) for the findings and
[NEXT-METHOD.md](NEXT-METHOD.md) for the proposed next calculation and its
unresolved validity obligations. No public test was added.

The main frozen study has 36,000 response datasets over nine fixed designs.
A separately specified reference-precision check adds 40,000 datasets for two
of those designs. The latter reuses the original evaluation samples and is
not independent confirmation. The earlier random-design study remains in
`../study/` unchanged.

Run from the package root with the development package installed. Set
`FASTGLM_STUDY_LIBRARY` only if selecting a separate installed R library.
The frozen manifests require the recorded package/source/runtime identities;
do not overwrite them to resume with different software. For a new study,
use a new output directory and record the changed inputs.

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/validate.R
Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/run.R audit
Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/run.R summarize
Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/reference-check.R
Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/diagnose.R
Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/interpret.R
python3 benchmarks/ordinal-proportional-odds-development/mechanisms/check-regeneration.py
```

`run.R pilot` runs only the separate timing sample; `run.R run` generates
missing original banks with three processes. `reference-check.R` generates
missing precision-check banks with two processes. Existing banks are reused.
Auditing and summarizing completed banks require no simulation. The identity
and source checks are backed by `manifest.rds` and `reference-manifest.rds`;
`raw-hashes.csv` records SHA-256 hashes of the saved banks and designs.

`PROTOCOL.md` and `THEORY.md` specify the main comparisons and derivations.
`REFERENCE-CHECK.md` records the later check and its reason. Raw banks retain
all slope contrasts, statistics, category counts, seeds, errors, warnings,
fit residuals, covariance diagnostics and accumulated covariance matrices.
`summary.csv`, `paired.csv` and `moments.csv` contain the full results.
`mechanism-rejection.pdf` and `.png` visualize the small-sample comparisons.
No oracle calculation should be interpreted as a method available to an
analyst who does not know the generating model.

Empty-interior covariance statistics in the unrestricted population can pass
factorization solely because of roundoff; `empty-interior-diagnostics.csv`
documents this. Observed-category comparisons exclude these datasets.
Independent simulation-covariance calculations retain a fixed contrast
dimension. Evaluation Monte Carlo intervals condition on shared moment and
reference banks, as explained in the assessment.
