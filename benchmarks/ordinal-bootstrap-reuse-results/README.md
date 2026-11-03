# Reusing ordinal setup during parametric bootstrap

This bounded benchmark compares the existing ordinal parametric bootstrap with
the same procedure reusing design setup across refits. Each operation performs
19 refits. The cache starts empty for every bootstrap invocation, so its first
setup cost is included. The fitted distribution, random draws, starting values,
statistic, category policy, and convergence controls are unchanged. Pair and
cluster bootstrap paths are outside this optimization.

The four cases use 20,000 rows, either three predictors and four categories or
twelve predictors and eight categories, and both ML and mean bias reduction.
The inputs are the frozen fixtures in
[`../ordinal-prepared-results/fixtures.rds`](../ordinal-prepared-results/fixtures.rds).
Each case uses the first response in that fixture, its original integer
frequency weights (including zero-frequency rows), and its location offsets.
The baseline is the preceding fastglm 0.1.3 development snapshot with public
ordinal preparation but without bootstrap reuse. The candidate changes only
the ordinal bootstrap refit path; the compiled native library is identical.

| Predictors / categories | Method | Baseline median | Reuse median | Baseline R allocation | Reuse R allocation |
|---|---|---:|---:|---:|---:|
| 3 / 4 | ML | 0.586 s | 0.549 s | 829.6 MB | 676.5 MB |
| 3 / 4 | Mean BR | 0.903 s | 0.819 s | 829.6 MB | 676.5 MB |
| 12 / 8 | ML | 1.427 s | 1.266 s | 1906.0 MB | 1395.7 MB |
| 12 / 8 | Mean BR | 2.174 s | 1.979 s | 1906.0 MB | 1395.6 MB |

All four complete-output comparisons passed exactly. Median runtime fell by
6–11% in this panel, and cumulative R allocation fell by 18–27%. The totals
above cover the complete 19-refit invocation, including its first design setup.
The same seeded refits are repeated for timing; these are not independent
replicates of a statistical coverage study.

Run the baseline first and then the candidate, on a quiet machine, with separate
installed libraries and a common fresh output directory:

```sh
Rscript benchmarks/ordinal-bootstrap-reuse-results/run.R baseline /path/to/baseline-library /path/to/output
Rscript benchmarks/ordinal-bootstrap-reuse-results/run.R candidate /path/to/candidate-library /path/to/output
```

Before timing, the candidate must reproduce the entire saved baseline
bootstrap result identically, except its recorded call. This includes
replicates, seeds, diagnostics, failure records, standard errors, and intervals.
The fixed seed is 8402 and the runner specifies the random-number generators.
Results retain the three individual elapsed-time observations per case;
allocation profiling uses an additional, separate invocation.

Only three timing repetitions are used, with baseline runs preceding candidate
runs. This small panel supports a bounded computational comparison, not broad
runtime or statistical coverage claims. `Rprofmem()` records cumulative
allocations managed by R; it does not measure peak memory, resident set size,
or all native allocations. Reuse retains canonical input and transformed-design
arrays until that bootstrap invocation ends. Each worker has its own cache;
there is no shared cache across worker processes or bootstrap invocations.

The source fingerprints and installed-code metadata identify both development
snapshots; their shared version number alone does not. The installed libraries
and source archives are not bundled. The package's existing `benchmarks`
exclusion keeps this development evidence out of the distributed R package.

`baseline.rds` and `candidate.rds` retain complete comparison outputs and
installation metadata. The CSV files expose case summaries and all 24 individual
elapsed-time observations. `*-Rprofmem.out.gz` files retain losslessly compressed
allocation traces; the runner creates ordinary uncompressed traces for new
runs. `allocation-profile-fingerprints.csv` records each original trace's size
and SHA-256 fingerprint.
An initial baseline run is preserved under `preliminary-startup-overlap`: a
separate provenance process was finishing when it began, so the entire
baseline/candidate panel was subsequently run in a quiet window. Only that final
panel supplies the table above; no run was selected by its observed speed.

Focused serial and worker validation logs are in `validation/`. The final
worker run passed all 151 assertions without failures, test warnings, or skips.
These checks confirm nondefault-generator reproducibility, cache persistence
within each worker, and complete formula-model identity, including field order.
Earlier validation is retained separately; `validation/metadata.json` records
the exact source identities and test counts. `artifact-sha256.csv` verifies the
durable evidence files.
