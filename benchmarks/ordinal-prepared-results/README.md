# Reusing ordinal design setup

This bounded implementation benchmark compares independent ordinal fits with
repeated fits using a prepared design. It uses two frozen responses per design,
20,000 rows, integer frequencies including zero-frequency rows, and fixed
location offsets. Both maximum likelihood and mean bias reduction are included.
The fitted coefficients and covariance matrices agree exactly with the frozen
baseline in these cases.

| Predictors / categories | Method | Baseline, two fits | Prepared reuse, two fits | Baseline R allocations | Prepared R allocations |
|---|---|---:|---:|---:|---:|
| 3 / 4 | ML | 41 ms | 39 ms | 23.59 MB | 8.02 MB |
| 3 / 4 | Mean BR | 74 ms | 71 ms | 23.59 MB | 8.02 MB |
| 12 / 8 | ML | 102 ms | 91 ms | 61.76 MB | 9.37 MB |
| 12 / 8 | Mean BR | 184 ms | 173 ms | 61.76 MB | 9.37 MB |

The clearest result is the 66–85% reduction in repeated R allocation. Timing
medians use only three repetitions of short operations, with the baseline run
before the candidate. The small elapsed-time differences are indicative, not a
precise general speedup estimate. No uncertainty intervals or broad performance
claims are inferred from this panel. Preparation alone was below the timer's
resolution in these runs and is not reported as having zero cost.

`Rprofmem()` measures cumulative allocations tracked by R, separately from the
timed repetitions. These totals are not peak memory, resident set size (RSS), or
all native allocations. A prepared object retains a transformed design for
reuse, so reduced allocation does not imply lower resident memory. The CSV
files separately report the first fit including preparation; no first-fit
acceleration is claimed. Direct and warm-reuse operations contain two fits,
whereas first-fit and preparation-only operations contain one.

## Reproduce

Install the baseline and candidate into separate libraries. The baseline was
the preceding fastglm 0.1.3 development snapshot; the candidate added ordinal
design preparation. Both used the same compiled native library. Source SHA-256
fingerprints, the baseline source-archive fingerprint, installed R database and
DLL fingerprints, and session information identify the measured snapshots.
Version number alone is insufficient to identify either snapshot. The libraries
and source archive themselves are not bundled here.

Run from a quiet machine, using a fresh output directory shared by these two
commands:

```sh
Rscript benchmarks/ordinal-prepared-results/run.R baseline /path/to/baseline-library /path/to/output
Rscript benchmarks/ordinal-prepared-results/run.R candidate /path/to/candidate-library /path/to/output
```

The runner uses the included `fixtures.rds`, verifies convergence and agreement
before timing, and writes all three elapsed-time observations plus allocation
profiles. `generate-fixtures.R` documents fixture generation and accepts a new
output filename; the bundled fixture fixes the actual inputs independently of
later R random-number defaults.

The runner is the original measurement protocol with portable input/output
paths and compact result storage. It was parsed after packaging; no additional
fits or timings were run to create this directory. The archived CSV and RDS
results retain the original measurements. `benchmark-baseline.rds` and
`benchmark-candidate.rds` contain compact coefficient/covariance references,
individual elapsed times, allocation totals, and installation metadata.
`timings.csv` exposes every elapsed-time observation. `allocation-profiles/`
contains the underlying R allocation traces. `source-fingerprints.csv` records
the source snapshots, and `artifact-sha256.csv` checks this evidence bundle.

The existing package-build exclusion for `benchmarks` excludes these development
artifacts from the distributed R package.
