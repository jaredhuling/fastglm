# Ordinal score-test performance

The development prototype exploits the tridiagonal threshold information in
the existing expected-information statistic. It changes the arithmetic and
shares work across requested hypotheses; it does not change the statistic,
its null estimator, or its first-order chi-square reference.

`development.R` and `run-development.R` retain the original implementation
experiment, its three data configurations, timings, and environment. The
confirmation below uses the installed production implementation and new data.

## Frozen confirmation design

Specified before running `run-confirmation.R` on either implementation:

* Baseline: package source at `c437a26e5778a127408c92aa2bca22096d05ff0c`, installed separately. Candidate:
  current implementation, identified by the retained source-file checksums and
  exact dumps of the functions under comparison. Both are version 0.1.3.
* Seed 111501, reset to seed plus case number for each configuration.
* Four configurations `(rows, predictors, categories)`: `(5000,3,4)`,
  `(30000,6,5)`, `(30000,12,8)`, and `(1000000,3,4)`.
* Independent standard normal predictors; logistic latent noise plus
  `0.35*x1 - 0.2*x2`; equally spaced interior cutoffs from -1.4 to 1.4.
  Dense individual responses, no weights or offsets, logit link.
* Both ML and mean-bias-reduced null estimation, tolerance 1e-10 and at most
  500 iterations. The original fit is outside the measured work.
* Three operations: score/information calculation alone; one public joint
  test including its null refit; joint plus all separate predictor tests.
  The baseline makes independent public calls for the last operation; the
  candidate uses `by.term=TRUE` with the same hypotheses and null estimator.
* One untimed warm-up, followed by five elapsed-time repetitions for each
  operation. Garbage collection precedes each measured repetition. Compare
  medians; retain every repetition, including unsuccessful results.
* Run baseline and candidate sequentially without other package tests or
  rendering jobs. This is a single-machine throughput comparison, not a claim
  about universal speed or latency.
* Record R-managed allocation totals and largest individual allocations for
  the score-only operation at `(30000,12,8)` using `Rprofmem`. This is neither
  peak resident memory nor native C++ allocation measurement.
* Check equality of input hashes, coefficients, test availability, degrees of
  freedom, statistics and p-values. Numerical tolerance is 1e-8 relative to
  `max(1,abs(baseline))`. Do not discard failures or alter the configurations
  after observing timings.

Run from the repository root, supplying the two separate installed-library
directories:

```sh
Rscript benchmarks/ordinal-score-performance/run-confirmation.R baseline /path/to/baseline-library /path/to/new-results
Rscript benchmarks/ordinal-score-performance/run-confirmation.R candidate /path/to/candidate-library /path/to/new-results
Rscript benchmarks/ordinal-score-performance/summarize.R /path/to/new-results
```

The runner refuses to overwrite existing implementation records. The exact
runner used for the retained study is saved as `confirmation/runner.R`; the
entry point adds the configurable output directory and overwrite protection.
To check and regenerate the retained summaries without rerunning timings:

```sh
Rscript benchmarks/ordinal-score-performance/summarize.R
```

The baseline can be reconstructed by exporting commit `c437a26` to a temporary
directory and installing it into its own R library. To reconstruct the candidate,
export the same commit and apply `confirmation/candidate-source.patch` to that
export. The patch contains the source and namespace changes required for the
timing comparison. Verify its files against `candidate-source-md5.csv`, then
install into a different R library. The confirmation directory
retains raw records, method functions, input and coefficient hashes, and R
session information. The source checksums identify the code used before later
documentation-only changes. Numerical derivative and reference-package tests
in the package test suite validate the new links and Wald interface separately;
this benchmark makes no claims about their runtime or finite-sample calibration.
