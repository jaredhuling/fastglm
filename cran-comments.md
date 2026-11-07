## Update to 'fastglm' 0.1.3

This release adds cumulative-link ordinal models, including partial proportional
odds, mean bias reduction, and blockwise fitting; extends inference for count and
binomial models; and corrects several numerical and display issues. See NEWS.md.

Some calls that previously returned arbitrary coefficients for singular designs now
report an error. NEWS.md lists these behavior changes.

## Test environments

* local macOS (aarch64, R 4.5.1)
* R-hub and win-builder: results to be added before submission.

## R CMD check results

Local `R CMD check --as-cran`: 0 errors | 0 warnings | 0 notes.

## Reverse dependencies

Results to be added before submission.
