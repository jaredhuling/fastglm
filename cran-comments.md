## Update to 'fastglm' 0.1.3

This release adds cumulative-link ordinal models, including partial proportional
odds, mean bias reduction, and blockwise fitting; extends inference for count and
binomial models; and corrects several numerical and display issues. See NEWS.md.

Aliased coefficients from rank-deficient designs are now reported as `NA`, as in
`glm()`, instead of 0. NEWS.md lists this and the other behavior changes.

## Test environments

* local macOS (aarch64, R 4.5.1)
* R-hub and win-builder: results to be added before submission.

## R CMD check results

Local `R CMD check --as-cran`: 0 errors | 0 warnings | 0 notes.

## Reverse dependencies

We checked all 18 reverse dependencies with revdepcheck, comparing the CRAN
version of 'fastglm' with this release. 16 show no new problems.

Two have new problems, both caused by aliased coefficients now being `NA`:

* DRDID: one test expects the word "singular" in an error message. DRDID's
  estimators still reject the rank-deficient design, with DRDID's own message
  about `NA` coefficients.
* FBMS: a vignette stores coefficients of rank-deficient candidate models, and
  the `NA` values propagate to its predictions.

We notified both maintainers before submission.
