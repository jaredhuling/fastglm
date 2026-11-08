## Update to 'fastglm' 0.1.3

This release adds cumulative-link ordinal models, including partial proportional
odds, mean bias reduction, and fitting from data read in blocks of rows; extends
inference for count and binomial models; and corrects several numerical and display issues. See NEWS.md.

Aliased coefficients from rank-deficient designs are now reported as `NA`, as in
`glm()`, instead of 0. NEWS.md lists this and the other behavior changes.

## Test environments

* local macOS (aarch64), R 4.5.1
* win-builder: R-devel
* R-hub: Linux, Windows, and macOS (x86_64), R-devel
* GitHub Actions: Ubuntu, Windows, and macOS, R-release; Ubuntu, R-devel, R 4.4,
  R 4.3, R 4.2, and R 4.1

## R CMD check results

0 errors | 0 warnings | 0 notes on win-builder R-devel. The local check has one
NOTE, which says the local HTML Tidy is too old to validate the HTML manual.

## Reverse dependencies

We checked all 18 reverse dependencies with revdepcheck, comparing the CRAN
version of 'fastglm' with this release. 16 show no new problems.

Two have new problems, both caused by aliased coefficients now being `NA`:

* DRDID: one test expects the word "singular" in an error message. DRDID's
  estimators still reject the rank-deficient design, with DRDID's own message
  about `NA` coefficients.
* FBMS: a vignette stores coefficients of rank-deficient candidate models, and
  the `NA` values propagate to its predictions.

We notified both maintainers before submission. The DRDID maintainer has
merged a corrected test into DRDID's development version.
