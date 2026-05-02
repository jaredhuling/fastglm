
## CRAN submission for 'fastglm' 0.1.0

This release adds:

* `fastglm_nb()` — joint (beta, theta) MLE for negative-binomial GLMs.
* `fastglm_hurdle()` and `fastglm_zi()` — hurdle and zero-inflated
  count models (Poisson / NB) with C++ drivers and Louis-formula vcov.
* Firth bias-reduced logistic regression (`firth = TRUE`).
* Native fast paths for tier-2 families: negative binomial,
  quasi-binomial, quasi-poisson, Tweedie.
* HC and cluster-robust covariance now registered on
  `sandwich::vcovHC()` / `sandwich::estfun()` / `sandwich::bread()`
  via delayed S3 registration in `.onLoad()`, so user code uses
  `sandwich::vcovHC(fit)` rather than a shadowed local generic.

## Test environments

* local Mac OSX Sequoia (R 4.5.1)
* Rhub linux, macos-arm64, m1-san, windows
* Rhub atlas / c23 / clang-asan (R-devel)

## R CMD check results

0 errors | 0 warnings | 1 note

The note flags `fastglm.fit` and `fastglm.control` as apparent S3
methods of the generic `fastglm()` because of the dot-naming pattern;
they are intentionally standalone exported functions (mirroring
`stats::glm.fit` / `stats::glm.control`) and their signatures
deliberately differ from the generic. The names predate the package's
adoption of `fastglm()` as a generic and are preserved for backwards
compatibility with downstream callers using
`glm(..., method = fastglm.fit)`.

## Reverse dependency results

revdepcheck against 16 CRAN reverse dependencies: 15 OK, 1 with a
snapshot-test failure.

* `SEQTaRget 1.4.1` — one snapshot test
  (`test_multinomial.R:92`, "Multinomial Censoring Excused
  Post-Expansion") compares fitted coefficients to a hardcoded
  expected list with `tolerance = 1e-2`. Several coefficients drift
  1-5% between fastglm 0.0.4 (against which the snapshot was
  generated) and 0.1.0. Both versions match `stats::glm()` to
  approximately 1e-15 on non-degenerate fits; the SEQTaRget test
  exercises a near-separated quasi-binomial scenario (one fitted
  coefficient is ~10.2 on the logit scale) where the likelihood is
  nearly flat near the boundary and IRLS endpoints are not unique to
  floating-point precision. 
