
## CRAN submission for 'fastglm' 0.1.1

This is a minor release that extends Firth bias reduction to all
standard GLM families and backends, improves numerical stability for
negative-binomial models, and adds SQUAREM acceleration for
zero-inflated EM.

### New features

* Firth bias-reduced GLMs (`firth = TRUE`) now work for all standard
  families (gaussian, binomial, poisson, Gamma, inverse.gaussian) on
  dense, sparse, and streaming backends. In 0.1.0 Firth was limited to
  `binomial(link = "logit")` on dense designs only.
* SQUAREM acceleration (Varadhan and Roland, 2008) for the
  `fastglm_zi()` EM driver, converting linear EM convergence to
  near-quadratic.

### Numerical stability improvements

* Negative-binomial initialization in `fastglm_nb()` now uses a
  moment-based mu/theta seed instead of always running a full pilot
  Poisson fit, improving convergence on overdispersed data and near
  boundary cases.
* Increased the default `outer.maxit` for `fastglm_nb()` from 25 to
  50 to allow convergence in harder NB problems.
* Clamping guards in native C++ family kernels for inverse-link and
  sqrt-link families (Gamma, inverse.gaussian, Tweedie) to prevent
  overflow when `eta` is near zero.
* Floor on `mu.eta` for Tweedie inverse and sqrt links to avoid
  division by zero in IRLS weights.
* Sign correction in the Tweedie sqrt `mu.eta` kernel.

### Bug fixes

* Replaced `Rf_error()` with `Rcpp::stop()` in `bigmemory.cpp`
  (reported in GitHub issue).

### Documentation

* New vignette `firth-fastglm` covering Firth bias-reduced GLMs for
  all supported families.
* New vignettes `nb-convergence-fastglm` and `nb-stability-fastglm`
  demonstrating `fastglm_nb()` convergence and stability relative to
  `MASS::glm.nb()` on challenging datasets.
* Expanded benchmarks vignette with additional model classes.

### Internal

* C++ override annotations on all virtual method overrides to satisfy
  `-Winconsistent-missing-override` under clang.
* Shared Brent root-finder extracted to `inst/include/brent.h`,
  replacing three duplicated implementations.
* Expanded test suite (107 test blocks / 361 expectations, up from
  54 / 131 in 0.1.0), including new suites for numerical edge cases
  and Firth across all families.

## Test environments

* local macOS Sequoia (R 4.5.1)
* Rhub:
    - linux, ubuntu 24.04: R Under development (unstable) (2026-04-30 r89987)
    - macos-arm64: R Under development (unstable) (2026-04-30 r89988)
    - m1-san: R Under development (unstable) (2026-04-30 r89988)
    - windows: R Under development (unstable) (2026-04-30 r89987 ucrt)
* Rhub:
    - atlas (fedora linux): R Under development (unstable) (2026-04-30 r89987)

## R CMD check results

0 errors | 0 warnings | 0 notes

## Reverse dependency results

- passes all