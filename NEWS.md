# fastglm 0.1.3

## Behavior changes

* Aliased coefficients from rank-deficient designs are reported as `NA`, as in
  `glm()`, instead of 0.
* When the Cholesky factorization of a dense design is singular or
  ill-conditioned, `method = 2` and `method = 3` now refit with pivoted QR and
  warn, instead of returning arbitrary coefficients or invalid covariance.
  Sparse and `big.matrix` designs, which cannot use QR, report an error.
* Indefinite mixture information no longer produces standard errors from absolute
  values, and information matrices are evaluated at the reported estimates.
* Streaming fits now validate every chunk and propagate source errors.
* Printed output for models, summaries, diagnostics, predictions, bootstraps, and
  tests now shares one format. Numeric fields are unchanged, and test results and
  predictions support `as.data.frame()`.
* The plot method for ordinal assessments uses *ggplot2* when it is installed.

## Ordinal models

* Added `fastglm_ordinal()` for cumulative-link models with logit, probit, and
  complementary log-log links, by maximum likelihood or mean bias reduction, with
  formula and matrix interfaces, frequency counts, and offsets.
* Added threshold-specific effects through `nonparallel`, including partial
  proportional odds. Fits whose best solution lies on the ordering boundary stop
  with `convergence.reason = "ordering_boundary"` and an explanation.
* Added category and cumulative predictions with standard errors, average
  probabilities and contrasts, `fastglm_marginal_effects()`, simulation, residuals,
  and optional broom methods.
* Added `fastglm_ordinal_assess()` for prediction scores and calibration,
  `fastglm_po_test()` for common-slope tests, and boundary-row diagnostics through
  `fastglm_diagnostics(boundary = TRUE)`.
* Added robust and clustered covariance, refitting bootstraps for common-slope fits,
  profile-likelihood intervals, and score tests.
* Added `fastglm_ordinal_prepare()` and `fastglm_ordinal_batch()` for repeated
  responses, and `fastglm_ordinal_streaming()` for data read in blocks.

## GLMs and count models

* Added a formula interface, named and automatic solver choices, prediction with
  offsets, leverage, simulation, Wald intervals, numerical diagnostics, and optional
  broom methods.
* Added `fastglm_prepare()` and `fastglm_batch()` for repeated outcomes on a common
  design, including bias reduction, optional process workers, and
  `inference = "none"` to skip covariance.
* Added `fastglm_truncated()` for zero-truncated Poisson and negative-binomial
  regression, and `fastglm_twopart()` for nonnegative continuous outcomes.
* Count models now provide component and marginal predictions, probability masses,
  delta-method standard errors, residuals, simulation, and profile intervals for
  theta. Negative-binomial, hurdle, and two-part fits also accept sparse designs and
  repeatable sources.
* Added joint robust and clustered covariance for count and two-part models, average
  response contrasts, and refitting bootstraps.

## Inference

* Added `fastglm_wald_test()` for joint term tests and named contrasts in GLMs,
  ordinal, truncated, hurdle, zero-inflated, and two-part models.
* Added profile-likelihood intervals and `fastglm_profile_test()` for dense
  binomial-logit Firth fits.

## Large data

* Added `method = "qr"` to streaming GLMs, including mean bias reduction.
* Added resettable sources with fixed formula schemas, optional DBI and Parquet
  adapters, sinks for predictions and diagnostics, robust and clustered covariance,
  and average predictions for streamed fits.

## Numerical corrections

* Binomial probit and complementary log-log bias reduction now evaluates tail
  probabilities, scores, and information accurately for nearly deterministic
  outcomes, and limits oversized scoring updates.
* Gamma mean bias reduction now solves the adjusted dispersion equation and guards
  against overshooting initial steps. Dense Firth fits with a zero-rank working
  design no longer report convergence.
* Negative-binomial theta scores and likelihoods are stable near the Poisson limit.
  Boundary theta estimates have unavailable standard errors, and hurdle
  negative-binomial covariance uses joint observed information.
* Corrected rank-deficient QR updates, aliased coefficient reporting, mapped matrix
  views, offset null models, zero-weight degrees of freedom, grouped-binomial
  likelihoods, and simulation of numeric binomial proportions.
* `fastglm_ordinal()` now stops when named count-matrix columns do not match `levels`
  in order. These columns were previously assigned by position, which could silently
  reverse a fit.
* Printed tables without p-values no longer round small estimates, standard errors,
  or bootstrap limits to zero, and models without a size parameter no longer print a
  missing `Theta` row.
* Streamed fits no longer retain the last source block, so saving a fit does not
  write those rows.

## Performance

* Ordinal maximum likelihood begins with observed-information Newton steps, whose
  cost does not grow with the number of categories. Expected information still
  certifies convergence and gives the reported covariance.
  `control = list(observed = FALSE)` restores expected scoring throughout.
* Ordinal fits with many categories use a block factorization, and setup, robust
  covariance, common-slope tests, and constrained fits run in C++.
* Negative-binomial, hurdle, and truncated-count calculations for dense designs run
  in C++, and Firth fits reuse constant matrix calculations where possible.

# fastglm 0.1.2

## Bug fixes

* Compilation now works under libc++ with clang 23 and newer, which
  no longer supplies `<iterator>` and `<algorithm>` as transitive
  includes. The `bigmemory` header `BigMatrix.h`, pulled in by the
  big.matrix backend, uses `std::back_inserter` and `std::copy` without
  including those headers itself, so the two source files that include
  it (`fit_glm_dense.cpp` and `bigmemory.cpp`) now include `<iterator>`
  and `<algorithm>` explicitly beforehand. Reported by CRAN's
  clang-trunk checks.

# fastglm 0.1.1

## New features

* Firth bias-reduced GLMs (`firth = TRUE`) now work for all standard
  families (gaussian, binomial, Poisson, Gamma, inverse.gaussian) on
  dense, sparse, and streaming backends. In 0.1.0 Firth was limited to
  `binomial(link = "logit")` on dense designs only.

* SQUAREM acceleration (Varadhan and Roland, 2008) for the
  `fastglm_zi()` EM driver, converting linear EM convergence to
  near-quadratic.

## Numerical stability

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

## Bug fixes

* Replaced `Rf_error()` with `Rcpp::stop()` in `bigmemory.cpp`
  (reported in GitHub issue).

## Documentation

* New vignette `firth-fastglm` covering Firth bias-reduced GLMs for
  all supported families.

* New vignettes `nb-convergence-fastglm` and `nb-stability-fastglm`
  demonstrating `fastglm_nb()` convergence and stability relative to
  `MASS::glm.nb()` on challenging datasets.

* Expanded benchmarks vignette with additional model classes.

## Internal

* Shared Brent root-finder moved to `inst/include/brent.h`,
  replacing three duplicated implementations.

* Expanded test suite (107 test blocks / 361 expectations, up from
  54 / 131 in 0.1.0), including new suites for numerical edge cases
  and Firth across all families.

# fastglm 0.1.0

## Breaking changes

* `fastglm.fit()` and `fastglm.control()` have been renamed to
  `fastglm_fit()` and `fastglm_control()`. The old dot-form names
  triggered an apparent-S3-method `R CMD check` NOTE because `fastglm`
  is itself a generic. Existing code that passes `method = fastglm.fit`
  to `glm()` should be updated to `method = fastglm_fit`.

## New features

* New top-level function `fastglm_nb()` for negative-binomial
  regression with the dispersion `theta` estimated jointly with the
  regression coefficients. Plays the same role as `MASS::glm.nb()`,
  but the IRLS loop, the inner Brent root-find for `theta`, and the
  outer `(beta, theta)` alternation all run entirely in C++.

* New top-level function `fastglm_hurdle()` for two-part count models
  with a binary zero / non-zero component and a zero-truncated Poisson
  or NB count component. Same model as `pscl::hurdle()`; the joint fit
  runs in a single C++ driver that calls the existing IRLS solver
  twice and runs an inner Brent MLE for `theta` in the unknown-`theta`
  NB case.

* New top-level function `fastglm_zi()` for zero-inflated Poisson and
  NB regression. Same model as `pscl::zeroinfl()`; the entire EM
  driver runs in C++, including closed-form posterior responsibilities,
  weighted IRLS for both M-steps, the inner Brent MLE for `theta`, and
  the analytical observed-information `vcov`.

* New `firth = TRUE` argument to `fastglm()` and `fastglm_fit()` for
  Firth's bias-reducing penalty on the score. Currently supported for
  `family = binomial(link = "logit")` on dense designs; converges in
  finite steps under separation, where unpenalized IRLS would diverge.
  Coefficients agree with `logistf::logistf()` to 1e-7 on the standard
  Heinze-Schemper test cases.

* New built-in `negbin(theta, link)` family for negative-binomial
  regression with known dispersion, dispatched to a native C++ kernel
  on the `log`, `sqrt`, and `identity` links. Drop-in for
  `MASS::negative.binomial(theta, link)` without a hard dependency on
  *MASS*.

* Native fast paths for the `family$family == "Negative Binomial(K)"`
  form produced by `MASS::negative.binomial()`, the
  `quasibinomial()` / `quasipoisson()` families, and `statmod::tweedie()`.
  Detection is automatic.

## Documentation

* New vignette `count-firth-fastglm` covering `fastglm_nb()`,
  `fastglm_hurdle()`, `fastglm_zi()`, and the `firth = TRUE` flag, with
  small inline accuracy and timing comparisons against `MASS::glm.nb`,
  `pscl::hurdle`, `pscl::zeroinfl`, and `logistf::logistf`.

* New vignette `benchmarks-fastglm` providing a more comprehensive
  benchmark study at larger sample sizes, covering all six model
  classes. Pre-compiled at maintainer build time so it does not run
  during `R CMD check` / `R CMD build` and adds essentially zero time
  to the package check budget.

* `fastglm-overview` updated with a short section introducing the new
  count-data and Firth entry points.

## Other new features

* Real `vcov()` for `"fastglm"` objects: the unscaled covariance is now
  computed in C++ from the IRLS factorisation directly and exposed as
  `$cov.unscaled`. `summary.fastglm()`, `vcov.fastglm()`, and
  `predict.fastglm(se.fit = TRUE)` now all work without a refit hack;
  `vcov.fastglmFit()` no longer re-runs `glm.fit` to recover a `qr` slot.

* `S3` methods registered on `sandwich::vcovHC()` and `sandwich::vcovCL()`
  for heteroskedasticity-consistent (HC0–HC3) and cluster-robust covariance
  matrices. After `library(sandwich)`, `vcovHC(fit)` and `vcovCL(fit, ...)`
  match `sandwich::vcovHC.glm()` / `sandwich::vcovCL.glm()` to
  floating-point precision and work for sparse, `big.matrix`, and
  in-memory fits. (Earlier development versions shipped local
  `vcovHC()` / `vcovCL()` generics; those have been removed in favor of
  registering on the canonical *sandwich* generics.)

* Sparse design matrices (`Matrix::dgCMatrix`) are now supported directly
  by `fastglm()` and `fastglm_fit()` for the LLT (`method = 2`) and LDLT
  (`method = 3`) Cholesky paths. Other decompositions are rejected with
  an informative error rather than silently densified.

* `bigmemory::big.matrix` inputs now stream the design matrix in
  row-blocks of `FASTGLM_CHUNK_ROWS` rows (default `16384`,
  user-configurable via the environment variable). Filebacked
  `big.matrix` objects no longer have to be materialised in RAM.

* New top-level function `fastglm_streaming(chunk_callback, n_chunks,
  family, ...)` for fitting GLMs on data sources that do not fit in
  memory: Arrow datasets, Parquet files, DuckDB queries, CSV streamers,
  and any other chunk-yielding iterator. The IRLS loop, step-halving,
  and Cholesky solve all run in C++; the closure is invoked only to
  deliver one row-block at a time.

## Speed

* For standard families (gaussian / binomial / poisson / Gamma /
  inverse.gaussian on their common links) the per-iteration calls to
  `family$variance`, `mu.eta`, `linkinv`, and `dev.resids` are now
  evaluated in inline C++ rather than via R callbacks. Detection is
  automatic; non-standard families fall back to the previous R-callback
  path with no user-facing change.

* The IRLS solver pre-allocates its working buffers across iterations
  and uses `noalias()` writes throughout `solve_wls()`. Eigen's
  parallelism is no longer disabled.

* On large `n` the combined effect is roughly a 1.5×–2× speed-up over
  fastglm 0.0.4 on the same hardware, on top of the existing 3×–10×
  advantage over `stats::glm()`.

## Documentation

* New vignette `fastglm-overview` providing a single high-level entry
  point covering all of the package's functionality.

* New vignette `large-data-fastglm` walking through the three
  large-data paths (sparse, `big.matrix`, streaming callback) end to
  end, including the Arrow / Parquet recipe.

## Internal

* New `tests/testthat/` suite covering `vcov()`, native vs callback
  family dispatch, sparse fits, `big.matrix` streaming, callback-based
  streaming, and robust SE.

# fastglm 0.0.4 and earlier

* Added a `NEWS.md` file to track changes to the package.

* Added `fastglm.fit()` to be used as a fitting method for `glm()`.

* Documentation and vignette updates.
