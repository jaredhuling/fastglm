# fastglm 0.1.0

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

* New `firth = TRUE` argument to `fastglm()` and `fastglm.fit()` for
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

* New methods `vcovHC.fastglm()` and `vcovCL.fastglm()` for
  heteroskedasticity-consistent (HC0–HC3) and cluster-robust covariance
  matrices. Results match `sandwich::vcovHC()` / `sandwich::vcovCL()` to
  floating-point precision and work for sparse, `big.matrix`, and
  in-memory fits.

* Sparse design matrices (`Matrix::dgCMatrix`) are now supported directly
  by `fastglm()` and `fastglm.fit()` for the LLT (`method = 2`) and LDLT
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
