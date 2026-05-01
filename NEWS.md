# fastglm 0.0.5

## New features

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
