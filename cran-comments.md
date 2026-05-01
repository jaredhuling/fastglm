
## CRAN submission for 'fastglm' 0.0.5

This release adds:

* Real `vcov()` / `predict(se.fit = TRUE)` for `"fastglm"` objects
  (no more refit hack on `vcov.fastglmFit()`).
* HC and cluster-robust covariance methods registered on
  `sandwich::vcovHC()` / `sandwich::vcovCL()`.
* Native (inline-C++) family/link dispatch for standard GLMs
  (gaussian / binomial / poisson / Gamma / inverse.gaussian on their
  common links); roughly 1.5×-2× faster than 0.0.4 on large `n`.
* Sparse design matrix support via `Matrix::dgCMatrix`.
* Filebacked `bigmemory::big.matrix` is now streamed in row-blocks
  rather than materialised in RAM.
* New `fastglm_streaming()` for fitting on Arrow / Parquet / DuckDB /
  CSV-stream sources via a user-supplied chunk callback.
* New testthat suite and two new vignettes
  (`fastglm-overview`, `large-data-fastglm`).

## Test environments

* local Mac OSX Sequoia (R 4.5.1)
* Rhub linux, macos-arm64, m1-san, windows
* Rhub atlas / c23 / clang-asan (R-devel)

## R CMD check results

── R CMD check results ────────────────────────────────────── fastglm 0.0.5 ────

0 errors | 0 warnings | 0 notes

R CMD check succeeded
