## Update to 'fastglm' 0.1.4

This release fixes the problems reported for 0.1.3.

* The M1mac ASAN/UBSAN check reported "applying non-zero offset 8 to null
  pointer" in `positive_rows()` (`inst/include/ordinal_constrained.h`), reached
  when a constrained cumulative-link fit has no predictor columns. The function
  now returns before forming row blocks of a matrix with no columns. With the
  package compiled with `-fsanitize=undefined`, the full test suite now gives no
  sanitizer reports; the 0.1.3 code gives the two reported ones.
* The same check failed `test-sources.R` because 'arrow' there was built without
  Parquet support. The test is now skipped in that case.
* `Eigen::MappedSparseMatrix` is replaced by `Eigen::Map<Eigen::SparseMatrix>`,
  as requested by the 'RcppEigen' maintainers for the upcoming Eigen 5 release.

## Test environments

* local macOS (aarch64), R 4.5.1, including a UBSAN build of the package

## R CMD check results

## Reverse dependencies

The changes do not alter results, so reverse dependencies are affected as for
0.1.3.
