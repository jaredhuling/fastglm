## Update to 'fastglm' 0.1.4

This release fixes the problems reported for 0.1.3, which is why it follows
that release so closely.

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
* win-builder: R-devel
* GitHub Actions: Ubuntu, Windows, and macOS, R-release; Ubuntu, R-devel, R 4.4,
  R 4.3, R 4.2, and R 4.1

## R CMD check results

0 errors | 0 warnings locally, on win-builder, and on GitHub Actions. The only
NOTE on win-builder is the days since the last update. The local check has two
NOTEs about the local machine: the current time could not be verified, and the
local HTML Tidy is too old to validate the HTML manual.

## Reverse dependencies

The changes do not alter results, so reverse dependencies are affected as for
0.1.3.
