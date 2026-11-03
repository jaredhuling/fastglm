# Local verification

The records here accompany the source and numerical comparisons in
`../confirmation/`. They were generated with R 4.5.1 on macOS arm64.

* `install.log`: successful native compilation and installation into a temporary
  library. The source checksums match `candidate-source-md5.csv`.
* `full-tests.csv` and `full-tests.log`: the full package test suite. Three
  tests marked for execution outside CRAN are exercised separately in
  `parallel-tests.log`; the optional Arrow test is skipped because Arrow is not
  installed. Testthat was built under R 4.5.2, producing its startup warning.
* `new-links-tests.log`: focused new-link, ordinal inference, and joint-test
  checks, including analytic derivatives and tail calculations.
* `build.log` and `check.log`: source build with `--no-build-vignettes --no-manual`
  and `R CMD check --no-install --no-tests --no-manual --ignore-vignettes`, using
  the installed candidate. Installation, tests, and HTML rendering were run
  separately. The check finished with status OK; this is a local development
  check, not a multi-platform release check.
* `render.log`: successful rendering of `fastglm-overview.Rmd` and
  `ordinal-fastglm.Rmd` to HTML in the repository's `vignettes` directory.
  Quiet rendering produces an empty success log.
* `baseline-run.log` and `candidate-run.log`: completion of every timing case.
* `artifacts-md5.csv`: integrity checksums for the retained confirmation records,
  protocol, and generated assessment, relative to this study's directory.

Source reconstruction was checked by exporting baseline commit `c437a26`,
applying `candidate-source.patch`, and verifying all 54 source checksums. A
copy of the retained records in a fresh temporary directory regenerated the
four derived CSV files and `ASSESSMENT.md` byte-for-byte. The timing runner's
overwrite guard was also exercised and rejected the existing result directory.

The parallel tests were run with local socket connections enabled.
