## Update to 'fastglm' 0.1.3

This release adds cumulative-link ordinal models, including partial proportional
odds, mean bias reduction, and blockwise fitting; extends inference for count and
binomial models; and corrects several numerical and display issues. See NEWS.md.

## Test environments

* local macOS (aarch64, R 4.5.1)
* R-hub: not yet run for this version. Run the workflow in
  `.github/workflows/rhub.yaml` and list its environments here before submission.

## R CMD check results

Local `R CMD check --as-cran`: 0 errors | 0 warnings | 1 note.

The note, "unable to verify current time", comes from the local check machine.
Package `arrow` (Suggests) was not installed for this check, so its tests were
skipped.
