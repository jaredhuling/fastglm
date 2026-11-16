## Update to 'fastglm' 0.1.5

This update follows 0.1.4 (published 2026-11-11) closely because it corrects
fits that could be reported as converged when they were not at the maximum of
the likelihood, so that earlier versions could return incorrect estimates
without a warning. We think users should receive the correction promptly; the
change log in NEWS.md lists each case.

* Convergence is now reported only when the predicted improvement from one
  more step (the Fisher or Newton decrement) is negligible, rather than when
  the last step changed little. Previously a step shortened to nothing could
  be reported as converged away from the maximum, for dense, sparse, and
  streamed generalized linear models, the zero-truncated count models, and
  bias-reduced fits.
* Models whose maximum lies on the boundary of the valid region (for example
  an identity or square-root link with a fitted mean of zero, or a fitted
  probability of one under a log link) now reach that maximum and report
  `convergence.reason = "converged_boundary"` with a warning.
* Negative binomial, hurdle, zero-inflated, and partial proportional odds
  models use the same rule for their outer iterations; negative binomial fits
  can return `theta = Inf` (the Poisson limit) when the likelihood is highest
  there, and a rounding error that could push the zero-inflated negative
  binomial `theta` to its upper bound is fixed.
* A new vignette, `stability-fastglm`, documents these checks with a
  prespecified simulation study; it replaces two earlier vignettes.

Results change only where earlier versions stopped before the maximum, did
not converge, or failed with an error; fits that converged at the maximum are
unchanged within the convergence tolerance.

## Test environments

* local macOS (aarch64), R 4.5.1
* win-builder: R-devel (PENDING)
* R-hub (PENDING)

## R CMD check results

0 errors | 0 warnings | 2 notes (local, `R CMD check --as-cran`)

* "Days since last update: 5" (see above).
* The local HTML Tidy is too old to validate the HTML manual (local machine
  only).

## Reverse dependencies

PENDING. There are 12 reverse dependencies (Depends, Imports, LinkingTo):
crctStepdown, ddecompose, did, DRDID, FBMS, GWASinlps, HTRX, SEQTaRget,
sffdr, simstudy, swaglm, triplediff; and 6 that suggest fastglm: btergm,
familiar, mfp2, parglm, plsRglm, registr.
