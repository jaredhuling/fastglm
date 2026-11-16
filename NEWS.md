# fastglm 0.1.5

## Convergence and starting values

* A GLM fit is reported as converged only when the Fisher decrement at the
  estimate, the deviance decrease predicted by a full scoring step, is below
  the tolerance. Previously a step halved to nothing could satisfy the
  deviance-change rule anywhere, so a fit could report convergence at a
  poor estimate: for one Gamma model with a square-root link, fastglm 0.1.4
  reported convergence at deviance 5293 when the optimum is 1787.
  Unsuccessful step searches now end with `converged = FALSE` and
  `convergence.reason = "step_halving_failed"`.
* Step-halving after an increase in deviance now stops once the deviance no
  longer increases (Marschner 2011), instead of requiring a decrease larger
  than the tolerance, which cannot occur near the optimum.
* The default starting coefficients are the intercept-only fit instead of
  zeros, so an invalid first step is halved back toward a valid fit. Poisson
  and Gamma models with identity links, which previously stopped with
  "cannot find valid starting values", now fit without user-supplied starts.
* The convergence pass supplies the information matrix at the reported
  estimate, so covariance no longer requires a separate pass over the data.
* Dense and `big.matrix` fits return `convergence.reason`.

## Boundary optima

* The boundary step releases a held row when its Karush-Kuhn-Tucker
  multiplier, computed from the deviance gradient, has the wrong sign. Rows
  were held whenever the scoring step pushed them out of the valid region,
  which does not show that holding them is optimal: in one simulated
  square-root negative binomial data set a dense fit held two rows on the
  boundary and was reported converged 0.042 log-likelihood units below the
  maximum, which holds one.
* Streamed fits with an offset now have an intercept-only fallback start
  (previously only without an offset), so an invalid first update can be
  halved back into the valid region instead of stopping with an error.
* Sparse fits whose Cholesky factorization fails although the design has
  full rank (working weights near the boundary of the valid region) are refit
  with a dense copy and pivoted QR, with a warning, when the copy has at most
  1e8 entries; streamed fits with the default method refit with the streaming
  QR decomposition. Singular designs, and explicitly requested Cholesky
  methods for streamed fits, still stop with an error.
* Fisher scoring converges only linearly for non-canonical links, more slowly
  as fitted means approach a bound; such fits can need more than the default
  `maxit` and are then reported as not converged.

* Sparse (`dgCMatrix`) designs now use the convergence rule of dense fits
  (deviance change confirmed by the Fisher decrement, with step-halving until
  the deviance does not increase) and the boundary step. Earlier 0.1.5
  development builds applied the new rule to dense, big.matrix, and streamed
  fits only; sparse fits used the deviance-change rule of 0.1.4 and could
  report convergence away from the optimum (a Poisson model with a
  square-root link was reported as converged at deviance 654.32 against an
  optimum of 652.39). Sparse and dense fits now agree to rounding error on
  the boundary designs used in development.
* Partial proportional odds models (`fastglm_ordinal()` with `nonparallel`)
  now reach maxima on the ordering boundary, where cumulative probabilities
  touch at some rows. When the scoring step would bring a row's cumulative
  predictors to cross, the row-pair reached first is held at half its gap
  (or where it is, once the gap is below 1e-10) and the step is recomputed
  on the remaining directions; convergence there is certified by that
  constrained step and reported as `convergence.reason = "converged_boundary"`
  with a warning that Wald inference is not reliable. Previously such fits
  stopped as `"ordering_boundary"`, not converged, up to 8.1 log-likelihood
  units below the constrained maximum (68 units in one test design), and
  three boundary maxima in 480 simulated fits were reported with an ordinary
  convergence reason and ordinary standard errors, because the expected
  information diverges as a category probability tends to zero. In those
  simulations, 83 of 85 fits with a boundary maximum now converge there (none
  did before), with dense and streamed fits agreeing; proportional odds fits
  are unchanged.
* Bias-reduced fits (`firth = TRUE`) keep their coefficient-change rule but
  no longer count a step that was halved: at the boundary of the valid
  region such steps shrink to nothing, and the rule could report convergence
  wherever the iteration stalled (an identity-link Poisson fit was reported
  as converged at deviance 913.601, above the optimum 913.487). Bias-reduced
  fits whose estimate lies on that boundary (identity or square-root links,
  log-binomial) are now reported as not converged; the boundary step is not
  used for them.
* `fastglm_zi()` and `fastglm_nb()`: where the log likelihood is flat in
  `log(theta)` (large `theta`), the `theta` information need not be positive;
  convergence is then certified on the other parameters when the `theta`
  score on the log scale is below the tolerance. The zero-inflated Newton
  finish works on `log(theta)` and allows up to 50 steps. Sparse and streamed
  negative binomial fits report `convergence.reason = "iteration_limit"` when
  the outer loop does not converge (previously the last inner fit's reason).
* `fastglm_zi()`: the negative binomial `theta` score is computed with the
  cancellation-free series used by `fastglm_nb()`. The direct form lost its
  sign to rounding error near `theta = 1e8`, which pushed `theta` to that
  bound, and the EM rule then reported convergence at the zero-inflated
  Poisson fit (in 17 of 40 simulated weakly identified data sets, up to 5.1
  log-likelihood units below the maximum). EM is now followed by safeguarded
  Newton steps, and convergence is reported only when the Newton decrement is
  small (returned as `em.decrement`), so a stalled EM is not reported as
  converged and a slow one finishes. SQUAREM extrapolates `theta` on the log
  scale, the zero-model fits use the pivoted QR decomposition (the Cholesky
  factorization failed when the zero-inflation probability approached 0), and
  results carry `convergence.reason`. In those simulations false convergences
  fell from 21 to 6 of 400; the remaining 6 are local maxima other than the
  best one found (certified stationary, with positive definite information).
* The zero-truncated count solver (`fastglm_hurdle()`, `fastglm_truncated()`)
  also confirms convergence with the Fisher decrement and halves until the
  deviance does not increase.

* Fits whose optimum lies on the boundary of the family's valid region (a
  fitted mean of zero under an identity or square-root link, a fitted
  probability of one under a log link) now converge to that optimum. When the
  full update leaves the region, the row that leaves first is held halfway to
  its bound and the weighted least squares problem is solved again over the
  remaining directions; the step is used only if it is a descent direction
  for the deviance. Previously, halving the whole step stalled such fits above
  the optimum (for a Poisson model with a square-root link, 0.07% to 1% above
  the optimal deviance in every one of 100 simulated data sets), reported as
  not converged. Applies to dense and streamed fits, with identical results.
* These fits report `convergence.reason = "converged_boundary"` and a warning
  that Wald standard errors and tests are not reliable at such an estimate;
  `print()` notes convergence on the boundary.
* Streamed fits make at most `maxit` coefficient updates in total, as dense
  fits do; previously the first update from the starting means was not
  counted against `maxit`.

## Negative binomial convergence

* New vignette `stability-fastglm`: how fits are certified as converged,
  with a prespecified simulation study across generalized linear, negative
  binomial, hurdle, zero-inflated, and ordinal models against `glm.fit()`,
  glm2, `MASS::glm.nb()`, pscl, and ordinal, and the twelve negative binomial
  scenarios of the former stability benchmark. It replaces the
  `nb-stability-fastglm` vignette.
* `fastglm_nb()` with `link = "sqrt"` or `"identity"` now fits data whose
  optimum lies on the boundary `eta = 0`; previously it usually stopped with
  "no valid set of coefficients" (in 63 of 80 simulated data sets). An invalid
  least-squares start is halved toward the intercept-only fit, the boundary
  step also applies to the warm-started coefficient fit after each `theta`
  update, the outer decrement uses the coefficient update actually taken
  (which tends to zero at a boundary optimum, unlike the score), a Cholesky
  failure refits with the QR decomposition, and a boundary fit reports
  `convergence.reason = "converged_boundary"` with a warning. In 80 simulated
  data sets every fit converges, within 7e-7 of the maximum log likelihood
  found by an active-set constrained optimizer. With a sparse design, the
  identity link can still stop when the sparse Cholesky factorization is
  ill-conditioned (1 of 20 simulated data sets), because no QR fallback is
  available for sparse designs.

* The `nb-convergence-fastglm` vignette is merged into `nb-stability-fastglm`,
  which now compares log-likelihoods and theta estimates with
  `MASS::glm.nb()` across all twelve scenarios where both converge (both
  evaluated with `dnbinom()`, since `logLik()` for `glm.nb` loses precision
  at very large theta).

* `fastglm_nb()` decides outer-loop convergence by the joint Fisher
  decrement of the coefficients and `theta` (the deviance decrease predicted
  by one more scoring step, relative to the deviance), the rule the IRLS
  solver uses, instead of the raw change in the coefficients plus the
  relative change in `theta`. The old rule could not be met when the
  likelihood is flat in some direction of the coefficients (very small
  `theta`, outliers): the coefficients kept moving by about 1e-5 per outer
  iteration while the log-likelihood changed by 1e-9, and the fit stopped at
  `outer.maxit` as not converged. Two polishing rounds after the rule is met
  keep coefficients and `theta` within about 5e-8 (relative) of a tightly
  converged reference, as before. On the 600 simulated data sets of the
  nb-stability vignette, 599 fits now converge (563 before), with a median of
  2 outer iterations (9 before). `outer.tol` keeps its default and now
  applies to the decrement, which is returned as `outer.decrement`.
* `fastglm_nb()` checks the `theta = Inf` limit, the Poisson model
  (`poisson.limit = TRUE`, the default). The profile likelihood of `theta`
  need not be unimodal, and the fit could stop at a finite local maximum
  below the Poisson likelihood (17 log-likelihood units below it in one
  simulated data set) or at the cap `theta = 1e8`, short of the Poisson
  limit when fitted means are large. When the Poisson likelihood is higher,
  the fit is restarted from the Poisson coefficients if the data show
  overdispersion there, and otherwise the Poisson limit is returned with
  `theta = Inf`, `convergence.reason = "theta_infinite"`, the Poisson family,
  and a warning. In the nb-stability simulations this applies to 78 of 600
  data sets, all with no overdispersion at the Poisson fit.
* The zero-truncated negative binomial models (`fastglm_hurdle()`,
  `fastglm_truncated()`, dense and source-backed) use the same decrement
  rule, with a joint Newton step in the coefficients and `log(theta)` in
  every outer iteration: under truncation the two are coupled, and
  alternation alone could creep. In simulated hurdle models with
  `theta = 0.05` and `0.2`, 39 and 40 of 40 fits converge (10 and 11
  before), with log-likelihoods never lower than before.

## Streaming fits

* `fastglm_streaming()` uses the same convergence check, step-halving rule,
  and starting values as dense fits: convergence is confirmed by the Fisher
  decrement, an exhausted step search ends with
  `convergence.reason = "step_halving_failed"` instead of an error or a
  false convergence, and an invalid first step is halved toward the
  intercept-only fit. Gamma models with a square-root link and Poisson
  models with an identity link that previously stopped with "invalid linear
  predictor" or "invalid mean" now fit. Streamed and dense fits agree to
  rounding error.
* Streamed LLT and LDLT fits read the data fewer times: the totals come from
  the validation pass, and the deviance, Pearson statistic, and null
  deviance are accumulated in the last iteration's pass instead of a
  separate final pass (a typical logistic fit reads the data 6 times instead
  of 8). QR streaming fits now do the same.
* Streamed fits no longer copy each block of rows: the C++ reader uses the
  block returned by the source in place, and the weighted cross products are
  accumulated over row sub-blocks instead of from a weighted copy of the
  block (also in the validation pass, through a new internal
  `weighted_crossprod()`). For a logistic model of 7.3 million flights read
  from 12 Parquet files (61 coefficients), peak memory fell from 4.1 to
  3.0 GB and fitting time from 96 to 58 seconds.

* Streamed fits count `iter` as dense fits and `glm.fit()` do (coefficient
  updates, including the first update from the usual starting means), for
  both `covariance` settings. `passes` reports reads of the data.
* `?fastglm` has a new section, "Convergence and starting values", and the
  documentation of `method = "auto"` states that dense designs are rescaled
  column by column before the Cholesky factorization.

## Speed

* For LLT and LDLT fits the Fisher decrement is computed from X'WX, so the
  convergence check does not add a pass over the design.
* AIC for the standard binomial, Poisson, and Gamma families, the residuals,
  Pearson statistic, and null deviance for natively evaluated families, the
  binomial `initialize` step, and input validation are computed in C++, with
  the same values as before. A logistic fit with n = 300,000 and p = 10 takes
  about 15 percent less time.
* Family-name checks use `startsWith()` instead of regular expressions, and
  the stats reference families used to detect native families are cached.

## Interfaces

* `fastglm_nb()` accepts a model formula with `data`, `subset`, `weights`,
  `offset`, and `na.action`, and supports `predict(newdata = )`.
* `sandwich::vcovHC()` defaults to HC0 for count models, for which leverage
  corrections are not defined, and has methods for truncated, hurdle, and
  zero-inflated fits. Clustered covariance for these models is available
  through `fastglm_vcov(cluster = )`.

## Two-part model labels (breaking change)

* Coefficients of the positive (Gamma) part of `fastglm_twopart()` models are
  now named `positive_*` instead of `count_*`, in `coef()`, `vcov()`, and
  printed summaries. Code that indexes two-part coefficients by the old names
  must use the new ones. Hurdle and zero-inflated models keep `count_*`.
* For two-part models, `coef()`, `vcov()`, `predict()`, and `model.matrix()`
  accept `"positive"` and `"binary"` as well as `"count"` and `"zero"`, as
  `fastglm_wald_test(component = )` already did.

## Covariance convention

* New argument `covariance` in `fastglm()`, `fastglmPure()`,
  `fastglm_control()` (for `glm(method = fastglm_fit)`), and
  `fastglm_streaming()`. The default, `"estimate"`, evaluates the information
  matrix at the reported coefficients. `"previous"` uses the factorization from
  the final iteration, the convention of `glm.fit()`: it saves one pass over
  the data (roughly 10 percent of the fitting time for large data) and changes
  standard errors by a relative amount of order 1e-6 at the default
  tolerance.

# fastglm 0.1.4

* Sparse designs are now read with `Eigen::Map<Eigen::SparseMatrix>` instead of
  `Eigen::MappedSparseMatrix`, which Eigen 5 removes, so the package builds with
  both the current and the upcoming 'RcppEigen'.
* Constrained cumulative-link fits with no predictor columns no longer form a
  row block of an empty matrix, which offset a null pointer.
* The count model kernels no longer call `realloc()` with size 0 when a data
  block has no rows with positive weight.
* The Parquet source test is skipped when 'arrow' is built without Parquet
  support.

# fastglm 0.1.3

## Behavior changes

* Requires R 4.1.0 or later.
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
