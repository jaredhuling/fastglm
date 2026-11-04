# fastglm 0.1.3

## Model interfaces and inference

* Added a sufficient preflight check for saturated, fully varying logit models
  with mean bias reduction. An absent intermediate category within a predictor
  pattern can force a boundary solution despite all categories being observed
  overall. Certified cases now explain the unavailable interior solution before
  iteration; other boundary failures retain their existing diagnostics.
* Partial ordinal fitting reuses the predictor-step bound already calculated
  for its first scoring direction. This removes a duplicate source scan while
  preserving direction scaling, probability ordering, and fallback checks.

* Extended `fastglm_po_test()` to partial ordinal models. Complete terms can
  be tested for common slopes while other varying effects remain unrestricted,
  with joint or term-specific score tests and an ML likelihood-ratio option.
  The meanBR null option retains its ordinary projected-score convention;
  full-model adjusted-score hypotheses remain available through
  `fastglm_score_test()`.
* Added HC0/HC1 and clustered covariance for streamed common-slope and partial
  ordinal fits. One source pass accumulates the same observed-information
  and ordinary-score contributions as dense fits. Cluster labels are matched
  across blocks without rounding numeric identifiers, and storage depends on
  the block, coefficient dimension, and number of clusters.

* Added selected threshold-specific effects through `nonparallel` in dense, prepared,
  and streamed ordinal fits. Partial proportional odds and nonparallel probit
  and complementary log-log models support maximum likelihood and full-model
  mean bias reduction, formula terms or matrix column selections, grouped
  frequencies, and offsets. Training and prediction checks enforce valid
  cumulative probabilities at the relevant predictor values.
* Added readable partial-model summaries, full expected-information covariance,
  Wald intervals and term tests, probability standard errors, averages, and
  prediction assessment. The ordinal vignette explains the model, prediction
  domain, and bias-reduced wine example. Retained partial fits also support
  robust and clustered covariance, automatic marginal effects, ML profiles,
  and ordinary or full-model adjusted-score tests. Linear profile contrasts
  can test equality of selected varying slopes while other effects remain
  unrestricted. Partial models still require every declared category to have
  positive frequency; refitting bootstrap support remains limited to common
  slopes.
* Corrected cumulative Pearson residuals for ordinal probit and complementary
  log-log fits to use the binomial variance of each cumulative indicator.

* Added ML profile-likelihood intervals and joint likelihood-ratio tests for
  retained dense ordinal fits, and `fastglm_score_test()` for ordinary or
  full-model mean-bias-adjusted ordinal scores under linear constraints.
  The procedures use first-order chi-square references, retain constrained-fit
  diagnostics, and require every category to be observed. Profile intervals
  have a compact shared display; failed limits remain missing.
* Added `fastglm_ordinal_streaming()` for repeatable sources, with logit,
  probit, and complementary log-log links, maximum likelihood and mean bias
  reduction, frequency counts, and offsets. Category and rank checks combine
  all blocks. The bias adjustment uses information from the complete source.
  Predictions, residuals, and weights support block destinations without
  collecting the complete result. Model-based covariance is retained;
  robust covariance rereads the source, while constrained coefficient
  inference requires retained data.
* Reduced binomial probit and complementary log-log post-fit overhead by
  combining accurate likelihood and residual summaries, reusing the solver's
  current deviance, and calculating prediction derivatives without allocating
  unused probability summaries. The estimating equations are unchanged.

* Binomial probit and complementary log-log bias reduction now evaluates tail
  probabilities, scores, and information without separately flooring means and
  derivatives. This corrects the estimating equations for nearly deterministic
  outcomes. QR and SVD adjustments avoid amplification of roundoff in rows with
  tiny working weights. Likelihoods, residuals, and diagnostics use the same
  calculations across dense, sparse, mapped, and streamed fits.
* Logistic ordinal fitting reuses threshold-gap calculations and algebraic
  probability identities to reduce exponential and logarithm evaluations without
  storing observation-level probability arrays. Binary probit and complementary
  log-log bias reduction uses bounded Fisher scoring, retaining checks of the
  adjusted score and remaining predictor change.

* Binomial probit and complementary log-log mean bias reduction now limits
  oversized scoring updates on the linear-predictor scale. Sparse, mapped, and
  streamed fits use current information for these adjustments, avoiding delayed
  adjustment oscillation. Diagnostics report reduced steps. Streaming null
  deviances handle all-zero/all-one binomial outcomes with offsets directly.
* Extended `fastglm_wald_test()` to truncated, hurdle, zero-inflated, and two-part
  fits. Component selection, complete formula terms, and named contrasts retain
  joint regression covariance, including supplied cross-component covariance.
  Estimated negative-binomial size boundaries return unavailable tests.

* Ordinal models now support probit and complementary log-log links with
  maximum likelihood and mean bias reduction. Predictions, marginal effects,
  expected and robust covariance, assessment, common-slope score tests, prepared
  designs, and bootstrap refits retain the selected link.
* Dense Firth GLMs with a numerically zero-rank working design no longer report
  successful convergence from a zero coefficient update.

* Added `fastglm_wald_test()` for joint formula-term tests and named linear
  coefficient contrasts in GLMs and ordinal models, with model-based or supplied
  covariance and the shared test display. Nonestimability and unavailable
  covariance remain explicit; the chi-square reference is a first-order approximation.
* Proportional-odds score tests now use the tridiagonal threshold information
  directly, reducing intermediate allocations. `fastglm_po_test(by.term=TRUE)`
  shares one null fit and one information calculation across joint and term-specific
  tests, with readable tables and numeric extraction.

* Unified printed model, summary, diagnostics, prediction, bootstrap, and hypothesis-test output with
  common headings, aligned tables, display precision, and fitting status.
  Test results retain numeric fields and provide `as.data.frame()` and optional
  `broom::tidy()` extraction, including failure explanations.
  Diagnostics use readable solver names and labeled numerical checks.
  Predictions with standard errors print aligned tables and support
  `as.data.frame()`, while retaining the original `fit` and `se.fit` components.
  Vignettes use these displays directly.
* Added `fastglm_po_test()` for direct proportional-odds testing and an explicit
  `null.estimation = "meanBR"` option (`po.estimation` within assessment).
  This evaluates the ordinary nuisance-projected score at a bias-reduced null;
  its chi-square reference has first-order justification. A sufficient
  separation check identifies two-valued predictor groups observed only in an
  endpoint category. ML separation is reported before optimization; empty
  categories and deficient rank remain unsupported.

* Added an optional analytic proportional-odds score test to
  `fastglm_ordinal_assess()`, using an ML null refit and expected Fisher
  information. Joint and selected-term tests support frequency counts and
  offsets. Empty categories, deficient designs, failed null fits, and unusable
  information return explicit diagnostics and unavailable p-values. The test
  uses a first-order chi-square reference without resampling.

* Added `fastglm_marginal_effects()` for average derivatives of ordinal category
  and cumulative probabilities with respect to a continuous predictor. Formula
  interactions and smooth transformations are differentiated together, with
  offsets held fixed and delta-method uncertainty for thresholds and slopes.
* Added ordinal prediction assessment with logarithmic and ranked probability
  scores, cumulative calibration plots, and descriptive threshold-specific
  slope comparisons. Frequency counts contribute individual-observation scores,
  and supplied evaluation outcomes are kept separate from fitting.
* Ordinal parametric bootstraps now reuse response-independent design setup
  within each process while retaining replicate-specific fitting diagnostics.

* Added `fastglm_ordinal_prepare()` and `fastglm_ordinal_batch()` for repeated
  ordinal responses on a common predictor design. Saved centering, scaling,
  and rank checks are reused when predictors and effective row frequencies
  agree; changed inputs rebuild the preparation. Formula metadata and ordinary
  R serialization are retained, with separate category mappings and convergence
  diagnostics for each fit.
* Added ordinal simulation, probability-scale and cumulative residuals, and
  optional broom tidy, glance, and augment methods. Simulation preserves
  effective category frequencies without expanding grouped observations.
* Added HC0/HC1 and cluster covariance for ordinal fits, using the observed
  ordinary-score Jacobian and explicit frequency semantics. For mean bias
  reduction this is a first-order sandwich convention, with regularity and
  positive-category requirements. Supplied covariance matrices can be used
  in summaries, tidy output, and average probability contrasts.
* Added ordinal pairs, cluster, and parametric refitting bootstraps. Replicate
  diagnostics retain failures and category changes. Uncertainty summaries
  remain unavailable when refits fail or an interior category is empty. For
  mean bias-reduced parametric bootstraps, empty endpoints are allowed when
  all interior categories remain represented in the original fit and every
  refit.

* Added cumulative-logit proportional-odds models through `fastglm_ordinal()`,
  with formula and matrix interfaces, maximum likelihood and mean bias
  reduction, frequency counts, offsets, and native fitting.
  Empty interior categories retain their labels through an explicit reduced
  model mapping; tied-threshold Wald uncertainty is unavailable.
  Category/cumulative predictions, model-based standard errors, and average
  probability contrasts use the common model interface.

* Added `method = "qr"` to streaming GLMs, including mean bias reduction.
  Incremental QR avoids weighted crossproducts in initialization and fitting,
  retaining a triangular factor for covariance, prediction uncertainty,
  leverage, robust covariance, and average-prediction inference. Rank is
  assessed across the complete source; globally redundant columns still
  produce an explicit error. The default Cholesky solver is unchanged.

* Added penalized profile-likelihood intervals through
  `confint(..., method = "profile")` and joint coefficient tests through
  `fastglm_profile_test()` for retained dense binomial-logit Firth fits.
  Constrained fits retain the full original Jeffreys penalty. The procedures
  support binomial counts and frequency weights, use an asymptotic chi-square
  reference, and report numerical failures through missing results and diagnostics.
* Added `firth` to prepared designs and batch fits. Dense Gaussian-identity,
  Gamma-log, and Poisson-square-root batches reuse common setup with the
  automatic or LLT solver; other solvers retain their requested decomposition.
  Modified predictors or weights invalidate the saved setup, and prepared
  objects remain serializable across R sessions and process workers.

* Extended estimated-theta negative-binomial, hurdle, and two-part fitting
  to sparse matrices and repeatable block readers, with consistent offsets,
  formula transformations, and joint theta uncertainty.
* Added joint HC0/HC1 and cluster covariance for count and two-part models,
  average response contrasts, and refitting bootstrap uncertainty. Bootstrap
  output records failed fits, preserves reproducible replicate seeds, and
  supports sparse designs and repeatable sources.

* Added a direct formula interface, named and automatic solver choices,
  prediction with offsets and term standard errors, leverage, simulation,
  Wald intervals, numerical diagnostics, and optional broom methods.
* Added reusable designs and batch fitting, including a shared Gaussian
  factorization, optional process workers, and controls for retained data
  and inference. `inference = "none"` skips covariance calculation.
* Added standalone zero-truncated Poisson/NB regression and two-part
  binomial/Gamma regression for nonnegative continuous outcomes.
* Count models now provide component and marginal predictions, probability
  masses, delta-method standard errors, summaries, residuals, and simulation.
  Theta profile intervals are available for ordinary, truncated, and hurdle NB.
* Added resettable sources with fixed formula schemas and optional DBI and
  Parquet adapters. Streaming fits support prediction and diagnostic sinks,
  robust/cluster covariance, and average predictions and contrasts.

## Numerical corrections

* Streaming QR Firth convergence now checks the actual remaining changes in
  fitted linear predictors, joint adjusted scores, and dispersion.
  The criterion avoids requiring tiny changes to large cancelling coefficients.
  Additional precision is used to verify highly ill-conditioned designs;
  fits that fail the verified criteria still report nonconvergence.

* Safeguarded Gamma-log mean bias reduction against initial coefficient steps
  that overshoot the adjusted-score solution. Firth diagnostics now distinguish
  ordinary and adjusted coefficient scores, report the adjusted dispersion
  score where applicable, and retain the solver's convergence reason.
  Source initialization uses consistent fitted means and derivatives for very
  small responses. Gamma-log null models with offsets use their explicit score
  solution, avoiding a separate unstable intercept-only iteration.

* Stabilized negative-binomial theta scores and likelihood calculations near
  the Poisson limit. Boundary theta estimates have explicitly unavailable
  theta standard errors; `SE.theta` elsewhere uses joint observed information.
  The default beta-only NB covariance retains the usual expected-information
  convention, while `vcov(..., full = TRUE)` includes theta jointly.
* Truncated NB means and likelihood now correspond to the returned coefficients
  and theta even when the outer iteration limit is reached.
* Preserved sequential readers during component fits and bootstrap resampling,
  retained external source offsets in new-data prediction, and corrected
  simulation of numeric binomial proportions with integer trial weights.

* Fixed rank-deficient QR updates and aliased coefficient reporting. Explicit
  Cholesky solves reject singular or ill-conditioned information matrices;
  automatic dense fits can fall back to pivoted QR.
  Calls that previously returned arbitrary coefficients or invalid covariance
  matrices can now fail explicitly. Callers with redundant predictors should
  use pivoted QR or `method = "auto"` and handle aliased coefficients as `NA`.
* Unified matrix/formula initialization and model accounting, including
  grouped-binomial trials, zero-weight degrees of freedom, offset null models,
  fixed dispersion, NB AIC, and working residuals.
* Native family dispatch preserves modified-family callback semantics and
  exact negative-binomial theta parameters.
* Fixed mapped matrix views and matrix products involving row/column offsets.
* Gamma mean bias reduction now solves the adjusted dispersion equation.
  Supported families and the distinction between adjusted scores and penalized
  likelihood are documented explicitly.
* Hurdle NB covariance now uses joint coefficient–theta observed information.
  Truncated-count iterations check likelihood improvement, and small-rate
  derivatives use the correct limits. Component offsets are handled separately.
* Information matrices are evaluated at the reported estimate. Indefinite
  mixture information no longer produces standard errors from absolute values.
* Streaming validates every chunk, uses family-specific initialization, and
  propagates source errors. Grouped-binomial likelihoods retain trial totals.

## Performance and verification

* Added a reproducible ordinal uncertainty development study comparing
  model-based, sandwich, and bootstrap intervals across small samples, rare
  categories, correlated predictors, and independent clusters. Results retain
  unavailable intervals and failures separately from conditional coverage.

* Added a reproducible ordinal development study spanning small samples,
  rare categories, separation, correlated predictors, grouped frequencies,
  and larger predictor/category dimensions. It distinguishes convergence,
  reduced-category fits, and regular interval coverage, with simulation error
  reported separately from computational performance.

* Replaced numerical differentiation in joint count-model information with
  analytic derivatives, retaining coefficient--theta dependence, mixture
  cross-component terms, and the existing covariance validity checks.

* Reused constant Firth information matrices in sparse, mapped, and callback
  fits. Gamma-log fitting also reuses the common adjustment after its initial
  calculation. Gamma callback fits combine repeated prior weights within each
  block while preserving sequential reads and bounded block storage.
* Included `NEWS.md` in source packages so the README link resolves in the
  packaged source; this repairs the incoming-check note from Windows.

* Accelerated Gamma mean bias reduction by combining identical prior weights
  in its dispersion equation and replacing fixed bisection with a safeguarded
  Brent solve. Constant-weight Firth fits reuse matrix decompositions and
  leverage calculations without changing the estimating equations.
* Final dense Cholesky inference omits an unnecessary weighted-response solve,
  and intercept detection stops scanning a column once it is nonconstant.
* Revised vignette explanations to define technical terms and their practical
  consequences, corrected outdated interface descriptions, and distinguished
  historical benchmark outputs from current measurements.

* Removed sparse intercept densification and duplicate covariance inversions.
* Reused constant-weight factors and accepted streaming weighted systems;
  omitted covariance calculations from mixture intermediate fits.
* Added regression tests against base R, brglm2, and pscl, an accuracy-checked
  timing script, and automatic Linux/macOS/Windows package checks.
* Expanded the vignettes and restored the static benchmark regeneration script.


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
