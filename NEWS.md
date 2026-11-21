# fastglm 0.1.5

## Threads

* The compiled computations of every model class can run on several
  threads through OpenMP. `fastglm_threads()` reports or sets the number of
  threads for the session (also settable as `options(fastglm.threads = )`).
  `fastglm()`, `fastglmPure()`, `fastglm_control()`, `fastglm_streaming()`,
  `fastglm_nb()`, `fastglm_truncated()`, `fastglm_hurdle()`, `fastglm_zi()`,
  `fastglm_twopart()`, and `fastglm_batch()` take `threads`;
  `fastglm_multinom()`, `fastglm_ordinal()`, and
  `fastglm_ordinal_streaming()` take `control = list(threads = )`. Each fit
  records the number it used in `threads`, and the request covers every pass
  of the call, including pilot, component, covariance, and null-deviance
  fits.
* The default is half the physical cores (half the logical cores when the
  physical count is unavailable), at most 8 and at least 1. It is reduced to
  `OMP_THREAD_LIMIT` and `OMP_NUM_THREADS` when they are set, and to at most 2
  when `_R_CHECK_LIMIT_CORES_` is set to anything but `"false"`, as during
  `R CMD check`; that cap also applies to explicit requests. A request is never
  above the number of logical cores.
* Results are identical for every number of threads: coefficients, deviance
  or log likelihood, iterations, passes, and covariance. Rows are divided into
  blocks whose size is set by the dimensions of the problem and by block-size
  settings (the `block.size` argument, the blocks of a source, and the
  `FASTGLM_CHUNK_ROWS` environment variable), never by the number of threads,
  and the block sums are added in a fixed order. Changing a block-size setting
  can therefore change results at the level of rounding error. When
  the accumulated matrix has at most 1024 rows (8 copies of it take at most
  64 MiB), blocks are added into 8 fixed partial sums; above that, the matrix
  is formed in tiles of 64 columns.
* Because sums now run over row blocks in this fixed order, estimates differ
  from 0.1.4 at the level of rounding error. The largest differences measured
  were 2.6e-15 in GLM coefficients (dense, `big.matrix`, and streamed fits,
  with deviances within 1.6e-14 relative), 1.9e-14 in sparse GLM coefficients,
  2.2e-16 in ordinal coefficients, and 1e-9 in a hurdle negative binomial
  coefficient at n = 1e6. Iteration counts were unchanged in these
  comparisons. A line search near a step bound can occasionally take one or
  two evaluations more or fewer, and so one or two passes; on one ordinal
  logit fit at n = 1e6 the number of evaluations went from 8 to 9.
* Speed with 4 threads compared with 1, at n = 1e6 (logistic GLMs with
  Cholesky methods), measured on a shared machine: dense p = 20 2.1 times
  faster and p = 200 3.6 times; `big.matrix` 1.8 and 2.9 times; streamed
  p = 200 2.1 times; sparse (p = 200, 5 percent nonzero) 1.8 times; negative
  binomial 1.8 times, zero-inflated Poisson 2.5 times, hurdle negative
  binomial 2.9 times, zero-truncated Poisson 1.5 times; multinomial (K = 5)
  3.4 to 3.6 times and (K = 50, n = 2.5e5) 3.7 to 3.9 times in the compiled
  fit; ordinal logit 3.7 times and probit 2.0 times in the compiled fit.
* Streamed GLM fits with few columns gain less, about 1.3 times at p = 20
  with 4 threads; they stay threaded because results do not change and they
  were not slower than one thread in our measurements. Two-part fits gain about 1.4 times,
  through their GLM components.
* With one thread, GLM fits with at most 1024 columns ran within about 5
  percent of the speed of 0.1.4 in our measurements, count fits at most 4.7
  percent slower, and multinomial fits at most 3.0 percent slower. An ordinal
  logit fit at n = 1e6 took 15 percent longer on one thread because it made
  one extra line-search evaluation, a rounding-level outcome to which 0.1.4
  is equally subject; over 12 seeds at n = 2e5 the one-thread total was 2.5
  percent longer. Ordinal probit fits were 4.2 percent slower.
* For GLMs whose accumulated matrix has more than 1024 rows (more than 1024
  columns), the column-tile scheme made fits 9 to 13 percent slower than
  0.1.4 on one thread, and 2.9 to 3.3 times faster than one thread of the
  same build on 4 threads. One-thread timings above 1024 coefficients were
  measured for GLMs only.
* Sparse (`dgCMatrix`) GLM fits form X'WX from a row-compressed copy of the
  design built once per fit, instead of transposing the design at every
  iteration. A logistic fit with n = 1e6, p = 200, and 5 percent nonzeros
  is about 7 times faster on one thread (0.66 seconds instead of 4.8), and
  peak memory fell from 1.56 to 1.30 GB; designs with thousands of columns
  gain little (p = 2000: 2 percent faster). The row-compressed copy (12 bytes
  per nonzero and 8 per row) is held for the whole fit, and with at most 1024
  columns the reduction holds up to about 72 MiB of extra working memory.
* Left on one thread:
  * dense QR and SVD methods (`method` 0, 1, 4, and 5), whose
    decompositions are serial; such fits record `threads = 1`, and so does a
    Cholesky fit that falls back to QR;
  * the evaluation of probit links and other calculations that use R's
    distribution functions (GLMs, zero-inflated models, the negative binomial
    joint covariance, `big.matrix` count blocks, and ordinal fits, where this
    caps probit near 2 times); the cross products of such fits are still
    threaded;
  * families written in R, whose functions are evaluated on one thread (the
    cross products are still threaded);
  * the row loops of sparse, `big.matrix`, and source count fits, which are
    written in R and gain only through their inner GLM fits;
  * sparse Cholesky factorizations and covariance inverses, so sparse designs
    with thousands of columns gain little;
  * the information of sparse multinomial fits with (K - 1) p above 1024 and
    of nonparallel ordinal models with more than 1024 coefficients;
  * ordinal robust covariances, score tests, and profile scans;
  * reading a source block, which happens on one thread before its rows are
    processed in parallel.
* With the default `block.size`, large multinomial and ordinal problems use
  row blocks smaller than `block.size`, so their estimates may change in the
  last bits.
* Worker processes of `fastglm_bootstrap()` and `fastglm_batch(workers > 1)`
  use one thread each, so the number of processes times threads never exceeds
  the request. A `threads` entry in the bootstrap `control` (or, for ordinal
  fits, in `control$control`) sets the refits' threads when `workers = 1`; with
  `workers > 1` a value other than 1 is ignored with a warning. Refits never
  inherit the original fit's thread count.
* An allocation failure inside a parallel computation stops the fit with
  "not enough memory: an allocation failed inside a parallel computation",
  the same message for every number of threads.
* On macOS the `configure` script probes for the OpenMP runtime that ships
  with R and builds without threads when it is not found. On Linux it uses
  R's OpenMP flags without a probe, so a compiler without OpenMP support fails
  at link time. Setting `FASTGLM_DISABLE_OPENMP=1` at installation forces a
  build without threads (not on Windows). In such a build
  `fastglm_threads()` reports 1 and a request for more warns once per
  session.

## Progress and interrupts

* `trace = TRUE` (or `control = list(trace = TRUE)` for `fastglm_multinom()`
  and ordinal fits, or `options(fastglm.trace = TRUE)`) prints one line per
  iteration: the iteration, the passes over the data so far, the deviance or
  log likelihood, the step halvings, and the elapsed seconds. It is on by
  default only for streamed fits in an interactive session that is not
  knitting a document, and off otherwise. `trace` was added to `fastglm()`,
  `fastglmPure()`, `fastglm_control()` (so
  `glm(method = fastglm_fit, control = list(trace = TRUE))` works),
  `fastglm_streaming()`, `fastglm_nb()`, `fastglm_truncated()`,
  `fastglm_hurdle()`, `fastglm_zi()`, and `fastglm_twopart()`. Tracing does
  not change any result.
* What `passes` counts depends on the model: reads of the source for streamed
  fits, evaluations of the deviance or log likelihood for fits in memory, and
  inner iterations plus `theta` updates for the outer loops of negative
  binomial and truncated negative binomial fits.
* Tracing a negative binomial fit computes the log likelihood at each outer
  iteration, so a traced source-backed fit reads the source once more per
  outer iteration and reports more `passes` than an untraced one.
* The `trace` controls of `fastglm_ordinal()` and
  `fastglm_ordinal_streaming()`, and the trace of multinomial fits, now print
  the common one-line format; the former score, step-bound, boundary-step, and
  solver text is gone. The ordinal `control$trace` default is now `NULL` (the
  option), and fits store `control$trace = FALSE`, so refits from
  `object$control` are silent.
* Bootstrap and batch refits are not traced (except that
  `fastglm_ordinal_batch()` honors an explicit `control$trace = TRUE`), and
  inference functions that
  refit internally (profile, score, and likelihood-ratio tests, profile
  intervals, `anova()`, `drop1()`, `add1()`, and `profile()`) do not trace
  their refits.
* Fits check for a user interrupt every 8 row blocks within a pass, instead of
  once per pass, and at each outer iteration of negative binomial,
  zero-inflated, and truncated negative binomial fits. The thread setting and
  the trace option are restored after an interrupt.

## Convergence

* A fit is now reported as converged only when one more step would barely
  improve it. Previously a fit whose steps had shrunk to nothing could be
  reported as converged well short of the maximum. Such fits now return
  `converged = FALSE`.
* Every model class returns `convergence.reason`, which says why fitting
  stopped (for example `"converged"`, `"converged_boundary"`,
  `"iteration_limit"`, or `"step_halving_failed"`).
* Without user-supplied starting values, an invalid first update is halved
  toward the intercept-only fit, so identity-link Poisson and Gamma models no
  longer need starting values.
* This applies to dense, sparse, `big.matrix`, and streamed fits, and to the
  negative binomial, hurdle, truncated, zero-inflated, and ordinal models.
* Sparse and streamed fits fall back to a QR decomposition when the Cholesky
  factorization fails on a full-rank design.

## Estimates on the boundary

* Models whose maximum lies on the edge of the valid parameter space now reach
  it: for example a fitted mean of zero under an identity or square-root link,
  a fitted probability of one under a log-binomial model, or touching
  cumulative probabilities in a partial proportional odds model. Previously
  these fits stopped short and were reported as not converged. This includes
  partial proportional odds maxima where many observations touch at once,
  for example when a category is empty for one value of a covariate.
* Such fits report `convergence.reason = "converged_boundary"` with a warning,
  because Wald standard errors and tests are not reliable there.
* Bias-reduced fits (`firth = TRUE`) do not handle such boundary estimates and
  report them as not converged.

## Negative binomial and zero-inflated models

* `fastglm_nb()` converges in cases where it previously stopped at the
  iteration limit (very small `theta`, outliers), and the outer convergence
  measure is returned as `outer.decrement`. On data where 0.1.4 already
  converged, these checks make fits about 20 percent slower.
* `fastglm_nb()` checks the Poisson limit (`poisson.limit = TRUE`). When the
  likelihood is highest there, it returns `theta = Inf` with the Poisson fit,
  `convergence.reason = "theta_infinite"`, and a warning.
* `fastglm_nb()` with square-root and identity links now fits data whose
  maximum is on the boundary; previously it usually stopped with an error.
* `fastglm_zi()` with `dist = "negbin"` no longer drifts to `theta = 1e8`
  because of a rounding error, and confirms convergence with Newton steps
  (`em.decrement`). It still finds a local maximum; the likelihood can have
  others.
* Hurdle and truncated negative binomial fits converge more reliably when
  `theta` is small.

## Streaming

* `fastglm_streaming()` uses the same convergence rules and starting values as
  dense fits, and agrees with them to rounding error.
* Streamed fits read the data fewer times and no longer copy each block. For
  a logistic model of 7.3 million rows read from Parquet files, peak memory
  fell from 4.1 to 3.0 GB and fitting time from 96 to 58 seconds.
* `iter` counts coefficient updates as dense fits do, and `passes` counts
  reads of the data.
* For binomial, Poisson, Gaussian, Gamma, and inverse Gaussian families,
  streamed fits compute the log-likelihood and AIC from the final deviance
  and so read the data one time fewer.
* `fastglm_source()`, `fastglm_source_dbi()`, and `fastglm_source_arrow()`
  have an opt-in block cache (`cache = TRUE`): each block is written to a
  local file the first time it is read, and later passes read that file
  instead of repeating the query. Memory use is unchanged; for a database
  source it made a logistic fit about three times faster.
* `fastglm_streaming()` accepts a design matrix (dense, sparse, or a
  file-backed `big.matrix`) with `y`, `weights`, and `offset`, read in blocks
  of `block.size` rows. For a `big.matrix` this bounds memory by the block
  size; `fastglm()` on a `big.matrix` remains faster but holds vectors of
  length n during the fit.

## Multinomial models

* New vignette, "Multinomial logit models with fastglm"
  (`vignette("multinom-fastglm")`), walks through fitting, printed output,
  prediction, robust and bootstrap inference, probability summaries,
  grouped data, separation, and large data, and reports timing comparisons
  with `nnet::multinom()` and `VGAM::vglm()`.
* New `fastglm_multinom()` fits multinomial logit models. It accepts a formula
  with `data`, a numeric or sparse (`dgCMatrix`) design matrix, a
  `big.matrix`, or a `fastglm_source()` read block by block. The response is a
  factor or a matrix of category counts, and `weights`, `offset` (one column
  per non-reference category is allowed), the reference category, and
  bias-reduced estimation (`firth = TRUE`) are supported. Fits have
  `summary()`, `coef()`, `vcov()`, `confint()`, `predict()` (with
  `se.fit = TRUE`), `residuals()`, `fastglm_wald_test()`,
  `fastglm_diagnostics()`, `fastglm_marginal_effects()`, `fastglm_average()`,
  `fastglm_bootstrap()`, `fastglm_vcov()` (robust and
  cluster-robust covariances), and `broom::tidy()` and `broom::glance()`.
  Some of these are limited for fits over a `fastglm_source()`, which keeps
  the source rather than its rows: `residuals()` and evaluation without
  `newdata` are not available, marginal
  effects require `newdata`. `fastglm_vcov()` offers
  the HC0 and HC1 types and cluster-robust versions for these fits.
* `fastglm_bootstrap()` refits `big.matrix` and source multinomial fits
  without copying the data: every row is kept and its weight is multiplied by
  the number of times it was drawn, which gives the same estimates as
  refitting the drawn rows. A source refit reads the blocks in order on each
  pass, and cluster identifiers for a source are a function of the block
  index.
* `confint()` of a multinomial fit stops when a `method` other than `"wald"`
  is requested, instead of returning Wald intervals.
* A source with a character or logical response no longer needs `levels`:
  one extra pass collects the distinct values, sorted as `factor()` sorts
  them.
* `fastglm_multinom()` has a new `control = list(solver = )` option. With
  `"cg"`, each Fisher scoring step is solved by preconditioned conjugate
  gradients on information-vector products, so the full information is not
  formed during the conjugate gradient iterations, which keeps fits fast when
  the number of categories is large. The information is still formed to
  confirm convergence, in the exact steps that follow, and for the returned
  covariance, so peak memory is that of the exact solver. The default
  `"auto"` uses it when the design has at least 8 columns and the number of
  design columns times `K - 1` is large, and the exact solver otherwise and
  for Firth fits (`"cg"` with `firth = TRUE` is an error). Convergence is
  declared by the same rule, and the fit records the `solver` used, the
  number of `products`, and the `accumulation` of the information.
* The fit uses a general vector GLM engine: weighted cross products are
  accumulated for each pair of linear predictors, one block of rows at a time,
  and safeguarded Fisher scoring with the Fisher-decrement stopping rule is
  run on the stacked coefficients.
* Without `firth = TRUE`, a multinomial fit with separation stops by the usual
  rule and warns that fitted probabilities are numerically 0 or 1, using the
  same threshold as `glm.fit()`; the warning suggests `firth = TRUE`.
* `fastglm_source()` accepts several offset names for a one-part formula, one
  per linear predictor, when the source is used by `fastglm_multinom()`. Other
  models still accept one offset column (two for two-part formulas). The error
  for any other number of offset columns now appears at the start of the fit,
  with the same message, rather than when the source is created.
* For a count-matrix (grouped) response, the deviance and null deviance of a
  multinomial fit subtract the saturated log likelihood, as for a grouped
  binomial `glm()`; the log likelihood, AIC, and BIC are unchanged. In formula
  fits, unused levels of factor predictors are dropped, as in `glm()`; only
  factors with unused levels are changed, so contrasts set on the others are
  kept, and a factor that loses its contrasts by dropping levels gives a
  warning, as `model.frame()` does.

## Model comparison

* `anova()` compares nested models of every class by likelihood ratio
  tests: GLMs (dense, sparse, `big.matrix`, streamed, and formula fits),
  negative binomial, truncated, zero-inflated, hurdle, two-part, ordinal
  (including partial proportional odds), and multinomial fits. The degrees
  of freedom are the differences in the `df` attribute of `logLik()`; GLMs
  with a dispersion estimated by moments get the F test of
  `stats::anova.glm()`. Calls with only formula `fastglm()` fits still go to
  `stats::anova.glm()` unchanged.
* Before giving a p-value, `anova()` checks that the models use the same
  data (number of observations, sum of prior weights, response, and
  offsets, without regard to row order), the same family and link, and that
  they are nested: for fits held in memory with rows in the same order, by
  the column space of the designs, so `y ~ poly(x, 2)` is nested in
  `y ~ poly(x, 3)`; otherwise, and for streamed, `big.matrix`, and sparse
  fits, by coefficient names, with a warning. A failed check is an error.
  Streamed fits are compared through a fingerprint of the response recorded
  at fit time (number of observations, sum of weights, and the weighted sums
  of the response and of its square), so no comparison reads the source.
* A Poisson model can be compared with a negative binomial model whose mean
  model contains it, and likewise the Poisson and negative binomial versions
  of truncated, hurdle, and zero-inflated models. The p-value uses the
  boundary-corrected reference, the 50:50 mixture of chi-square(r) and
  chi-square(r + 1) when the larger model adds r mean parameters (half the
  chi-square(1) p-value for the same mean model), and the table says so. A
  zero-inflated model is not compared with a model without zero inflation;
  the error points to `AIC()`.
* Bias-reduced (Firth) binomial-logit fits held in memory are compared by
  the penalized likelihood ratio test of `fastglm_profile_test()` on the
  larger model, as `logistf`'s nested `anova()`, not by the difference of
  two penalized log likelihoods. Other bias-reduced fits have no likelihood
  ratio comparison, and a Firth fit is never compared with a maximum
  likelihood fit.
* `drop1()` for every class with a formula, and for design-matrix fits
  through groups of columns. GLM and negative binomial fits follow
  `stats::drop1.glm()` (theta held at the full model's estimate, as for
  `MASS::glm.nb()`); other classes refit without the term, as `drop1()` does
  for `MASS::polr()` and `nnet::multinom()`. Models with two linear
  predictors get one row per term per part (`count: a`, `zero: a`), and a
  partial proportional odds model lists a nonparallel term as `a (all
  effects)` and `a (nonparallel)`. The arguments follow `stats::drop1()`:
  the default `test` is `"none"`, `"LRT"` (or `"Chisq"`) gives likelihood
  ratio tests, `"Rao"` is available where `fastglm_score_test()` supports
  the fit, and `k` and `scale` work as in `stats::drop1()`.
* `anova()` of one model gives the sequential table for models with a
  formula and one linear predictor.
* Streamed and `big.matrix` fits: `drop1()` and the sequential `anova()`
  print one line before starting that states what will be read: the number
  of refits, each a streamed fit that makes several passes over the data
  with the fit's `threads`, and, for a cached source, the one pass that
  builds the cache of the call; `trace = TRUE` traces the refits.
  `fastglm_source()` has a new argument `reusable`; a source declared with
  `reusable = FALSE` gives an error before any refit.
* Refits by formula of a source made with `cache` (zero-inflated, hurdle,
  two-part, truncated, ordinal, and multinomial fits) rebuild each block from
  the raw data, which the source's cache of built blocks does not hold. For
  such a source, `drop1()` and `anova()` read the source once at the start
  of the call into a block cache of the raw data in `tempdir()`, run every
  refit from that cache, and remove it when the call ends (also on an error
  or an interrupt), so the source's callback is read once per call rather
  than on every pass of every refit. Sources without `cache` are read by
  each refit as before.
* `drop1()` and the sequential `anova()` keep only the numbers each row
  needs (deviance, rank, log likelihood, AIC, test statistics) and discard
  each refit before the next one, as `stats::drop1.glm()` does. For a
  negative binomial fit with 2e5 rows and 20 terms, the peak memory of
  `drop1()` falls from about 1.05 GB to 0.31 GB.
* `drop1()`, the sequential `anova()`, and `step()` of a negative binomial
  fit whose theta estimate is infinite (the Poisson limit, where
  `fastglm_nb()` returns the Poisson fit) refit Poisson models, instead of
  stopping with "'theta' must be a positive finite scalar".
* In `drop1()` of `fastglm_twopart()` fits, the rows of the first part are
  named `positive: a` (they were `count: a`), matching the coefficient names
  and the emmeans mode; a list `scope` takes `positive` (or `count`) and
  `zero`.
* `extractAIC()` methods for every class, so `stats::step()` and
  `MASS::stepAIC()` run on fits held in memory, including multinomial fits.
  The fit `step()` returns can be evaluated again from its call and
  updated: `fastglm_nb()` and `fastglm_truncated()` take the model formula
  as `formula` as well as `x` (as `step()` writes it into the call), and
  `fastglm_ordinal()` uses a `formula` given next to `x`. For streamed fits,
  a `formula` given next to the source (a new argument of
  `fastglm_streaming()`, `fastglm_ordinal_streaming()`, `fastglm_nb()`, and
  `fastglm_truncated()`, and accepted by `fastglm_multinom()`) rebuilds the
  source with that formula, `update()` of a streamed ordinal fit with a
  formula source now refits on the rebuilt source (it was an error), and
  streamed truncated fits have a `terms()` method, so `step()` runs on
  streamed GLM, negative binomial, truncated, ordinal, and multinomial fits.
  `step()` and `MASS::stepAIC()` do not select terms of zero-inflated,
  hurdle, and two-part fits: they return the fit unchanged, now with a
  warning that points to `drop1()` for tests of the terms of each part.
  These fitting functions accept the list of count and zero formulas that
  `step()` leaves in the call. `extractAIC()` counts
  the rank, where `extractAIC()` of `glm()` and `MASS::glm.nb()` fits with
  zero prior weights also counts the zero-weight rows; the difference is the
  same for every model, so `step()` selects the same models.
* `formula()` of negative binomial, truncated, zero-inflated, hurdle,
  two-part, ordinal, multinomial, and streamed fits returns the model formula
  (the two-part formula `y ~ x | z` for models with two linear predictors),
  not the stored terms object with its attributes.
* `nobs()` of multinomial fits, and the number of observations in their
  `logLik()`, `BIC()`, `glance()`, and `summary()`, is now the total
  frequency (the sum of the prior weights, times the row totals for a
  count-matrix response), as for ordinal fits and `MASS::polr()` with
  frequency weights, so `AIC()` of ordinal and multinomial fits of the same
  weighted data no longer warns. It was the number of rows with positive
  weight, which the element `nobs` and `df.residual` still use. Unweighted
  individual data are unchanged; BIC of weighted or grouped multinomial fits
  changes.
* `fastglm_multinom()` formula fits keep their model frame (element
  `model`, used by `drop1()`, `anova()`, and emmeans); the new argument
  `model = FALSE` leaves it out.
* `update()` works for `fastglm_nb()`, `fastglm_truncated()`, and streamed
  fits (it failed with "invalid formula" for streamed fits in 0.1.4), and
  follows `stats::update()`. A new `update()` method for sources rebuilds a
  formula source with a new formula.
* `logLik()`, `AIC()`, and `BIC()` of `fastglm_twopart()` fits now use the
  Gamma shape at its maximum likelihood estimate given the fitted means, as
  `MASS::gamma.shape()` gives it, instead of the plug-in shape (the total
  weight over the deviance) of 0.1.4. Two-part AIC values therefore change
  (they are lower), and differences of log likelihoods between nested
  two-part fits are likelihood ratio statistics. AIC values built from
  `glm()` Gamma components, which use the plug-in dispersion, no longer
  agree with them.

## Other packages: emmeans, broom, and parsnip

* emmeans support (`?fastglm-emmeans`) for negative binomial, zero-inflated,
  hurdle, two-part, truncated, cumulative-link (including threshold-specific
  slopes), and multinomial fits, with the modes emmeans provides for
  `MASS::glm.nb()`, `pscl`, `MASS::polr()`, and `nnet::multinom()`.
  Ordinal standard errors use fastglm's expected-information covariance, so
  they differ from those of `MASS::polr()` and `ordinal::clm()` (observed
  information) by up to a few percent; another covariance can be given
  through the `vcov.` argument of emmeans. The `"prob0"` mode of
  zero-inflated fits and the `"zero"` mode of hurdle fits use the complete
  delta-method derivative, which the emmeans support for `pscl` fits
  (emmeans 1.11 and 2.0) does not, so their standard errors differ; the
  hurdle `"response"` mode uses the derivative of the fitted binary link
  for links other than the logit. Formula `fastglm()` fits use the `glm`
  support of emmeans.
* `broom::augment()` methods (`?fastglm-augment`) for every in-memory class,
  returning a tibble. GLM and negative binomial fits get the columns of
  `broom::augment.glm()`, including `.hat`, `.sigma`, `.cooksd`, and
  `.std.resid` (formula fits are served by fastglm, since
  `broom::augment.glm()` fails for them); multinomial fits get one
  `.prob_<level>` column per category and `.class`; ordinal fits keep their
  matrix column `.fitted` and also gain `.prob_<level>` columns. Streamed
  and `big.matrix` fits accept `newdata`.
* parsnip engines `"fastglm"` (`?fastglm-parsnip`) for `logistic_reg()`,
  `linear_reg()`, `poisson_reg()`, `multinom_reg()`, and `ordinal_reg()`,
  registered when parsnip (or poissonreg, or ordered) is loaded, with engine
  arguments `method`, `firth`, `threads`, and `link`. An engine fit uses one
  thread unless `threads` is given, so that resampling in several worker
  processes does not multiply processes by threads. `predict(type =
  "conf_int")` is available for linear and logistic models.
* parsnip engine fits store a call whose `data` argument refers to the
  training data through a private environment (`data = <environment>$data`),
  so `update()`, `step()`, and `anova()` of an updated engine fit work (they
  failed with "'data' must be a data.frame, environment, or list"). The
  model formula keeps its own environment, so `predict()` with new data that
  lack a predictor still stops with "object 'x' not found". A family
  of the stats package is stored in the call as a short call such as
  `stats::binomial(link = "logit")`, so a printed workflow no longer shows
  the family object. A `.` in the model formula is expanded against the
  training data.
* `augment()` of ordinal fits returns a tibble, as the other methods do. All
  `augment()` methods refuse a streamed fit without `newdata` with the same
  message.
* The emmeans error for a streamed, `big.matrix`, or matrix-interface fit
  names the alternatives that work for it: for a streamed fit,
  `fastglm_average()` and `fastglm_marginal_effects()` with `newdata`.
  `fastglm_average()` of a `big.matrix` GLM fit without `newdata` now reads
  the design in blocks of rows (it stopped with "non-numeric argument to
  binary operator").
* `residuals(type = "pearson")` for weighted `fastglm_hurdle()`,
  `fastglm_zi()`, `fastglm_twopart()`, and `fastglm_truncated()` fits now
  includes the square root of the prior weight, as `glm()` and `pscl` do;
  unweighted fits are unchanged.
* New vignette `ecosystem-fastglm`, "Using fastglm with emmeans, broom, and
  tidymodels". The overview vignette has a section on comparing models, and
  the large-data vignette shows comparisons of streamed fits.

## Other changes

* `fastglm_zi()` and `fastglm_hurdle()` with `link = "log"` now start the
  zero model from valid values; in 0.1.4 they stopped with "cannot find valid
  starting values". This applies to the formula, matrix, sparse,
  `big.matrix`, and source interfaces. The start needs an intercept in the
  zero-model design; without one the fit still stops with that message.
  `fastglm_twopart()` with `link = "log"` also passes a valid start to its
  binary part.
* With the log zero link and a maximum on the boundary (a zero probability of
  one), step halving keeps the fitted probabilities below one, and the fit
  reports non-convergence and an unavailable covariance instead of stopping.
* The starting linear predictor of the zero model in `fastglm_zi()` and
  `fastglm_hurdle()` is now computed on the scale of the chosen link (it was
  on the logit scale for every link). Logit fits are unchanged; probit and
  cloglog estimates can differ from 0.1.4 within the convergence tolerance,
  up to about 1e-5 at the default `tol`.
* `fastglm()` with a varying offset no longer fails when computing the null
  deviance if the default start of `glm.fit()` is invalid for the link (for
  example `binomial("log")`, which stopped with "no valid set of coefficients
  has been found"). The intercept-only model is then refitted from a valid
  start; fits that worked before are unchanged.
* Sparse zero-truncated fits whose design has an unnamed column (for example
  from `cbind(1, x)`) no longer stop with "subscript out of bounds".
* `drop1()`, `add1()`, and `profile()` now work on fits from
  `glm(method = fastglm_fit)`; they failed with an unused-argument error from
  `glm.control()`. They refit by maximum likelihood with `glm.fit()`, so they
  stop with an error for bias-reduced (`firth = TRUE`) fits. `drop1()`,
  `add1()`, and `anova()` match `stats::glm()` to about 1e-8. `profile()`
  matches to about 1e-8 with `covariance = "previous"` and to about 1e-5 with
  the default `covariance = "estimate"`, because its grid of steps is set by
  the standard errors, which differ at the 1e-6 level.
* Multinomial fits compute fitted probabilities faster (row maxima without
  `apply()`).
* Printing a `fastglm_bootstrap()` result for a GLM, count, or multinomial
  fit no longer reports "Inference: Unavailable" when every refit
  succeeded.
* `fastglm_ordinal(formula = y ~ x, ...)` now works; previously only a
  formula given as the first, unnamed argument was dispatched.
* A `fastglm()` fit to a `big.matrix` that was saved and reloaded (the
  `big.matrix` does not survive `saveRDS()`) now stops with an error in
  `model.matrix()` and `fastglm_vcov()` instead of crashing R.

* Printed calls show the fitting function without the `fastglm::` prefix,
  as `glm()` does; the stored call keeps it, so `update()` works when the
  package is not attached. GLM warnings start with `fastglm:` instead of the
  internal name `fit_glm:`.

* New argument `covariance` in `fastglm()`, `fastglmPure()`,
  `fastglm_control()`, and `fastglm_streaming()`. The default, `"estimate"`,
  computes the covariance at the reported coefficients; `"previous"` follows
  `glm.fit()` and saves one pass over the data.
* `fastglm_nb()` accepts a formula with `data`, `subset`, `weights`,
  `offset`, and `na.action`, and supports `predict(newdata = )`.
* `sandwich::vcovHC()` defaults to HC0 for count models and has methods for
  truncated, hurdle, and zero-inflated fits.
* `predict()` accepts a sparse `newdata` matrix; it previously stopped with an
  error.
* For hurdle, zero-inflated, and two-part models, `predict(newdata = )` stops
  with an error when the fit's `offset` argument refers to the fitting data
  (for example `log(data$exposure)`). Previously the fitting offset was
  silently reused, giving wrong predictions.
* `predict(type = "binary")` for these models returns the probability of a
  positive outcome, the quantity the binary part models.
* `fastglm_diagnostics()` supports truncated, hurdle, zero-inflated, and
  two-part fits, and `broom::tidy()` and `broom::glance()` support them too.
* `fastglm_marginal_effects()` supports GLMs and the count and two-part models,
  not only ordinal models.
* A file-backed `big.matrix` can be used with `fastglm_nb()`,
  `fastglm_truncated()`, `fastglm_hurdle()`, `fastglm_twopart()`, and
  `fastglm_ordinal()`; the matrix is read in blocks of rows.
* `fastglm_truncated()` and `fastglm_zi()` accept sparse matrices,
  file-backed `big.matrix` objects, and data sources, read in blocks of rows,
  and `fastglm_zi()` also accepts dense design matrices (`response` and `z`
  arguments, as in `fastglm_hurdle()`). Blockwise zero-inflated fits use the
  same convergence rule as dense fits and agree with them.
* `confint(fit, method = "profile")` gives profile likelihood intervals for
  maximum-likelihood GLMs, `fastglm_profile_test()` gives likelihood-ratio
  tests, and `fastglm_score_test()` gives score tests; these match
  `stats::confint()` and `stats::anova()`.
* `fastglm_bootstrap()` supports partial proportional odds models.
* `fastglm_batch()` fits negative binomial, zero-truncated, hurdle,
  zero-inflated, and two-part models to many responses on a shared design
  (`model` argument, with a prepared binary-part design `z`).
* Predictions of hurdle, zero-inflated, and two-part models from new data no
  longer warn about factors that appear in only one component.
* `predict()`, `residuals()`, and `simulate()` work for hurdle and two-part
  models fit to sparse matrices; they previously stopped with an error.
* Sparse and streamed hurdle models no longer report convergence when the
  count-part step was halved to almost nothing.
* Data sources print a short summary, and count model summaries print `theta`
  and the log-likelihood with more digits.
* More post-fit calculations (AIC, residuals, null deviance) run in C++, so
  ordinary fits are somewhat faster.
* Logit ordinal fits with many categories are about twice as fast, from a
  faster calculation of the information matrix, which involves every category
  for every observation.
* New vignette `stability-fastglm`, a simulation study of convergence against
  `glm.fit()`, glm2, `MASS::glm.nb()`, pscl, and ordinal. It replaces the
  `nb-stability-fastglm` and `nb-convergence-fastglm` vignettes.
* `?fastglm` has a new section, "Convergence and starting values".

## Breaking change

* Coefficients of the positive (Gamma) part of `fastglm_twopart()` models are
  named `positive_*` instead of `count_*`. Code that uses the old names must
  be updated. Hurdle and zero-inflated models keep `count_*`. The two-part
  accessors accept `"positive"` and `"binary"` as well as `"count"` and
  `"zero"`.

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
