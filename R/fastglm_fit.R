#' Fast generalized linear model fitting
#'
#' `fastglm_fit()` is a fitting method for [glm()]. It works like `glm.fit()`, i.e., by being supplied to the `method` argument of `glm()`.
#' 
#' @param x a design matrix of dimension `n * p`. Can also be a `big.matrix` object from \pkg{bigmemory}.
#' @param y a vector of observations of length `n`.
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be `NULL` or a numeric vector.
#' @param start optional starting values for the parameters in the linear predictor.
#' @param etastart optional starting values for the linear predictor.
#' @param mustart optional starting values for the vector of means.
#' @param offset this can be used to specify an *a priori* known component to be included in the linear predictor during fitting. This should be `NULL` or a numeric vector of length equal to the number of cases.
#' @param family a description of the error distribution and link function to be used in the model. This must be a family function or the result of a call to a family function. (See [`family`] for details of family functions.)
#' @param control a list of parameters for controlling the fitting process. This is passed to `fastglm_control()`.
#' @param singular.ok,intercept See [glm.fit()].
#' @param fastmethod `integer`; the method used for fitting. Allowable values include 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition, 2 for the LLT Cholesky, 3 for the LDLT Cholesky, 4 for the full pivoted QR decomposition, and 5 for the Bidiagonal Divide and Conquer SVD. Default is 0. Can also be supplied as `method` when not supplied directly as an argument from `glm()` (see Examples).
#' @param tol `numeric`; threshold tolerance for convergence.
#' @param maxit `integer`; the maximum number of IRLS iterations.
#' @param covariance where the information matrix behind the standard errors
#'   is evaluated. `"estimate"` (default) evaluates it at the reported
#'   coefficients, which takes one more pass over the data after convergence.
#'   `"previous"` uses the factorization from the final iteration, the
#'   convention of [stats::glm.fit()]; it saves that pass (roughly 10 percent
#'   of the fitting time for large data) and changes standard errors by a
#'   relative amount of order 1e-6 at the default tolerance. Coefficients agree
#'   to within the convergence tolerance. Bias-reduced fits (`firth = TRUE`)
#'   are not affected.
#' @param firth `logical`; if `TRUE` apply the Kosmidis--Firth mean
#'   bias-reducing adjusted score (AS_mean) to the IRLS iteration.  Supported
#'   for all standard GLM families on dense `x`.
#' @param threads the number of threads for the compiled computations, or
#'   `NULL` (default) for the session setting of [fastglm_threads()]. Results
#'   are identical for every number of threads.
#' @param trace `TRUE` to print one line per iteration (the iteration,
#'   the passes over the data so far, the deviance, the step halvings of the
#'   iteration, and the elapsed seconds), `FALSE` for none, or `NULL`
#'   (default) for `getOption("fastglm.trace", FALSE)`. Lines are printed
#'   between iterations, so tracing off costs nothing.
#' @param object a `fastglmFit` object; the output of a call to `glm()` with `method = fastglm_fit`.
#' @param \dots for `vcov()` and `summary()`, other arguments passed downstream.
#'
#' @details
#' The purpose of the functions documented on this page is to facilitate integration with existing [glm()] utilities in base R. `fastglm_fit()` is just a wrapper for [fastglmPure()] with some additional quality-of-life features. The `vcov()` and `summary()` methods use the unscaled coefficient covariance matrix returned directly from the C++ solver, so no refit is required.
#' 
#' @examples
#' set.seed(1234)
#' n <- 1e4
#' x <- matrix(rnorm(n * 25), ncol = 25)
#' eta <- 0.1 + 0.25 * x[,1] - 0.25 * x[,3] + 0.75 * x[,5] -0.35 * x[,6]
#' dat <- as.data.frame(x)
#' 
#' # binomial
#' dat$y <- rbinom(n, 1, pnorm(eta))
#' 
#' system.time({
#'     gl <- glm(y ~ ., data = dat,
#'               family = binomial)
#' })
#' 
#' system.time({
#'     gf0 <- glm(y ~ ., data = dat,
#'                family = binomial,
#'                method = fastglm_fit)
#' })
#' 
#' system.time({
#'     gf1 <- glm(y ~ ., data = dat,
#'                family = binomial,
#'                method = fastglm_fit,
#'                fastmethod = 1)
#' })
#' 
#' # poisson
#' dat$y <- rpois(n, eta^2)
#' 
#' system.time({
#'     gl <- glm(y ~ ., data = dat,
#'               family = poisson)
#' })
#' 
#' system.time({
#'     gf0 <- glm(y ~ ., data = dat,
#'                family = poisson,
#'                method = fastglm_fit)
#' })
#' 
#' system.time({
#'     gf1 <- glm(y ~ ., data = dat,
#'                family = poisson,
#'                method = fastglm_fit,
#'                fastmethod = 1)
#' })
#' 
#' # gamma
#' dat$y <- rgamma(n, exp(eta) * 1.75, 1.75)
#' 
#' system.time({
#'     gl <- glm(y ~ ., data = dat,
#'               family = Gamma(link = "log"))
#' })
#' 
#' system.time({
#'     gf0 <- glm(y ~ ., data = dat,
#'                family = Gamma(link = "log"),
#'                method = fastglm_fit)
#' })
#' 
#' system.time({
#'     gf1 <- glm(y ~ ., data = dat,
#'                family = Gamma(link = "log"),
#'                method = fastglm_fit,
#'                fastmethod = 1)
#' })
#' 
#' # Different (equivalent) ways of supplying
#' # control arguments:
#' gf1 <- glm(y ~ ., data = dat,
#'            family = Gamma(link = "log"),
#'            method = fastglm_fit,
#'            fastmethod = 1)
#' 
#' gf1 <- glm(y ~ ., data = dat,
#'            family = Gamma(link = "log"),
#'            method = fastglm_fit,
#'            control = list(fastmethod = 1))
#' 
#' gf1 <- glm(y ~ ., data = dat,
#'            family = Gamma(link = "log"),
#'            method = fastglm_fit,
#'            control = list(method = 1))

#' @export
fastglm_fit <- function(x, y,
                        weights  = rep(1, NROW(y)),
                        start    = NULL,
                        etastart = NULL,
                        mustart  = NULL,
                        offset   = rep(0, NROW(y)),
                        family   = gaussian(),
                        control  = list(),
                        intercept = TRUE,
                        singular.ok = TRUE,
                        firth = FALSE)
{
    control <- do.call(fastglm_control, control)
    firth <- firth || isTRUE(control$firth)
    # With an offset, stats::glm() refits the intercept-only model for the
    # null deviance with the same control (x is its "(Intercept)" column and
    # mustart the fitted values of the main fit); that refit is not traced.
    if (intercept && identical(colnames(x), "(Intercept)") && !is.null(mustart))
        control$trace <- FALSE
    res <- fastglmPure(x, y, .resolve_family(family),
                      if (is.null(weights)) rep(1, NROW(y)) else weights,
                      if (is.null(offset)) rep(0, NROW(y)) else offset,
                      start, etastart, mustart, control$method, control$tol,
                      control$maxit, firth = firth, covariance = control$covariance,
                      threads = control$threads, trace = control$trace)
    # The post-fit pass runs at the fit's thread count.
    restore.threads <- .fastglm_with_threads(res$threads)
    on.exit(restore.threads(), add = TRUE)
    if (!singular.ok && res$rank < ncol(x)) stop("singular fit encountered", call. = FALSE)
    res <- .finish_glm(res, if (is.null(offset)) rep(0, NROW(y)) else offset, intercept)
    res$class <- "fastglmFit"
    res
}

#' @export
#' @rdname fastglm_fit
fastglm_control <- function(fastmethod = 0L, tol = 1e-7, maxit = 100L, firth = FALSE,
                            covariance = c("estimate", "previous"), threads = NULL,
                            trace = NULL, ...) {
    covariance <- match.arg(covariance)
    if (...length() > 0L && "method" %in% ...names() && identical(fastmethod, 0L)) {
        fastmethod <- ...elt(which(...names() == "method")[1L])
        fastmethod_name <- "method"
    }
    else {
        fastmethod_name <- "fastmethod"
    }
    
    if (!identical(fastmethod, "auto")) fastmethod <- .solver_method(fastmethod)
    .check_iterations(tol, maxit)

    if (!is.null(threads)) .fastglm_resolve_threads(threads)   # validates the request
    if (!is.null(trace)) .fastglm_resolve_trace(trace)         # validates the request
    list(method = fastmethod, tol = tol, maxit = maxit, firth = firth, covariance = covariance,
         threads = threads, trace = trace)
}

#' @exportS3Method stats::vcov fastglmFit
#' @rdname fastglm_fit
vcov.fastglmFit <- function(object, ...) vcov.fastglm(object, ...)

#' @method summary fastglmFit
#' @exportS3Method summary fastglmFit
#' @rdname fastglm_fit
summary.fastglmFit <- function(object, ...) {
    summary.fastglm(object, ...)
}

#' @exportS3Method print fastglmFit
print.fastglmFit <- function(x,...) print.fastglm(x,...)

## Methods of stats that refit a formula fit (anova(), drop1(), add1(),
## profile()). anova.glm() refits with method = fastglm_fit and the fit's
## controls; drop1.glm(), add1.glm(), and profile.glm() refit with
## stats::glm.fit(), which accepts only glm.control() fields, so they get the
## fit's tolerance and iteration limit in that form. The refits are not
## traced, and refits by glm.fit() (maximum likelihood) are refused for
## bias-reduced fits. The anova.glm() refits use .fastglm_fit_threads().
.fastglm_stats_refit <- function(object, what, glm.fit = TRUE)
{
    ctl <- object$control
    if (glm.fit) {
        if (isTRUE(ctl$firth) || isTRUE(object$firth))
            stop(sprintf("%s() refits by maximum likelihood with stats::glm.fit() and is not available for bias-reduced (firth = TRUE) fits", what),
                 call. = FALSE)
        object$control <- stats::glm.control(
            epsilon = if (is.null(ctl$tol)) 1e-8 else ctl$tol,
            maxit = if (is.null(ctl$maxit)) 100L else ctl$maxit)
    } else if (is.list(ctl)) {
        ctl$trace <- FALSE
        # The refits use the fit's threads, at most the session setting.
        ctl$threads <- .fastglm_fit_threads(object)
        object$control <- ctl
    }
    object
}

## Calls with only glm objects go to stats::anova.glm unchanged; a call with
## another fastglm class, or with a bias-reduced fit, uses the comparison
## engine of ?anova.fastglm (never a silent anova.glm result).
#' @rdname anova.fastglm
#' @exportS3Method stats::anova fastglmFit
anova.fastglmFit <- function(object, ...)
{
    dots <- list(...)
    is.model <- vapply(dots, function(x) inherits(x, .fastglm_model_classes), NA)
    if (any(is.model) && .fastglm_anova_needs_engine(c(list(object), dots[is.model]))) {
        mc <- match.call(expand.dots = FALSE)
        labels <- .fastglm_anova_labels(mc)
        nm <- if (is.null(names(dots))) rep_len("", length(dots)) else names(dots)
        opt <- dots[nm %in% c("test", "dispersion")]
        keep <- !(nm %in% c("test", "dispersion"))
        m <- .fastglm_anova_collect(object, dots[keep], labels[c(TRUE, keep)])
        return(.fastglm_lrt(m$models, test = opt$test, labels = m$labels, dispersion = opt$dispersion))
    }
    # A bias-reduced formula fit alone: the sequential table of penalized
    # likelihood ratio tests (never the deviances of separate Firth fits).
    if (!any(is.model) && .fastglm_anova_needs_engine(list(object))) {
        nm <- if (is.null(names(dots))) rep_len("", length(dots)) else names(dots)
        if (any(!nzchar(nm)) || any(!(nm %in% c("test", "dispersion", "trace"))))
            warning("arguments to 'anova' that are not fitted models, 'test', 'dispersion', or 'trace' are ignored",
                    call. = FALSE)
        return(.fastglm_anova_sequential(object, test = dots$test, dispersion = dots$dispersion,
                                         label = deparse1(substitute(object)), trace = dots$trace))
    }
    # Formula fits compared among themselves (or with stats::glm fits): the
    # call goes to stats::anova.glm unchanged.
    object <- .fastglm_stats_refit(object, "anova", glm.fit = FALSE)
    no.trace <- .fastglm_scope_trace(FALSE)
    on.exit(no.trace$restore(), add = TRUE)
    NextMethod()
}

## Bias-reduced fits use the penalized likelihood ratio tests of
## ?drop1.fastglm; other formula fits go to stats::drop1.glm unchanged.
#' @rdname drop1.fastglm
#' @exportS3Method stats::drop1 fastglmFit
drop1.fastglmFit <- function(object, ...)
{
    if (isTRUE(object$firth) || isTRUE(object$control$firth))
        return(.fastglm_drop1_entry(object, ..., .label = deparse1(substitute(object))))
    object <- .fastglm_stats_refit(object, "drop1")
    no.trace <- .fastglm_scope_trace(FALSE)
    on.exit(no.trace$restore(), add = TRUE)
    NextMethod()
}

#' @exportS3Method stats::add1 fastglmFit
add1.fastglmFit <- function(object, ...)
{
    object <- .fastglm_stats_refit(object, "add1")
    no.trace <- .fastglm_scope_trace(FALSE)
    on.exit(no.trace$restore(), add = TRUE)
    NextMethod()
}

#' @exportS3Method stats::profile fastglmFit
profile.fastglmFit <- function(fitted, ...)
{
    fitted <- .fastglm_stats_refit(fitted, "profile")
    no.trace <- .fastglm_scope_trace(FALSE)
    on.exit(no.trace$restore(), add = TRUE)
    NextMethod()
}
