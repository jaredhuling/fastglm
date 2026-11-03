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
#' @param firth `logical`; if `TRUE` apply the Kosmidis--Firth mean
#'   bias-reducing adjusted score (AS_mean) to the IRLS iteration.  Supported
#'   for all standard GLM families on dense `x`.
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
    res <- fastglmPure(x, y, .resolve_family(family),
                      if (is.null(weights)) rep(1, NROW(y)) else weights,
                      if (is.null(offset)) rep(0, NROW(y)) else offset,
                      start, etastart, mustart, control$method, control$tol,
                      control$maxit, firth = firth)
    if (!singular.ok && res$rank < ncol(x)) stop("singular fit encountered", call. = FALSE)
    res <- .finish_glm(res, if (is.null(offset)) rep(0, NROW(y)) else offset, intercept)
    res$class <- "fastglmFit"
    res
}

#' @export
#' @rdname fastglm_fit
fastglm_control <- function(fastmethod = 0L, tol = 1e-7, maxit = 100L, firth = FALSE, ...) {
    if (...length() > 0L && "method" %in% ...names() && identical(fastmethod, 0L)) {
        fastmethod <- ...elt(which(...names() == "method")[1L])
        fastmethod_name <- "method"
    }
    else {
        fastmethod_name <- "fastmethod"
    }
    
    if (!identical(fastmethod, "auto")) fastmethod <- .solver_method(fastmethod)
    .check_iterations(tol, maxit)

    list(method = fastmethod, tol = tol, maxit = maxit, firth = firth)
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