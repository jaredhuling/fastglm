#' Fit a negative-binomial GLM with simultaneous (beta, theta) MLE
#'
#' `fastglm_nb()` fits a negative-binomial regression model, jointly
#' maximising the likelihood over the regression coefficients `beta` and the
#' NB2 dispersion `theta`. It is the fastglm analogue of [MASS::glm.nb()],
#' built on top of the native NB family kernel introduced in 0.0.6 so that
#' all numerical loops -- IRLS, the inner theta MLE Brent root-find, and
#' the outer (beta, theta) alternation -- run entirely in C++.
#'
#' @param x design matrix (numeric matrix, dgCMatrix not yet supported here).
#' @param y non-negative integer response vector.
#' @param weights optional prior weights vector of length `length(y)`.
#' @param offset optional offset vector of length `length(y)`.
#' @param start optional starting values for `beta`.
#' @param init.theta optional starting value for `theta`. If `NULL`, uses
#'   the method-of-moments estimator from a Poisson pilot fit.
#' @param link character, one of `"log"` (default), `"sqrt"`, `"identity"`.
#' @param method integer; `0..5`, see [fastglm()].
#' @param tol convergence tolerance for the IRLS inner loop.
#' @param maxit maximum number of inner-loop IRLS iterations.
#' @param outer.maxit maximum number of `(beta, theta)` outer iterations.
#' @param outer.tol convergence tolerance for the outer loop on
#'   `||Δbeta||∞ + |Δtheta|/theta`.
#' @param theta.tol Brent tolerance for the inner theta MLE.
#' @param theta.maxit max iterations for the inner theta MLE.
#'
#' @returns A list of class `c("fastglm_nb", "fastglm")` with the usual
#'   fastglm components plus `theta`, `SE.theta`, `iter.theta`, and
#'   `twologlik` (twice the maximised NB log-likelihood).
#'
#' @examples
#' set.seed(1)
#' n <- 500
#' x <- cbind(1, matrix(rnorm(n * 2), n, 2))
#' eta <- x %*% c(0.3, 0.5, -0.2)
#' mu  <- exp(eta)
#' if (requireNamespace("MASS", quietly = TRUE)) {
#'   y   <- MASS::rnegbin(n, mu = mu, theta = 2)
#'   fit <- fastglm_nb(x, y)
#'   c(theta = fit$theta, fit$coefficients)
#' }
#'
#' @export
fastglm_nb <- function(x, y,
                       weights     = NULL,
                       offset      = NULL,
                       start       = NULL,
                       init.theta  = NULL,
                       link        = c("log", "sqrt", "identity"),
                       method      = 2L,
                       tol         = 1e-8,
                       maxit       = 100L,
                       outer.maxit = 25L,
                       outer.tol   = 1e-7,
                       theta.tol   = 1e-8,
                       theta.maxit = 100L)
{
    link <- match.arg(link)
    if (!is.matrix(x)) stop("'x' must be a numeric matrix.", call. = FALSE)
    if (!is.numeric(y) || any(y < 0))
        stop("'y' must be non-negative.", call. = FALSE)
    n <- length(y)
    if (nrow(x) != n)
        stop("nrow(x) and length(y) disagree.", call. = FALSE)

    if (is.null(weights)) weights <- rep(1, n)
    if (is.null(offset))  offset  <- rep(0, n)
    weights <- as.numeric(weights); offset <- as.numeric(offset)

    # Pilot Poisson fit to seed mu / eta and (if needed) theta.
    pilot <- fastglmPure(x, y,
                         family  = poisson(link = link),
                         weights = weights,
                         offset  = offset,
                         start   = start,
                         method  = as.integer(method),
                         tol     = tol,
                         maxit   = maxit)
    mu_init  <- as.numeric(pilot$fitted.values)
    eta_init <- as.numeric(pilot$linear.predictors)
    beta_init <- if (!is.null(start)) as.numeric(start) else as.numeric(pilot$coefficients)

    # Theta init: caller-supplied, or MoM (signalled to C++ via theta <= 0).
    init_th <- if (!is.null(init.theta)) {
        if (!is.numeric(init.theta) || length(init.theta) != 1L ||
            !is.finite(init.theta) || init.theta <= 0)
            stop("'init.theta' must be a positive finite scalar.", call. = FALSE)
        init.theta
    } else {
        -1
    }

    # The C++ driver uses the native NB kernels for {log,sqrt,identity}, but
    # still requires an R family object for the callback fallback signature.
    fam_dummy <- if (requireNamespace("MASS", quietly = TRUE)) {
        MASS::negative.binomial(theta = max(init_th, 1), link = link)
    } else {
        negbin(theta = max(init_th, 1), link = link)
    }

    fc <- switch(link,
                 "log"      = 17L,
                 "sqrt"     = 18L,
                 "identity" = 19L)

    res <- fit_glm_nb(
        x       = x,
        y       = as.numeric(y),
        weights = weights,
        offset  = offset,
        start   = beta_init,
        mu_init = mu_init,
        eta_init = eta_init,
        var_fun         = fam_dummy$variance,
        mu_eta_fun      = fam_dummy$mu.eta,
        linkinv_fun     = fam_dummy$linkinv,
        dev_resids_fun  = fam_dummy$dev.resids,
        valideta_fun    = fam_dummy$valideta,
        validmu_fun     = fam_dummy$validmu,
        type        = as.integer(method),
        tol         = as.double(tol),
        maxit       = as.integer(maxit),
        fam_code    = fc,
        init_theta  = as.double(init_th),
        theta_tol   = as.double(theta.tol),
        theta_maxit = as.integer(theta.maxit),
        outer_maxit = as.integer(outer.maxit),
        outer_tol   = as.double(outer.tol)
    )

    # Cosmetic post-processing
    cnames <- colnames(x)
    if (is.null(cnames)) cnames <- paste0("x", seq_len(ncol(x)))
    names(res$coefficients) <- cnames
    names(res$se)           <- cnames
    rownames(res$cov.unscaled) <- colnames(res$cov.unscaled) <- cnames

    final_family <- if (requireNamespace("MASS", quietly = TRUE)) {
        MASS::negative.binomial(theta = res$theta, link = link)
    } else {
        negbin(theta = res$theta, link = link)
    }
    res$family        <- final_family
    res$y             <- as.numeric(y)
    res$x             <- x
    res$prior.weights <- weights
    res$dispersion    <- 1                       # NB likelihood is fully specified
    res$df.null       <- n - 1L
    res$null.deviance <- sum(final_family$dev.resids(
        as.numeric(y),
        rep(if (sum(weights) > 0) sum(weights * as.numeric(y))/sum(weights) else mean(y), n),
        weights))
    res$call <- match.call()
    class(res) <- c("fastglm_nb", "fastglm")
    res
}

#' @export
logLik.fastglm_nb <- function(object, ...)
{
    df <- length(object$coefficients) + 1L          # +1 for theta
    structure(object$twologlik / 2, df = df,
              nobs = length(object$y), class = "logLik")
}
