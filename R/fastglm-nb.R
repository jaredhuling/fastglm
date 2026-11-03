#' Fit a negative-binomial GLM with simultaneous (beta, theta) MLE
#'
#' `fastglm_nb()` fits a negative-binomial regression model, jointly
#' maximising the likelihood over the regression coefficients `beta` and the
#' NB2 dispersion `theta`. It is the fastglm analogue of [MASS::glm.nb()],
#' Dense fits use native coefficient and size updates. Sparse and streamed
#' fits alternate the existing GLM solver with size updates accumulated over
#' row blocks, without loading the full design into a dense matrix.
#'
#' @param x numeric matrix, `dgCMatrix`, or [fastglm_source()] supplying rows
#'   on demand. A source must supply its own responses, weights, and offsets.
#' @param y non-negative integer response vector; omitted for a data source.
#' @param weights optional prior weights vector of length `length(y)`.
#' @param offset optional offset vector of length `length(y)`.
#' @param start optional starting values for `beta`.
#' @param init.theta optional starting value for `theta`. If `NULL`, dense
#'   fits use a moment initializer; sparse and source fits start at one after
#'   fitting the Poisson mean model.
#' @param link character, one of `"log"` (default), `"sqrt"`, `"identity"`.
#' @param method integer; `0..5`, see [fastglm()].
#' @param tol convergence tolerance for the IRLS inner loop.
#' @param maxit maximum number of inner-loop IRLS iterations.
#' @param outer.maxit maximum number of `(beta, theta)` outer iterations.
#' @param outer.tol convergence tolerance for the outer loop on the
#'   sup-norm of the beta update plus the relative change in `theta`.
#' @param theta.tol tolerance for the inner theta update.
#' @param theta.maxit max iterations for the inner theta MLE.
#' @param block.size maximum rows processed together for sparse-model size
#'   estimation and joint inference. A source determines its own block sizes.
#'
#' @returns A list of class `c("fastglm_nb", "fastglm")` with the usual
#'   fastglm components plus `theta`, `SE.theta`, `iter.theta`, and
#'   `twologlik` (twice the maximized NB log-likelihood).
#'   `SE.theta` uses the joint observed information for coefficients and theta.
#'   The default coefficient covariance retains the GLM expected-information
#'   convention; `vcov(fit, full=TRUE)` returns the joint observed covariance.
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
fastglm_nb <- function(x, y = NULL,
                       weights     = NULL,
                       offset      = NULL,
                       start       = NULL,
                       init.theta  = NULL,
                       link        = c("log", "sqrt", "identity"),
                       method      = 2L,
                       tol         = 1e-8,
                       maxit       = 100L,
                       outer.maxit = 50L,
                       outer.tol   = 1e-7,
                       theta.tol   = 1e-8,
                       theta.maxit = 100L,
                       block.size  = 8192L)
{
    link <- match.arg(link)
    if(inherits(x,"dgCMatrix") || inherits(x,"fastglm_source")) {
        out <- .count_backend_nb(x,y,weights,offset,start,init.theta,link,method,tol,maxit,
                                 outer.maxit,outer.tol,theta.tol,theta.maxit,block.size)
        out$call <- match.call()
        return(out)
    }
    if (!is.matrix(x)) stop("'x' must be a numeric matrix.", call. = FALSE)
    if(is.integer(x)) storage.mode(x)<-"double"
    if (!is.numeric(y) || !is.null(dim(y)) || any(!is.finite(y) | y < 0 | y != floor(y)))
        stop("'y' must contain finite nonnegative integers.", call. = FALSE)
    n <- length(y)
    if (nrow(x) != n)
        stop("nrow(x) and length(y) disagree.", call. = FALSE)

    if (is.null(weights)) weights <- rep(1, n)
    if (is.null(offset))  offset  <- rep(0, n)
    weights <- as.numeric(weights); offset <- as.numeric(offset)

    .validate_glm_data(x, y, weights, offset)
    .check_iterations(tol, maxit); .check_iterations(outer.tol, outer.maxit)
    .check_iterations(theta.tol, theta.maxit)
    method <- .solver_method(method)
    if (!is.null(start) && (length(start) != ncol(x) || any(!is.finite(start)))) stop("invalid starting coefficients", call. = FALSE)
    if (!is.null(start)) {
        beta_init <- as.numeric(start)
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
    } else {
        mustart   <- y + 0.1667 * (y == 0)
        lnk       <- make.link(link)
        eta_init  <- as.numeric(lnk$linkfun(mustart))
        mu_init   <- mustart
        beta_init <- .lm.fit(x, eta_init - offset)$coefficients
        beta_init[is.na(beta_init)] <- 0
        eta_init  <- as.numeric(x %*% beta_init + offset)
        mu_init   <- as.numeric(lnk$linkinv(eta_init))
    }

    init_th <- if (!is.null(init.theta)) {
        if (!is.numeric(init.theta) || length(init.theta) != 1L ||
            !is.finite(init.theta) || init.theta <= 0)
            stop("'init.theta' must be a positive finite scalar.", call. = FALSE)
        init.theta
    } else {
        -1  # C++ MoM estimator from mu_init
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
    res$family <- final_family
    res$y <- as.numeric(y); res$x <- x; res$n <- n
    res$prior.weights <- weights
    res$intercept <- any(.intercept_columns(x))
    res <- .finish_glm(res, offset)
    keep <- weights>0
    res$twologlik <- 2*sum(weights[keep]*dnbinom(y[keep],mu=res$fitted.values[keep],size=res$theta,log=TRUE))
    res$aic <- -res$twologlik+2*(res$rank+1L)
    # The beta-only covariance retains the usual expected-information GLM
    # convention. The joint covariance and theta standard error use observed
    # information, including the beta/theta cross derivatives.
    res$theta.boundary <- res$theta<=1e-6*(1+theta.tol) || res$theta>=1e8/(1+theta.tol)
    if(res$theta.boundary) {
        joint <- matrix(NA_real_,length(res$coefficients)+1L,length(res$coefficients)+1L)
        joint[seq_along(res$coefficients),seq_along(res$coefficients)] <- res$cov.unscaled
    } else {
        reader <- .count_backend_reader(x,y,weights=weights,offset=offset,block.size=block.size)
        joint <- .count_backend_joint(reader,res$coefficients,res$theta,FALSE,link)
    }
    dimnames(joint) <- rep(list(c(names(res$coefficients),"theta")),2L)
    res$vcov.full <- joint
    res$SE.theta <- sqrt(joint[length(res$coefficients)+1L,length(res$coefficients)+1L])
    if (!res$converged) warning("negative-binomial fit did not converge", call. = FALSE)
    res$call <- match.call()
    res$theta.estimated <- TRUE
    res$refit.controls <- list(link=link,method=method,tol=tol,maxit=maxit,outer.maxit=outer.maxit,
        outer.tol=outer.tol,theta.tol=theta.tol,theta.maxit=theta.maxit,block.size=block.size)
    class(res) <- c("fastglm_nb", "fastglm")
    res
}

#' @export
logLik.fastglm_nb <- function(object, ...)
{
    df <- object$rank + 1L          # +1 for theta
    structure(object$twologlik / 2, df = df,
              nobs = object$nobs, class = "logLik")
}
