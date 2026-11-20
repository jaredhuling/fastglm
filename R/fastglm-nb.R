#' Fit a negative-binomial GLM with simultaneous (beta, theta) MLE
#'
#' `fastglm_nb()` fits a negative-binomial regression model, jointly
#' maximising the likelihood over the regression coefficients `beta` and the
#' NB2 dispersion `theta`. It is the fastglm analogue of [MASS::glm.nb()],
#' Dense fits use native coefficient and size updates. Sparse and streamed
#' fits alternate the existing GLM solver with size updates accumulated over
#' row blocks, without loading the full design into a dense matrix.
#'
#' @param x a model formula, numeric matrix, `dgCMatrix`, file-backed
#'   `big.matrix` (read in row blocks), or [fastglm_source()]
#'   supplying rows on demand. A source must supply its own responses,
#'   weights, and offsets.
#' @param y non-negative integer response vector; omitted for a formula or a
#'   data source.
#' @param weights optional prior weights: a vector of length `length(y)`, or
#'   for a formula, a variable in `data`.
#' @param offset optional offset: a vector of length `length(y)`, or for a
#'   formula, a variable in `data`. A formula may also contain `offset()`
#'   terms.
#' @param start optional starting values for `beta`.
#' @param init.theta optional starting value for `theta`. If `NULL`, dense
#'   fits use a moment initializer; sparse and source fits start at one after
#'   fitting the Poisson mean model.
#' @param link character, one of `"log"` (default), `"sqrt"`, `"identity"`.
#' @param method integer; `0..5`, see [fastglm()].
#' @param tol convergence tolerance for the IRLS inner loop.
#' @param maxit maximum number of inner-loop IRLS iterations.
#' @param outer.maxit maximum number of `(beta, theta)` outer iterations.
#' @param outer.tol convergence tolerance for the outer loop. The loop stops
#'   when the joint Fisher decrement at the current coefficients and `theta`
#'   (the deviance decrease predicted by one more scoring step in both) is
#'   below `outer.tol` times `0.1 + |deviance|`. Because the expected
#'   information between the coefficients and `theta` is zero, the decrement
#'   is the sum of the coefficient decrement and the `theta` decrement. The
#'   value is returned as `outer.decrement`.
#' @param theta.tol tolerance for the inner theta update.
#' @param theta.maxit max iterations for the inner theta MLE.
#' @param block.size maximum rows processed together for sparse-model size
#'   estimation and joint inference. A source determines its own block sizes.
#' @param poisson.limit whether to compare the fit with its `theta = Inf`
#'   limit, the Poisson model (see Details).
#' @param threads the number of threads for the compiled computations of this
#'   fit, including its pilot, Poisson-limit, and covariance passes, or `NULL`
#'   (default) for the session setting of [fastglm_threads()]. Results are
#'   identical for every number of threads.
#' @param trace `TRUE` to print one line per outer (coefficient and theta)
#'   iteration (the iteration, the passes over the data so far, the log
#'   likelihood at the updated theta, the step halvings of the coefficient
#'   fit, and the elapsed seconds; the log likelihood takes one more pass
#'   over the data per iteration, made only when tracing),
#'   `FALSE` for none, or `NULL` (default) for `getOption("fastglm.trace")`,
#'   which defaults to `TRUE` for a [fastglm_source()] in an interactive
#'   session and `FALSE` otherwise.
#' @param data,subset,na.action,contrasts model-frame arguments used when `x`
#'   is a formula, as in [stats::glm()].
#'
#' @details
#' The fit alternates coefficient updates at fixed `theta` (iteratively
#' reweighted least squares) with maximum likelihood updates of `theta` at
#' fixed coefficients, and stops when the joint Fisher decrement is below
#' `outer.tol` (see that argument).
#'
#' The parameter space includes `theta = Inf`, where the model is the Poisson
#' model, and the profile likelihood of `theta` need not be unimodal: the
#' alternation can stop at a finite local maximum while the likelihood is
#' higher in the Poisson limit, or stop at the upper cap `theta = 1e8` short
#' of that limit. With `poisson.limit = TRUE`, the Poisson fit is computed and
#' compared. If its log likelihood is higher and the data show overdispersion
#' at the Poisson fit (`sum(w * ((y - mu)^2 - y)) > 0`, the sign of the
#' derivative of the profile likelihood as `theta` grows without bound), the
#' fit is restarted from the Poisson coefficients with the moment estimate of
#' `theta` and kept if it improves on the Poisson likelihood. Otherwise the
#' Poisson limit is returned: `theta = Inf`, the Poisson coefficients and
#' covariance, the Poisson family, `SE.theta = NA`,
#' `convergence.reason = "theta_infinite"`, and a warning.
#'
#' With `link = "sqrt"` or `"identity"` the optimum can lie on the boundary
#' `eta = 0` of the valid region. The coefficient fits then use the boundary
#' step described in [fastglm()] (Convergence and starting values), and a fit
#' that converges there reports `convergence.reason = "converged_boundary"`
#' with a warning that Wald inference is not reliable. If the Cholesky
#' factorization of a dense design is ill-conditioned near the boundary, the
#' fit is repeated with the QR decomposition; sparse designs have no such
#' fallback.
#'
#' @returns A list of class `c("fastglm_nb", "fastglm")` with the usual
#'   fastglm components plus `theta`, `SE.theta`, `iter.theta`,
#'   `twologlik` (twice the maximized NB log-likelihood), and `threads` (the
#'   thread setting of the fit).
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
#'
#'   # Formula interface
#'   d <- data.frame(y = y, x1 = x[, 2], x2 = x[, 3])
#'   coef(fastglm_nb(y ~ x1 + x2, data = d))
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
                       block.size  = 8192L,
                       poisson.limit = TRUE,
                       threads     = NULL,
                       trace       = NULL,
                       data        = NULL,
                       subset, na.action, contrasts = NULL)
{
    link <- match.arg(link)
    # The thread setting holds for every fit and pass below, including the
    # pilot, Poisson-limit, and covariance passes after the main fit.
    scope <- .fastglm_scope_threads(threads)
    on.exit(scope$restore(), add = TRUE)
    threads <- scope$threads
    tr <- .fastglm_scope_trace(trace, streamed = inherits(x, "fastglm_source"))
    on.exit(tr$restore(), add = TRUE)
    if (inherits(x, "formula")) {
        # Build the design from the formula, fit with the matrix interface,
        # and keep what predict() and model.matrix() need.
        mc <- match.call(expand.dots = FALSE)
        keep <- match(c("x", "data", "subset", "weights", "na.action", "offset"), names(mc), 0L)
        mf <- mc[c(1L, keep)]
        names(mf)[names(mf) == "x"] <- "formula"
        if (is.null(mf$data)) mf$data <- environment(x)
        mf$drop.unused.levels <- TRUE
        mf[[1L]] <- quote(stats::model.frame)
        mf <- eval(mf, parent.frame())
        mt <- attr(mf, "terms")
        yy <- stats::model.response(mf, "numeric")
        X <- stats::model.matrix(mt, mf, contrasts)
        w <- as.vector(stats::model.weights(mf))
        off <- as.vector(stats::model.offset(mf))
        fit <- fastglm_nb(X, yy, weights = w, offset = off, start = start, init.theta = init.theta,
                          link = link, method = method, tol = tol, maxit = maxit,
                          outer.maxit = outer.maxit, outer.tol = outer.tol, theta.tol = theta.tol,
                          theta.maxit = theta.maxit, block.size = block.size, poisson.limit = poisson.limit,
                          threads = threads, trace = tr$trace)
        fit$terms <- mt
        fit$model <- mf
        fit$xlevels <- stats::.getXlevels(mt, mf)
        fit$contrasts <- attr(X, "contrasts")
        fit$call <- match.call()
        return(fit)
    }
    if(inherits(x,c("dgCMatrix","big.matrix","fastglm_source"))) {
        out <- .count_backend_nb(x,y,weights,offset,start,init.theta,link,method,tol,maxit,
                                 outer.maxit,outer.tol,theta.tol,theta.maxit,block.size,poisson.limit)
        out$call <- match.call()
        out$threads <- threads
        out$compare.controls <- list(poisson.limit = poisson.limit)
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
    method.supplied <- !missing(method)
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
        # Under the sqrt and identity links the least-squares start can give
        # invalid linear predictors (eta <= 0), from which the first IRLS step
        # cannot be halved back into the valid region. Halve it toward the
        # intercept-only fit, which is valid when the design has an intercept,
        # as fastglm() does for an invalid first update.
        pfam <- poisson(link = link)
        valid_eta <- function(e) isTRUE(pfam$valideta(e)) && isTRUE(pfam$validmu(pfam$linkinv(e)))
        if (!valid_eta(eta_init)) {
            anchor <- .feasible_start(x, y, weights, offset, pfam, pfam$valideta, pfam$validmu,
                                      .intercept_columns(x))
            if (any(anchor != 0) && valid_eta(as.numeric(x %*% anchor + offset))) {
                for (k in seq_len(60L)) {
                    beta_init <- 0.5 * (beta_init + anchor)
                    eta_init  <- as.numeric(x %*% beta_init + offset)
                    if (valid_eta(eta_init)) break
                }
                if (!valid_eta(eta_init)) {
                    beta_init <- anchor
                    eta_init  <- as.numeric(x %*% beta_init + offset)
                }
            }
        }
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

    run_nb <- function(type) fit_glm_nb(
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
        type        = as.integer(type),
        tol         = as.double(tol),
        maxit       = as.integer(maxit),
        fam_code    = fc,
        init_theta  = as.double(init_th),
        theta_tol   = as.double(theta.tol),
        theta_maxit = as.integer(theta.maxit),
        outer_maxit = as.integer(outer.maxit),
        outer_tol   = as.double(outer.tol)
    )
    # A row held near the bound of the sqrt or identity link can carry a
    # working weight near 1 / mu, which makes the Cholesky factorization
    # ill-conditioned; refit with the column-pivoted QR decomposition, as
    # fastglm() does (with a warning when a Cholesky method was requested).
    res <- tryCatch(run_nb(method), error = function(e) {
        if (!(method %in% 2:3) || !grepl("Cholesky factorization is singular", conditionMessage(e))) stop(e)
        if (method.supplied)
            warning("Cholesky factorization is singular or ill-conditioned; refitting with method = 'qr'",
                    call. = FALSE)
        method <<- 0L
        run_nb(0L)
    })

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
    # The native family code lets .finish_glm() compute residuals, the
    # Pearson statistic, and the null deviance in one C++ pass; the AIC is
    # set below from the negative binomial log likelihood.
    attr(res, "family_code") <- fc
    res <- .finish_glm(res, offset, aic = FALSE)
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
    res$call <- match.call()
    res$theta.estimated <- TRUE
    res$refit.controls <- list(link=link,method=method,tol=tol,maxit=maxit,outer.maxit=outer.maxit,
        outer.tol=outer.tol,theta.tol=theta.tol,theta.maxit=theta.maxit,block.size=block.size)
    class(res) <- c("fastglm_nb", "fastglm")
    if (poisson.limit) {
        # The Poisson fit starts from the negative binomial coefficients, which
        # estimate the same mean model, so it needs few iterations; if that
        # start fails (for example a row held at the boundary of the sqrt or
        # identity link), it uses its default start. The comparison needs only
        # the fitted values, so the light fastglmPure() fit is used, and the
        # full fastglm() object is built only when the Poisson limit is higher
        # and may be returned.
        fit_poisson <- function(start, fitter) suppressWarnings(fitter(x, y, family = poisson(link = link),
            weights = weights, offset = offset, start = start, method = method, tol = tol, maxit = maxit))
        pstart <- if (all(is.finite(res$coefficients))) unname(res$coefficients)
        pois <- if (!is.null(pstart)) tryCatch(fit_poisson(pstart, fastglmPure), error = function(e) NULL)
        if (is.null(pois) || !isTRUE(pois$converged)) { pstart <- NULL; pois <- fit_poisson(NULL, fastglmPure) }
        keep <- weights > 0
        pois_ll <- sum(weights[keep] * dpois(y[keep], pois$fitted.values[keep], log = TRUE))
        if (isTRUE(pois$converged) && is.finite(pois_ll) && pois_ll > res$twologlik / 2)
        {
            pois <- fit_poisson(pstart, fastglm)
            pois_ll <- sum(weights[keep] * dpois(y[keep], pois$fitted.values[keep], log = TRUE))
        }
        refit <- function(theta0) {
            f <- tryCatch(suppressWarnings(fastglm_nb(x, y, weights = weights, offset = offset,
                start = unname(coef(pois)), init.theta = theta0, link = link, method = method,
                tol = tol, maxit = maxit, outer.maxit = outer.maxit, outer.tol = outer.tol,
                theta.tol = theta.tol, theta.maxit = theta.maxit, block.size = block.size,
                poisson.limit = FALSE)), error = function(e) NULL)
            if (!is.null(f)) f$call <- res$call
            f
        }
        mu <- pois$fitted.values
        res <- .nb_poisson_limit(res, pois, pois_ll,
                                 excess = sum(weights[keep] * ((y[keep] - mu[keep])^2 - y[keep])),
                                 musq = sum(weights[keep] * mu[keep]^2), link, refit)
    }
    res$threads <- threads
    res$compare.controls <- list(poisson.limit = poisson.limit)
    if (!res$converged) warning("negative-binomial fit did not converge", call. = FALSE)
    if (identical(res$convergence.reason, "converged_boundary"))
        warning("negative-binomial fit: ", BOUNDARY_WARNING, call. = FALSE)
    res
}

# The theta = Inf limit of a negative binomial fit (see ?fastglm_nb). nb: the
# fit; pois: the Poisson fit with the same design, weights, offset, and link;
# pois_ll: its log likelihood; excess = sum(w ((y - mu)^2 - y)) and
# musq = sum(w mu^2) at the Poisson fit; refit(theta0): the NB fit restarted
# from the Poisson coefficients with initial theta theta0 (NULL on failure).
.nb_poisson_limit <- function(nb, pois, pois_ll, excess, musq, link, refit) {
    nb_ll <- nb$twologlik / 2
    if (!isTRUE(pois$converged) || !is.finite(pois_ll) || !(pois_ll > nb_ll)) return(nb)
    if (is.finite(excess) && excess > 0) {
        theta0 <- min(1e8, max(1e-6, musq / excess))
        f <- refit(theta0)
        if (!is.null(f) && isTRUE(f$converged) && f$twologlik / 2 > pois_ll) return(f)
    }
    out <- nb
    for (nm in c("coefficients", "se", "cov.unscaled", "fitted.values", "linear.predictors",
                 "weights", "deviance", "null.deviance", "residuals", "rank", "df.residual",
                 "df.null", "iter"))
        if (!is.null(pois[[nm]])) out[[nm]] <- pois[[nm]]
    names(out$coefficients) <- names(out$se) <- names(nb$coefficients)
    dimnames(out$cov.unscaled) <- dimnames(nb$cov.unscaled)
    out$family <- poisson(link = link)
    out$theta <- Inf; out$SE.theta <- NA_real_; out$theta.boundary <- TRUE
    out$twologlik <- 2 * pois_ll
    out$aic <- -2 * pois_ll + 2 * (out$rank + 1L)
    p <- length(out$coefficients)
    full <- matrix(NA_real_, p + 1L, p + 1L)
    full[seq_len(p), seq_len(p)] <- out$cov.unscaled
    dimnames(full) <- rep(list(c(names(out$coefficients), "theta")), 2L)
    out$vcov.full <- full
    out$converged <- isTRUE(pois$converged)
    out$convergence.reason <- "theta_infinite"
    out$outer.decrement <- NA_real_
    warning("the theta estimate is infinite: the log likelihood is highest in the Poisson limit, ",
            "and the returned fit is the Poisson fit (convergence.reason = 'theta_infinite')",
            call. = FALSE)
    out
}

#' @export
logLik.fastglm_nb <- function(object, ...)
{
    df <- object$rank + 1L          # +1 for theta
    structure(object$twologlik / 2, df = df,
              nobs = object$nobs, class = "logLik")
}
