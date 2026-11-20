# Shared validation and model accounting for all in-memory GLM frontends.
.resolve_family <- function(family, envir = parent.frame()) {
    if (is.character(family)) family <- get(family, mode = "function", envir = envir)
    if (is.function(family)) family <- family()
    if (!inherits(family, "family") || !is.function(family$linkinv) ||
        !is.function(family$variance)) stop("'family' must be a GLM family", call. = FALSE)
    family
}

.solver_method <- function(method, x = NULL) {
    if (is.character(method)) {
        method <- match.arg(method, c("qr", "qr_unpivoted", "llt", "ldlt", "qr_full", "svd", "auto"))
        if (method == "auto") return(2L)
        return(match(method, c("qr", "qr_unpivoted", "llt", "ldlt", "qr_full", "svd")) - 1L)
    }
    if (!is.numeric(method) || length(method) != 1L || is.na(method) || !(method %in% 0:5))
        stop("'method' must name a solver or be an integer between 0 and 5", call. = FALSE)
    as.integer(method)
}

.check_iterations <- function(tol, maxit) {
    if (length(tol) != 1L || !is.numeric(tol) || !is.finite(tol) || tol <= 0)
        stop("'tol' must be a positive finite scalar", call. = FALSE)
    if (length(maxit) != 1L || !is.numeric(maxit) || !is.finite(maxit) ||
        maxit < 1 || maxit != as.integer(maxit))
        stop("'maxit' must be a positive integer", call. = FALSE)
}

.validate_glm_data <- function(x, y, weights, offset) {
    n <- NROW(y)
    if (n < 1L || nrow(x) != n) stop("'x' and 'y' must have the same positive number of rows", call. = FALSE)
    if (!is.numeric(y) && !is.factor(y)) stop("'y' must be numeric or a binomial factor", call. = FALSE)
    if (is.factor(y) && anyNA(y)) stop("non-finite response", call. = FALSE)
    for (nm in c("weights", "offset")) {
        z <- get(nm)
        if (!is.numeric(z) || length(z) != n)
            stop(sprintf("'%s' must contain one finite number per observation", nm), call. = FALSE)
    }
    # Finiteness and weight checks in one pass (src/validate_glm.cpp).
    status <- validate_glm_vectors(if (is.numeric(y)) as.double(y) else numeric(0),
                                   as.double(weights), as.double(offset))
    if (status[1L] == 1) stop("non-finite response", call. = FALSE)
    if (status[1L] == 2) stop("'weights' must contain one finite number per observation", call. = FALSE)
    if (status[1L] == 3) stop("'offset' must contain one finite number per observation", call. = FALSE)
    if (status[1L] == 4) stop("weights must be nonnegative with positive total weight", call. = FALSE)
    if (is.matrix(x) && (!is.numeric(x) || !all_finite_dense(x))) stop("'x' must contain finite numeric values", call. = FALSE)
    values <- if (inherits(x, "sparseMatrix")) x@x else NULL
    if (!is.null(values) && (!is.numeric(values) || any(!is.finite(values))))
        stop("'x' must contain finite numeric values", call. = FALSE)
    if (ncol(x) > status[2L])
        stop("more coefficients than positive-weight observations", call. = FALSE)
}

.intercept_columns <- function(x) {
    if (inherits(x, "dgCMatrix")) {
        # A nonzero constant sparse column must store every row. No dense copy.
        return(vapply(seq_len(ncol(x)), function(j) {
            lo <- x@p[j] + 1L; hi <- x@p[j + 1L]
            if (hi - lo + 1L != nrow(x)) return(FALSE)
            z <- x@x[lo:hi]
            z[1L] != 0 && all(z == z[1L])
        }, logical(1)))
    }
    if (inherits(x, "big.matrix")) {
        hi <- big.colMax(x); lo <- big.colMin(x)
    } else {
        return(constant_columns_dense(x))
    }
    hi == lo & hi != 0
}

.fixed_dispersion <- function(family) {
    if (!is.null(family$dispersion) && is.finite(family$dispersion)) return(family$dispersion)
    if (family$family %in% c("poisson", "binomial") || startsWith(family$family, "Negative Binomial")) return(1)
    NA_real_
}

# Log of sum(w*y*exp(-offset)), optionally combined with preceding blocks.
# Evaluating the sum on the log scale also handles very small Gamma responses.
.gamma_log_null_total <- function(y,weights,offset,previous=-Inf) {
    keep <- weights>0
    terms <- c(previous,log(weights[keep])+log(y[keep])-offset[keep])
    largest <- max(terms)
    if(!is.finite(largest)) return(largest)
    largest+log(sum(exp(terms-largest)))
}

.finish_glm <- function(res, offset, intercept = res$intercept, null = TRUE, aic = TRUE) {
    fam <- res$family; y <- res$y; w <- res$prior.weights
    # Family code supplied by fastglmPure() (attribute removed below); NULL
    # for other callers, which use the R path throughout.
    fc <- attr(res, "family_code"); attr(res, "family_code") <- NULL
    positive <- w > 0
    if (anyNA(res$coefficients) && !is.null(res$x)) {
        good <- !is.na(res$coefficients)
        xx <- as.matrix(res$x[positive, , drop = FALSE])
        scales <- apply(abs(xx[, good, drop = FALSE]), 2L, max)
        scales[scales == 0] <- 1
        relation <- qr.coef(qr(sweep(xx[, good, drop = FALSE], 2L, scales, "/"), tol = 1e-12), xx[, !good, drop = FALSE])
        res$alias.relation <- relation / scales
    }
    # As in earlier releases, a fit without an offset stores NULL. Callers
    # use a non-NULL offset to detect that the model was fit with one.
    res$offset <- if (length(offset) && any(offset != 0)) offset else NULL
    res$intercept <- intercept
    res$nobs <- sum(positive)
    res$df.residual <- res$nobs - res$rank
    res$df.null <- res$nobs - as.integer(intercept)
    binomial.code <- .stable_binomial_code(res)
    # Natively evaluated families: residuals, working weights, Pearson
    # statistic, and null deviance in one C++ pass (src/finish_glm.cpp).
    varying.offset <- intercept && any(offset != offset[1L])
    native <- !is.null(fc) && fc >= 0L && binomial.code < 0L && is.numeric(y) && is.null(dim(y)) &&
        length(res$linear.predictors) == length(y)
    if (native) {
        null.mean <- if (intercept) sum(w * y) / sum(w) else NA_real_
        fast <- finish_glm_native(fc, as.numeric(family_params(fam)), as.numeric(y), res$fitted.values,
                                  res$linear.predictors, as.numeric(w), as.numeric(offset),
                                  null.mean, identical(res$inference, "none"))
    }
    if(binomial.code>=0L) {
        moments <- binomial_firth_summary(res$linear.predictors,as.numeric(y),w,res$n,binomial.code)
        res$weights <- moments$working
        res$residuals <- moments$residuals
        names(res$residuals) <- names(y)
        res$deviance <- moments$deviance
    } else if (native) {
        res$residuals <- fast$residuals
        names(res$residuals) <- names(y)
        if (identical(res$inference, "none")) res$weights <- fast$weights
        res$weights[!positive] <- 0
    } else {
        derivative <- fam$mu.eta(res$linear.predictors)
        res$residuals <- (y - res$fitted.values) / derivative
        res$residuals[!positive] <- 0
        if (identical(res$inference, "none"))
            res$weights <- w * derivative^2 / fam$variance(res$fitted.values)
        res$weights[!positive] <- 0
    }
    disp <- .fixed_dispersion(fam)
    res$estimated.dispersion <- is.na(disp)
    if (is.na(disp)) disp <- if (res$df.residual > 0) {
        if (native) fast$pearson / res$df.residual
        else sum(w * (y - res$fitted.values)^2 / fam$variance(res$fitted.values)) / res$df.residual
    } else NaN
    if (isTRUE(res$firth) && !is.null(res$bias.dispersion)) disp <- res$bias.dispersion
    res$dispersion <- disp
    if (length(res$cov.unscaled)) {
        res$se <- sqrt(diag(res$cov.unscaled) * disp)
        names(res$se) <- names(res$coefficients)
        dimnames(res$cov.unscaled) <- rep(list(names(res$coefficients)), 2L)
    } else { res$cov.unscaled <- NULL; res$se <- NULL }
    if (null) {
        if (intercept && any(offset != offset[1L])) {
            if(family_code(fam)==10L) {
                null.intercept <- .gamma_log_null_total(y,w,offset)-log(sum(w))
                res$null.deviance <- sum(fam$dev.resids(y,exp(offset+null.intercept),w))
            } else {
                nf <- .null_glm_fit(y, w, offset, fam)
                res$null.deviance <- nf$deviance
            }
        } else if (native) {
            res$null.deviance <- fast$null.deviance
        } else {
            mu0 <- if (intercept) sum(w * y) / sum(w) else fam$linkinv(offset)
            res$null.deviance <- sum(fam$dev.resids(y, mu0, w))
        }
    }
    # aic = FALSE: the caller sets res$aic itself (fastglm_nb() does, from
    # the negative binomial log likelihood).
    if (!aic) {
    } else if(binomial.code>=0L) {
        res$aic <- moments$aic+2*res$rank
    } else if (all(positive)) {
        res$aic <- .family_aic(fam,y, res$n, res$fitted.values, w, res$deviance) + 2 * res$rank
    } else {
        trials <- if (length(res$n) == length(y)) res$n[positive] else res$n
        res$aic <- .family_aic(fam,y[positive], trials, res$fitted.values[positive], w[positive], res$deviance) + 2 * res$rank
    }
    if (isTRUE(res$firth) && !(family_code(fam) %in% c(0L,3L,7L)))
        res$penalized.deviance <- NULL
    res$boundary <- any(!is.finite(res$coefficients[!is.na(res$coefficients)])) ||
        (fam$family == "binomial" && any(res$fitted.values < 10 * .Machine$double.eps | res$fitted.values > 1 - 10 * .Machine$double.eps))
    res
}

# Evaluate standard NB likelihoods without cancelling large log-gamma terms.
# User-supplied AIC functions keep their original semantics.
.family_aic <- function(family,y,n,mu,weights,dev) {
    if(startsWith(family$family,"Negative Binomial")) {
        theta <- unname(family_params(family)["theta"])
        ref <- tryCatch(negbin(theta,family$link),error=function(e) NULL)
        standard <- !is.null(ref) && .same_kernel(family$aic,ref$aic)
        if(!standard && requireNamespace("MASS",quietly=TRUE)) {
            ref <- tryCatch(MASS::negative.binomial(theta,family$link),error=function(e) NULL)
            standard <- !is.null(ref) && .same_kernel(family$aic,ref$aic)
        }
        if(standard) return(-2*sum(weights*dnbinom(y,mu=mu,size=theta,log=TRUE)))
    }
    code <- .standard_aic_code(family)
    if (code > 0L)
        return(family_aic_cpp(code, as.numeric(y), as.numeric(rep_len(n, length(y))),
                              as.numeric(mu), as.numeric(weights), as.numeric(dev)))
    family$aic(y,n,mu,weights,dev)
}

# AIC kernels of the unmodified stats families, computed in C++ by
# family_aic_cpp() with the same densities and expressions. A family whose
# aic function differs from the stats definition keeps its R function.
.aic_reference <- local({
    bodies <- NULL
    function() {
        if (is.null(bodies)) bodies <<- list(binomial = body(stats::binomial()$aic),
                                             poisson = body(stats::poisson()$aic),
                                             Gamma = body(stats::Gamma()$aic))
        bodies
    }
})
.standard_aic_code <- function(family) {
    code <- switch(family$family, binomial = 1L, poisson = 2L, Gamma = 3L, 0L)
    if (code == 0L || !is.function(family$aic)) return(0L)
    if (identical(body(family$aic), .aic_reference()[[family$family]])) code else 0L
}

# Compare kernel definitions and the closure variables they actually reference.
# A family with modified kernels must retain R callback semantics.
.same_kernel <- function(f, g) {
    if (!is.function(f) || !is.function(g)) return(identical(f, g))
    if (!identical(body(f), body(g)) || !identical(formals(f), formals(g))) return(FALSE)
    vars <- setdiff(all.vars(body(f)), names(formals(f)))
    all(vapply(vars, function(v) {
        ef <- environment(f); eg <- environment(g)
        if (!exists(v, ef, inherits = TRUE) || !exists(v, eg, inherits = TRUE)) return(TRUE)
        identical(get(v, ef, inherits = TRUE), get(v, eg, inherits = TRUE))
    }, logical(1)))
}

.native_family <- function(family) {
    name <- family$family
    ref <- tryCatch({
        if (name %in% c("gaussian", "binomial", "quasibinomial", "poisson", "quasipoisson", "Gamma", "inverse.gaussian")) {
            .reference_family(name, family$link)
        } else if (startsWith(name, "Negative Binomial")) {
            theta <- unname(family_params(family)["theta"])
            if (.same_kernel(family$variance, negbin(theta)$variance)) negbin(theta, family$link)
            else if (requireNamespace("MASS", quietly = TRUE)) MASS::negative.binomial(theta, family$link)
        } else if (name == "Tweedie" && requireNamespace("statmod", quietly = TRUE)) {
            statmod::tweedie(var.power = tweedie_var_power(family), link.power = tweedie_link_power(family))
        }
    }, error = function(e) NULL)
    !is.null(ref) && all(vapply(c("linkinv", "mu.eta", "variance", "dev.resids", "validmu", "valideta"),
                               function(nm) .same_kernel(family[[nm]], ref[[nm]]), logical(1)))
}

# Default starting coefficients for fastglm(): the intercept-only fit,
# mu = weighted mean of y, placed on the first intercept column. The IRLS
# loop halves an invalid first step back toward these coefficients, so they
# must give a valid fit. Falls back to zeros (the earlier default) when there
# is no intercept column, the mean is on the boundary of the link's domain,
# or the offsets make the intercept-only fit invalid.
.feasible_start <- function(x, y, weights, offset, family, valideta, validmu, is.int) {
    start <- numeric(ncol(x))
    j <- which(is.int)[1L]
    if (is.na(j) || !any(weights > 0)) return(start)
    b <- suppressWarnings(family$linkfun(sum(weights * y) / sum(weights)))
    cval <- as.numeric(x[1L, j])
    if (length(b) != 1L || !is.finite(b) || !is.finite(cval) || cval == 0) return(start)
    # Without offsets the intercept-only linear predictor is one value, so
    # validity is checked once instead of per observation.
    eta <- if (any(offset != 0)) offset + b else b
    ok <- tryCatch({ mu <- family$linkinv(eta)
                     isTRUE(valideta(eta)) && isTRUE(validmu(mu)) && all(is.finite(mu)) },
                   error = function(e) FALSE)
    if (!ok) return(start)
    start[j] <- b / cval
    start
}

# stats family objects used as references by .native_family(). They do not
# change within a session, so each name and link is constructed once.
.reference_family <- local({
    cache <- new.env(parent = emptyenv())
    function(name, link) {
        key <- paste0(name, ":", link)
        if (is.null(cache[[key]])) cache[[key]] <- get(name, asNamespace("stats"))(link = link)
        cache[[key]]
    }
})

# Convergence rule for the IRLS solvers selected by `covariance`:
# "estimate" (rule 1) or "previous" (rule 2; see inst/include/conv_rule.h).
# The rule is set for the duration of one fit and restored on exit.
.covariance_rule <- function(covariance) {
    covariance <- match.arg(covariance, c("estimate", "previous"))
    if (covariance == "previous") 2L else 1L
}

# Warning for fits that converge with rows held near a bound of the family's
# valid region (see inst/include/boundary_step.h). The score is not zero at
# such an estimate, so the usual Wald inference does not apply.
BOUNDARY_WARNING <- paste("the estimate is on the boundary of the valid region for this family and link",
                          "(convergence.reason = 'converged_boundary');",
                          "Wald standard errors and tests are not reliable")

# Intercept-only start for a streamed fit with an offset: an intercept b0
# with b0 + o valid for every offset o in [off.lo, off.hi]. The valid linear
# predictors of each family form an interval, so checking the two ends
# suffices. Candidates centre linkfun(ybar) on the mean, the midpoint, and
# each end of the offset range; NULL if none is valid.
.feasible_start_offset <- function(low, is.int, ybar, off.lo, off.hi, off.mean, family) {
    j <- which(is.int)[1L]
    if (is.na(j) || !is.finite(off.lo) || !is.finite(off.hi)) return(NULL)
    cval <- as.numeric(low[j])
    c0 <- suppressWarnings(family$linkfun(ybar))
    if (!is.finite(c0) || !is.finite(cval) || cval == 0) return(NULL)
    valideta <- if (is.null(family$valideta)) function(x) TRUE else family$valideta
    validmu <- if (is.null(family$validmu)) function(x) TRUE else family$validmu
    ok <- function(b0) tryCatch({
        e <- c(b0 + off.lo, b0 + off.hi); m <- family$linkinv(e)
        isTRUE(valideta(e)) && isTRUE(validmu(m)) && all(is.finite(m))
    }, error = function(e) FALSE)
    for (b0 in c0 - c(off.mean, (off.lo + off.hi) / 2, off.lo, off.hi)) if (ok(b0)) {
        start <- numeric(length(low)); start[j] <- b0 / cval
        return(start)
    }
    NULL
}

# The intercept-only fit with offsets, for the null deviance. glm.fit's
# default start (the family's starting means) can be invalid with a bounded
# link and a varying offset (for example binomial("log"), where it can put a
# fitted probability above one, and glm.fit then stops with "no valid set of
# coefficients has been found"). It is then repeated from the first
# intercept that is valid in every row among a few candidates, including
# linkfun(ybar / 2) - max(offset), which is valid for the log link.
.null_glm_fit <- function(y, w, offset, fam) {
    fit <- function(start) stats::glm.fit(matrix(1, length(y), 1L), y, weights = w,
                                          offset = offset, family = fam, start = start,
                                          control = stats::glm.control(maxit = 100))
    first <- tryCatch(fit(NULL), error = function(e) e)
    if (!inherits(first, "error")) return(first)
    ybar <- sum(w * y) / sum(w)
    cand <- suppressWarnings(c(fam$linkfun(ybar) - mean(offset), fam$linkfun(ybar / 2) - max(offset),
                               fam$linkfun(ybar) - max(offset), fam$linkfun(ybar) - min(offset)))
    valideta <- if (is.null(fam$valideta)) function(e) TRUE else fam$valideta
    validmu <- if (is.null(fam$validmu)) function(m) TRUE else fam$validmu
    for (b in cand[is.finite(cand)]) {
        ok <- tryCatch({ eta <- offset + b
                         isTRUE(valideta(eta)) && isTRUE(validmu(fam$linkinv(eta))) },
                       error = function(e) FALSE)
        if (ok) return(fit(b))
    }
    stop(first)
}
