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
    if (anyNA(y) || (is.numeric(y) && any(!is.finite(y)))) stop("non-finite response", call. = FALSE)
    for (nm in c("weights", "offset")) {
        z <- get(nm)
        if (!is.numeric(z) || length(z) != n || any(!is.finite(z)))
            stop(sprintf("'%s' must contain one finite number per observation", nm), call. = FALSE)
    }
    if (any(weights < 0) || !any(weights > 0)) stop("weights must be nonnegative with positive total weight", call. = FALSE)
    if (is.matrix(x) && (!is.numeric(x) || !all_finite_dense(x))) stop("'x' must contain finite numeric values", call. = FALSE)
    values <- if (inherits(x, "sparseMatrix")) x@x else NULL
    if (!is.null(values) && (!is.numeric(values) || any(!is.finite(values))))
        stop("'x' must contain finite numeric values", call. = FALSE)
    if (ncol(x) > sum(weights > 0))
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
    if (family$family %in% c("poisson", "binomial") || grepl("^Negative Binomial", family$family)) return(1)
    NA_real_
}

.finish_glm <- function(res, offset, intercept = res$intercept, null = TRUE) {
    fam <- res$family; y <- res$y; w <- res$prior.weights
    positive <- w > 0
    if (anyNA(res$coefficients) && !is.null(res$x)) {
        good <- !is.na(res$coefficients)
        xx <- as.matrix(res$x[positive, , drop = FALSE])
        scales <- apply(abs(xx[, good, drop = FALSE]), 2L, max)
        scales[scales == 0] <- 1
        relation <- qr.coef(qr(sweep(xx[, good, drop = FALSE], 2L, scales, "/"), tol = 1e-12), xx[, !good, drop = FALSE])
        res$alias.relation <- relation / scales
    }
    res$offset <- offset
    res$intercept <- intercept
    res$nobs <- sum(positive)
    res$df.residual <- res$nobs - res$rank
    res$df.null <- res$nobs - as.integer(intercept)
    derivative <- fam$mu.eta(res$linear.predictors)
    res$residuals <- (y - res$fitted.values) / derivative
    res$residuals[!positive] <- 0
    if (identical(res$inference, "none"))
        res$weights <- w * derivative^2 / fam$variance(res$fitted.values)
    res$weights[!positive] <- 0
    disp <- .fixed_dispersion(fam)
    res$estimated.dispersion <- is.na(disp)
    if (is.na(disp)) disp <- if (res$df.residual > 0) sum(w * (y - res$fitted.values)^2 / fam$variance(res$fitted.values)) / res$df.residual else NaN
    if (isTRUE(res$firth) && !is.null(res$bias.dispersion)) disp <- res$bias.dispersion
    res$dispersion <- disp
    if (length(res$cov.unscaled)) {
        res$se <- sqrt(diag(res$cov.unscaled) * disp)
        names(res$se) <- names(res$coefficients)
        dimnames(res$cov.unscaled) <- rep(list(names(res$coefficients)), 2L)
    } else { res$cov.unscaled <- NULL; res$se <- NULL }
    if (null) {
        if (intercept && any(offset != offset[1L])) {
            nf <- stats::glm.fit(matrix(1, length(y), 1L), y, weights = w,
                                 offset = offset, family = fam,
                                 control = stats::glm.control(maxit = 100))
            res$null.deviance <- nf$deviance
        } else {
            mu0 <- if (intercept) sum(w * y) / sum(w) else fam$linkinv(offset)
            res$null.deviance <- sum(fam$dev.resids(y, mu0, w))
        }
    }
    if (all(positive)) {
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
    family$aic(y,n,mu,weights,dev)
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
            get(name, asNamespace("stats"))(link = family$link)
        } else if (grepl("^Negative Binomial", name)) {
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
