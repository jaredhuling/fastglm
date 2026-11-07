# Map a family object to the integer FamilyCode enum understood by the C++
# solver.  Returns -1L when the (family, link) pair has no native fast path,
# in which case the solver falls back to the per-iteration R callbacks.
family_code <- function(family) {
    if (is.null(family) || is.null(family$family) || is.null(family$link))
        return(-1L)

    if (!.native_family(family)) return(-1L)
    famname <- family$family
    link    <- family$link

    # Quasi-binomial / quasi-poisson share C++ kernels with binomial / poisson.
    # Dispersion handling is done R-side based on family$family.
    if (famname == "quasibinomial") famname <- "binomial"
    if (famname == "quasipoisson")  famname <- "poisson"

    # MASS::negative.binomial(theta) sets family$family to "Negative Binomial(K)".
    if (grepl("^Negative Binomial", famname)) famname <- "negbin"

    # statmod::tweedie() uses a generic "power" link; map to the standard four.
    # var.power / link.power are not direct slots on the family object — they
    # live in the closure environment of family$variance / family$linkfun.
    if (famname == "Tweedie") {
        lp <- tweedie_link_power(family)
        if (is.null(lp) || !is.finite(lp)) lp <- 0
        link <- switch(as.character(lp),
                       "0"   = "log",
                       "1"   = "identity",
                       "-1"  = "inverse",
                       "0.5" = "sqrt",
                       link)
    }

    key <- paste0(famname, ":", link)
    code <- switch(key,
        "gaussian:identity"          = 0L,
        "gaussian:log"               = 1L,
        "gaussian:inverse"           = 2L,
        "binomial:logit"             = 3L,
        "binomial:probit"            = 4L,
        "binomial:cloglog"           = 5L,
        "binomial:log"               = 6L,
        "poisson:log"                = 7L,
        "poisson:identity"           = 8L,
        "poisson:sqrt"               = 9L,
        "Gamma:log"                  = 10L,
        "Gamma:inverse"              = 11L,
        "Gamma:identity"             = 12L,
        "inverse.gaussian:1/mu^2"    = 13L,
        "inverse.gaussian:log"       = 14L,
        "inverse.gaussian:identity"  = 15L,
        "inverse.gaussian:inverse"   = 16L,
        "negbin:log"                 = 17L,
        "negbin:sqrt"                = 18L,
        "negbin:identity"            = 19L,
        "Tweedie:log"                = 20L,
        "Tweedie:identity"           = 21L,
        "Tweedie:inverse"            = 22L,
        "Tweedie:sqrt"               = 23L,
        -1L)
    code
}

# Extract parameters needed by params-aware families.  Returns a length-3
# numeric vector (theta, var.power, link.power); inert defaults are used
# for params-free families.
family_params <- function(family) {
    out <- c(theta = 1.0, var_power = 0.0, link_power = 0.0)
    if (is.null(family) || is.null(family$family)) return(out)

    fam <- family$family

    # Negative binomial: theta is encoded in the family$family string,
    # e.g. "Negative Binomial(2.34)".  family$theta is also set by
    # MASS::negative.binomial() / fastglm's negbin().
    if (grepl("^Negative Binomial", fam)) {
        if (!is.null(family$theta) && is.finite(family$theta) && family$theta > 0) {
            out["theta"] <- family$theta
        } else {
            e <- environment(family$variance)
            if (!is.null(e) && exists(".Theta", e, inherits = FALSE))
                out["theta"] <- get(".Theta", e)
            else if (!is.null(e) && exists("theta", e, inherits = FALSE))
                out["theta"] <- get("theta", e)
            else stop("negative-binomial family must retain its exact theta parameter")
        }
    }

    if (fam == "Tweedie") {
        vp <- tweedie_var_power(family)
        if (is.numeric(vp) && length(vp) == 1L && is.finite(vp))
            out["var_power"]  <- vp
        lp <- tweedie_link_power(family)
        if (is.numeric(lp) && length(lp) == 1L && is.finite(lp))
            out["link_power"] <- lp
    }

    out
}

# statmod::tweedie() hides var.power / link.power inside the closure
# environment of family$variance / family$linkfun.  Extract them defensively;
# fall back to parsing the link string ("mu^0" -> 0).
tweedie_var_power <- function(family) {
    if (!is.null(family$var.power))
        return(family$var.power)
    e <- tryCatch(environment(family$variance), error = function(e) NULL)
    if (!is.null(e) && exists("var.power", envir = e, inherits = FALSE))
        return(get("var.power", envir = e, inherits = FALSE))
    if (!is.null(e) && exists("p", envir = e, inherits = FALSE))
        return(get("p", envir = e, inherits = FALSE))
    NA_real_
}

tweedie_link_power <- function(family) {
    if (!is.null(family$link.power))
        return(family$link.power)
    e <- tryCatch(environment(family$linkfun), error = function(e) NULL)
    if (!is.null(e) && exists("link.power", envir = e, inherits = FALSE))
        return(get("link.power", envir = e, inherits = FALSE))
    # Fallback: parse the link string.  statmod sets it to e.g. "mu^0".
    if (!is.null(family$link) && grepl("^mu\\^", family$link)) {
        v <- suppressWarnings(as.numeric(sub("^mu\\^", "", family$link)))
        if (is.finite(v)) return(v)
    }
    NA_real_
}

#' Fast generalized linear model fitting
#'
#' @param x input model matrix. Must be a matrix object
#' @param y numeric response vector of length nobs.
#' @param family a description of the error distribution and link function to be used in the model.
#' For \code{fastglmPure} this can only be the result of a call to a family function.
#' (See \code{\link[stats]{family}} for details of family functions.)
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be a numeric vector.
#' @param offset this can be used to specify an a priori known component to be included in the linear predictor during fitting.
#' This should be a numeric vector of length equal to the number of cases
#' @param start starting values for the parameters in the linear predictor.
#' @param etastart starting values for the linear predictor.
#' @param mustart values for the vector of means.
#' @param method a decomposition name (`qr`, `qr_unpivoted`, `llt`, `ldlt`,
#'   `qr_full`, `svd`, or `auto`) or an integer scalar with value 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition,
#' 2 for the LLT Cholesky, 3 for the LDLT Cholesky, 4 for the full pivoted QR decomposition, 5 for the Bidiagonal Divide and
#' Conquer SVD. For a dense design whose Cholesky factorization is singular or
#' ill-conditioned, methods 2 and 3 refit with the column-pivoted QR decomposition
#' and warn; `auto` does so silently.
#' @param tol threshold tolerance for convergence. Should be a positive real number
#' @param maxit maximum number of IRLS iterations. Should be an integer
#' @param inference retain full covariance, diagonal standard errors, or no inference.
#'   The no-inference option skips the final covariance calculation.
#' @param firth logical; if `TRUE` apply the Kosmidis--Firth mean
#'   bias-reducing adjusted score (AS_mean) to the IRLS iteration.  The
#'   adjustment modifies each working response by
#'   \eqn{\xi_i = \varphi \, h_i \, (d^2\mu/d\eta^2)_i \, V(\mu_i) /
#'   (2\, w_i\, (d\mu/d\eta)_i^3)}{xi_i = phi * h_i * d2mu * V(mu) /
#'   (2 * pw * dmu^3)}, where \eqn{\varphi}{phi} is the dispersion
#'   (1 for binomial/Poisson, estimated iteratively for other families).
#'   For `binomial(logit)` this reduces to the familiar
#'   \eqn{h_i (0.5 - \mu_i) / (\mu_i (1 - \mu_i))}.  Supported for all
#'   supported unmodified likelihood families on dense, sparse (`dgCMatrix`), and
#'   `big.matrix` designs, and with any decomposition method.
#'   Sparse and `big.matrix` paths use lagged leverages from the
#'   previous IRLS iteration; at convergence the result is identical
#'   to the exact-leverage dense path.  Convergence is checked on the
#'   sup-norm of the coefficient update. Gamma-log fits also check the remaining
#'   adjusted-score update, so a small step caused by step halving does not
#'   suffice for convergence.
#' @return A list with the elements
#' \item{coefficients}{a vector of coefficients}
#' \item{se}{a vector of the standard errors of the coefficient estimates}
#' \item{rank}{a scalar denoting the computed rank of the model matrix}
#' \item{df.residual}{a scalar denoting the degrees of freedom in the model}
#' \item{residuals}{the vector of residuals}
#' \item{s}{a numeric scalar - the root mean square for residuals}
#' \item{fitted.values}{the vector of fitted values}
#' @seealso [fastglm_fit()]
#' @export
#' @examples
#'
#' set.seed(1)
#' x <- matrix(rnorm(1000 * 25), ncol = 25)
#' eta <- 0.1 + 0.25 * x[,1] - 0.25 * x[,3] + 0.75 * x[,5] -0.35 * x[,6] #0.25 * x[,1] - 0.25 * x[,3]
#' y <- 1 * (eta > rnorm(1000))
#'
#' yp <- rpois(1000, eta ^ 2)
#' yg <- rgamma(1000, exp(eta) * 1.75, 1.75)
#'
#' # binomial
#' system.time(gl1 <- glm.fit(x, y, family = binomial()))
#'
#' system.time(gf1 <- fastglmPure(x, y, family = binomial(), tol = 1e-8))
#'
#' system.time(gf2 <- fastglmPure(x, y, family = binomial(), method = 1, tol = 1e-8))
#'
#' system.time(gf3 <- fastglmPure(x, y, family = binomial(), method = 2, tol = 1e-8))
#'
#' system.time(gf4 <- fastglmPure(x, y, family = binomial(), method = 3, tol = 1e-8))
#'
#' max(abs(coef(gl1) - gf1$coef))
#' max(abs(coef(gl1) - gf2$coef))
#' max(abs(coef(gl1) - gf3$coef))
#' max(abs(coef(gl1) - gf4$coef))
#'
#' # poisson
#' system.time(gl1 <- glm.fit(x, yp, family = poisson(link = "log")))
#'
#' system.time(gf1 <- fastglmPure(x, yp, family = poisson(link = "log"), tol = 1e-8))
#'
#' system.time(gf2 <- fastglmPure(x, yp, family = poisson(link = "log"), method = 1, tol = 1e-8))
#'
#' system.time(gf3 <- fastglmPure(x, yp, family = poisson(link = "log"), method = 2, tol = 1e-8))
#'
#' system.time(gf4 <- fastglmPure(x, yp, family = poisson(link = "log"), method = 3, tol = 1e-8))
#'
#' max(abs(coef(gl1) - gf1$coef))
#' max(abs(coef(gl1) - gf2$coef))
#' max(abs(coef(gl1) - gf3$coef))
#' max(abs(coef(gl1) - gf4$coef))
#'
#' # gamma
#' system.time(gl1 <- glm.fit(x, yg, family = Gamma(link = "log")))
#'
#' system.time(gf1 <- fastglmPure(x, yg, family = Gamma(link = "log"), tol = 1e-8))
#'
#' system.time(gf2 <- fastglmPure(x, yg, family = Gamma(link = "log"), method = 1, tol = 1e-8))
#'
#' system.time(gf3 <- fastglmPure(x, yg, family = Gamma(link = "log"), method = 2, tol = 1e-8))
#'
#' system.time(gf4 <- fastglmPure(x, yg, family = Gamma(link = "log"), method = 3, tol = 1e-8))
#'
#' max(abs(coef(gl1) - gf1$coef))
#' max(abs(coef(gl1) - gf2$coef))
#' max(abs(coef(gl1) - gf3$coef))
#' max(abs(coef(gl1) - gf4$coef))
#'
fastglmPure <- function(x, y,
                        family   = gaussian(),
                        weights  = rep(1, NROW(y)),
                        offset   = rep(0, NROW(y)),
                        start    = NULL,
                        etastart = NULL,
                        mustart  = NULL,
                        method   = 0L,
                        tol      = 1e-7,
                        maxit    = 100L,
                        firth    = FALSE,
                        inference = c("full", "diagonal", "none"))
{
    if (!is.logical(firth) || length(firth) != 1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE.", call. = FALSE)
    inference <- match.arg(inference)
    if(is.matrix(x) && is.integer(x)) storage.mode(x)<-"double"
    auto <- identical(method, "auto")
    original.x <- x
    scaling <- NULL
    if (auto && is.matrix(x) && nrow(x) > 0L && ncol(x) > 0L && is.numeric(x) && all_finite_dense(x)) {
        scaling <- as.numeric(pmax(abs(colMax_dense(x)), abs(colMin_dense(x))))
        scaling[scaling == 0] <- 1
        x <- sweep(x, 2L, scaling, "/")
        if (!is.null(start)) start <- start * scaling
    }
    method <- .solver_method(method, x)
    .check_iterations(tol, maxit)
    .validate_glm_data(x, y, weights, offset)
    weights <- as.vector(weights)
    offset  <- as.vector(offset)

    is_sparse_matrix <- inherits(x, "dgCMatrix")
    if (is_sparse_matrix)
    {
        is_big_matrix <- FALSE
        if (method != 2 & method != 3)
        {
            stop("for sparse (dgCMatrix) objects, 'method' must be 2 (LLT) or 3 (LDLT). ",
                 "QR / SVD on sparse matrices is not supported by this package.")
        }
    } else if (is.big.matrix(x))
    {
        is_big_matrix <- TRUE
        if (method != 2 & method != 3)
        {
            stop("for big.matrix objects, 'method' must be 2 (LLT) or 3 (LDLT). ",
                 "QR / SVD methods would force the matrix to be fully read into RAM, ",
                 "defeating the purpose of bigmemory.")
        }
    } else if (is.matrix(x))
    {
        is_big_matrix <- FALSE
    } else
    {
        stop("x must be a matrix, a big.matrix object, or a Matrix::dgCMatrix")
    }


    stopifnot(is.numeric(y) || is.factor(y),
              is.numeric(weights),
              is.numeric(offset),
              NROW(y) == nrow(x),
              NROW(y) == NROW(weights),
              NROW(y) == NROW(offset),
              is.numeric(method),
              is.numeric(tol),
              is.numeric(maxit),
              tol[1] > 0,
              maxit[1] > 0
              )

    if (ncol(x) == 0L) {
        res <- stats::glm.fit(as.matrix(x), y, weights = weights, offset = offset,
                             family = family, control = stats::glm.control(epsilon = tol, maxit = maxit))
        res$cov.unscaled <- matrix(numeric(), 0, 0)
        res$se <- numeric(); res$x <- x
        res$n <- if (is.matrix(y)) rowSums(y) else rep(1, NROW(y))
        res$intercept <- FALSE; res$inference <- inference; res$solver <- "empty"
        return(res)
    }
    nobs  <- n <- NROW(y)
    nvars <- NCOL(x)
    if(is.null(family$family))
    {
        print(family)
        stop("'family' not recognized")
    }

    if( any(weights < 0) ) stop("negative weights not allowed")

    if (method[1] > 5L || method[1] < 0)
    {
        stop("Invalid decomposition method specified. Choose from 0, 1, 2, 3, 4, or 5.")
    }

    cnames <- colnames(x)

    # from glm
    variance    <- family$variance
    dev.resids  <- family$dev.resids
    aic         <- family$aic
    linkinv     <- family$linkinv
    mu.eta      <- family$mu.eta

    unless.null <- function(x, if.null) if(is.null(x)) if.null else x
    valideta    <- unless.null(family$valideta, function(eta) TRUE)
    validmu     <- unless.null(family$validmu,  function(mu)  TRUE)


    if(is.null(mustart))
    {
        ## calculates mustart and may change y and weights and set n (!)
        eval(family$initialize)
    } else
    {
        mukeep <- mustart
        eval(family$initialize)
        mustart <- mukeep
    }

    y <- as.numeric(y)

    coefold <- NULL
    eta <-
        if(!is.null(etastart)) {
            etastart
        } else if(!is.null(start))
            {
                if (length(start) != nvars)
                {
                    stop(gettextf("length of 'start' should equal %d", nvars),
                         domain = NA)
                } else
                {
                    coefold <- start
                    offset + as.vector(x %*% start)
                }
            } else family$linkfun(mustart)
    mu <- linkinv(eta)

    if (!(validmu(mu) && valideta(eta)))
        stop("cannot find valid starting values: please specify some", call. = FALSE)

    if (is.null(start)) start <- rep(0, nvars)

    fc <- family_code(family)
    fp <- family_params(family)
    if (firth && fc %in% c(0:2,10:16) && !is.null(family$dispersion) && is.finite(family$dispersion))
        stop("mean bias reduction with an externally fixed dispersion is not implemented for this family", call. = FALSE)
    if (firth && (fc < 0L || fc >= 20L || grepl("^quasi", family$family)))
        stop("mean bias reduction currently requires an unmodified binomial, Poisson, Gaussian, Gamma, inverse Gaussian, or fixed-theta negative-binomial family", call. = FALSE)

    res <- tryCatch({
    if (is_sparse_matrix)
    {
        res <- fit_sparse_glm(x, drop(y), drop(weights), drop(offset),
                              drop(start), drop(mu), drop(eta),
                              family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                              family$valideta, family$validmu,
                              as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                              as.integer(fc), fp, firth = firth, inference = inference != "none")
        res$intercept <- any(is.int <- .intercept_columns(x))
    } else if (!is_big_matrix && firth)
    {
        res <- fit_glm_firth(x, drop(y), drop(weights), drop(offset),
                             drop(start), drop(mu), drop(eta),
                             family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                             family$valideta, family$validmu,
                             as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                             as.integer(fc), fp, inference = inference != "none")
        res$intercept <- any(is.int <- .intercept_columns(x))
    } else if (!is_big_matrix)
    {
        res <- fit_glm(x, drop(y), drop(weights), drop(offset),
                       drop(start), drop(mu), drop(eta),
                       family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                       family$valideta, family$validmu,
                       as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                       as.integer(fc), fp, inference = inference != "none")

        res$intercept <- any(is.int <- .intercept_columns(x))
    } else
    {
        res <- fit_big_glm(x@address, drop(y), drop(weights), drop(offset),
                           drop(start), drop(mu), drop(eta),
                           family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                           family$valideta, family$validmu,
                           as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                           as.integer(fc), fp, firth = firth, inference = inference != "none")

        res$intercept <- any(is.int <- .intercept_columns(x))
    }

    res
    }, error = function(e) {
        if (is.matrix(x) && grepl("Cholesky factorization", conditionMessage(e), fixed = TRUE)) {
            if (!auto)
                warning("Cholesky factorization is singular or ill-conditioned; ",
                        "the model was fit with pivoted QR (method = 0) instead.", call. = FALSE)
            method <<- 0L
            is.int <<- .intercept_columns(x)
            ans <- if (firth) fit_glm_firth(x, y, weights, offset, start, mu, eta,
                family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                valideta, validmu, 0L, tol, as.integer(maxit), fc, fp, inference = inference != "none") else
                fit_glm(x, y, weights, offset, start, mu, eta,
                family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                valideta, validmu, 0L, tol, as.integer(maxit), fc, fp, inference = inference != "none")
            ans$intercept <- any(is.int)
            ans$solver.fallback <- conditionMessage(e)
            return(ans)
        }
        stop(e)
    })

    if (!res$converged)
    {
        warning("fit_glm: algorithm did not converge", call. = FALSE)
    }

    eps <- 10*.Machine$double.eps
    if (family$family == "binomial")
    {
        if (any(res$fitted.values > 1 - eps) || any(res$fitted.values < eps))
            warning("fit_glm: fitted probabilities numerically 0 or 1 occurred", call. = FALSE)
    }
    if (family$family == "poisson")
    {
        if (any(res$fitted.values < eps))
            warning("fit_glm: fitted rates numerically 0 occurred", call. = FALSE)
    }

    if (is.null(cnames)) {
        cnames <- paste0("X", seq_len(ncol(x)))
        cnames[is.int] <- if (sum(is.int) == 1L) "(Intercept)" else paste0("(Intercept)", seq_len(sum(is.int)))
    }
    names(res$coefficients) <- cnames
    res$solver <- c("qr", "qr_unpivoted", "llt", "ldlt", "qr_full", "svd")[method + 1L]
    res$family <- family
    res$prior.weights <- weights
    res$y <- y
    res$n <- n
    if (!is.null(scaling)) {
        res$coefficients <- res$coefficients / scaling
        if (!is.null(res$adjusted.score)) res$adjusted.score <- res$adjusted.score * scaling
        if (!is.null(res$adjusted.information.diagonal))
            res$adjusted.information.diagonal <- res$adjusted.information.diagonal * scaling^2
        if (length(res$cov.unscaled)) res$cov.unscaled <- res$cov.unscaled / outer(scaling, scaling)
        res$se <- res$se / scaling
        if (!is.null(res$log.det.XtWX)) res$log.det.XtWX <- res$log.det.XtWX + 2 * sum(log(scaling))
        if (!is.null(res$penalized.deviance)) res$penalized.deviance <- res$penalized.deviance - 2 * sum(log(scaling))
        res$column.scale <- scaling
    }
    res$inference <- inference
    res$x <- original.x
    res
}

#' Fast generalized linear model fitting
#'
#' @param x input model matrix. Must be a matrix object
#' @param y numeric response vector of length nobs.
#' @param family a description of the error distribution and link function to be used in the model.
#' For \code{fastglm} this can be a character string naming a family function, a family function or the
#' result of a call to a family function. For \code{fastglmPure} only the third option is supported.
#' (See \code{\link[stats]{family}} for details of family functions.)
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be a numeric vector.
#' @param offset this can be used to specify an a priori known component to be included in the linear predictor during fitting.
#' This should be a numeric vector of length equal to the number of cases
#' @param start starting values for the parameters in the linear predictor.
#' @param etastart starting values for the linear predictor.
#' @param mustart values for the vector of means.
#' @param method a decomposition name (`qr`, `qr_unpivoted`, `llt`, `ldlt`,
#'   `qr_full`, `svd`, or `auto`) or an integer scalar with value 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition,
#' 2 for the LLT Cholesky, or 3 for the LDLT Cholesky. For a dense design whose
#' Cholesky factorization is singular or ill-conditioned, methods 2 and 3 refit
#' with the column-pivoted QR decomposition and warn; `auto` does so silently.
#' @param tol threshold tolerance for convergence. Should be a positive real number
#' @param maxit maximum number of IRLS iterations. Should be an integer
#' @param inference retain full covariance, diagonal standard errors, or no inference.
#'   The no-inference option skips the final covariance calculation.
#' @param firth logical; if `TRUE` apply the Kosmidis--Firth mean
#'   bias-reducing adjusted score. Supported for unmodified binomial, Poisson,
#'   Gaussian, Gamma, inverse Gaussian, and fixed-theta negative-binomial families
#'   on dense, sparse, and `big.matrix` designs, and with any decomposition
#'   method.
#' @return A list with the elements
#' \item{coefficients}{a vector of coefficients}
#' \item{se}{a vector of the standard errors of the coefficient estimates}
#' \item{rank}{a scalar denoting the computed rank of the model matrix}
#' \item{df.residual}{a scalar denoting the degrees of freedom in the model}
#' \item{residuals}{the vector of residuals}
#' \item{s}{a numeric scalar - the root mean square for residuals}
#' \item{fitted.values}{the vector of fitted values}
#' @seealso [fastglm_fit()]
#' @export
#' @examples
#'
#' x <- matrix(rnorm(10000 * 100), ncol = 100)
#' y <- 1 * (0.25 * x[,1] - 0.25 * x[,3] > rnorm(10000))
#'
#' system.time(gl1 <- glm.fit(x, y, family = binomial()))
#'
#' system.time(gf1 <- fastglm(x, y, family = binomial()))
#'
#' system.time(gf2 <- fastglm(x, y, family = binomial(), method = 1))
#'
#' system.time(gf3 <- fastglm(x, y, family = binomial(), method = 2))
#'
#' system.time(gf4 <- fastglm(x, y, family = binomial(), method = 3))
#'
#' max(abs(coef(gl1) - gf1$coef))
#' max(abs(coef(gl1) - gf2$coef))
#' max(abs(coef(gl1) - gf3$coef))
#' max(abs(coef(gl1) - gf4$coef))
#'
#'
#' \dontrun{
#' nrows <- 50000
#' ncols <- 50
#' bkFile <- "bigmat2.bk"
#' descFile <- "bigmatk2.desc"
#' bigmat <- filebacked.big.matrix(nrow=nrows, ncol=ncols, type="double",
#'                                 backingfile=bkFile, backingpath=".",
#'                                 descriptorfile=descFile,
#'                                 dimnames=c(NULL,NULL))
#' for (i in 1:ncols) bigmat[,i] = rnorm(nrows)*i
#' y <- 1*(rnorm(nrows) + bigmat[,1] > 0)
#'
#' system.time(gfb1 <- fastglm(bigmat, y, family = binomial(), method = 3))
#' }
#'
fastglm <- function(x, ...)
{
    if (missing(x)) {
        mc <- match.call(expand.dots = TRUE)
        if (is.null(mc$formula)) stop("supply a design matrix or model formula", call. = FALSE)
        mc$x <- mc$formula; mc$formula <- NULL
        mc[[1L]] <- quote(fastglm::fastglm)
        return(eval(mc, parent.frame()))
    }
    UseMethod("fastglm")
}


#' bigLm default
#'
#' @param retain.data retain the design and response for subsequent inference.
#' @param ... additional arguments
#' @rdname fastglm
#' @method fastglm default
#' @exportS3Method fastglm default
fastglm.default <- function(x, y,
                            family = gaussian(),
                            weights = NULL,
                            offset = NULL,
                            start    = NULL,
                            etastart = NULL,
                            mustart  = NULL,
                            method = 0L, tol = 1e-8, maxit = 100L,
                            firth = FALSE,
                            inference = c("full", "diagonal", "none"),
                            retain.data = TRUE,
                            ...)
{
    family <- .resolve_family(family)
    if (is.null(weights)) weights <- rep(1, NROW(y))
    if (is.null(offset)) offset <- rep(0, NROW(y))
    res <- fastglmPure(x, y, family, weights, offset, start, etastart,
                      mustart, method, tol, maxit, firth = firth, inference = match.arg(inference))
    res <- .finish_glm(res, offset)
    if (res$inference != "full") res$cov.unscaled <- NULL
    if (!retain.data) { res$x <- NULL; res$y <- NULL }
    res$call <- match.call()
    class(res) <- if (isTRUE(res$firth)) c("fastglm_firth", "fastglm") else "fastglm"
    res
}
