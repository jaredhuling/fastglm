#' Fast generalized linear model fitting
#' 
#' `fastglm.fit()` is a fitting method for [glm()]. It works like `glm.fit()`, i.e., by being supplied to the `method` argument of `glm()`.
#' 
#' @param x a design matrix of dimension `n * p`. Can also be a `big.matrix` object from \pkg{bigmemory}.
#' @param y a vector of observations of length `n`.
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be `NULL` or a numeric vector.
#' @param start optional starting values for the parameters in the linear predictor.
#' @param etastart optional starting values for the linear predictor.
#' @param mustart optional starting values for the vector of means.
#' @param offset this can be used to specify an *a priori* known component to be included in the linear predictor during fitting. This should be `NULL` or a numeric vector of length equal to the number of cases.
#' @param family a description of the error distribution and link function to be used in the model. This must be a family function or the result of a call to a family function. (See [`family`] for details of family functions.)
#' @param control a list of parameters for controlling the fitting process. This is passed to `fastglm.control()`.
#' @param singular.ok,intercept See [glm.fit()].
#' @param fastmethod `integer`; the method used for fitting. Allowable values include 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition, 2 for the LLT Cholesky, 3 for the LDLT Cholesky, 4 for the full pivoted QR decomposition, and 5 for the Bidiagonal Divide and Conquer SVD. Default is 0. Can also be supplied as `method` when not supplied directly as an argument from `glm()` (see Examples).
#' @param tol `numeric`; threshold tolerance for convergence.
#' @param maxit `integer`; the maximum number of IRLS iterations.
#' @param object a `fastglmFit` object; the output of a call to `glm()` with `method = fastglm.fit`.
#' @param refit `logical`; whether to refit the model using `glm()` with `method = "glm.fit"`. If `TRUE`, the model will be refit using the estimated coefficients as starting values for a single IRLS iteration in order to produce the usual coefficient covariance matrix. If `FALSE`, `vcov` will only produce the diagonal of the covariance matrix.
#' @param \dots for `vcov()` and `summary()`, other arguments passed to [vcov.glm()] and [summary.glm()] when `refit = TRUE`.
#' 
#' @details
#' The purpose of the functions documented on this page is to facilitate integration with existing [glm()] utilities in base R. `fastglm.fit()` is just a wrapper for [fastglmPure()] with some additional quality-of-life features. The `vcov()` and `summary()` methods are quick hacks to use the existing architecture for these functions in base R. Because of this, they involve refitting the GLM with the estimated coefficients as starting values.
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
#'                method = fastglm.fit)
#' })
#' 
#' system.time({
#'     gf1 <- glm(y ~ ., data = dat,
#'                family = binomial,
#'                method = fastglm.fit,
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
#'                method = fastglm.fit)
#' })
#' 
#' system.time({
#'     gf1 <- glm(y ~ ., data = dat,
#'                family = poisson,
#'                method = fastglm.fit,
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
#'                method = fastglm.fit)
#' })
#' 
#' system.time({
#'     gf1 <- glm(y ~ ., data = dat,
#'                family = Gamma(link = "log"),
#'                method = fastglm.fit,
#'                fastmethod = 1)
#' })
#' 
#' # Different (equivalent) ways of supplying
#' # control arguments:
#' gf1 <- glm(y ~ ., data = dat,
#'            family = Gamma(link = "log"),
#'            method = fastglm.fit,
#'            fastmethod = 1)
#' 
#' gf1 <- glm(y ~ ., data = dat,
#'            family = Gamma(link = "log"),
#'            method = fastglm.fit,
#'            control = list(fastmethod = 1))
#' 
#' gf1 <- glm(y ~ ., data = dat,
#'            family = Gamma(link = "log"),
#'            method = fastglm.fit,
#'            control = list(method = 1))

#' @export `fastglm.fit`
fastglm.fit <- function(x, y, 
                        weights  = rep(1, NROW(y)), 
                        start    = NULL,
                        etastart = NULL,
                        mustart  = NULL,
                        offset   = rep(0, NROW(y)), 
                        family   = gaussian(),
                        control  = list(),
                        intercept = TRUE,
                        singular.ok = TRUE)
{
    control <- do.call("fastglm.control", control)
    
    if (bigmemory::is.big.matrix(x))
    {
        is_big_matrix <- TRUE
        if (!(control$method %in% c(2, 3)))
        {
            method <- 3L
            warning("for big.matrix objects, 'method' must either be 2 (for LLT) or 3 (for LDLT) -- 'method' changed to 3.")
        }
    } else if (is.matrix(x))
    {
        is_big_matrix <- FALSE
    } else
    {
        stop("x must be either a matrix or a big.matrix object")
    }
    
    xnames <- colnames(x)
    ynames <- if (is.matrix(y)) rownames(y) else names(y)
    
    nobs <- NROW(y)
    nvars <- ncol(x)
    
    if (is.null(weights)) 
        weights <- rep.int(1, nobs)
    else
        weights <- as.vector(weights)
    
    if (any(weights < 0))
        stop("negative weights not allowed")
    
    if (is.null(offset)) 
        offset <- rep.int(0, nobs)
    else
        offset  <- as.vector(offset)
    
    stopifnot(is.numeric(y), 
              is.numeric(weights),
              is.numeric(offset),
              NROW(y) == nrow(x),
              NROW(y) == NROW(weights),
              NROW(y) == NROW(offset)             
    )
    
    variance <- family$variance
    linkinv <- family$linkinv
    
    if (!is.function(variance) || !is.function(linkinv)) 
        stop("'family' argument seems not to be a valid family object", 
             call. = FALSE)
    
    if (is.null(family$family)) 
    {
        print(family)
        stop("'family' not recognized")
    }
    
    # from glm
    dev.resids  <- family$dev.resids
    aic         <- family$aic
    mu.eta      <- family$mu.eta 
    
    unless.null <- function(x, if.null) if (is.null(x)) if.null else x
    valideta    <- unless.null(family$valideta, function(eta) TRUE)
    validmu     <- unless.null(family$validmu,  function(mu)  TRUE)
    
    if (is.null(mustart)) 
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
        if (!is.null(etastart)) {
            etastart
        } else if (!is.null(start))
        {    
            if (length(start) != nvars)
            {
                stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s", 
                              nvars, toString(xnames)), 
                     domain = NA)
            }
            
            coefold <- start
            offset + as.vector(x %*% start)
            
        } else family$linkfun(mustart)
    
    mu <- linkinv(eta)
    
    if (!(validmu(mu) && valideta(eta)))
        stop("cannot find valid starting values: please specify some", call. = FALSE)
    
    if (is.null(start)) start <- rep(0, nvars)
    
    if (!is_big_matrix)
    {
        res <- fit_glm(x, drop(y), drop(weights), drop(offset), 
                       drop(start), drop(mu), drop(eta),
                       variance, mu.eta, linkinv, dev.resids, 
                       valideta, validmu,
                       as.integer(control$method), as.double(control$tol), as.integer(control$maxit))
        
        res$intercept <- any(is.int <- colMax_dense(x) == colMin_dense(x))
    } else
    {
        res <- fit_big_glm(x@address, drop(y), drop(weights), drop(offset), 
                           drop(start), drop(mu), drop(eta),
                           variance, mu.eta, linkinv, dev.resids, 
                           valideta, validmu,
                           as.integer(control$method), as.double(control$tol), as.integer(control$maxit))
        
        res$intercept <- any(is.int <- big.colMax(x) == big.colMin(x))
    }
    
    if (!res$converged)
    {
        warning("fastglm.fit: algorithm did not converge", call. = FALSE)
    }
    
    eps <- 10 * .Machine$double.eps
    if (family$family == "binomial") 
    {
        if (any(res$fitted.values > 1 - eps) || any(res$fitted.values < eps))
            warning("fastglm.fit: fitted probabilities numerically 0 or 1 occurred", call. = FALSE)
    }
    if (family$family == "poisson") 
    {
        if (any(res$fitted.values < eps))
            warning("fastglm.fit: fitted rates numerically 0 occurred", call. = FALSE)
    }
    
    if (is.null(xnames))
    {
        ncx <- ncol(x)
        if (res$intercept)
        {
            which.int <- which(is.int)
            xnames    <- paste0("X", seq_len(ncx - 1L))
            names(res$coefficients) <- seq_len(ncx)
            names(res$coefficients)[-which.int] <- xnames
            names(res$coefficients)[which.int]  <- "(Intercept)"
        } else
        {
            names(res$coefficients) <- paste0("X", seq_len(ncx))
        }
    } else
    {
        names(res$coefficients) <- xnames
    }
    
    wtdmu         <- if (res$intercept) sum(weights * y) / sum(weights) else family$linkinv(offset)

    n.ok          <- nobs - sum(weights == 0)
    
    res$df.null   <- n.ok - as.integer(res$intercept)
    res$null.deviance <- sum(family$dev.resids(y, wtdmu, weights))
    res$family <- family
    res$prior.weights <- weights
    
    res$aic <- aic(y, nobs, res$fitted.values, res$prior.weights, res$deviance) + 2 * res$rank
    
    res$y <- y
    res$n <- nobs
    res$class <- "fastglmFit"
    
    res
}

#' @export `fastglm.control`
#' @rdname fastglm.fit
fastglm.control <- function(fastmethod = 0L, tol = 1e-7, maxit = 100L, ...) {
    if (...length() > 0L && "method" %in% ...names() && identical(fastmethod, 0L)) {
        fastmethod <- ...elt(which(...names() == "method")[1L])
        fastmethod_name <- "method"
    }
    else {
        fastmethod_name <- "fastmethod"
    }
    
    if (!is.numeric(fastmethod) || length(fastmethod) != 1L || !(fastmethod %in% 0:5)) 
        stop(sprintf("'%s' must be an interger between 0 and 5", fastmethod_name))
    
    if (!is.numeric(tol) || length(tol) != 1L || tol <= 0) 
        stop("value of 'tol' must be > 0")
    
    if (!is.numeric(maxit) || length(maxit) != 1L || maxit <= 0) 
        stop("maximum number of iterations must be > 0")
    
    list(method = fastmethod, tol = tol, maxit = maxit)
}

#' @exportS3Method stats::vcov fastglmFit
#' @rdname fastglm.fit
vcov.fastglmFit <- function(object, refit = TRUE, ...) {
    if (isTRUE(refit)) {
        object <- do.call("update", list(object, start = coef(object),
                                         method = "glm.fit",
                                         control = list(maxit = 1L)))
        
        return(vcov(object, ...))
    }
    
    v <- diag(object[["se"]]^2)
    rownames(v) <- colnames(v) <- names(coef(object))
    
    v
}

#' @exportS3Method summary fastglmFit
#' @rdname fastglm.fit
summary.fastglmFit <- function(object, refit = TRUE, ...) {
    if (isTRUE(refit)) {
        call <- getCall(object)
        
        object <- do.call("update", list(object, start = coef(object),
                                         method = "glm.fit",
                                         control = list(maxit = 1L)))
        
        object$call <- call
        
        return(summary(object, ...))
    }
    
    summary.fastglm(object, ...)
}