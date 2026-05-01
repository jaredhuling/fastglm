# Map a family object to the integer FamilyCode enum understood by the C++
# solver.  Returns -1L when the (family, link) pair has no native fast path,
# in which case the solver falls back to the per-iteration R callbacks.
family_code <- function(family) {
    if (is.null(family) || is.null(family$family) || is.null(family$link))
        return(-1L)

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
            m <- regmatches(fam, regexec("\\(([^)]+)\\)", fam))[[1]]
            if (length(m) >= 2L) {
                v <- suppressWarnings(as.numeric(m[2]))
                if (is.finite(v) && v > 0) out["theta"] <- v
            }
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
#' @param method an integer scalar with value 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition,
#' 2 for the LLT Cholesky, 3 for the LDLT Cholesky, 4 for the full pivoted QR decomposition, 5 for the Bidiagonal Divide and 
#' Conquer SVD
#' @param tol threshold tolerance for convergence. Should be a positive real number
#' @param maxit maximum number of IRLS iterations. Should be an integer
#' @param firth logical; if `TRUE` apply Firth's (1993) bias-reducing penalty
#'   to the score function. Currently supported only for
#'   `family = binomial(link = "logit")` on dense `x`. The penalty modifies
#'   the working response by `h_i (0.5 - mu_i) / (mu_i (1 - mu_i))` where
#'   `h_i` is the leverage; convergence is checked on `||\Delta\beta||_\infty`.
#'   See `logistf::logistf()` for the canonical reference implementation.
#' @return A list with the elements
#' \item{coefficients}{a vector of coefficients}
#' \item{se}{a vector of the standard errors of the coefficient estimates}
#' \item{rank}{a scalar denoting the computed rank of the model matrix}
#' \item{df.residual}{a scalar denoting the degrees of freedom in the model}
#' \item{residuals}{the vector of residuals}
#' \item{s}{a numeric scalar - the root mean square for residuals}
#' \item{fitted.values}{the vector of fitted values}
#' @seealso [fastglm.fit()]
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
                        firth    = FALSE)
{
    if (!is.logical(firth) || length(firth) != 1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE.", call. = FALSE)
    if (firth) {
        if (is.null(family$family) || family$family != "binomial" ||
            is.null(family$link) || family$link != "logit")
            stop("'firth = TRUE' currently requires family = binomial(link = \"logit\").",
                 call. = FALSE)
        if (inherits(x, "dgCMatrix") || (requireNamespace("bigmemory", quietly = TRUE) &&
                                          bigmemory::is.big.matrix(x)))
            stop("'firth = TRUE' is not supported for sparse or big.matrix designs.",
                 call. = FALSE)
        # Firth uses the same Cholesky factor for the leverage and the WLS
        # update; force LLT internally regardless of caller's `method`.
        method <- 2L
    }
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
    
    
    stopifnot(is.numeric(y), 
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

    if (is_sparse_matrix)
    {
        res <- fit_sparse_glm(x, drop(y), drop(weights), drop(offset),
                              drop(start), drop(mu), drop(eta),
                              family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                              family$valideta, family$validmu,
                              as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                              as.integer(fc), fp)
        # Detect intercept-like columns (all entries identical) by max == min.
        col_max <- apply(x, 2, max)
        col_min <- apply(x, 2, min)
        res$intercept <- any(is.int <- (col_max == col_min))
    } else if (!is_big_matrix && firth)
    {
        res <- fit_glm_firth(x, drop(y), drop(weights), drop(offset),
                             drop(start), drop(mu), drop(eta),
                             family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                             family$valideta, family$validmu,
                             as.double(tol[1]), as.integer(maxit[1]),
                             as.integer(fc), fp)
        res$intercept <- any(is.int <- colMax_dense(x) == colMin_dense(x))
    } else if (!is_big_matrix)
    {
        res <- fit_glm(x, drop(y), drop(weights), drop(offset),
                       drop(start), drop(mu), drop(eta),
                       family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                       family$valideta, family$validmu,
                       as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                       as.integer(fc), fp)

        res$intercept <- any(is.int <- colMax_dense(x) == colMin_dense(x))
    } else
    {
        res <- fit_big_glm(x@address, drop(y), drop(weights), drop(offset),
                           drop(start), drop(mu), drop(eta),
                           family$variance, family$mu.eta, family$linkinv, family$dev.resids,
                           family$valideta, family$validmu,
                           as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]),
                           as.integer(fc), fp)

        res$intercept <- any(is.int <- big.colMax(x) == big.colMin(x))
    }
    
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
    
    if (is.null(cnames))
    {
        ncx <- ncol(x)
        if (res$intercept)
        {
            which.int <- which(is.int)
            cnames    <- paste0("X", 1:(ncx - 1) )
            names(res$coefficients) <- 1:ncx
            names(res$coefficients)[-which.int] <- cnames
            names(res$coefficients)[which.int]  <- "(Intercept)"
        } else
        {
            names(res$coefficients) <- paste0("X", 1:ncx)
        }
    } else
    {
        names(res$coefficients) <- cnames
    }
    
    res$family <- family
    res$prior.weights <- weights
    res$y <- y
    res$n <- n
    res$x <- x        # reference to the model matrix; used by vcovHC / vcovCL
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
#' @param method an integer scalar with value 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition,   
#' 2 for the LLT Cholesky, or 3 for the LDLT Cholesky
#' @param tol threshold tolerance for convergence. Should be a positive real number
#' @param maxit maximum number of IRLS iterations. Should be an integer
#' @return A list with the elements
#' \item{coefficients}{a vector of coefficients}
#' \item{se}{a vector of the standard errors of the coefficient estimates}
#' \item{rank}{a scalar denoting the computed rank of the model matrix}
#' \item{df.residual}{a scalar denoting the degrees of freedom in the model}
#' \item{residuals}{the vector of residuals}
#' \item{s}{a numeric scalar - the root mean square for residuals}
#' \item{fitted.values}{the vector of fitted values}
#' @seealso [fastglm.fit()]
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
    UseMethod("fastglm")
}


#' bigLm default
#'
#' @param ... not used
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
                            ...)
{
    ## family
    if(is.character(family))
    {
        family <- get(family, mode = "function", envir = parent.frame())
    }
    if(is.function(family)) family <- family()
    if(is.null(family$family)) 
    {
        print(family)
        stop("'family' not recognized")
    }
    
    #y             <- as.numeric(y)
    
    ## avoid problems with 1D arrays, but keep names
    if(length(dim(y)) == 1L) 
    {
        nm <- rownames(y)
        dim(y) <- NULL
        if(!is.null(nm)) names(y) <- nm
    }
    
    nobs <- NROW(y)
    
    aic <- family$aic
    
    if (is.null(weights)) weights <- rep(1, nobs)
    if (is.null(offset))  offset  <- rep(0, nobs)
    
    res     <- fastglmPure(x, y, family, weights, offset,
                           start, etastart, mustart,
                           method, tol, maxit, firth = firth)
    y <- res$y
    
    res$residuals <- (y - res$fitted.values) / family$mu.eta(res$linear.predictors)
    #res$y         <- y
    
    # from summary.glm()
    dispersion <-
        if(family$family %in% c("poisson", "binomial"))  1
        else if(res$df.residual > 0)
        {
            est.disp <- TRUE
            if(any(weights == 0))
                warning("observations with zero weight not used for calculating dispersion")
            sum((res$weights*res$residuals ^ 2)[weights > 0]) / res$df.residual
        } else
        {
            est.disp <- TRUE
            NaN
        }
    
    res$dispersion <- dispersion
    
    if (!is.nan(dispersion)) res$se <- res$se * sqrt(dispersion)
    
    wtdmu         <- if (res$intercept) sum(weights * y) / sum(weights) else family$linkinv(offset)
    nulldev       <- sum(family$dev.resids(y, wtdmu, weights))
    
    n.ok          <- nobs - sum(weights == 0)
    nulldf        <- n.ok - as.integer(res$intercept)
    res$df.null   <- nulldf
    
    res$null.deviance <- nulldev
    
    rank <- res$rank
    dev  <- res$deviance
    
    aic.model <- aic(y, res$n, res$fitted.values, res$prior.weights, dev) + 2 * rank
    
    res$aic <- aic.model
    
    # will change later
    boundary <- FALSE
    
    if (boundary)
    {
        warning("fit_glm: algorithm stopped at boundary value", call. = FALSE)
    }
    
    
    res$call      <- match.call()

    class(res)    <- if (isTRUE(res$firth)) c("fastglm_firth", "fastglm") else "fastglm"
    res
}
