

#' fast generalized linear model fitting
#'
#' @param X input model matrix. Must be a matrix object 
#' @param y numeric response vector of length nobs.
#' @param family a description of the error distribution and link function to be used in the model. 
#' For \code{fastglmPure} this can only be the result of a call to a family function. 
#' (See \code{\link[stats]{family}} for details of family functions.)
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be a numeric vector.
#' @param offset this can be used to specify an a priori known component to be included in the linear predictor during fitting. 
#' This should be a numeric vector of length equal to the number of cases
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
#' @export
#' @examples
#' 
#' x <- matrix(rnorm(10000 * 100), ncol = 100)
#' y <- 1 * (0.25 * x[,1] - 0.25 * x[,3] > rnorm(10000))
#' 
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
fastglmPure <- function(X, y, 
                        family = gaussian(),
                        weights = rep(1, NROW(y)), 
                        offset = rep(0, NROW(y)), 
                        method = 0L,
                        tol = 1e-7,
                        maxit = 100L)
{
    weights <- as.vector(weights)
    offset  <- as.vector(offset)
    stopifnot(is.matrix(X), 
              is.numeric(y), 
              is.numeric(weights),
              is.numeric(offset),
              NROW(y) == nrow(X),
              NROW(y) == NROW(weights),
              NROW(y) == NROW(offset),
              is.numeric(method),
              is.numeric(tol),
              is.numeric(maxit),
              tol[1] > 0,
              maxit[1] > 0              
              )
    
    if(is.null(family$family)) 
    {
        print(family)
        stop("'family' not recognized")
    }
    
    if( any(weights < 0) ) stop("negative weights not allowed")
    
    if (method[1] > 3L)
    {
        stop("Invalid decomposition method specified. Choose from 0, 1, 2, or 3.")
    }
    
    cnames <- colnames(X)
    
    
    res <- fit_glm(X, y, weights, offset, 
                   family$variance, family$mu.eta, family$linkinv, family$dev.resids, 
                   as.integer(method[1]), as.double(tol[1]), as.integer(maxit[1]) )
    
    res$intercept <- any(is.int <- colMax_dense(X) == colMin_dense(X))
    
    conv <- res$iter < maxit[1]
    res$conv <- conv
    
    if (!conv)
    {
        warning("glm.fit: algorithm did not converge", call. = FALSE)
    }

    eps <- 10*.Machine$double.eps
    if (family$family == "binomial") 
    {
        if (any(res$fitted.values > 1 - eps) || any(res$fitted.values < eps))
            warning("glm.fit: fitted probabilities numerically 0 or 1 occurred", call. = FALSE)
    }
    if (family$family == "poisson") 
    {
        if (any(res$fitted.values < eps))
            warning("glm.fit: fitted rates numerically 0 occurred", call. = FALSE)
    }
    
    if (is.null(cnames))
    {
        if (res$intercept)
        {
            which.int <- which(is.int)
            cnames    <- paste0("X", 1:(ncol(X) - 1) )
            names(res$coefficients) <- 1:ncol(X)
            names(res$coefficients)[-which.int] <- cnames
            names(res$coefficients)[which.int]  <- "(Intercept)"
        } else
        {
            names(res$coefficients) <- paste0("X", 1:ncol(X))
        }
    }
    
    res$family <- family
    res
}

#' fast generalized linear model fitting
#'
#' @param X input model matrix. Must be a matrix object 
#' @param y numeric response vector of length nobs.
#' @param family a description of the error distribution and link function to be used in the model. 
#' For \code{fastglm} this can be a character string naming a family function, a family function or the 
#' result of a call to a family function. For \code{fastglmPure} only the third option is supported. 
#' (See \code{\link[stats]{family}} for details of family functions.)
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be a numeric vector.
#' @param offset this can be used to specify an a priori known component to be included in the linear predictor during fitting. 
#' This should be a numeric vector of length equal to the number of cases
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
fastglm <- function(X, ...) UseMethod("fastglm")


#' bigLm default
#'
#' @param ... not used
#' @rdname fastglm
#' @method fastglm default
#' @export
fastglm.default <- function(X, y, 
                            family = gaussian(),
                            weights = NULL, 
                            offset = NULL, 
                            method = 0L, tol = 1e-8, maxit = 100L,
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
    
    y             <- as.numeric(y)
    
    ## avoid problems with 1D arrays, but keep names
    if(length(dim(y)) == 1L) 
    {
        nm <- rownames(y)
        dim(y) <- NULL
        if(!is.null(nm)) names(y) <- nm
    }
    
    nobs <- NROW(y)
    
    if (is.null(weights)) weights <- rep(1, nobs)
    if (is.null(offset))  offset  <- rep(0, nobs)
    
    res     <- fastglmPure(X, y, family, weights, offset, method, tol, maxit)
    
    wtdmu   <- if (res$intercept) sum(weights * y) / sum(weights) else family$linkinv(offset)
    nulldev <- sum(family$dev.resids(y, wtdmu, weights))
    
    n.ok        <- nobs - sum(weights == 0)
    nulldf      <- n.ok - as.integer(res$intercept)
    res$df.null <- nulldf
    
    res$null.deviance <- nulldev
    
    # will change later
    boundary <- FALSE
    
    if (boundary)
    {
        warning("glm.fit: algorithm stopped at boundary value", call. = FALSE)
    }
    
    
    res$call      <- match.call()
    
    class(res)    <- "fastglm"
    res
}
