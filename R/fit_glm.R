

#' fast and memory efficient linear model fitting
#'
#' @param X input model matrix. Must be a matrix object 
#' @param y numeric response vector of length nobs.
#' @param weights an optional vector of 'prior weights' to be used in the fitting process. Should be a numeric vector.
#' @param offset this can be used to specify an a priori known component to be included in the linear predictor during fitting. 
#' This should be a numeric vector of length equal to the number of cases
#' @param type an integer scalar with value 0 for the column-pivoted QR decomposition, 1 for the unpivoted QR decomposition,   
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
#' system.time(gf2 <- fastglmPure(x, y, family = binomial(), type = 1, tol = 1e-8))
#' 
#' system.time(gf3 <- fastglmPure(x, y, family = binomial(), type = 2, tol = 1e-8))
#' 
#' system.time(gf4 <- fastglmPure(x, y, family = binomial(), type = 3, tol = 1e-8))
#' 
#' max(abs(coef(gl1) - gf1$coef))
#' max(abs(coef(gl1) - gf2$coef))
#' max(abs(coef(gl1) - gf3$coef))
#' max(abs(coef(gl1) - gf4$coef))
#' 
fastglmPure <- function(X, y, 
                        weights = rep(1, NROW(y)), 
                        offset = rep(0, NROW(y)), 
                        family = gaussian(),
                        type = 0L,
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
              is.numeric(type),
              is.numeric(tol),
              is.numeric(maxit),
              tol[1] > 0,
              maxit[1] > 0              
              )
    
    fit_glm(X, y, weights, offset, family$variance, family$mu.eta, family$linkinv, family$dev.resids, 
            as.integer(type[1]), as.double(tol[1]), as.integer(maxit[1]) )
}