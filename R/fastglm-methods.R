#' `summary()` method for `fastglm` fitted objects
#'
#' @param object `fastglm` fitted object
#' @param dispersion the dispersion parameter for the family used. Either a single numerical value or `NULL` (the default), when it is inferred from `object`.
#' @param ... not used
#' 
#' @returns A `summary.fastglm` object
#' 
#' @seealso [summary.glm()]
#' 
#' @method summary fastglm
#' 
#' @examples
#' x <- matrix(rnorm(10000 * 10), ncol = 10)
#' y <- 1 * (0.25 * x[,1] - 0.25 * x[,3] > rnorm(10000))
#' 
#' fit <- fastglm(x, y, family = binomial())
#' 
#' summary(fit)

#' @exportS3Method summary fastglm
summary.fastglm <- function(object, dispersion = NULL, ...)
{
    p <- object$rank
    
    est.disp <- FALSE
    df.r <- object$df.residual
    
    if(is.null(dispersion)) 
    {
        if (!(object$family$family %in% c("poisson", "binomial"))) est.disp <- TRUE
        dispersion <- object$dispersion
    }
    
    aliased <- is.na(coef(object))  # used in print method

    if (p > 0)
    {
        coef   <- object$coefficients
        se     <- object$se
        tvalue <- coef / se
        
        #coef.table <- cbind(Estimate     = coef,
        #                    "Std. Error" = se,
        #                    "t value"    = tval,
        #                    "Pr(>|t|)"   = 2*pt(-abs(tval), df = object$df))
        
        dn <- c("Estimate", "Std. Error")
        if(!est.disp) 
        { # known dispersion
            pvalue <- 2 * pnorm(-abs(tvalue))
            coef.table <- cbind(coef, se, tvalue, pvalue)
            dimnames(coef.table) <- list(names(coef),
                                         c(dn, "z value","Pr(>|z|)"))
        } else if(df.r > 0) 
        {
            pvalue <- 2 * pt(-abs(tvalue), df.r)
            coef.table <- cbind(coef, se, tvalue, pvalue)
            dimnames(coef.table) <- list(names(coef),
                                         c(dn, "t value","Pr(>|t|)"))
        } else 
        { # df.r == 0
            coef.table <- cbind(coef, NaN, NaN, NaN)
            dimnames(coef.table) <- list(names(coef),
                                         c(dn, "t value","Pr(>|t|)"))
        }
    
        df.f <- length(aliased)
    } else 
    {
        coef.table <- matrix(0, 0L, 4L)
        dimnames(coef.table) <-
            list(NULL, c("Estimate", "Std. Error", "t value", "Pr(>|t|)"))
        covmat.unscaled <- covmat <- matrix(0, 0L, 0L)
        df.f <- length(aliased)
    }
    df.int <- if (object$intercept) 1L else 0L
    
    ## these need not all exist, e.g. na.action.
    keep <- match(c("call","terms","family","deviance", "aic",
                    "contrasts", "df.residual","null.deviance","df.null",
                    "iter", "na.action"), names(object), 0L)
    ans <- c(object[keep],
             list(deviance.resid = residuals(object, type = "deviance"),
                  coefficients = coef.table,
                  aliased = aliased,
                  dispersion = dispersion,
                  df = c(object$rank, df.r, df.f)))
                  #cov.unscaled = covmat.unscaled,
                  #cov.scaled = covmat))
    
    ## will do this later    
    # if(correlation && p > 0) 
    # {
    #     dd <- sqrt(diag(covmat.unscaled))
    #     ans$correlation <-
    #         covmat.unscaled/outer(dd,dd)
    #     ans$symbolic.cor <- symbolic.cor
    # }
    class(ans) <- "summary.glm"
    return(ans)
}

#' @exportS3Method print fastglm
print.fastglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) 
{
    cat("\nCall:  ", paste(deparse(x$call), sep = "\n", collapse = "\n"), 
        "\n\n", sep = "")
    
    if (length(coef(x)) > 0L) {
        cat("Coefficients")
        if (is.character(co <- x$contrasts)) {
            cat("  [contrasts: ", apply(cbind(names(co), co), 1L, paste, collapse = "="), "]")
        }
        
        cat(":\n")
        print.default(format(x$coefficients, digits = digits), 
                      print.gap = 2, quote = FALSE)
    }
    else {
        cat("No coefficients\n\n")
    }
}

#' @exportS3Method residuals fastglm
residuals.fastglm <- function(object, 
                              type = c("deviance", "pearson", "working", "response", "partial"), 
                              ...)
{
    class(object) <- "glm"
    
    residuals(object, type = type, ...)
}

#' @exportS3Method logLik fastglm
logLik.fastglm <- function(object, ...)
{
    class(object) <- "glm"
    
    logLik(object, ...)
}

#' @exportS3Method deviance fastglm
deviance.fastglm <- function(object, ...)
{
    class(object) <- "glm"
    
    deviance(object, ...)
}

#' @exportS3Method family fastglm
family.fastglm <- function(object, ...)
{
    class(object) <- "glm"
    
    family(object, ...)
}


#' Obtains predictions and optionally estimates standard errors of those predictions from a fitted generalized linear model object.
#' @param object a fitted object of class inheriting from `"fastglm"`.
#' @param newdata a matrix to be used for prediction.
#' @param type the type of prediction required. The default is on the scale of the linear predictors;
#' the alternative "\code{response}" is on the scale of the response variable. Thus for a default binomial
#' model the default predictions are of log-odds (probabilities on logit scale) and \code{type = "response"}
#'  gives the predicted probabilities. The "\code{terms}" option returns a matrix giving the fitted values of each
#'  term in the model formula on the linear predictor scale.
#'
#' The value of this argument can be abbreviated.
#' @param se.fit logical switch indicating if standard errors are required.
#' @param dispersion the dispersion of the GLM fit to be assumed in computing the standard errors.
#' If omitted, that returned by \code{summary} applied to the object is used.
#' @param ... further arguments passed to or from other methods.
#' @export
predict.fastglm <- function(object,
                            newdata = NULL,
                            type = c("link", "response"),
                            se.fit = FALSE,
                            dispersion = NULL, ...)
{
    type <- match.arg(type)
    
    eta <- predict_fastglm_lm(object, newdata, se.fit, scale = 1, ...)
    if (type == "response")
    {
        eta <- family(object)$linkinv(eta)
    }
    eta
}


predict_fastglm_lm <- function(object, newdata, se.fit = FALSE, scale = 1)
{
    if (se.fit)
    {
        stop("confidence/prediction intervals not available yet")
    }
    dims <- dim(newdata)
    if (is.null(dims))
    {
        newdata <- as.matrix(newdata)
        dims <- dim(newdata)
    }
    beta <- object$coefficients
    
    if (dims[2L] != length(beta))
    {
        stop("newdata provided does not match fitted model 'object'")
    }
    eta <- drop(newdata %*% beta)
    eta
}
