
#' print method for fastglm objects
#'
#' @param x object to print
#' @param ... not used
#' @rdname print
#' @method print fastglm
#' @export
print.fastglm <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\nCoefficients:\n")
    print(x$coefficients, digits=5)
}


#' summary method for fastglm fitted objects
#'
#' @param object fastglm fitted object
#' @param dispersion the dispersion parameter for the family used. 
#' Either a single numerical value or \code{NULL} (the default), when it is inferred from \code{object}.
#' @param ... not used
#' @return a summary.fastglm object
#' @rdname summary
#' @method summary fastglm
#' @export
#' @examples
#' 
#' x <- matrix(rnorm(10000 * 10), ncol = 10)
#' y <- 1 * (0.25 * x[,1] - 0.25 * x[,3] > rnorm(10000))
#' 
#' fit <- fastglm(x, y, family = binomial())
#' 
#' #summary(fit)
#'
#'
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


#' residuals method for fastglm fitted objects
#'
#' @param object fastglm fitted object
#' @param type type of residual to be returned
#' @param ... not used
#' @return a vector of residuals
#' @rdname residuals
#' @method residuals fastglm
#' @export
residuals.fastglm <- function(object, 
                              type = c("deviance", "pearson", "working", "response", "partial"), 
                              ...)
{
    residuals.glm(object, type, ...)
}

#' Obtains predictions and optionally estimates standard errors of those predictions from a fitted generalized linear model object.
#' @param object a fitted object of class inheriting from "\code{fastglm}".
#' @param newdata optionally, a data frame in which to look for variables with which to predict. 
#' If omitted, the fitted linear predictors are used.
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
#' @param terms with \code{type = "terms"} by default all terms are returned. A character vector specifies 
#' which terms are to be returned
#' @param na.action function determining what should be done with missing values in \code{newdata}. 
#' The default is to predict \code{NA}.
#' @param ... further arguments passed to or from other methods.
#' @export
predict.fastglm <- function(object, 
                            newdata = NULL, 
                            type = c("link", "response", "terms"),
                            se.fit = FALSE, 
                            dispersion = NULL, 
                            terms = NULL,
                            na.action = na.pass, ...)
{
    #class(object) <- c("glm", "lm")
    #predict.glm(object, newdata, type, se.fit, dispersion, terms, na.action, ...)
}


predict_fastglm_lm <- function(object, se.fit, scale = 1, type, terms, na.action)
{
    
}


