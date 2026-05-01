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

#' @param correlation logical; if `TRUE`, the correlation matrix of the estimated parameters is returned.
#' @param symbolic.cor logical; if `TRUE`, print the correlations in a symbolic form (see `symnum`) rather than as numbers.
#' @exportS3Method summary fastglm
summary.fastglm <- function(object, dispersion = NULL,
                            correlation = FALSE, symbolic.cor = FALSE, ...)
{
    p <- object$rank

    est.disp <- FALSE
    df.r <- object$df.residual

    if (is.null(dispersion))
    {
        if (!(object$family$family %in% c("poisson", "binomial"))) est.disp <- TRUE
        dispersion <- object$dispersion
    }

    aliased <- is.na(coef(object))  # used in print method

    covmat.unscaled <- object$cov.unscaled
    if (is.null(covmat.unscaled) && p > 0)
    {
        # Fallback for older fitted objects without cov.unscaled
        covmat.unscaled <- diag(object$se ^ 2 / max(dispersion, .Machine$double.eps),
                                nrow = length(object$coefficients))
    }
    if (!is.null(covmat.unscaled) && length(object$coefficients) > 0L)
    {
        nms <- names(object$coefficients)
        rownames(covmat.unscaled) <- colnames(covmat.unscaled) <- nms
        # Match base R summary.glm: drop aliased rows/cols (rank x rank)
        if (any(aliased))
            covmat.unscaled <- covmat.unscaled[!aliased, !aliased, drop = FALSE]
    }
    covmat <- if (!is.null(covmat.unscaled)) dispersion * covmat.unscaled else NULL

    if (p > 0)
    {
        coef   <- object$coefficients
        se     <- object$se
        tvalue <- coef / se

        dn <- c("Estimate", "Std. Error")
        if (!est.disp)
        { # known dispersion
            pvalue <- 2 * pnorm(-abs(tvalue))
            coef.table <- cbind(coef, se, tvalue, pvalue)
            dimnames(coef.table) <- list(names(coef),
                                         c(dn, "z value","Pr(>|z|)"))
        } else if (df.r > 0)
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
                  df = c(object$rank, df.r, df.f),
                  cov.unscaled = covmat.unscaled,
                  cov.scaled = covmat))

    if (correlation && p > 0 && !is.null(covmat.unscaled))
    {
        dd <- sqrt(diag(covmat.unscaled))
        ans$correlation <- covmat.unscaled / outer(dd, dd)
        ans$symbolic.cor <- symbolic.cor
    }
    class(ans) <- "summary.glm"
    return(ans)
}

#' `vcov()` method for `fastglm` fitted objects
#'
#' @param object a fitted object of class inheriting from `"fastglm"`.
#' @param ... additional arguments (currently unused).
#'
#' @returns The estimated variance-covariance matrix of the fitted coefficients.
#' For rank-deficient fits, rows and columns corresponding to aliased
#' coefficients are filled with `NA`.
#'
#' @method vcov fastglm
#' @exportS3Method stats::vcov fastglm
vcov.fastglm <- function(object, ...)
{
    cov.unscaled <- object$cov.unscaled
    if (is.null(cov.unscaled))
    {
        v <- diag(object[["se"]]^2, nrow = length(object$coefficients))
        rownames(v) <- colnames(v) <- names(coef(object))
        return(v)
    }
    disp <- if (is.null(object$dispersion) || is.nan(object$dispersion)) 1 else object$dispersion
    v <- disp * cov.unscaled
    nms <- names(coef(object))
    rownames(v) <- colnames(v) <- nms
    v
}

#' Heteroskedasticity-consistent (HC) variance estimators for `fastglm` objects
#'
#' Methods for `sandwich::vcovHC()` on objects of class `"fastglm"` and
#' `"fastglmFit"`. Load `sandwich` (`library(sandwich)`) before calling
#' `vcovHC(fit)`; otherwise no `vcovHC` generic is in scope.
#'
#' @param object a fitted object of class `"fastglm"` or `"fastglmFit"`.
#' @param type one of `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`. Default `"HC3"` matches
#'   `sandwich::vcovHC.glm`.
#' @param ... not used.
#'
#' @returns A `p x p` heteroskedasticity-consistent variance-covariance matrix.
#'
#' @details
#' Computes the Eicker-Huber-White sandwich estimator
#' `bread %*% meat %*% bread`, where `bread = (X' W X)^{-1}` (already stored as
#' `cov.unscaled`) and `meat = X' diag(omega_i) X`. With `s_i = w_i^2 * r_i`
#' the score contribution from observation `i`, the omegas are:
#' \describe{
#'   \item{`HC0`}{`omega_i = s_i^2`}
#'   \item{`HC1`}{`HC0` rescaled by `n / (n - p)`}
#'   \item{`HC2`}{`omega_i = s_i^2 / (1 - h_i)`}
#'   \item{`HC3`}{`omega_i = s_i^2 / (1 - h_i)^2`}
#' }
#' where `r_i` is the working residual `(y - mu) / mu.eta(eta)`,
#' `w_i^2 = prior.weight * mu.eta(eta)^2 / variance(mu)` is the IRLS working
#' weight, and `h_i = w_i^2 * x_i' (X' W X)^(-1) x_i` is the IRLS leverage.
#' Equivalent to `sandwich::vcovHC.glm`.
#'
#' Requires the model matrix `x` stored on the fitted object (set automatically
#' by `fastglm()`, `fastglmPure()`, and `fastglm.fit()` since version 0.0.6).
#'
#' @examples
#' if (requireNamespace("sandwich", quietly = TRUE)) {
#'   x <- cbind(1, matrix(rnorm(500 * 4), ncol = 4))
#'   y <- rbinom(500, 1, plogis(x %*% c(0.2, 0.3, -0.4, 0.1, 0.2)))
#'   fit <- fastglm(x, y, family = binomial())
#'   sandwich::vcovHC(fit)
#'   sandwich::vcovHC(fit, type = "HC0")
#' }
#'
#' @name vcovHC.fastglm
NULL

#' @rdname vcovHC.fastglm
#' @exportS3Method sandwich::vcovHC fastglm
vcovHC.fastglm <- function(object, type = c("HC3", "HC2", "HC1", "HC0"), ...)
{
    type <- match.arg(type)
    .vcov_hc_fastglm(object, type)
}

#' @rdname vcovHC.fastglm
#' @exportS3Method sandwich::vcovHC fastglmFit
vcovHC.fastglmFit <- function(object, type = c("HC3", "HC2", "HC1", "HC0"), ...)
{
    type <- match.arg(type)
    .vcov_hc_fastglm(object, type)
}

#' Cluster-robust variance estimator for `fastglm` objects
#'
#' Methods for `sandwich::vcovCL()` on objects of class `"fastglm"` and
#' `"fastglmFit"`. Load `sandwich` (`library(sandwich)`) before calling
#' `vcovCL(fit, cluster)`; otherwise no `vcovCL` generic is in scope.
#'
#' @param object a fitted object of class `"fastglm"` or `"fastglmFit"`.
#' @param cluster a vector identifying the cluster for each observation. Length
#'   must equal the number of rows of the design matrix.
#' @param type one of `"HC0"`, `"HC1"`. `"HC1"` (the default) applies the small-sample
#'   degrees-of-freedom adjustment `(G / (G - 1)) * ((n - 1) / (n - p))`, where
#'   `G` is the number of clusters.
#' @param ... not used.
#'
#' @returns A `p x p` cluster-robust variance-covariance matrix.
#'
#' @details
#' The Liang-Zeger (1986) cluster-robust sandwich:
#' `bread %*% (sum_g s_g s_g') %*% bread`, where `s_g = sum_{i in g} w_i r_i x_i`
#' is the score contribution from cluster `g`.
#'
#' @name vcovCL.fastglm
NULL

#' @rdname vcovCL.fastglm
#' @exportS3Method sandwich::vcovCL fastglm
vcovCL.fastglm <- function(object, cluster, type = c("HC1", "HC0"), ...)
{
    type <- match.arg(type)
    .vcov_cl_fastglm(object, cluster, type)
}

#' @rdname vcovCL.fastglm
#' @exportS3Method sandwich::vcovCL fastglmFit
vcovCL.fastglmFit <- function(object, cluster, type = c("HC1", "HC0"), ...)
{
    type <- match.arg(type)
    .vcov_cl_fastglm(object, cluster, type)
}

# Internal: pull (X, working residuals, working weights, bread) off a fitted
# fastglm/fastglmFit object.  Returns a list with elements x, r, w, bread, p.
.vcov_hc_meat_inputs <- function(object)
{
    x <- object$x
    if (is.null(x))
        stop("vcovHC/vcovCL require the design matrix to be stored on the fitted object. ",
             "Refit with fastglm/fastglmPure/fastglm.fit (>= 0.0.6).", call. = FALSE)
    if (inherits(x, "big.matrix"))
        x <- x[]
    if (inherits(x, "dgCMatrix"))
        x <- as.matrix(x)

    bread <- object$cov.unscaled
    if (is.null(bread))
        stop("'cov.unscaled' missing from fitted object; refit with current fastglm.", call. = FALSE)

    fam <- object$family
    eta <- object$linear.predictors
    mu  <- object$fitted.values
    pw  <- object$prior.weights
    if (is.null(pw)) pw <- rep(1, length(eta))

    mu_eta_vec <- fam$mu.eta(eta)
    var_mu_vec <- fam$variance(mu)
    y <- object$y
    r <- (y - mu) / mu_eta_vec                              # working residual
    w2 <- pw * mu_eta_vec^2 / var_mu_vec                    # working weight (W in (X'WX))

    # GLM score scalar: s_i = w^2_i * r_i = pw_i * mu_eta_i * (y_i - mu_i) / V(mu_i).
    # Note: do NOT divide by dispersion here. sandwich::estfun.glm divides by an
    # internal dispersion estimate, but sandwich::bread.glm multiplies by the
    # SAME dispersion, so the two cancel in the final sandwich formula.
    s <- w2 * r

    list(x = x, r = r, s = s, w2 = w2, bread = bread,
         p = length(object$coefficients), n = length(y))
}

.vcov_hc_fastglm <- function(object, type)
{
    inp <- .vcov_hc_meat_inputs(object)
    x <- inp$x; s <- inp$s; w2 <- inp$w2; bread <- inp$bread
    n <- inp$n; p <- inp$p

    omega <- s^2

    if (type %in% c("HC2", "HC3"))
    {
        # IRLS leverage h_i = w_i^2 * x_i' (X'WX)^{-1} x_i
        h <- w2 * rowSums((x %*% bread) * x)
        h <- pmin(h, 1 - 1e-12)
        if (type == "HC2") omega <- omega / (1 - h)
        else               omega <- omega / (1 - h)^2
    }

    XtMX <- crossprod(x, omega * x)
    V <- bread %*% XtMX %*% bread

    if (type == "HC1")
        V <- V * (n / max(n - p, 1))

    nms <- names(object$coefficients)
    rownames(V) <- colnames(V) <- nms
    V
}

.vcov_cl_fastglm <- function(object, cluster, type)
{
    inp <- .vcov_hc_meat_inputs(object)
    x <- inp$x; s <- inp$s; bread <- inp$bread
    n <- inp$n; p <- inp$p

    if (length(cluster) != n)
        stop(sprintf("length(cluster) (%d) does not match number of observations (%d)",
                     length(cluster), n), call. = FALSE)
    cl <- as.factor(cluster)
    G <- nlevels(cl)

    estfun <- s * x                                          # n x p
    s_g <- rowsum(estfun, cl, reorder = FALSE)               # G x p
    meat <- crossprod(s_g)
    V <- bread %*% meat %*% bread

    if (type == "HC1")
        V <- V * (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))

    nms <- names(object$coefficients)
    rownames(V) <- colnames(V) <- nms
    V
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

    if (is.null(dispersion))
        dispersion <- if (is.null(object$dispersion) || is.nan(object$dispersion)) 1 else object$dispersion

    pred <- predict_fastglm_lm(object, newdata, se.fit, dispersion = dispersion, ...)

    if (type == "response")
    {
        fam <- family(object)
        if (se.fit)
        {
            mu_eta <- fam$mu.eta(pred$fit)
            pred$fit    <- fam$linkinv(pred$fit)
            pred$se.fit <- pred$se.fit * abs(mu_eta)
        } else
        {
            pred <- fam$linkinv(pred)
        }
    }
    pred
}


predict_fastglm_lm <- function(object, newdata, se.fit = FALSE, dispersion = 1)
{
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

    if (!se.fit) return(eta)

    cov.unscaled <- object$cov.unscaled
    if (is.null(cov.unscaled))
        stop("standard errors of predictions require 'cov.unscaled' from a refit; reinstall fastglm and refit")

    cov.scaled <- dispersion * cov.unscaled
    se <- sqrt(rowSums((newdata %*% cov.scaled) * newdata))
    list(fit = eta, se.fit = se, residual.scale = sqrt(dispersion))
}
