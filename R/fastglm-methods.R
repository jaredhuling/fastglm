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
    if (identical(object$inference, "none")) stop("summary requires inference; refit with inference = 'full' or 'diagonal'", call. = FALSE)
    p <- object$rank

    est.disp <- FALSE
    df.r <- object$df.residual

    if (is.null(dispersion))
    {
        est.disp <- is.na(.fixed_dispersion(object$family))
        dispersion <- object$dispersion
    }

    aliased <- is.na(coef(object))  # used in print method

    covmat.unscaled <- object$cov.unscaled
    if (is.null(covmat.unscaled) && correlation)
        stop("correlations require full covariance; refit with inference = 'full'", call. = FALSE)
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
        coef   <- object$coefficients[!aliased]
        se     <- if (!is.null(covmat)) sqrt(diag(covmat)) else
            object$se[!aliased] * sqrt(dispersion / object$dispersion)
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
    ans$converged <- object$converged
    ans$convergence.reason <- object$convergence.reason
    ans$firth <- object$firth
    ans$theta <- object$theta
    ans$SE.theta <- object$SE.theta
    ans$title <- if(inherits(object,"fastglm_nb")) "Negative-binomial regression" else "Generalized linear model"
    class(ans) <- c("summary.fastglm","summary.glm")
    return(ans)
}

#' `vcov()` method for `fastglm` fitted objects
#'
#' @param object a fitted object of class inheriting from `"fastglm"`.
#' @param dispersion scale used for model-based covariance.
#' @param type optional HC correction for robust covariance.
#' @param cluster optional cluster identifiers for robust covariance.
#' @param ... arguments to [fastglm_vcov()].
#'
#' @returns The estimated variance-covariance matrix of the fitted coefficients.
#' For rank-deficient fits, rows and columns corresponding to aliased
#' coefficients are filled with `NA`.
#'
#' @method vcov fastglm
#' @exportS3Method stats::vcov fastglm
vcov.fastglm <- function(object, dispersion = object$dispersion, type = NULL,
                         cluster = NULL, ...) {
    if (!is.null(type) || !is.null(cluster))
        return(fastglm_vcov(object, if (is.null(type)) "HC1" else type, cluster, ...))
    if (is.null(object$cov.unscaled)) stop("full covariance was not retained; refit with inference = 'full'", call. = FALSE)
    v <- dispersion * object$cov.unscaled
    dimnames(v) <- rep(list(names(coef(object))), 2L)
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
#' by `fastglm()`, `fastglmPure()`, and `fastglm_fit()` since version 0.0.6).
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
    fastglm_vcov(object, type)
}

#' @rdname vcovHC.fastglm
#' @exportS3Method sandwich::vcovHC fastglmFit
vcovHC.fastglmFit <- function(object, type = c("HC3", "HC2", "HC1", "HC0"), ...)
{
    type <- match.arg(type)
    fastglm_vcov(object, type)
}

#' Empirical estimating-function and bread methods for `fastglm` objects
#'
#' Methods for `sandwich::estfun()` and `sandwich::bread()`, which let
#' `sandwich::vcovCL()`, `sandwich::vcovBS()`, and the rest of the *sandwich*
#' machinery work directly on `fastglm` and `fastglmFit` objects.  Load
#' `sandwich` (`library(sandwich)`) before calling them.
#'
#' @param x a fitted object of class `"fastglm"` or `"fastglmFit"`.
#' @param ... not used.
#'
#' @details
#' `estfun(x)` returns the per-observation empirical estimating-function
#' contributions, an `n x p` matrix whose `i`-th row is
#' `(y_i - mu_i) * mu.eta_i / variance(mu_i) * x_i / dispersion`. `bread(x)`
#' returns `(X' W X)^(-1) * n * dispersion` (the *sandwich* convention).
#' For poisson, binomial, and negative-binomial families the dispersion is
#' fixed at 1, exactly matching `sandwich::estfun.glm()` /
#' `sandwich::bread.glm()`.
#'
#' @returns `estfun()` returns an `n x p` matrix; `bread()` returns a
#'   `p x p` matrix.
#'
#' @name fastglm-sandwich
NULL

# Internal: dispersion used by sandwich's GLM bread/estfun convention.
.sandwich_dispersion <- function(object, inp)
{
    fam <- object$family$family
    if (substr(fam, 1L, 17L) %in% c("poisson", "binomial", "Negative Binomial"))
        return(1)
    sum_w2 <- sum(inp$w2, na.rm = TRUE)
    if (sum_w2 <= 0) return(1)
    sum(inp$s^2, na.rm = TRUE) / sum_w2
}

#' @rdname fastglm-sandwich
#' @exportS3Method sandwich::estfun fastglm
estfun.fastglm <- function(x, ...)
{
    inp  <- .vcov_hc_meat_inputs(x)
    disp <- .sandwich_dispersion(x, inp)
    rval <- (inp$s / disp) * inp$x
    colnames(rval) <- names(x$coefficients)
    rval
}

#' @rdname fastglm-sandwich
#' @exportS3Method sandwich::estfun fastglmFit
estfun.fastglmFit <- estfun.fastglm

#' @rdname fastglm-sandwich
#' @exportS3Method sandwich::bread fastglm
bread.fastglm <- function(x, ...)
{
    inp  <- .vcov_hc_meat_inputs(x, design = FALSE)
    disp <- .sandwich_dispersion(x, inp)
    nms  <- names(x$coefficients)
    out  <- inp$bread * inp$n * disp
    rownames(out) <- colnames(out) <- nms
    out
}

#' @rdname fastglm-sandwich
#' @exportS3Method sandwich::bread fastglmFit
bread.fastglmFit <- bread.fastglm

# Internal: pull (X, working residuals, working weights, bread) off a fitted
# fastglm/fastglmFit object.  Returns a list with elements x, r, w, bread, p.
.vcov_hc_meat_inputs <- function(object, design = TRUE)
{
    x <- object$x
    if (is.null(x))
        stop("vcovHC/vcovCL require the design matrix to be stored on the fitted object. ",
             "Refit with fastglm/fastglmPure/fastglm_fit (>= 0.0.6).", call. = FALSE)
    if (design && inherits(x, "big.matrix"))
        x <- x[]
    if (design && inherits(x, "dgCMatrix"))
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
    code <- .stable_binomial_code(object)
    if(code>=0L) {
        moments <- binomial_firth_moments(as.numeric(eta),as.numeric(y),pw,code)
        w2 <- moments$working;s <- moments$score
        r <- .binomial_working_residual(moments,y,eta,code);r[pw==0] <- 0
    }

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

#' @exportS3Method print fastglm
print.fastglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...)
{
    digits <- .fastglm_print_digits(digits)
    title <- if(inherits(x,"fastglm_truncated")) "Zero-truncated regression" else
        if(inherits(x,"fastglm_nb")) "Negative-binomial regression" else
        if(inherits(x,"fastglm_stream")) "Streaming generalized linear model" else "Generalized linear model"
    .fastglm_print_header(title,x$call)
    if(!is.null(x$family)) .fastglm_print_field("Family",paste0(x$family$family," (",x$family$link," link)"))
    if(isTRUE(x$firth)) .fastglm_print_field("Estimation","mean bias reduction")
    .fastglm_print_coefficients(coef(x),digits=digits,...)
    cat("\n")
    if(isTRUE(is.finite(x$theta))) .fastglm_print_table(data.frame(Theta=x$theta,`Std. Error`=x$SE.theta,check.names=FALSE),digits)
    .fastglm_print_convergence(x$converged,x$iter,x$convergence.reason)
    invisible(x)
}

#' @exportS3Method residuals fastglm
residuals.fastglm <- function(object, 
                              type = c("deviance", "pearson", "working", "response", "partial"), 
                              ...)
{
    type <- match.arg(type)
    code <- .stable_binomial_code(object)
    if(code>=0L && type %in% c("deviance","pearson","response")) {
        y <- object$y
        if(is.null(y)) {
            eta <- object$linear.predictors
            moments <- binomial_firth_moments(eta,numeric(length(eta)),object$prior.weights,code)
            y <- moments$mean+object$residuals*moments$derivative
            if(any(!is.finite(y))) stop("response data are required for these residuals; refit with retain.data=TRUE",call.=FALSE)
        }
        moments <- binomial_firth_moments(object$linear.predictors,as.numeric(y),object$prior.weights,code)
        result <- .binomial_tail_residual(moments,y,object$prior.weights,type)
        names(result) <- names(object$residuals)
        if(!is.null(object$na.action)) result <- naresid(object$na.action,result)
        return(result)
    }
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
#' @param newdata a model matrix, or a data frame for a formula fit.
#' @param offset optional prediction offsets.
#' @param terms optional subset of model terms for term predictions.
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
#' @returns Predictions on the requested scale. With `se.fit=TRUE`, a
#'   `fastglm_prediction` list containing `fit`, `se.fit`, and `residual.scale`;
#'   see \link{fastglm-prediction-methods} for printing and table extraction.
#' @export
predict.fastglm <- function(object, newdata = NULL,
                            type = c("link", "response", "terms"),
                            se.fit = FALSE, dispersion = NULL, offset = NULL,
                            terms = NULL, ...) {
    type <- match.arg(type)
    if (se.fit && is.null(object$cov.unscaled)) stop("prediction standard errors require full covariance", call. = FALSE)
    if (is.null(dispersion)) dispersion <- object$dispersion
    if (is.null(newdata) && !se.fit && type != "terms" && !is.null(object$linear.predictors)) {
        eta <- object$linear.predictors
        out <- if (type == "response") object$family$linkinv(eta) else eta
    } else {
        design <- .prediction_design(object, newdata, offset)
        x <- design$x; b <- coef(object); good <- !is.na(b)
        nonestimable <- rep(FALSE, nrow(x))
        if (!is.null(object$alias.relation) && !all(good)) {
            implied <- x[, good, drop = FALSE] %*% object$alias.relation
            observed <- x[, !good, drop = FALSE]
            nonestimable <- rowSums(abs(observed-implied)) > 1e-7*(1+rowSums(abs(observed)))
            if (any(nonestimable)) warning("prediction is not estimable for some rows of the rank-deficient design", call. = FALSE)
        }
        if (!all(good)) x <- x[, good, drop = FALSE]
        eta <- .model_eta(x, object, design$offset)
        v <- dispersion * object$cov.unscaled[good, good, drop = FALSE]
        if (type == "terms") {
            if (is.null(object$terms)) stop("term prediction requires a formula fit", call. = FALSE)
            aa <- attr(design$x, "assign")[good]
            labs <- attr(object$terms, "term.labels")
            center <- colMeans(as.matrix(model.matrix(object)[, good, drop = FALSE]))
            xx <- sweep(as.matrix(x), 2L, center)
            out <- se <- matrix(0, nrow(x), length(labs), dimnames = list(rownames(x), labs))
            for (j in seq_along(labs)) {
                ix <- which(aa == j)
                out[, j] <- drop(xx[, ix, drop = FALSE] %*% b[good][ix])
                if (se.fit) se[, j] <- sqrt(.quad_rows(xx[, ix, drop = FALSE], v[ix, ix, drop = FALSE]))
            }
            attr(out, "constant") <- sum(center * b[good])
            if (!is.null(terms)) { out <- out[, terms, drop = FALSE]; se <- se[, terms, drop = FALSE] }
        } else {
            eta[nonestimable] <- NA_real_
            out <- eta
            if (se.fit) se <- sqrt(pmax(0, dispersion * .model_quad_rows(x, object)))
            if (se.fit) se[nonestimable] <- NA_real_
            if (type == "response") {
                if (se.fit) se <- se * abs(.glm_mu_eta(object,eta))
                out <- object$family$linkinv(eta)
            }
        }
    }
    if (is.null(newdata) && !is.null(object$na.action)) {
        out <- napredict(object$na.action, out)
        if (se.fit) se <- napredict(object$na.action, se)
    }
    if (se.fit) .fastglm_prediction(list(fit = out, se.fit = se, residual.scale = sqrt(dispersion)),type) else out
}
