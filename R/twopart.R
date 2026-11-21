#' Two-part regression for nonnegative continuous outcomes
#'
#' Fits a binomial model for a positive outcome and a Gamma model with a log
#' link among positive outcomes. The marginal mean is the product of the
#' probability of a positive outcome and the conditional positive mean.
#' @param formula numeric matrix, `dgCMatrix`, `big.matrix`, [fastglm_source()], or a formula
#'   with one or two right-hand sides, as in [fastglm_hurdle()].
#'   A list with elements `count` and `zero` (formulas, as [stats::step()]
#'   writes the terms of the model into its call) is taken as the formula
#'   `y ~ count terms | zero terms`.
#' @param data,subset,na.action,weights,offset model-frame arguments. A vector
#'   offset applies to the positive component; a two-column matrix supplies
#'   positive and binary offsets.
#' @param link link for the probability of a positive outcome.
#' @param method,tol,maxit solver and convergence controls for [fastglm()].
#' @param firth apply mean bias reduction to both components.
#' @param response nonnegative continuous response for a matrix design.
#' @param z optional binary-component design for matrix inputs; defaults to
#'   the positive-component matrix. Source blocks can supply `Z` and `X`.
#' @param block.size maximum number of matrix rows processed together for
#'   model validation. A source determines its own block sizes.
#' @param threads the number of threads for the compiled computations of both
#'   component fits, or `NULL` (default) for the session setting of
#'   [fastglm_threads()]. Results are identical for every number of threads.
#' @param trace `TRUE` to print one line per iteration of each component
#'   fit (the binary part, then the positive part; see [fastglm()]), `FALSE`
#'   for none, or `NULL` (default) for `getOption("fastglm.trace")`, which
#'   defaults to `TRUE` for a [fastglm_source()] in an interactive session
#'   and `FALSE` otherwise.
#' @details [logLik()] (and hence [AIC()], [BIC()], and `extractAIC()`) is the
#'   Gamma log likelihood of the positive observations with the shape at its
#'   maximum likelihood estimate given the fitted means, plus the weighted
#'   Bernoulli log likelihood of the binary part. With weights \eqn{w}{w}
#'   (total \eqn{W}{W}) and deviance \eqn{D}{D} of the positive part, the
#'   shape \eqn{\nu}{nu} solves
#'   \eqn{\log\nu - \psi(\nu) = D/(2W)}{log(nu) - digamma(nu) = D / (2 W)}, the
#'   estimate that `MASS::gamma.shape()` gives for the Gamma GLM with unit
#'   weights. It is counted as one parameter in the `df` of [logLik()].
#'   [stats::glm()] instead evaluates the Gamma likelihood at the plug-in shape
#'   \eqn{W/D}{W / D} (version 0.1.4 of this package did the same), which gives
#'   a smaller log likelihood whose differences between fits are not
#'   likelihood ratio statistics.
#' @returns A `fastglm_twopart` model supporting prediction, delta-method
#'   standard errors, simulation, and component summaries. Its element
#'   `threads` records the thread setting of the fit, and `shape` the maximum
#'   likelihood Gamma shape.
#' @export
fastglm_twopart <- function(formula, data, subset, na.action, weights, offset,
                            link = "logit", method = "auto", tol = 1e-8,
                            maxit = 100L, firth = FALSE,
                            response = NULL, z = NULL, block.size = 8192L, threads = NULL,
                            trace = NULL) {
    # A list of count and zero formulas (as stats::step() writes the terms of
    # a two-part model into the call) is the two-part formula.
    formula <- .two_part_formula_arg(formula)
    scope <- .fastglm_scope_threads(threads)
    on.exit(scope$restore(), add = TRUE)
    threads <- scope$threads
    # The component fits are traced (nested = TRUE).
    tr <- .fastglm_scope_trace(trace, streamed = inherits(formula, "fastglm_source"), nested = TRUE)
    on.exit(tr$restore(), add = TRUE)
    if(is.matrix(formula) || inherits(formula,c("dgCMatrix","big.matrix","fastglm_source"))) {
        if(!missing(subset) || !missing(na.action))
            stop("subset and na.action apply only to formula fits; prepare matrix or source rows first",call.=FALSE)
        out <- .count_backend_twopart(formula,response,z,
            if(missing(weights)) NULL else weights,if(missing(offset)) NULL else offset,
            link,method,tol,maxit,firth,block.size)
        out$call <- match.call()
        out$threads <- threads
        return(out)
    }
    cl <- match.call(); f <- Formula::Formula(formula)
    if (missing(data)) data <- environment(formula)
    mc <- match.call(expand.dots = FALSE)
    mc <- mc[c(1L, match(c("formula", "data", "subset", "na.action", "weights", "offset"), names(mc), 0L))]
    mc[[1L]] <- quote(stats::model.frame); mc$formula <- f
    mf <- eval(mc, parent.frame())
    tt <- list(count = terms(f, data=data, rhs=1L), zero = terms(f, data=data, rhs=if (length(f)[2L] > 1L) 2L else 1L))
    x <- lapply(tt, model.matrix, data=mf); y <- model.response(mf)
    w <- model.weights(mf); if (is.null(w)) w <- rep(1, length(y))
    off <- .component_offsets(mf, tt)
    .validate_glm_data(x$count, y, w, off$count)
    if (any(y < 0)) stop("two-part responses must be nonnegative", call. = FALSE)
    positive <- y > 0 & w > 0
    if (!any(positive) || all(y[w > 0] > 0)) stop("two-part models require zeros and positive observations", call. = FALSE)
    binary <- fastglm(x$zero, as.numeric(y > 0), family=binomial(link), weights=w, offset=off$zero,
                      method=method, tol=tol, maxit=maxit, firth=firth)
    fit <- fastglm(x$count[positive,,drop=FALSE], y[positive], family=Gamma("log"),
                   weights=w[positive], offset=off$count[positive], method=method, tol=tol, maxit=maxit, firth=firth)
    pc <- length(coef(fit)); pz <- length(coef(binary))
    v <- matrix(0, pc+pz, pc+pz); v[seq_len(pc),seq_len(pc)] <- vcov(fit)
    v[pc+seq_len(pz),pc+seq_len(pz)] <- vcov(binary)
    cn <- c(paste0("positive_", names(coef(fit))),paste0("zero_", names(coef(binary))))
    dimnames(v) <- list(cn,cn)
    ll <- .twopart_loglik(fit, binary)
    out <- list(coefficients=list(count=coef(fit),zero=coef(binary)),
        vcov=v, vcov.full=v, positive=fit, binary=binary,
        loglik=ll$loglik, loglik_count=ll$loglik_count, loglik_zero=ll$loglik_zero, shape=ll$shape,
        converged=fit$converged && binary$converged, dist="Gamma", link=link,
        model=mf, terms=tt, assign=lapply(x,attr,"assign"),formula=f, x=x, y=y, weights=w, offset=off,
        levels=lapply(mf[vapply(mf,is.factor,logical(1))],levels),
        contrasts=lapply(x,attr,"contrasts"), na.action=attr(mf,"na.action"),
        call=cl, df.residual=sum(w>0)-pc-pz-1, firth=firth,
        refit.controls=list(link=link,method=method,tol=tol,maxit=maxit,firth=firth,block.size=block.size),
        threads=threads)
    class(out) <- c("fastglm_twopart","fastglm_hurdle")
    out
}
#' @export
logLik.fastglm_twopart <- function(object, ...) {
    structure(object$loglik, df=length(coef(object))+1L, nobs=nobs(object), class="logLik")
}

## ---- Gamma shape at its maximum likelihood estimate ------------------------------

## h(nu) = log(nu) - digamma(nu) and h'(nu), with the asymptotic series for
## large nu, where the difference of the two terms would lose precision.
.gamma_shape_h <- function(nu)
{
    if (nu >= 1e3) 1 / (2 * nu) + 1 / (12 * nu^2) - 1 / (120 * nu^4) + 1 / (252 * nu^6)
    else log(nu) - digamma(nu)
}
.gamma_shape_dh <- function(nu)
{
    if (nu >= 1e3) -1 / (2 * nu^2) - 1 / (6 * nu^3) + 1 / (30 * nu^5) - 1 / (42 * nu^7)
    else 1 / nu - trigamma(nu)
}

## q(nu) = nu log(nu) - nu - lgamma(nu), written as
## log(nu / (2 pi)) / 2 - stirlerr(nu) so that large nu keeps its precision.
.gamma_shape_q <- function(nu)
{
    s <- if (nu >= 10) 1 / (12 * nu) - 1 / (360 * nu^3) + 1 / (1260 * nu^5) - 1 / (1680 * nu^7) +
             1 / (1188 * nu^9)
         else lgamma(nu) - (nu - 0.5) * log(nu) + nu - 0.5 * log(2 * pi)
    0.5 * log(nu / (2 * pi)) - s
}

## The maximum likelihood Gamma shape given the fitted means: the root of
## h(nu) = D / (2 W), with D the deviance and W the total of the (frequency)
## weights of the positive observations. h decreases from infinity to 0, so
## the root is unique. Newton's method on log(nu), from the approximation
## nu = (3 - t + sqrt((t - 3)^2 + 24 t)) / (12 t), t = D / (2 W), with a
## bracketing search as fallback.
.gamma_ml_shape <- function(D, W)
{
    t <- D / (2 * W)
    if (!is.finite(t) || t <= 0) return(NA_real_)
    u <- log((3 - t + sqrt((t - 3)^2 + 24 * t)) / (12 * t))
    for (it in seq_len(100L)) {
        nu <- exp(u)
        f <- .gamma_shape_h(nu) - t
        step <- f / (nu * .gamma_shape_dh(nu))
        if (!is.finite(step)) break
        u <- u - step
        if (abs(step) < 1e-14 * max(1, abs(u))) return(exp(u))
    }
    exp(stats::uniroot(function(v) .gamma_shape_h(exp(v)) - t, c(u - 1, u + 1), extendInt = "downX",
                       tol = 1e-14)$root)
}

## The log likelihood of the positive (Gamma) part with the shape at its
## maximum likelihood estimate given the fitted means. With frequency weights
## w (total W) and deviance D, the Gamma log likelihood at shape nu is
## W q(nu) - nu D / 2 - sum(w log y). logLik() of the Gamma GLM uses the
## plug-in shape W / D (the reciprocal of deviance over weight total, as
## stats::glm()), so the maximized value adds W q(nu.hat) - nu.hat D / 2 -
## (W q(W / D) - W / 2), which needs only D, W, and that log likelihood (the
## same for in-memory and streamed fits). Returns the log likelihood and the
## shape.
.twopart_positive_loglik <- function(positive)
{
    ll <- as.numeric(logLik(positive))
    D <- as.numeric(positive$deviance)
    W <- if (!is.null(positive$prior.weights)) sum(positive$prior.weights) else
        if (!is.null(positive$fingerprint.response)) positive$fingerprint.response[["wsum"]] else NA_real_
    nu <- .gamma_ml_shape(D, W)
    if (!is.finite(nu)) return(list(loglik = ll, shape = NA_real_))
    g <- function(v) W * .gamma_shape_q(v) - v * D / 2
    list(loglik = ll + g(nu) - g(W / D), shape = nu)
}

## The log likelihood of a two-part fit: the positive part at the maximum
## likelihood shape, plus the weighted Bernoulli log likelihood of the binary
## part, minus half its deviance for the 0/1 response (equal to logLik() of
## the binary GLM with integer weights, and the same for in-memory and
## streamed fits with any weights).
.twopart_loglik <- function(positive, binary)
{
    pos <- .twopart_positive_loglik(positive)
    list(loglik = pos$loglik - as.numeric(binary$deviance) / 2, loglik_count = pos$loglik,
         loglik_zero = -as.numeric(binary$deviance) / 2, shape = pos$shape)
}
#' @exportS3Method print fastglm_twopart
print.fastglm_twopart <- function(x, ...) {
    print(summary(x), ...); invisible(x)
}
