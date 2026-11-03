#' Fit a generalized linear model from a formula
#'
#' Constructs the model frame using the same conventions as [stats::glm()].
#' Factors, contrasts, offsets, missing observations, and prediction terms are
#' retained for subsequent inference and prediction.
#' @param x a model formula.
#' @param data,subset,na.action,weights,offset,contrasts See [stats::glm()].
#' @param family a GLM family.
#' @param method decomposition number or name; `"auto"` tries Cholesky and
#'   falls back to pivoted QR for a singular or ill-conditioned dense design.
#' @param tol,maxit convergence tolerance and iteration limit.
#' @param firth use mean bias reduction.
#' @param ... additional arguments to [stats::glm()].
#' @returns A fitted model inheriting from `fastglmFit`, `glm`, and `lm`.
#' @export
fastglm.formula <- function(x, data, family = gaussian(), subset, na.action,
                            weights, offset, contrasts = NULL, method = 0L,
                            tol = 1e-8, maxit = 100L, firth = FALSE, ...) {
    cl <- match.call(expand.dots = TRUE)
    cl[[1L]] <- quote(stats::glm)
    cl$formula <- cl$x
    cl$x <- TRUE
    cl$method <- quote(fastglm::fastglm_fit)
    cl$tol <- cl$maxit <- cl$firth <- NULL
    cl$control <- list(fastmethod = method, tol = tol, maxit = maxit, firth = firth)
    out <- eval(cl, parent.frame())
    out$call <- match.call()
    out$call[[1L]] <- quote(fastglm::fastglm)
    out$call$formula <- out$call$x
    out$call$x <- NULL
    out
}

.prediction_design <- function(object, newdata = NULL, offset = NULL) {
    if (is.null(newdata)) {
        x <- object$x
        if (is.null(x) && !is.null(object$terms)) x <- model.matrix(object)
        off <- object$offset
    } else if (!is.null(object$terms)) {
        tt <- delete.response(object$terms)
        mf <- model.frame(tt, newdata, na.action = na.pass, xlev = object$xlevels)
        x <- model.matrix(tt, mf, contrasts.arg = object$contrasts)
        off <- model.offset(mf)
        if (is.null(offset) && !is.null(object$call$offset)) {
            ext <- eval(object$call$offset, newdata, environment(object$terms))
            off <- if (is.null(off)) ext else off + ext
        }
        source.offset <- object$source$schema$offset
        if(is.null(offset) && length(source.offset)) {
            name <- source.offset[[1L]]
            if(!name %in% names(newdata)) stop("prediction data lack offset column '",name,"'",call.=FALSE)
            off <- if(is.null(off)) newdata[[name]] else off+newdata[[name]]
        }
    } else {
        x <- newdata
        if (is.null(dim(x))) x <- as.matrix(x)
        off <- NULL
    }
    if (is.null(x)) stop("prediction requires newdata or a retained design", call. = FALSE)
    if (ncol(x) != length(coef(object))) stop("prediction design has the wrong number of columns", call. = FALSE)
    if (!is.null(offset)) off <- offset
    if (is.null(off)) off <- rep(0, nrow(x))
    if (length(off) == 1L) off <- rep(off, nrow(x))
    if (length(off) != nrow(x)) stop("prediction offset has the wrong length", call. = FALSE)
    list(x = x, offset = off)
}

# Perform row calculations in blocks, including for sparse and mapped designs.
.design_blocks <- function(x, fun, block.size = 8192L) {
    n <- nrow(x)
    for (first in seq.int(1L, max(1L, n), by = block.size)) {
        if (n == 0L) break
        index <- first:min(n, first + block.size - 1L)
        z <- if (inherits(x, "big.matrix")) x[index, , drop = FALSE] else x[index, , drop = FALSE]
        fun(as.matrix(z), index)
    }
    invisible(NULL)
}

.quad_rows <- function(x, covariance) {
    out <- numeric(nrow(x))
    .design_blocks(x, function(z, i) out[i] <<- rowSums((z %*% covariance) * z))
    out
}

#' @export
predict.fastglmFit <- function(object, newdata = NULL,
                               type = c("link", "response", "terms"),
                               se.fit = FALSE, dispersion = NULL, offset = NULL,
                               terms = NULL, ...) {
    predict.fastglm(object, newdata, type, se.fit, dispersion, offset, terms, ...)
}

#' @export
model.matrix.fastglm <- function(object, ...) {
    if (!is.null(object$x)) return(object$x)
    if (is.null(object$terms)) stop("model matrix was not retained", call. = FALSE)
    stats::model.matrix(object$terms, model.frame(object), contrasts.arg = object$contrasts)
}

#' @export
model.matrix.fastglmFit <- model.matrix.fastglm

#' @export
nobs.fastglm <- function(object, ...) {
    if (!is.null(object$nobs)) object$nobs else sum(object$prior.weights > 0)
}

#' @export
nobs.fastglmFit <- nobs.fastglm

#' @export
hatvalues.fastglm <- function(model, ...) {
    good <- !is.na(coef(model))
    x <- model.matrix(model)
    if (!all(good)) x <- x[, good, drop = FALSE]
    h <- model$weights * .quad_rows(x, model$cov.unscaled[good, good, drop = FALSE])
    if (!is.null(model$na.action)) {
        h <- napredict(model$na.action, h)
        h[as.integer(model$na.action)] <- 0
    }
    h
}

#' @export
hatvalues.fastglmFit <- hatvalues.fastglm

#' Confidence intervals for GLM coefficients
#'
#' Returns Wald intervals by default. Binomial-logit Firth fits can instead
#' use penalized profile likelihood intervals; see [fastglm_profile_test()]
#' for assumptions, numerical controls, and diagnostics.
#' @param object a fitted fastglm model.
#' @param parm coefficient names or indices; defaults to all coefficients.
#' @param level confidence level.
#' @param ... additional arguments reserved for methods.
#' @param method `"wald"` or `"profile"`.
#' @param control numerical controls for profile inference.
#' @returns A matrix with lower and upper confidence limits. Profile intervals
#'   additionally carry convergence and endpoint diagnostics as attributes.
#' @export
confint.fastglm <- function(object, parm, level = 0.95, ...,
                            method = c("wald", "profile"), control = list()) {
    b <- coef(object)
    if (missing(parm)) parm <- seq_along(b)
    method <- match.arg(method)
    if (method == "profile") return(.fastglm_profile_confint(object, parm, level, control))
    if (is.character(parm)) parm <- match(parm, names(b))
    if (anyNA(parm)) stop("unknown coefficient in 'parm'", call. = FALSE)
    q <- if (is.na(.fixed_dispersion(object$family))) qt((1 + level) / 2, object$df.residual) else qnorm((1 + level) / 2)
    out <- b[parm] + outer(sqrt(diag(vcov(object)))[parm], c(-q, q))
    dimnames(out) <- list(names(b)[parm], paste0(100 * c((1-level)/2, (1+level)/2), " %"))
    out
}

#' @export
confint.fastglmFit <- confint.fastglm

#' @export
simulate.fastglm <- function(object, nsim = 1L, seed = NULL, ...) {
    if (!is.null(seed)) {
        had <- exists(".Random.seed", .GlobalEnv, inherits = FALSE)
        if (had) old <- get(".Random.seed", .GlobalEnv)
        on.exit(if (had) assign(".Random.seed", old, .GlobalEnv) else
            if (exists(".Random.seed", .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv))
        set.seed(seed)
    }
    if (is.null(object$fitted.values)) stop("simulation requires retained fitted values", call. = FALSE)
    fam <- object$family$family; mu <- object$fitted.values; n <- length(mu)
    w <- object$prior.weights
    draw <- function() {
        if (fam == "poisson") rpois(n, mu)
        else if (fam == "binomial") {
            size <- object$n
            if (length(size) != n) size <- rep(size, n)
            if(all(size==1)) {
                size <- w
                if(any(size>0 & abs(size-round(size))>1e-7))
                    stop("binomial simulation requires integer trial weights",call.=FALSE)
            }
            size[size==0] <- 1
            rbinom(n, size, mu) / size
        } else if (grepl("^Negative Binomial", fam)) rnbinom(n, mu = mu, size = family_params(object$family)["theta"])
        else if (fam == "gaussian") rnorm(n, mu, sqrt(object$dispersion / w))
        else if (fam == "Gamma") rgamma(n, shape = w / object$dispersion, scale = mu * object$dispersion / w)
        else stop("simulation is unavailable for this family", call. = FALSE)
    }
    out <- as.data.frame(replicate(nsim, draw()))
    names(out) <- paste0("sim_", seq_len(nsim))
    out
}

#' @export
simulate.fastglmFit <- simulate.fastglm

#' @export
weights.fastglm <- function(object,type=c("prior","working"),...) {
    type <- match.arg(type)
    if(type=="prior") object$prior.weights else object$weights
}
