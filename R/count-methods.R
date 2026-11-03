.component_offsets <- function(mf, terms) {
    n <- nrow(mf)
    one <- function(tt) {
        out <- rep(0, n)
        ix <- attr(tt, "offset")
        for (j in ix) {
            expr <- as.list(attr(tt, "variables"))[-1L][[j]]
            nm <- paste(deparse(expr, width.cutoff = 500L), collapse = "")
            z <- mf[[nm]]
            if (is.null(z)) z <- eval(expr, mf, environment(tt))
            out <- out + z
        }
        as.numeric(out)
    }
    out <- lapply(terms, one)
    ext <- mf[["(offset)"]]
    if (!is.null(ext)) {
        if (is.matrix(ext)) {
            if (ncol(ext) != 2L) stop("matrix offsets must have two columns", call. = FALSE)
            out$count <- out$count + ext[, 1L]; out$zero <- out$zero + ext[, 2L]
        } else out$count <- out$count + ext
    }
    out
}

.count_design <- function(object, newdata = NULL) {
    if (is.null(newdata)) {
        x <- object$x
        if (is.null(x)) {
            if (is.null(object$model)) stop("supply newdata: model frame was not retained", call. = FALSE)
            x <- lapply(names(object$terms), function(k) model.matrix(object$terms[[k]], object$model,
                    contrasts.arg = object$contrasts[[k]]))
            names(x) <- names(object$terms)
        }
        return(list(x = x, offset = object$offset))
    }
    if(is.matrix(newdata) || inherits(newdata,"Matrix") ||
       (is.list(newdata) && !is.data.frame(newdata) &&
        (!is.null(newdata$count) || !is.null(newdata$X)))) {
        z <- if(is.matrix(newdata) || inherits(newdata,"Matrix")) list(X=newdata,Z=newdata) else
            list(X=if(is.null(newdata$count)) newdata$X else newdata$count,
                 Z=if(is.null(newdata$zero)) newdata$Z else newdata$zero,
                 offset=newdata$offset)
        z <- .count_normalize_block(z,TRUE,materialize=FALSE)
        if(ncol(z$X)!=length(object$coefficients$count) ||
           ncol(z$Z)!=length(object$coefficients$zero) || nrow(z$X)!=nrow(z$Z))
            stop("prediction designs must match the fitted component columns and share row counts",call.=FALSE)
        return(list(x=list(count=z$X,zero=z$Z),offset=z$offset))
    }
    frames <- lapply(object$terms, function(tt) model.frame(delete.response(tt), newdata,
                        xlev = object$levels, na.action = na.pass))
    x <- lapply(names(frames), function(k) model.matrix(delete.response(object$terms[[k]]), frames[[k]],
                contrasts.arg = object$contrasts[[k]]))
    names(x) <- names(frames)
    off <- lapply(names(frames), function(k) {
        z <- model.offset(frames[[k]])
        if (is.null(z)) rep(0, nrow(frames[[k]])) else z
    }); names(off) <- names(frames)
    if (!is.null(object$call$offset)) {
        ext <- eval(object$call$offset, newdata, environment(object$formula))
        if (is.matrix(ext)) { off$count <- off$count + ext[, 1L]; off$zero <- off$zero + ext[, 2L] }
        else off$count <- off$count + ext
    }
    source.offset <- object$source$schema$offset
    if(length(source.offset)) {
        for(j in seq_along(source.offset)) {
            nm<-source.offset[[j]]
            if(!nm %in% names(newdata)) stop("prediction data lacks source offset column '",nm,"'",call.=FALSE)
            off[[j]]<-off[[j]]+newdata[[nm]]
        }
    }
    list(x = x, offset = off)
}

.count_moments <- function(mu, dist, theta = NULL) {
    lp0 <- if (dist == "poisson") -mu else -theta * log1p(mu / theta)
    q <- -expm1(lp0)
    m <- mu / q
    # Continuous limits avoid 0/0 for rates below machine range.
    m[mu == 0] <- 1
    variance <- if (dist == "poisson") mu else mu + mu^2 / theta
    vt <- (variance + mu^2) / q - m^2
    list(p0 = exp(lp0), q = q, mean = m, variance = pmax(0, vt))
}

.count_predict <- function(object, design, parameters, type, at) {
    pc <- length(object$coefficients$count); pz <- length(object$coefficients$zero)
    eta <- drop(design$x$count %*% parameters[seq_len(pc)]) + design$offset$count
    eta0 <- drop(design$x$zero %*% parameters[pc + seq_len(pz)]) + design$offset$zero
    mu <- exp(eta); pi <- make.link(object$link)$linkinv(eta0)
    theta <- if (object$dist == "negbin") utils::tail(parameters, 1L) else NULL
    zi <- inherits(object, "fastglm_zi")
    continuous <- inherits(object, "fastglm_twopart")
    if (type == "link") return(cbind(count = eta, zero = eta0))
    if (continuous) {
        if (type == "prob") stop("probability masses are only defined for count models", call. = FALSE)
        return(switch(type, response = pi * mu, count = mu, zero = 1 - pi))
    }
    mom <- .count_moments(mu, object$dist, theta)
    if (type == "zero") return(if (zi) pi + (1 - pi) * mom$p0 else 1 - pi)
    if (type == "count") return(if (zi) mu else mom$mean)
    if (type == "response") return(if (zi) (1 - pi) * mu else pi * mom$mean)
    out <- vapply(at, function(y) {
        d <- if (object$dist == "poisson") dpois(y, mu) else dnbinom(y, mu = mu, size = theta)
        if (zi) (1 - pi) * d + pi * (y == 0) else if (y == 0) 1 - pi else pi * d / mom$q
    }, numeric(length(mu)))
    colnames(out) <- as.character(at)
    out
}

#' Predictions and inference for two-component models
#'
#' The response prediction averages over both components. For hurdle models,
#' `count` is the conditional mean among positive counts and `zero` is the
#' probability of observing zero. For zero-inflated models, `count` is the
#' mean of the untruncated count distribution. Prediction standard errors use
#' the delta method and include uncertainty in theta when it is estimated.
#' @param object a hurdle, zero-inflated, or two-part fit.
#' @param newdata optional data frame or list of design matrices named `count`
#'   and `zero`, optionally with `offset=list(count=...,zero=...)`. With a
#'   streaming model, omitting it rereads the fitted source.
#' @param type prediction scale: marginal response, count-component mean,
#'   zero probability, probability masses, or component linear predictors.
#' @param se.fit return delta-method standard errors.
#' @param at nonnegative integer counts for `type = "prob"`.
#' @param sink optional function of `(value, chunk_index)` for a streaming fit.
#'   It receives each block of predictions so that results can be written or
#'   summarized without retaining predictions for all observations in memory.
#' @param ... additional arguments.
#' @returns Predicted values, or a `fastglm_prediction` list with `fit` and
#'   `se.fit`; see \link{fastglm-prediction-methods} for printing and table extraction.
#' @export
predict.fastglm_hurdle <- function(object, newdata = NULL,
                                   type = c("response", "count", "zero", "prob", "link"),
                                   se.fit = FALSE, at = 0:10, sink=NULL, ...) {
    type <- match.arg(type)
    if (any(!is.finite(at) | at < 0 | at != floor(at))) stop("'at' must contain nonnegative integers", call. = FALSE)
    if(is.null(newdata) && !is.null(object$source) && is.function(object$fetch)) {
        if(!is.null(sink) && !is.function(sink)) stop("'sink' must be a function",call.=FALSE)
        out <- if(is.null(sink)) vector("list",object$source$n_chunks)
        for(k in seq_len(object$source$n_chunks)) {
            z <- .count_normalize_block(object$fetch(k),TRUE)
            design <- list(x=list(count=z$X,zero=z$Z),offset=z$offset)
            value <- .count_prediction_result(object,design,type,se.fit,at)
            if(is.null(sink)) out[[k]] <- value else sink(value,k)
        }
        if(!is.null(sink)) return(invisible(object$source$n_chunks))
        bind <- function(x) if(is.null(dim(x[[1L]]))) unlist(x,use.names=FALSE) else do.call(rbind,x)
        if(se.fit) return(.fastglm_prediction(list(fit=bind(lapply(out,`[[`,"fit")),
            se.fit=bind(lapply(out,`[[`,"se.fit"))),type))
        return(bind(out))
    }
    if(!is.null(sink)) stop("a prediction sink requires a streaming fit and no newdata",call.=FALSE)
    design <- .count_design(object, newdata)
    value <- .count_prediction_result(object,design,type,se.fit,at)
    if(is.null(newdata) && !is.null(object$na.action)) {
        if(se.fit) {value$fit<-napredict(object$na.action,value$fit);value$se.fit<-napredict(object$na.action,value$se.fit)}
        else value<-napredict(object$na.action,value)
    }
    value
}
.count_prediction_result <- function(object,design,type,se.fit,at) {
    pars <- coef(object)
    if (object$dist == "negbin") pars <- c(pars, theta = object$theta)
    val <- .count_predict(object, design, pars, type, at)
    if (se.fit) {
        v <- object$vcov.full
        if (is.null(v)) v <- object$vcov
        grad <- matrix(0, length(val), length(pars))
        for (j in seq_along(pars)) {
            h <- .Machine$double.eps^(1/3) * (1 + abs(pars[j]))
            if (names(pars)[j] == "theta") h <- min(h, pars[j] / 4)
            a <- b <- pars; a[j] <- a[j] + h; b[j] <- b[j] - h
            grad[, j] <- (as.vector(.count_predict(object, design, a, type, at)) -
                          as.vector(.count_predict(object, design, b, type, at))) / (2 * h)
        }
        active <- which(colSums(abs(grad))>0)
        se <- if(length(active)) sqrt(pmax(0,rowSums((grad[,active,drop=FALSE] %*%
            v[active,active,drop=FALSE])*grad[,active,drop=FALSE]))) else rep(0,nrow(grad))
        if (!is.null(dim(val))) dim(se) <- dim(val)
    }
    if (se.fit) .fastglm_prediction(list(fit = val, se.fit = se),type) else val
}

#' @rdname predict.fastglm_hurdle
#' @export
predict.fastglm_zi <- predict.fastglm_hurdle

#' @export
fitted.fastglm_hurdle <- function(object, ...) predict(object, type = "response")
#' @export
fitted.fastglm_zi <- fitted.fastglm_hurdle
#' @export
nobs.fastglm_hurdle <- function(object, ...) if(!is.null(object$nobs)) object$nobs else sum(object$weights > 0)
#' @export
nobs.fastglm_zi <- nobs.fastglm_hurdle
#' @export
model.matrix.fastglm_hurdle <- function(object, model = c("count", "zero"), ...) {
    .count_design(object)$x[[match.arg(model)]]
}
#' @export
model.matrix.fastglm_zi <- model.matrix.fastglm_hurdle

#' @exportS3Method summary fastglm_hurdle
summary.fastglm_hurdle <- function(object, ...) {
    b <- coef(object); se <- sqrt(diag(vcov(object))); z <- b / se
    structure(list(call = object$call, title=if(inherits(object,"fastglm_twopart")) "Two-part binomial / Gamma regression" else
        if(inherits(object,"fastglm_zi")) "Zero-inflated regression" else "Hurdle regression",
        coefficients = cbind(Estimate = b, `Std. Error` = se, `z value` = z, `Pr(>|z|)` = 2 * pnorm(-abs(z))),
        loglik = as.numeric(logLik(object)), nobs = nobs(object), theta = object$theta,
        SE.theta = object$SE.theta, converged = object$converged), class = "summary.fastglm_count")
}
#' @exportS3Method summary fastglm_zi
summary.fastglm_zi <- summary.fastglm_hurdle
#' @exportS3Method print summary.fastglm_count
print.summary.fastglm_count <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(if(is.null(x$title)) "Count regression" else x$title,x$call)
    .fastglm_print_coefficients(x$coefficients,digits=digits,...)
    cat("\n")
    if(!is.null(x$theta) && is.finite(x$theta))
        .fastglm_print_table(data.frame(Theta=x$theta,`Std. Error`=x$SE.theta,check.names=FALSE),digits)
    .fastglm_print_field("Log-likelihood",format(x$loglik,digits=digits))
    .fastglm_print_field("Observations",x$nobs)
    .fastglm_print_convergence(x$converged)
    invisible(x)
}

#' @export
residuals.fastglm_hurdle <- function(object, type = c("response", "pearson"), sink=NULL, ...) {
    type <- match.arg(type)
    if(!is.null(object$source) && is.function(object$fetch)) {
        if(!is.null(sink) && !is.function(sink)) stop("'sink' must be a function",call.=FALSE)
        out<-if(is.null(sink)) vector("list",object$source$n_chunks)
        for(k in seq_len(object$source$n_chunks)) {
            z<-.count_normalize_block(object$fetch(k),TRUE)
            d<-list(x=list(count=z$X,zero=z$Z),offset=z$offset)
            r<-.count_residual_result(object,z$y,d,type)
            if(is.null(sink)) out[[k]]<-r else sink(r,k)
        }
        if(is.null(sink)) return(unlist(out,use.names=FALSE))
        return(invisible(object$source$n_chunks))
    }
    if(!is.null(sink)) stop("a residual sink requires a streaming fit",call.=FALSE)
    if (is.null(object$y)) stop("response was not retained", call. = FALSE)
    r<-.count_residual_result(object,object$y,.count_design(object),type)
    if (!is.null(object$na.action)) r <- naresid(object$na.action, r)
    r
}
.count_residual_result <- function(object,y,d,type) {
    pars <- coef(object)
    if (object$dist == "negbin") pars <- c(pars, theta = object$theta)
    m <- .count_predict(object, d, pars, "response", NULL)
    r <- y - m
    if (type == "pearson") {
        eta <- .count_predict(object, d, pars, "link", NULL)
        mu <- exp(eta[, 1]); pi <- make.link(object$link)$linkinv(eta[, 2])
        if (inherits(object, "fastglm_twopart")) variance <- pi * mu^2 * object$positive$dispersion + pi * (1-pi) * mu^2
        else if (inherits(object, "fastglm_zi")) {
            v <- if (object$dist == "poisson") mu else mu + mu^2/object$theta
            variance <- (1-pi)*v + pi*(1-pi)*mu^2
        } else {
            mm <- .count_moments(mu, object$dist, object$theta)
            variance <- pi*mm$variance + pi*(1-pi)*mm$mean^2
        }
        r <- r / sqrt(variance)
    }
    r
}
#' @export
residuals.fastglm_zi <- residuals.fastglm_hurdle

#' @export
simulate.fastglm_hurdle <- function(object, nsim = 1L, seed = NULL, ...) {
    if (!is.null(seed)) {
        had <- exists(".Random.seed", .GlobalEnv, inherits = FALSE)
        if (had) old <- get(".Random.seed", .GlobalEnv)
        on.exit(if (had) assign(".Random.seed", old, .GlobalEnv) else
            if (exists(".Random.seed", .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv))
        set.seed(seed)
    }
    eta <- predict(object, type = "link")
    mu <- exp(eta[, 1]); pi <- make.link(object$link)$linkinv(eta[, 2]); n <- length(mu)
    draw <- function() {
        if (inherits(object, "fastglm_twopart")) return(rbinom(n, 1, pi) * rgamma(n, shape = 1/object$positive$dispersion, scale = mu*object$positive$dispersion))
        if (inherits(object, "fastglm_zi")) {
            val <- if (object$dist == "poisson") rpois(n, mu) else rnbinom(n, mu = mu, size = object$theta)
            return(rbinom(n, 1, 1-pi) * val)
        }
        q <- .count_moments(mu, object$dist, object$theta)$q
        u <- runif(n) * q
        val <- if (object$dist == "poisson") qpois(u, mu, lower.tail = FALSE) else qnbinom(u, mu = mu, size = object$theta, lower.tail = FALSE)
        rbinom(n, 1, pi) * val
    }
    out <- as.data.frame(replicate(nsim, draw())); names(out) <- paste0("sim_", seq_len(nsim)); out
}
#' @export
simulate.fastglm_zi <- simulate.fastglm_hurdle
