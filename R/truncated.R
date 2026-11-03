#' Zero-truncated count regression
#'
#' Fits Poisson or negative-binomial regression conditional on a strictly
#' positive response. The linear predictor describes the mean of the
#' underlying untruncated distribution; response predictions describe the
#' conditional mean among positive observations.
#' @param x numeric model matrix or model formula.
#' @param y positive integer response for a matrix design.
#' @param data data frame for a formula.
#' @param dist `"poisson"` or `"negbin"`.
#' @param weights,offset prior weights and offsets.
#' @param theta fixed positive negative-binomial size; `NULL` estimates it.
#' @param start initial regression coefficients.
#' @param tol,maxit coefficient convergence controls.
#' @param outer.tol,outer.maxit joint coefficient--theta convergence controls.
#' @returns A `fastglm_truncated` model with joint covariance when theta is estimated.
#' @export
fastglm_truncated <- function(x, y = NULL, data = NULL, dist = c("poisson", "negbin"),
                              weights = NULL, offset = NULL, theta = NULL,
                              start = NULL, tol = 1e-8, maxit = 100L,
                              outer.tol = 1e-8, outer.maxit = 100L) {
    cl <- match.call(); dist <- match.arg(dist)
    frame <- tt <- NULL
    if (inherits(x, "formula")) {
        mc <- match.call(expand.dots = FALSE)
        mc <- mc[c(1L, match(c("x", "data", "weights", "offset"), names(mc), 0L))]
        mc[[1L]] <- quote(stats::model.frame); names(mc)[names(mc) == "x"] <- "formula"
        frame <- eval(mc, parent.frame()); tt <- attr(frame, "terms")
        x <- model.matrix(tt, frame); y <- model.response(frame)
        weights <- model.weights(frame); offset <- model.offset(frame)
    }
    if (is.null(weights)) weights <- rep(1, length(y))
    if (is.null(offset)) offset <- rep(0, length(y))
    if(!is.matrix(x) || !is.numeric(x) || ncol(x)<1L)
        stop("truncated regression requires a numeric matrix with at least one column",call.=FALSE)
    if(is.integer(x)) storage.mode(x)<-"double"
    if(!is.numeric(y) || !is.null(dim(y)))
        stop("truncated counts must be a numeric vector",call.=FALSE)
    .validate_glm_data(x, y, weights, offset)
    .check_iterations(tol, maxit); .check_iterations(outer.tol, outer.maxit)
    if (any(y < 1 | y != floor(y))) stop("truncated counts must be positive integers", call. = FALSE)
    if (!is.null(theta) && (length(theta) != 1L || !is.finite(theta) || theta <= 0)) stop("'theta' must be positive", call. = FALSE)
    if (is.null(start)) start <- .lm.fit(x * sqrt(weights), (log(y) - offset)*sqrt(weights))$coefficients
    if(!is.numeric(start) || length(start)!=ncol(x) || any(!is.finite(start)))
        stop("starting coefficients must be finite and match the design columns",call.=FALSE)
    res <- fit_glm_truncated(x, as.numeric(y), weights, offset, start,
        dist == "negbin", if (is.null(theta)) -1 else theta, is.null(theta),
        tol, as.integer(maxit), outer.tol, as.integer(outer.maxit))
    cn <- colnames(x); if (is.null(cn)) cn <- paste0("X", seq_len(ncol(x)))
    names(res$coefficients) <- names(res$se) <- cn
    dimnames(res$cov.unscaled) <- list(cn, cn)
    rn <- c(cn, if (dist == "negbin" && is.null(theta)) "theta")
    dimnames(res$vcov.full) <- list(rn, rn)
    res$x <- x; res$y <- y; res$prior.weights <- weights; res$offset <- offset
    res$dist <- dist; res$theta.estimated <- dist == "negbin" && is.null(theta)
    res$rank <- ncol(x); res$nobs <- sum(weights > 0); res$df.residual <- res$nobs - length(rn)
    res$dispersion <- 1; res$aic <- -2*res$loglik + 2*length(rn)
    res$terms <- tt; res$model <- frame; res$call <- cl
    if (!is.null(tt)) {
        res$xlevels <- .getXlevels(tt, frame); res$contrasts <- attr(x, "contrasts")
        res$na.action <- attr(frame, "na.action")
    }
    if (!res$converged) warning("truncated-count model did not converge", call. = FALSE)
    class(res) <- "fastglm_truncated"
    res$theta.boundary <- .count_theta_boundary(res)
    if(res$theta.boundary) {
        fixed<-res;fixed$theta.estimated<-FALSE
        v<-tryCatch(.count_information(fixed)$inverse,error=function(e)
            matrix(NA_real_,length(coef(res)),length(coef(res))))
        dimnames(v)<-list(cn,cn);res$cov.unscaled<-v;res$se<-sqrt(diag(v))
        res$vcov.full[,]<-NA_real_;res$vcov.full[seq_along(cn),seq_along(cn)]<-v
        res$SE.theta<-NA_real_
    }
    res
}

#' @export
vcov.fastglm_truncated <- function(object, full = FALSE,type=NULL,cluster=NULL,...) {
    if(!is.null(type) || !is.null(cluster))
        return(fastglm_vcov(object,if(is.null(type)) "HC0" else type,cluster=cluster,full=full,...))
    if (full) object$vcov.full else object$cov.unscaled
}
#' @export
logLik.fastglm_truncated <- function(object, ...) {
    structure(object$loglik, df = object$rank + object$theta.estimated,
              nobs = object$nobs, class = "logLik")
}
#' @export
nobs.fastglm_truncated <- nobs.fastglm
#' @export
model.matrix.fastglm_truncated <- model.matrix.fastglm
#' @exportS3Method print fastglm_truncated
print.fastglm_truncated <- print.fastglm
#' @export
predict.fastglm_truncated <- function(object, newdata = NULL,
                                      type = c("response", "link", "count", "prob"),
                                      se.fit = FALSE, at = 1:10, offset = NULL, ...) {
    type <- match.arg(type); design <- .prediction_design(object, newdata, offset)
    eta <- drop(design$x %*% coef(object)) + design$offset
    calc <- function(eta, theta) {
        mu <- exp(eta); mm <- .count_moments(mu, object$dist, theta)
        if (type == "link") eta else if (type == "count") mu else if (type == "response") mm$mean else
            vapply(at, function(y) {
                if (y == 0) return(rep(0, length(mu)))
                (if (object$dist == "poisson") dpois(y, mu) else dnbinom(y, mu=mu, size=theta)) / mm$q
            }, numeric(length(mu)))
    }
    fit <- calc(eta, object$theta)
    if (!se.fit) return(fit)
    if (type == "prob") stop("standard errors for truncated probability masses are not yet supported", call. = FALSE)
    h <- .Machine$double.eps^(1/3)
    de <- (calc(eta+h, object$theta)-calc(eta-h, object$theta))/(2*h)
    grad <- design$x * de
    if (object$theta.estimated) {
        ht <- min(h*(1+object$theta), object$theta/4)
        dt <- (calc(eta,object$theta+ht)-calc(eta,object$theta-ht))/(2*ht)
        grad <- cbind(grad, dt)
    }
    .fastglm_prediction(list(fit = fit, se.fit = sqrt(pmax(0, .quad_rows(grad, object$vcov.full)))),type)
}

#' @exportS3Method summary fastglm_truncated
summary.fastglm_truncated <- function(object,...) {
    b <- coef(object); se <- sqrt(diag(vcov(object))); z <- b/se
    structure(list(call=object$call, title="Zero-truncated regression", coefficients=cbind(Estimate=b,`Std. Error`=se,
        `z value`=z,`Pr(>|z|)`=2*pnorm(-abs(z))),loglik=object$loglik,
        nobs=object$nobs,theta=object$theta,SE.theta=object$SE.theta,
        converged=object$converged),class="summary.fastglm_count")
}
#' @export
residuals.fastglm_truncated <- function(object,type=c("response","pearson"),...) {
    type <- match.arg(type); r <- object$y-object$fitted.values
    if(type=="pearson") r <- r/sqrt(.count_moments(object$lambda,object$dist,object$theta)$variance)
    if(!is.null(object$na.action)) r <- naresid(object$na.action,r)
    r
}
#' @export
simulate.fastglm_truncated <- function(object,nsim=1L,seed=NULL,...) {
    if(!is.null(seed)) {
        had <- exists(".Random.seed",.GlobalEnv,inherits=FALSE)
        if(had) old <- get(".Random.seed",.GlobalEnv)
        on.exit(if(had) assign(".Random.seed",old,.GlobalEnv) else
            if(exists(".Random.seed",.GlobalEnv,inherits=FALSE)) rm(".Random.seed",envir=.GlobalEnv))
        set.seed(seed)
    }
    mu <- object$lambda; q <- .count_moments(mu,object$dist,object$theta)$q
    draw <- function() {
        u <- runif(length(mu))*q
        if(object$dist=="poisson") qpois(u,mu,lower.tail=FALSE) else qnbinom(u,mu=mu,size=object$theta,lower.tail=FALSE)
    }
    out <- as.data.frame(replicate(nsim,draw())); names(out) <- paste0("sim_",seq_len(nsim)); out
}
