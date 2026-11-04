#' Numerical diagnostics for a fitted GLM
#'
#' Reports whether the iteration converged, which coefficients can be estimated
#' separately, and whether fitted parameters are close to a boundary. The score
#' is the derivative of the log likelihood with respect to the coefficients; a large
#' value after accounting for predictor scale can indicate incomplete convergence.
#' Covariance conditioning describes how strongly nearly redundant coefficient
#' effects can amplify numerical error. For bias-reduced fits the ordinary score
#' need not vanish. The stored adjusted coefficient and dispersion scores instead
#' assess the equations used to fit the model at the reported estimates.
#' @param object a fitted fastglm GLM or ordinal model.
#' @returns A `fastglm_diagnostics` list with a compact printed summary.
#'   Existing components remain accessible with `$` or `[[` at full precision.
#'   `as.data.frame()` returns one row, with list columns for coefficient scores,
#'   category labels, and other nonscalar details. `score` omits division by the
#'   dispersion, and `standardized.score` uses the correspondingly unscaled
#'   information diagonal, retaining the ordinary GLM score convention. For a
#'   bias-reduced fit, `adjusted.score` contains the adjusted likelihood scores,
#'   including dispersion scaling, and `standardized.adjusted.score` is the
#'   largest absolute adjusted score divided by the square root of its expected
#'   information diagonal. `adjusted.dispersion.score` and
#'   `standardized.dispersion.score` describe the dispersion equation; they are
#'   missing for families whose dispersion is fixed. `convergence.reason`
#'   records the solver's stopping reason when available. Ordinary scores require
#'   retained data or a readable source. Stored adjusted scores remain available
#'   after data are omitted. With `inference = "none"`, adjusted scores are
#'   retained only when the existing constant-information calculation supplies
#'   the required leverages, including prepared Gamma-log fits. Other compact
#'   fits and older objects without stored adjusted scores report them as
#'   unavailable rather than computing an additional covariance matrix.
#'   QR source fits also report their remaining predictor change, joint
#'   information-scaled score per positive-weight row, and dispersion checks.
#'   For binomial probit and complementary log-log bias reduction,
#'   `step.reductions` counts scoring updates reduced to limit their largest
#'   fitted linear-predictor change to one. A reduced step alone cannot establish
#'   convergence. These safeguards do not guarantee convergence for every design.
#'   For these bias-reduced binomial fits, scores and information use stable
#'   probability-tail calculations; rounded fitted probabilities are not used
#'   to reconstruct the estimating equations.
#'   Ordinal diagnostics report scores in the effective category model after
#'   any empty interior categories have been collapsed; their standardized
#'   score is calculated in the centered/scaled fitting coordinates.
#' @md
#' @export
fastglm_diagnostics <- function(object) {
    if(inherits(object,"fastglm_ordinal"))
        return(structure(.ordinal_diagnostics(object),class="fastglm_diagnostics"))
    b <- coef(object); good <- !is.na(b)
    score <- numeric(sum(good)); info <- numeric(sum(good))
    ordinary.available <- FALSE
    add <- function(x,y,w,off) {
        x <- x[,good,drop=FALSE]; eta <- .model_eta(x,object,off)
        mu <- object$family$linkinv(eta); dm <- object$family$mu.eta(eta); v <- object$family$variance(mu)
        s <- w*dm*(y-mu)/v; s[w==0] <- 0
        ww <- w*dm^2/v; ww[w==0] <- 0
        code <- .stable_binomial_code(object)
        if(code>=0L) {
            moments <- binomial_firth_moments(as.numeric(eta),as.numeric(y),w,code)
            s <- moments$score;ww <- moments$working
        }
        score <<- score+drop(crossprod(x,s)); info <<- info+colSums(x^2*ww)
    }
    if(inherits(object,"fastglm_stream")) {
        for(k in seq_len(object$source$n_chunks)) { z <- object$fetch(k); add(z$X,z$y,z$weights,z$offset) }
        ordinary.available <- TRUE
    } else if(!is.null(object$y)) {
        x <- tryCatch(model.matrix(object), error=function(e) NULL)
        off <- object$offset
        if(!is.null(x)) {
            if(is.null(off)) off <- rep(0,nrow(x))
            .design_blocks(x,function(z,i) add(z,object$y[i],object$prior.weights[i],off[i]))
            ordinary.available <- TRUE
        }
    }
    if(!ordinary.available) score[] <- info[] <- NA_real_
    names(score) <- names(b)[good]
    max_abs <- function(z) if(length(z)) max(abs(z)) else 0
    v <- if(!is.null(object$cov.unscaled)) object$cov.unscaled[good,good,drop=FALSE]
    out <- list(converged=isTRUE(object$converged),iterations=object$iter,solver=object$solver,
        fallback=object$solver.fallback,rank=object$rank,aliased=names(b)[!good],
        score=score,score.type=if(isTRUE(object$firth)) "ordinary (before bias adjustment)" else "ordinary",
        ordinary.score.available=ordinary.available,
        standardized.score=if(ordinary.available) max_abs(score/sqrt(info)) else NA_real_,
        covariance.condition=if(length(v) && all(is.finite(v))) kappa(v,exact=TRUE) else NA_real_,
        boundary=isTRUE(object$boundary),theta.boundary=if(!is.null(object$theta)) object$theta<=1e-6 || object$theta>=1e8 else FALSE)
    out$convergence.reason <- if(!is.null(object$convergence.reason)) object$convergence.reason else
        if(isTRUE(object$converged)) "converged" else "not_converged"
    if(isTRUE(object$firth)) {
        available <- length(object$adjusted.score)==length(b) &&
            length(object$standardized.adjusted.score)==length(b) &&
            !is.null(object$adjusted.score)
        out$adjusted.score.available <- available
        out$adjusted.score <- setNames(if(available) object$adjusted.score[good] else
            rep(NA_real_,sum(good)),names(b)[good])
        out$standardized.adjusted.score <- if(available)
            max_abs(object$standardized.adjusted.score[good]) else NA_real_
        out$adjusted.dispersion.score <- if(available) object$adjusted.dispersion.score else NA_real_
        out$standardized.dispersion.score <- if(available) object$standardized.dispersion.score else NA_real_
        for(name in c("remaining.predictor.step","adjusted.score.rms",
                      "remaining.dispersion.step","adjusted.dispersion.score.rms",
                      "refined.convergence.verification","step.reductions"))
            if(!is.null(object[[name]])) out[[name]] <- object[[name]]
    }
    structure(out,class="fastglm_diagnostics")
}

#' Profile-likelihood interval for negative-binomial size
#'
#' Refits regression coefficients at fixed theta and inverts the likelihood-ratio
#' test on the log-theta scale. The interval may have an infinite upper endpoint
#' when the Poisson boundary cannot be excluded. Bias-reduced fits are excluded
#' because their estimates do not maximize this likelihood.
#' @param object a negative-binomial GLM, truncated model, or hurdle model.
#' @param level confidence level.
#' @param tol root-finding tolerance on log theta.
#' @param maxit maximum iterations for conditional regression fits.
#' @returns A named interval for theta with the evaluated profile attached.
#' @export
fastglm_confint_theta <- function(object,level=.95,tol=1e-5,maxit=200L) {
    if(is.null(object$theta) || !is.finite(object$theta) || object$theta<=0 ||
       identical(object$theta.estimated,FALSE)) stop("model must estimate theta",call.=FALSE)
    if(isTRUE(object$firth) || inherits(object,"fastglm_zi")) stop("theta profiles currently support ordinary NB, truncated NB, and hurdle NB fits",call.=FALSE)
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1) stop("'level' must lie between zero and one",call.=FALSE)
    .check_iterations(tol,maxit)
    is.hurdle <- inherits(object,"fastglm_hurdle")
    is.trunc <- inherits(object,"fastglm_truncated") || is.hurdle
    if(is.hurdle) {
        d <- .count_design(object); keep <- object$y>0 & object$weights>0
        x <- d$x$count[keep,,drop=FALSE]; y <- object$y[keep]
        w <- object$weights[keep]; off <- d$offset$count[keep]
        beta <- object$coefficients$count; llmax <- object$loglik_count
    } else {
        x <- model.matrix(object); y <- object$y; w <- object$prior.weights; off <- object$offset
        beta <- coef(object); llmax <- as.numeric(logLik(object))
    }
    if(is.null(y)) stop("profile likelihood requires a retained response",call.=FALSE)
    if(is.null(off)) off <- rep(0,length(y))
    trace <- new.env(parent=emptyenv()); trace$theta <- trace$loglik <- numeric()
    profile <- function(logtheta) {
        theta <- exp(logtheta)
        fit <- if(is.trunc) fastglm_truncated(x,y,dist="negbin",weights=w,offset=off,
                    theta=theta,start=beta,tol=1e-10,maxit=maxit) else
                fastglm(x,y,negbin(theta),weights=w,offset=off,start=beta,method="auto",tol=1e-10,maxit=maxit)
        if(!fit$converged) stop("conditional profile fit did not converge",call.=FALSE)
        ll <- as.numeric(logLik(fit)); trace$theta <- c(trace$theta,theta); trace$loglik <- c(trace$loglik,ll)
        2*(llmax-ll)-qchisq(level,1)
    }
    center <- log(object$theta)
    endpoint <- function(direction) {
        a <- center; fa <- profile(a)
        if(fa>0) stop("fitted theta is inconsistent with the conditional likelihood",call.=FALSE)
        for(j in 1:35) {
            b <- center+direction*j*log(2)
            if(b < log(1e-10)) return(0)
            if(b > log(1e9)) return(Inf)
            fb <- profile(b)
            if(fb>=0) return(exp(uniroot(profile,sort(c(a,b)),tol=tol)$root))
            a <- b
        }
        if(direction<0) 0 else Inf
    }
    out <- c(lower=endpoint(-1),upper=endpoint(1))
    attr(out,"level") <- level
    attr(out,"profile") <- data.frame(theta=trace$theta,logLik=trace$loglik)
    out
}
