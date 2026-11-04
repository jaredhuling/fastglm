# Use the same binomial tail calculations for fitting and post-fit quantities.
# The marker distinguishes new fits from saved objects using older equations.
.stable_binomial_code <- function(object) {
    if(!isTRUE(object$stable.binomial)) return(-1L)
    code <- family_code(object$family)
    if(code %in% c(4L,5L)) code else -1L
}

.binomial_working_residual <- function(moments,y,eta,code) {
    out <- numeric(length(y));success <- y>0;failure <- y<1
    if(code==5L) {
        out[success] <- y[success]*exp(-eta[success])
        t <- exp(eta[failure]);ratio <- expm1(t)/t
        ratio[t==0] <- 1;ratio[is.infinite(t)] <- Inf
        out[failure] <- out[failure]-(1-y[failure])*ratio
    } else {
        logd <- dnorm(eta,log=TRUE)
        out[success] <- y[success]*exp(moments$logq[success]-logd[success])
        out[failure] <- out[failure]-(1-y[failure])*exp(moments$logp[failure]-logd[failure])
    }
    out
}

.binomial_tail_aic <- function(moments,y,n,weights) {
    # Retain stats::binomial()$aic trial and frequency-weight conventions.
    m <- if(any(n>1)) n else weights
    m <- rep(m,length.out=length(y));positive <- m>0 & weights>0
    trials <- round(m);successes <- round(m*y);failures <- trials-successes
    value <- lchoose(trials,successes)
    take <- positive & successes>0
    value[take] <- value[take]+successes[take]*moments$logp[take]
    take <- positive & failures>0
    value[take] <- value[take]+failures[take]*moments$logq[take]
    -2*sum((weights[positive]/m[positive])*value[positive])
}

.binomial_tail_residual <- function(moments,y,weights,type) {
    r <- y-moments$mean
    r[y==1] <- exp(moments$logq[y==1])
    r[y==0] <- -exp(moments$logp[y==0])
    if(type=="response") return(r)
    logabs <- log(abs(r))
    logabs[y==1] <- moments$logq[y==1]
    logabs[y==0] <- moments$logp[y==0]
    direction <- sign(r);direction[y==1] <- 1;direction[y==0] <- -1
    out <- if(type=="deviance") direction*sqrt(pmax(0,moments$deviance)) else
        direction*exp(logabs+(log(weights)-moments$logp-moments$logq)/2)
    out[weights==0] <- 0
    out
}

.glm_mu_eta <- function(object,eta) {
    code <- .stable_binomial_code(object)
    if(code<0L) return(object$family$mu.eta(eta))
    n <- length(eta)
    out <- binomial_firth_moments(as.numeric(eta),numeric(n),numeric(n),code)$derivative
    out[is.infinite(eta)] <- 0
    out
}
