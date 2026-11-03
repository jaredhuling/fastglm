# Development-only ordinary likelihood score for departures from proportional odds.
# Parameter order: q cutpoints, p common slopes, p deviations at thresholds 2:q.
ordinal_score_parts <- function(x,y,alpha,beta,block.size=1024L) {
    n <- nrow(x); p <- ncol(x); q <- length(alpha); d0 <- q+p; d <- q*(p+1L)
    stopifnot(q>=2L,p>=1L,length(y)==n,length(beta)==p,all(diff(alpha)>0),
              all(is.finite(x)),all(y %in% seq_len(q+1L)))
    score <- numeric(d); information <- matrix(0,d,d)
    tgap <- exp(-diff(alpha))/(-expm1(-diff(alpha)))
    for(start in seq.int(1L,n,by=block.size)) {
        at <- seq.int(start,min(n,start+block.size-1L)); xx <- x[at,,drop=FALSE]; yy <- y[at]
        eta <- outer(-drop(xx%*%beta),alpha,"+"); F <- plogis(eta); S <- plogis(-eta)
        for(k in seq_len(q+1L)) {
            upper <- if(k<=q) S[,k]+if(k>1L) tgap[k-1L] else 0 else numeric(length(at))
            lower <- if(k>1L) -F[,k-1L]-if(k<=q) tgap[k-1L] else 0 else numeric(length(at))
            probability <- if(k==1L) F[,1L] else if(k==q+1L) S[,q] else
                F[,k]*S[,k-1L]*(-expm1(alpha[k-1L]-alpha[k]))
            if(any(!is.finite(probability)|probability<=0)) stop("nonpositive_model_probability")
            contribution <- matrix(0,length(at),d)
            if(k<=q) contribution[,k] <- upper
            if(k>1L) contribution[,k-1L] <- lower
            location <- (if(k>1L) F[,k-1L] else 0)-(if(k<=q) S[,k] else 0)
            contribution[,q+seq_len(p)] <- xx*location
            if(k>=2L && k<=q) contribution[,d0+(k-2L)*p+seq_len(p)] <- -xx*upper
            if(k>2L) contribution[,d0+(k-3L)*p+seq_len(p)] <- -xx*lower
            score <- score+colSums(contribution[yy==k,,drop=FALSE])
            information <- information+crossprod(contribution,probability*contribution)
        }
    }
    list(score=score,information=information,nuisance=seq_len(d0),tested=seq.int(d0+1L,d))
}

ordinal_score_statistic <- function(parts) {
    a <- parts$nuisance; b <- parts$tested; I <- parts$information; U <- parts$score
    A <- I[a,a,drop=FALSE]; C <- I[b,a,drop=FALSE]
    root <- chol(A); projection <- t(backsolve(root,forwardsolve(t(root),t(C))))
    efficient <- drop(U[b]-projection%*%U[a])
    J <- I[b,b,drop=FALSE]-projection%*%t(C); J <- (J+t(J))/2
    rootJ <- chol(J)
    Q <- sum(forwardsolve(t(rootJ),efficient)^2)
    nuisance.Q <- sum(forwardsolve(t(root),U[a])^2)
    list(statistic=Q,df=length(b),efficient=efficient,information=J,
         nuisance.statistic=nuisance.Q,condition=rcond(J))
}

ordinal_score_test <- function(x,y,K,firth=FALSE,block.size=1024L) {
    # Original empty categories do not define the regular full-dimensional null.
    if(any(tabulate(y,nbins=K)==0)) stop("empty_category")
    fit <- fastglm::fastglm_ordinal(x,ordered(y,levels=seq_len(K)),firth=firth,tol=1e-10,maxit=500L)
    if(!isTRUE(fit$converged)) stop("null_fit_not_converged")
    parts <- ordinal_score_parts(x,y,fit$alpha,fit$beta,block.size)
    answer <- ordinal_score_statistic(parts)
    answer$fit <- fit; answer
}
