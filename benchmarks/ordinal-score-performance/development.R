# Development prototype: exploit the tridiagonal threshold information before
# mapping it to common and threshold-specific predictor effects.
structured_moments <- function(x,response,offset,fit,selected,block.size) {
    n <- nrow(x); p <- ncol(x); q <- length(fit$alpha); s <- length(selected)
    d0 <- q+p; d <- d0+s*(q-1L); frequency <- fit$frequency
    alpha <- fit$working$alpha; beta <- fit$working$beta
    gap <- diff(alpha)
    if(any(!is.finite(gap)|gap<=0)) stop("the null fit requires finite, strictly ordered thresholds",call.=FALSE)
    tgap <- exp(-gap)/(-expm1(-gap))
    U <- numeric(d); I <- matrix(0,d,d)
    ia <- seq_len(q); ib <- q+seq_len(p)
    ig <- function(j) d0+(j-2L)*s+seq_len(s)
    for(first in seq.int(1L,n,by=block.size)) {
        at <- seq.int(first,min(n,first+block.size-1L))
        xx <- .ordinal_prediction_scale(x[at,,drop=FALSE],fit$working$center,fit$working$scale)
        zz <- xx[,selected,drop=FALSE]
        eta <- outer(-streaming_qr_eta(xx,beta,offset[at]),alpha,"+")
        F <- plogis(eta); S <- plogis(-eta); g <- F*S
        counts <- .ordinal_assessment_counts(response,at,q+1L)/frequency
        mass <- response$totals[at]/frequency
        threshold <- diagonal <- rowsum <- matrix(0,length(at),q)
        adjacent <- matrix(0,length(at),q-1L)
        location <- numeric(length(at))
        probability <- cbind(F[,1L],F[,-1L,drop=FALSE]*S[,-q,drop=FALSE]*
            rep(-expm1(-gap),each=length(at)),S[,q])
        if(any(!is.finite(probability)|probability<=0))
            stop("the fitted null has a category probability outside the representable positive range",call.=FALSE)
        for(j in seq_len(q)) {
            left <- S[,j]+if(j>1L) tgap[j-1L] else 0
            right <- F[,j]+if(j<q) tgap[j] else 0
            threshold[,j] <- counts[,j]*left-counts[,j+1L]*right
            diagonal[,j] <- mass*g[,j]*(left+right)
            # Direct row sums avoid cancellation when an interior category is narrow.
            rowsum[,j] <- mass*g[,j]*(probability[,j]+probability[,j+1L])
            if(j<q) adjacent[,j] <- -mass*g[,j+1L]*right
            location <- location+counts[,j+1L]*F[,j]-counts[,j]*S[,j]
        }
        U[ia] <- U[ia]+colSums(threshold)
        U[ib] <- U[ib]+drop(crossprod(xx,location))
        I[cbind(ia,ia)] <- I[cbind(ia,ia)]+colSums(diagonal)
        I[cbind(ia[-q],ia[-1L])] <- I[cbind(ia[-q],ia[-1L])]+colSums(adjacent)
        I[ia,ib] <- I[ia,ib]-crossprod(rowsum,xx)
        I[ib,ib] <- I[ib,ib]+crossprod(xx,xx*rowSums(rowsum))
        dg <- crossprod(diagonal,zz); ad <- crossprod(adjacent,zz)
        for(j in 2:q) {
            index <- ig(j)
            U[index] <- U[index]-drop(crossprod(zz,threshold[,j]))
            I[j,index] <- I[j,index]-dg[j,]
            I[j-1L,index] <- I[j-1L,index]-ad[j-1L,]
            if(j<q) I[j+1L,index] <- I[j+1L,index]-ad[j,]
            I[ib,index] <- I[ib,index]+crossprod(xx,zz*rowsum[,j])
            I[index,index] <- I[index,index]+crossprod(zz,zz*diagonal[,j])
            if(j<q) I[index,ig(j+1L)] <- I[index,ig(j+1L)]+crossprod(zz,zz*adjacent[,j])
        }
    }
    I[lower.tri(I)] <- t(I)[lower.tri(I)]
    if(any(!is.finite(U)) || any(!is.finite(I))) stop("score or expected information is nonfinite",call.=FALSE)
    list(score=U,information=I,nuisance=d0,frequency=frequency)
}

structured_score <- function(x,response,offset,fit,selected,block.size) {
    m <- structured_moments(x,response,offset,fit,selected,block.size)
    U <- m$score; I <- m$information; d0 <- m$nuisance; d <- length(U)
    scale <- sqrt(diag(I))
    if(any(!is.finite(scale)|scale<=0)) stop("expected information has a nonpositive diagonal",call.=FALSE)
    I <- I/outer(scale,scale); U <- U/scale
    nuisance <- seq_len(d0); tested <- seq.int(d0+1L,d)
    A <- I[nuisance,nuisance,drop=FALSE]; C <- I[tested,nuisance,drop=FALSE]
    root <- chol(A)
    projection <- t(backsolve(root,forwardsolve(t(root),t(C))))
    e <- drop(U[tested]-projection%*%U[nuisance])
    J <- I[tested,tested,drop=FALSE]-projection%*%t(C); J <- (J+t(J))/2
    js <- sqrt(diag(J))
    if(any(!is.finite(js)|js<=0)) stop("efficient expected information is not positive definite",call.=FALSE)
    J <- J/outer(js,js); e <- e/js
    root <- chol(J); condition <- min(rcond(A),rcond(J))
    if(!is.finite(condition) || condition<1e-14) stop("expected information is numerically singular",call.=FALSE)
    statistic <- m$frequency*sum(forwardsolve(t(root),e)^2)
    list(statistic=statistic,condition=condition)
}
