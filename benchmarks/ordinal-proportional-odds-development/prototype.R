# Development prototype: ordinary-score covariance at separate binary fits.
# Independent integer frequencies only. This is not an exported package test.
ordinal_po_covariance <- function(x,counts,offset=numeric(nrow(x)),firth=TRUE) {
    stopifnot(is.matrix(x),is.matrix(counts),nrow(x)==nrow(counts),
              all(is.finite(x)),all(is.finite(counts)),all(counts>=0),
              all(counts==floor(counts)),all(colSums(counts)>0),
              ncol(counts)>=3L,ncol(x)>0L,length(offset)==nrow(x),
              all(is.finite(offset)))
    keep <- rowSums(counts)>0
    x <- x[keep,,drop=FALSE]; counts <- counts[keep,,drop=FALSE]; offset <- offset[keep]
    z <- cbind(1,x); n <- nrow(z); d <- ncol(z); q <- ncol(counts)-1L
    m <- rowSums(counts)
    high <- vapply(seq_len(q),function(j) rowSums(counts[,seq.int(j+1L,q+1L),drop=FALSE]),numeric(n))
    low <- vapply(seq_len(q),function(j) rowSums(counts[,seq_len(j),drop=FALSE]),numeric(n))
    probability <- matrix(0,n,q); coefficients <- matrix(0,d,q)
    bread <- matrix(0,d*q,d*q); meat <- bread
    for(j in seq_len(q)) {
        fit <- fastglm::fastglm(z,cbind(high[,j],low[,j]),family=binomial(),
            offset=offset,firth=firth,method="auto",tol=1e-10,maxit=500)
        stopifnot(isTRUE(fit$converged),all(is.finite(coef(fit))))
        eta <- drop(z%*%coef(fit))+offset
        probability[,j] <- plogis(eta)
        a <- crossprod(z,(m*plogis(eta)*plogis(-eta))*z)
        at <- (j-1L)*d+seq_len(d)
        bread[at,at] <- chol2inv(chol(a))
        coefficients[,j] <- coef(fit)
    }
    # Sum individual score products algebraically, preserving frequency scale.
    for(j in seq_len(q)) for(k in seq_len(j)) {
        weight <- high[,max(j,k)]-probability[,k]*high[,j]-
            probability[,j]*high[,k]+m*probability[,j]*probability[,k]
        block <- crossprod(z,weight*z)
        jj <- (j-1L)*d+seq_len(d); kk <- (k-1L)*d+seq_len(d)
        meat[jj,kk] <- block; meat[kk,jj] <- t(block)
    }
    covariance <- bread%*%meat%*%bread
    # Compare all slopes with the first threshold; intercepts remain unrestricted.
    D <- matrix(0,(q-1L)*(d-1L),q*d)
    for(j in seq.int(2L,q)) for(k in seq.int(2L,d)) {
        row <- (j-2L)*(d-1L)+k-1L
        D[row,k] <- -1; D[row,(j-1L)*d+k] <- 1
    }
    difference <- drop(D%*%as.vector(coefficients))
    difference.covariance <- D%*%covariance%*%t(D)
    statistic <- drop(crossprod(difference,chol2inv(chol(difference.covariance))%*%difference))
    list(coefficients=coefficients,covariance=covariance,meat=meat,bread=bread,
         probability=probability,difference=difference,
         difference.covariance=difference.covariance,statistic=statistic,df=nrow(D))
}
