.ordinal_link <- function(object) if(is.null(object$link)) "logit" else object$link

.ordinal_link_code <- function(link) match(match.arg(link,c("logit","probit","cloglog")),
    c("logit","probit","cloglog"))-1L

.ordinal_start_thresholds <- function(cumulative,remaining,link) {
    if(link=="logit") return(log(cumulative+.5)-log(remaining+.5))
    total <- cumulative+remaining+1
    lp <- log(cumulative+.5)-log(total); ls <- log(remaining+.5)-log(total)
    lower <- lp < log(.5)
    if(link=="probit") return(ifelse(lower,qnorm(lp,log.p=TRUE),qnorm(ls,lower.tail=FALSE,log.p=TRUE)))
    out <- log(-ls)
    small <- lp < -36
    out[small] <- lp[small]
    at <- lower & !small
    out[at] <- log(-log1p(-exp(lp[at])))
    out
}

.ordinal_log1mexp <- function(x) {
    out <- log(-expm1(x))
    tail <- x < -log(2)
    out[tail] <- log1p(-exp(x[tail]))
    out
}

.ordinal_log_difference <- function(large,small) {
    equal <- large==small
    delta <- small-large
    delta[equal] <- 0
    large+.ordinal_log1mexp(delta)
}

# Normal and extreme-value cumulative distributions and their derivatives.
# Log differences preserve category probabilities in the distribution tails.
.ordinal_link_kernel <- function(alpha,location,link,scores=FALSE) {
    eta <- outer(-location,alpha,"+"); n <- length(location); q <- length(alpha)
    if(link=="probit") {
        lc <- pnorm(eta,log.p=TRUE); ls <- pnorm(eta,lower.tail=FALSE,log.p=TRUE)
        lg <- dnorm(eta,log=TRUE); h <- -eta
    } else {
        e <- exp(eta); ls <- -e; lc <- eta
        at <- eta>=-36; lc[at] <- .ordinal_log1mexp(-e[at])
        lg <- eta-e; h <- 1-e
    }
    F <- exp(lc); S <- exp(ls); g <- exp(lg); curvature <- g*h
    curvature[is.infinite(h) & g==0] <- 0
    logp <- matrix(NA_real_,n,q+1L); logp[,1L] <- lc[,1L]; logp[,q+1L] <- ls[,q]
    logden <- matrix(NA_real_,n,max(0L,q-1L))
    if(q>1L) for(j in 2:q) {
        upper.difference <- .ordinal_log_difference(ls[,j-1L],ls[,j])
        if(link=="cloglog") {
            ld <- eta[,j]+log(-expm1(alpha[j-1L]-alpha[j]))
            logden[,j-1L] <- ld
            at <- ld>=-36
            logden[at,j-1L] <- .ordinal_log1mexp(-exp(ld[at]))
            upper.difference <- ls[,j-1L]+logden[,j-1L]
        }
        logp[,j] <- ifelse(eta[,j]<=0,
            .ordinal_log_difference(lc[,j],lc[,j-1L]),upper.difference)
    }
    out <- list(cumulative=F,survival=S,derivative=g,curvature=curvature,
        probability=exp(logp),log.probability=logp,density.log.derivative=h)
    if(scores) {
        left <- exp(lg-logp[,seq_len(q),drop=FALSE])
        right <- exp(lg-logp[,seq_len(q)+1L,drop=FALSE])
        if(link=="cloglog") {
            right[,q] <- exp(eta[,q])
            if(q>1L) for(j in 2:q) {
                difference <- exp(eta[,j])*(-expm1(alpha[j-1L]-alpha[j]))
                left[,j] <- exp(eta[,j]-difference-logden[,j-1L])
                right[,j-1L] <- exp(eta[,j-1L]-logden[,j-1L])
            }
        }
        translation <- matrix(NA_real_,n,q+1L)
        translation[,1L] <- left[,1L]; translation[,q+1L] <- -right[,q]
        if(q>1L) for(j in 2:q) {
            gap <- alpha[j]-alpha[j-1L]
            delta <- if(link=="probit") -.5*gap*(eta[,j]+eta[,j-1L]) else
                gap-exp(eta[,j])*(-expm1(-gap))
            # Select the larger density to avoid subtracting two large ratios.
            high <- delta>=0
            translation[high,j] <- left[high,j]*(-expm1(-delta[high]))
            translation[!high,j] <- -right[!high,j-1L]*(-expm1(delta[!high]))
        }
        out$left <- left; out$right <- right; out$translation <- translation
        out$translation.difference <- translation[,seq_len(q),drop=FALSE]-translation[,seq_len(q)+1L,drop=FALSE]
    }
    out
}

.ordinal_link_sandwich <- function(x,alpha,beta,offset,codes,counts,weights,frequency,independent,link) {
    n <- nrow(x); p <- ncol(x); q <- length(alpha); d <- q+p
    location <- streaming_qr_eta(x,beta,offset)
    kernel <- .ordinal_link_kernel(alpha,location,link,scores=TRUE)
    if(link=="cloglog") exponential <- exp(outer(-location,alpha,"+"))
    information <- meat <- matrix(0,d,d); rows <- matrix(0,n,d)
    # For each category, -d2 log(p) = score score' - d2(p)/p.
    for(k in seq_len(q+1L)) {
        mass <- if(length(counts)) weights*counts[,k]/frequency else (weights/frequency)*(codes==k)
        if(!any(mass>0)) next
        score <- matrix(0,n,d)
        if(k<=q) score[,k] <- kernel$left[,k]
        if(k>1L) score[,k-1L] <- -kernel$right[,k-1L]
        if(p) score[,q+seq_len(p)] <- -x*kernel$translation[,k]
        weighted <- score*mass
        if(independent) meat <- meat+crossprod(score,weighted)
        rows <- rows+weighted
        if(link=="cloglog") {
            # Direct derivatives of log category probabilities avoid cancelling
            # two squared scores, which can erase all tail curvature.
            indices <- if(k==1L) 1L else if(k==q+1L) q else c(k-1L,k)
            if(k==1L) {
                rowsum <- matrix(kernel$left[,1L]*.ordinal_cloglog_curvature(exponential[,1L])$upper,n,1L)
                diagonal <- rowsum
            } else if(k==q+1L) {
                rowsum <- diagonal <- matrix(exponential[,q],n,1L)
            } else {
                difference <- exponential[,k]*(-expm1(alpha[k-1L]-alpha[k]))
                curvature <- .ordinal_cloglog_curvature(difference)
                rowsum <- cbind(kernel$right[,k-1L]*curvature$lower,kernel$left[,k]*curvature$upper)
                shared <- kernel$left[,k]*kernel$right[,k-1L]
                diagonal <- rowsum+shared
                information[k-1L,k] <- information[k-1L,k]-sum(mass*shared)
                information[k,k-1L] <- information[k-1L,k]
            }
            information[cbind(indices,indices)] <- information[cbind(indices,indices)]+colSums(diagonal*mass)
            if(p) {
                cross <- crossprod(rowsum*mass,x)
                information[indices,q+seq_len(p)] <- information[indices,q+seq_len(p),drop=FALSE]-cross
                information[q+seq_len(p),indices] <- information[q+seq_len(p),indices,drop=FALSE]-t(cross)
                information[q+seq_len(p),q+seq_len(p)] <- information[q+seq_len(p),q+seq_len(p),drop=FALSE]+
                    crossprod(x,x*(mass*rowSums(rowsum)))
            }
            next
        }
        information <- information+crossprod(score,weighted)
        for(j in c(if(k<=q) k,if(k>1L) k-1L)) {
            z <- matrix(0,n,d); z[,j] <- 1
            if(p) z[,q+seq_len(p)] <- -x
            curvature <- if(j==k) kernel$density.log.derivative[,j]*kernel$left[,j] else
                -kernel$density.log.derivative[,j]*kernel$right[,j]
            information <- information-crossprod(z,z*(mass*curvature))
        }
    }
    if(any(!is.finite(information)) || any(!is.finite(meat)) || any(!is.finite(rows)))
        stop("ordinal score or observed information is not finite",call.=FALSE)
    list(information=information,meat=meat,score=colSums(rows),rows=if(independent) NULL else rows)
}

.ordinal_cloglog_curvature <- function(difference) {
    upper <- difference/(-expm1(-difference))-1
    lower <- 1-difference/expm1(difference)
    small <- difference<1e-4
    z <- difference[small]
    upper[small] <- z/2+z^2/12-z^4/720
    lower[small] <- z/2-z^2/12+z^4/720
    list(upper=upper,lower=lower)
}
