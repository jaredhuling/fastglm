# Ordinary category scores and their observed negative Hessians. If r_j is
# the derivative of eta_j with respect to all working coefficients, an
# interior category has Hessian
#   h_upper r_upper r_upper' + h_lower r_lower r_lower'
#     + shared (r_upper-r_lower)(r_upper-r_lower)'.
# The common-slope columns cancel from the last direction exactly. This avoids
# subtracting large, nearly equal Hessian blocks for a narrow category.
.ordinal_partial_sandwich_block <- function(object,x,offset,codes,counts,weights,
        frequency=1,independent=TRUE,rows=NULL) {
    parts <- .ordinal_probability_block(object,x,offset,rows=rows)
    x <- parts$x;n <- nrow(x);q <- length(object$alpha);pc <- length(object$beta)
    pn <- length(object$nonparallel);d <- length(object$coefficients)
    link <- .ordinal_link(object);eta <- parts$eta
    common <- x[,object$common,drop=FALSE];varying <- x[,object$nonparallel,drop=FALSE]
    direction <- function(j) {
        out <- matrix(0,n,d);out[,j] <- 1
        if(pc) out[,q+seq_len(pc)] <- -common
        out[,q+pc+(j-1L)*pn+seq_len(pn)] <- -varying
        out
    }
    information <- meat <- matrix(0,d,d);row.score <- matrix(0,n,d)
    if(link=="probit") logdensity <- dnorm(eta,log=TRUE)
    if(link=="cloglog") exponential <- exp(eta)
    for(k in seq_len(q+1L)) {
        mass <- if(length(counts)) weights*counts[,k]/frequency else (weights/frequency)*(codes==k)
        active <- mass>0
        if(!any(active)) next
        upper <- if(k<=q) direction(k) else NULL
        lower <- if(k>1L) direction(k-1L) else NULL
        a <- b <- h.upper <- h.lower <- shared <- numeric(n)
        if(link=="logit") {
            if(k<=q) {a <- parts$survival[,k];h.upper <- parts$derivative[,k]}
            if(k>1L) {b <- parts$cumulative[,k-1L];h.lower <- parts$derivative[,k-1L]}
            translation <- a-b
            if(k>1L && k<=q) {
                t <- 1/expm1(parts$gap[,k-1L])
                a <- a+t;b <- b+t;shared <- t*(1+t)
            }
        } else if(link=="cloglog") {
            if(k==1L) {
                a <- exp(eta[,1L]-exponential[,1L]-parts$log.probability[,1L])
                curvature <- .ordinal_cloglog_curvature(exponential[,1L])$upper
                h.upper <- a*curvature
                h.upper[a==0 & is.infinite(curvature)] <- 0
                translation <- a
            } else if(k==q+1L) {
                b <- h.lower <- exponential[,q];translation <- -b
            } else {
                gap <- parts$gap[,k-1L]
                ld <- eta[,k]+log(-expm1(-gap));difference <- exp(ld)
                logden <- ld;large <- ld>=-36
                logden[large] <- .ordinal_log1mexp(-difference[large])
                a <- exp(eta[,k]-difference-logden)
                b <- exp(eta[,k-1L]-logden)
                delta <- gap-difference
                translation <- ifelse(delta>=0,a*(-expm1(-delta)),-b*(-expm1(delta)))
                curvature <- .ordinal_cloglog_curvature(difference)
                curvature$lower[is.infinite(difference)] <- 1
                h.upper <- a*curvature$upper
                h.upper[a==0 & is.infinite(curvature$upper)] <- 0
                h.lower <- b*curvature$lower;shared <- a*b
            }
        } else {
            lp <- parts$log.probability[,k]
            if(k<=q) a <- exp(logdensity[,k]-lp)
            if(k>1L) b <- exp(logdensity[,k-1L]-lp)
            if(k==1L) translation <- a else if(k==q+1L) translation <- -b else {
                delta <- -.5*parts$gap[,k-1L]*(eta[,k]+eta[,k-1L])
                translation <- ifelse(delta>=0,a*(-expm1(-delta)),-b*(-expm1(delta)))
                shared <- a*b
            }
            # These factors are the positive distances from a truncated-normal
            # mean to its endpoints. Refuse a subtraction whose roundoff can
            # dominate that distance; a ridge or sign repair would change bread.
            check.distance <- function(distance,j,ratio) {
                error <- 16*.Machine$double.eps*(1+abs(eta[,j])+abs(translation)*
                    (1+abs(logdensity[,j])+abs(lp)))
                bad <- active & (!is.finite(ratio) | (ratio>0 &
                    (!is.finite(distance) | !is.finite(error) | distance<=error)))
                if(any(bad)) stop("probit observed curvature is numerically unresolved; sandwich covariance is unavailable",call.=FALSE)
            }
            if(k<=q) {
                distance <- translation+eta[,k];check.distance(distance,k,a)
                h.upper <- a*distance
            }
            if(k>1L) {
                distance <- -translation-eta[,k-1L];check.distance(distance,k-1L,b)
                h.lower <- b*distance
            }
        }
        # Only categories observed in a row enter ordinary scores and bread.
        # Ignored cells cannot introduce 0*Inf into these aggregates.
        a[!active] <- b[!active] <- translation[!active] <- 0
        h.upper[!active] <- h.lower[!active] <- shared[!active] <- 0
        score <- matrix(0,n,d)
        if(k<=q) {
            score[,k] <- a
            score[,q+pc+(k-1L)*pn+seq_len(pn)] <- -varying*a
            information <- information+crossprod(upper,upper*(mass*h.upper))
        }
        if(k>1L) {
            score[,k-1L] <- -b
            score[,q+pc+(k-2L)*pn+seq_len(pn)] <- varying*b
            information <- information+crossprod(lower,lower*(mass*h.lower))
        }
        if(pc) score[,q+seq_len(pc)] <- -common*translation
        if(k>1L && k<=q) {
            difference <- upper-lower
            information <- information+crossprod(difference,difference*(mass*shared))
        }
        weighted <- score*mass;row.score <- row.score+weighted
        if(independent) meat <- meat+crossprod(score,weighted)
    }
    if(any(!is.finite(information)) || any(!is.finite(meat)) || any(!is.finite(row.score)))
        stop("nonparallel ordinal score or observed information is not finite",call.=FALSE)
    list(information=information,meat=meat,score=colSums(row.score),
        rows=if(independent) NULL else row.score)
}
