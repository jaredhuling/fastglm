# Conditional on a fixed full-rank individual-observation design.
po_information <- function(z,theta,adjusted=TRUE) {
    eta <- drop(z%*%theta); p <- plogis(eta); w <- p*plogis(-eta)
    A <- crossprod(z,w*z); inverse <- chol2inv(chol(A))
    zi <- z%*%inverse; v <- rowSums(zi*z); h <- w*v
    adjustment <- drop(crossprod(z,h*(.5-p)))
    H <- A
    if(adjusted) for(k in seq_len(ncol(z))) {
        dw <- w*(1-2*p)*z[,k]
        dA <- crossprod(z,dw*z)
        dh <- dw*v-w*rowSums((zi%*%dA)*zi)
        H[,k] <- A[,k]-drop(crossprod(z,dh*(.5-p)-h*w*z[,k]))
    }
    list(A=A,H=H,adjustment=adjustment,p=p)
}

po_contrasts <- function(p,q) {
    d <- p+1L; D <- matrix(0,p*(q-1L),d*q)
    for(j in seq.int(2L,q)) for(k in seq_len(p)) {
        row <- (j-2L)*p+k
        D[row,k+1L] <- -1; D[row,(j-1L)*d+k+1L] <- 1
    }
    D
}

po_score_variance <- function(z,probability) {
    d <- ncol(z); q <- ncol(probability); B <- matrix(0,d*q,d*q)
    for(j in seq_len(q)) for(k in seq_len(j)) {
        block <- crossprod(z,(probability[,max(j,k)]-probability[,j]*probability[,k])*z)
        jj <- (j-1L)*d+seq_len(d); kk <- (k-1L)*d+seq_len(d)
        B[jj,kk] <- block; B[kk,jj] <- t(block)
    }
    B
}

po_sandwich <- function(inverse,B,D) {
    if(is.null(inverse) || is.null(B)) return(NULL)
    L <- D%*%inverse
    V <- L%*%B%*%t(L)
    (V+t(V))/2
}

po_mechanism_fit <- function(design,y) {
    x <- design$x; z <- cbind(1,x); n <- nrow(x); d <- ncol(z); q <- design$q
    theta <- matrix(0,d,q); Ainv <- Hinv <- matrix(0,d*q,d*q)
    score <- matrix(0,n,d*q); score.max <- 0; adjustment.size <- 0
    h.ok <- TRUE; warnings <- character(); null.error <- ""
    capture <- function(expr) withCallingHandlers(expr,warning=function(w) {
        warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")
    })
    for(j in seq_len(q)) {
        binary <- as.numeric(y>j)
        fit <- capture(fastglm::fastglm(z,binary,family=binomial(),firth=TRUE,
            method="auto",tol=1e-10,maxit=500))
        if(!isTRUE(fit$converged) || any(!is.finite(coef(fit)))) stop("binary_fit_failed")
        theta[,j] <- coef(fit); a <- po_information(z,theta[,j])
        at <- (j-1L)*d+seq_len(d)
        Ainv[at,at] <- chol2inv(chol(a$A))
        H <- (a$H+t(a$H))/2
        hi <- tryCatch(chol2inv(chol(H)),error=function(e) NULL)
        if(is.null(hi)) h.ok <- FALSE else Hinv[at,at] <- hi
        score[,at] <- z*(binary-a$p)
        score.max <- max(score.max,max(abs(colSums(score[,at,drop=FALSE])+a$adjustment)))
        inverse.root <- backsolve(chol(a$A),diag(d))
        relative <- crossprod(inverse.root,H%*%inverse.root)
        adjustment.size <- max(adjustment.size,max(abs(eigen(relative,symmetric=TRUE,only.values=TRUE)$values-1)))
    }
    if(!h.ok) Hinv <- NULL
    D <- design$D; delta <- drop(D%*%as.vector(theta)); B.emp <- crossprod(score)
    null <- tryCatch(capture(fastglm::fastglm_ordinal(x,ordered(y,levels=seq_len(q+1L)),
        firth=TRUE,tol=1e-10,maxit=500)),error=function(e) {null.error <<- conditionMessage(e); NULL})
    B.null <- NULL
    if(!is.null(null) && isTRUE(null$converged)) {
        probability <- predict(null)
        survival <- vapply(seq_len(q),function(j) rowSums(probability[,seq.int(j+1L,q+1L),drop=FALSE]),numeric(n))
        B.null <- po_score_variance(z,survival)
    } else if(!nzchar(null.error)) null.error <- "null_fit_not_converged"
    V <- list(empirical_ordinary=po_sandwich(Ainv,B.emp,D),
        empirical_adjusted=po_sandwich(Hinv,B.emp,D),
        true_ordinary=po_sandwich(Ainv,design$B.true,D),
        true_adjusted=po_sandwich(Hinv,design$B.true,D),
        null_ordinary=po_sandwich(Ainv,B.null,D),
        null_adjusted=po_sandwich(Hinv,B.null,D),true_linear=design$V.linear)
    list(delta=delta,V=V,score.max=score.max,adjustment.size=adjustment.size,
         null.error=null.error,warning=paste(unique(warnings),collapse=" | "))
}

po_statistic <- function(delta,V) {
    if(is.null(V) || any(!is.finite(V))) return(NA_real_)
    tryCatch(sum(backsolve(chol(V),delta,transpose=TRUE)^2),error=function(e) NA_real_)
}
