test_that("QR Firth stopping checks the remaining score in predictor coordinates", {
    set.seed(92601)
    n <- 360L
    z <- cbind(1, rnorm(n, sd=.4), runif(n, -.5, .5), rnorm(n, sd=.3))
    offset <- runif(n, -.08, .08)
    families <- list(binomial(), poisson(), Gamma("log"), gaussian())
    for (family in families) {
        eta <- .4 + .3*z[,2] - .2*z[,3] + .25*z[,4] + offset
        y <- switch(family$family, binomial=rbinom(n,1,plogis(eta)),
            poisson=rpois(n,exp(eta)), Gamma=rgamma(n,shape=.7,scale=exp(eta)/.7),
            gaussian=eta+rnorm(n,sd=.4))
        w <- rep(if(family$family=="binomial") c(1,2,3) else c(.2,.9,2),length.out=n)
        w[1:9] <- 0
        reference <- fastglm(z,y,family,weights=w,offset=offset,method=0,firth=TRUE,tol=1e-11)
        for (scaled in c(FALSE,TRUE)) {
            x <- z; x[,4] <- z[,2] + 1e-6*z[,4]
            if(scaled) x <- sweep(x,2,c(1,1e-5,1e5,7),"*")
            for (block in c(7L,83L)) {
                groups <- split(seq_len(n),ceiling(seq_len(n)/block))
                last <- 0L
                reader <- function(k) {
                    if(k!=1L) stopifnot(k==last+1L)
                    last <<- k; i <- groups[[k]]
                    list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])
                }
                fit <- fastglm_streaming(reader,length(groups),family,method="qr",firth=TRUE,tol=1e-8)
                expect_true(fit$converged,info=paste(family$family,scaled,block))
                expect_lte(fit$remaining.predictor.step,1e-8)
                expect_lte(fit$adjusted.score.rms,1e-8)
                expect_equal(fastglm:::streaming_qr_eta(x,coef(fit),offset),
                             drop(z%*%coef(reference))+offset,tolerance=3e-8)
                if(family$family %in% c("Gamma","gaussian")) {
                    expect_lte(fit$remaining.dispersion.step,1e-8)
                    expect_lte(fit$adjusted.dispersion.score.rms,1e-8)
                    expect_equal(fit$dispersion,reference$dispersion,tolerance=3e-8)
                } else expect_true(is.na(fit$remaining.dispersion.step))
            }
        }
    }
})

test_that("QR Firth saved stopping quantities use current information and dispersion", {
    set.seed(92602)
    n <- 170L; x <- cbind(1,rnorm(n,sd=.5),rnorm(n,sd=.2))
    offset <- runif(n,-.1,.1); w <- rep(c(0,.4,1.2,2),length.out=n)
    y <- rgamma(n,shape=1.3,scale=exp(.2+.3*x[,2]+offset)/1.3)
    reader <- function(k) list(X=x,y=y,weights=w,offset=offset)
    expect_warning(fit <- fastglm_streaming(reader,1,Gamma("log"),method="qr",
                                            firth=TRUE,maxit=1,tol=1e-10),"did not converge")
    expect_false(fit$converged)
    expect_identical(fit$convergence.reason,"iteration_limit")
    expect_equal(fit$iter,1)
    active <- w>0; np <- sum(active); p <- ncol(x); phi <- fit$dispersion
    mu <- exp(drop(x%*%coef(fit))+offset)
    decomposition <- qr(sqrt(w[active])*x[active,,drop=FALSE])
    Q <- qr.Q(decomposition); h <- rowSums(Q^2)
    numerator <- w[active]*(y[active]/mu[active]-1)+phi*h/2
    q <- drop(crossprod(Q,numerator/sqrt(w[active])))
    step <- qr.coef(decomposition,numerator/sqrt(w[active]))
    expect_equal(fit$adjusted.score.rms,sqrt(sum(q*q)/(np*phi)),tolerance=1e-11)
    expect_equal(fit$remaining.predictor.step,max(abs(x[active,,drop=FALSE]%*%step)),tolerance=1e-11)
    a <- w[active]/phi
    d2 <- sum(w[active]^2*(2*trigamma(a)-2/a))
    d3 <- sum(w[active]^3*(-2*psigamma(a,2)-2/a^2))
    H <- fit$deviance-sum(2*w[active]*(log(a)-digamma(a)))+(p-2)*phi+d3/d2
    expect_equal(fit$remaining.dispersion.step,abs(H*phi/d2),tolerance=1e-11)
    expect_equal(fit$adjusted.dispersion.score.rms,abs(H)/sqrt(2*d2*np),tolerance=1e-11)
    expect_true(fit$remaining.predictor.step>1e-10 || fit$remaining.dispersion.step>1e-10)
})

test_that("QR Firth stopping handles separation and frequency weights honestly", {
    x <- cbind(1,seq(-2,2,length.out=90)); y <- as.numeric(x[,2]>0)
    w <- rep(c(1,2,4),length.out=nrow(x)); w[1:3] <- 0
    reader <- function(k) list(X=x,y=y,weights=w)
    fit <- fastglm_streaming(reader,1,binomial(),method="qr",firth=TRUE,tol=1e-8)
    expect_true(fit$converged)
    expect_true(all(is.finite(coef(fit))))
    expect_lte(fit$remaining.predictor.step,1e-8)
    rows <- rep(seq_len(nrow(x)),w)
    expanded <- function(k) list(X=x[rows,],y=y[rows])
    duplicate_fit <- fastglm_streaming(expanded,1,binomial(),method="qr",firth=TRUE,tol=1e-8)
    expect_true(duplicate_fit$converged)
    expect_equal(coef(fit),coef(duplicate_fit),tolerance=1e-7)
    expect_warning(short <- fastglm_streaming(reader,1,binomial(),method="qr",firth=TRUE,
                                               start=c(0,0),maxit=1,tol=1e-8),"did not converge")
    expect_false(short$converged)
    expect_gt(short$remaining.predictor.step,1e-8)
    expect_equal(short$iter,1)
    # A singular or numerically unresolved design cannot pass via small scores.
    singular <- function(k) list(X=cbind(x,x[,2]+1e-16*sin(seq_len(nrow(x)))),y=y)
    expect_error(fastglm_streaming(singular,1,binomial(),method="qr",firth=TRUE),"full column rank")
})

test_that("small marginal scores do not hide a remaining weak-direction step", {
    grid <- expand.grid(a=c(-.5,.5),z=c(-1,1),replicate=seq_len(30))
    x <- cbind(1,grid$a,grid$a+2^-30*grid$z)
    successes <- ifelse(grid$z>0,9,1)
    reader <- function(k) list(X=x,y=cbind(successes,10-successes))
    expect_warning(fit <- fastglm_streaming(reader,1,binomial(),method="qr",firth=TRUE,
        start=c(0,0,0),maxit=1,tol=1e-4),"did not converge")
    expect_false(fit$converged)
    expect_lt(max(abs(fit$standardized.adjusted.score)),1e-4)
    expect_gt(fit$remaining.predictor.step,.1)
    expect_gt(fit$adjusted.score.rms,.1)
})

test_that("ill-conditioned stopping verification preserves the actual design", {
    set.seed(93610)
    n <- 420L; p <- 4L
    z <- cbind(1,matrix(rnorm(n*(p-1),sd=.4),n)); x <- z
    x[,p] <- z[,2]+1e-9*z[,p]
    # Subtraction is exact for these close floating-point values. This defines
    # a well-conditioned equivalent design from the actual stored inputs.
    equivalent <- x; equivalent[,p] <- (x[,p]-x[,2])/1e-9
    offset <- runif(n,-.15,.15)
    eta <- .2+.3*z[,2]-.25*z[,3]+.2*z[,p]+offset
    w <- rep(c(.15,.7,3),length.out=n); w[1:11] <- 0
    y <- rpois(n,exp(eta)); groups <- split(seq_len(n),ceiling(seq_len(n)/97))
    reader <- function(k) {i<-groups[[k]];list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])}
    fit <- suppressWarnings(fastglm_streaming(reader,length(groups),poisson(),method="qr",firth=TRUE,tol=1e-8))
    expect_true(fit$refined.convergence.verification)
    mu <- exp(fastglm:::streaming_qr_eta(x,coef(fit),offset)); active <- w>0
    working <- w[active]*mu[active]
    factor <- qr(sqrt(working)*equivalent[active,,drop=FALSE]); Q <- qr.Q(factor)
    score <- w[active]*(y[active]-mu[active])+rowSums(Q^2)/2
    transformed <- drop(crossprod(Q,score/sqrt(working)))
    step <- qr.coef(factor,score/sqrt(working))
    max_step <- max(abs(equivalent[active,,drop=FALSE]%*%step))
    expect_lt(abs(fit$remaining.predictor.step-max_step),1e-11)
    expect_lt(abs(fit$adjusted.score.rms-sqrt(sum(transformed^2)/sum(active))),1e-11)
    if(fit$converged) expect_lte(max_step,1e-8+1e-12) else
        expect_identical(fit$convergence.reason,"iteration_limit")
})

test_that("the final allowed iterate is certified only with stationary dispersion", {
    set.seed(92603)
    x <- cbind(1,rnorm(100,sd=.4)); y <- .7+.2*x[,2]+rnorm(100,sd=.3)
    reader <- function(k) list(X=x,y=y)
    # The default Gaussian QR initialization already solves the coefficient
    # equation, so one accepted iteration also obtains the dispersion root.
    fit <- fastglm_streaming(reader,1,gaussian(),method="qr",firth=TRUE,maxit=1,tol=1e-10)
    expect_true(fit$converged)
    expect_equal(fit$iter,1)
    expect_identical(fit$convergence.reason,"converged")
    expect_lte(fit$remaining.predictor.step,1e-10)
    expect_lte(fit$remaining.dispersion.step,1e-10)
    # A supplied poor start reaches the same beta, but its one-step dispersion
    # update was computed at the old beta and must not pass final certification.
    expect_warning(short <- fastglm_streaming(reader,1,gaussian(),method="qr",firth=TRUE,
        start=c(0,0),maxit=1,tol=1e-10),"did not converge")
    expect_false(short$converged)
    expect_lte(short$remaining.predictor.step,1e-10)
    expect_gt(short$remaining.dispersion.step,1e-10)
})
