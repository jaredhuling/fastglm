test_that("streaming QR agrees with dense QR across native families", {
    skip_if_not_installed("MASS")
    set.seed(91401)
    n <- 240L
    x <- cbind("(Intercept)"=1, a=rnorm(n,sd=.12), b=runif(n,-.2,.2))
    offset <- runif(n,-.02,.02)
    families <- list(gaussian(),gaussian("log"),gaussian("inverse"),
        binomial(),binomial("probit"),binomial("cloglog"),binomial("log"),
        poisson(),poisson("identity"),poisson("sqrt"),
        Gamma("log"),Gamma("inverse"),Gamma("identity"),
        inverse.gaussian("1/mu^2"),inverse.gaussian("log"),
        inverse.gaussian("identity"),inverse.gaussian("inverse"),
        MASS::negative.binomial(2),MASS::negative.binomial(2,"sqrt"),
        MASS::negative.binomial(2,"identity"))
    groups <- list(1:2,3:7,8:61,62:137,138:n)
    for(family in families) {
        intercept <- if(family$family=="binomial" && family$link=="log") -.8 else 1.2
        mu <- family$linkinv(intercept+.15*x[,2]-.1*x[,3]+offset)
        if(family$family=="binomial") {
            trials <- rep(c(2,3,4),length.out=n)
            successes <- rbinom(n,trials,mu)
            y <- cbind(successes,trials-successes)
            w <- rep(c(1,2,1,1),length.out=n)
        } else {
            y <- if(family$family=="gaussian") mu+rnorm(n,sd=.08) else
                if(family$family=="poisson") rpois(n,mu) else
                if(grepl("Negative Binomial",family$family)) rnbinom(n,mu=mu,size=2) else
                rgamma(n,shape=8,scale=mu/8)
            w <- rep(c(.25,.7,1.5,2),length.out=n)
        }
        w[1:7] <- 0
        for(firth in c(FALSE,TRUE)) {
            last <- 0L; reads <- 0L
            source <- function(k) {
                if(k!=1L) stopifnot(k==last+1L)
                last <<- k; reads <<- reads+1L
                i <- groups[[k]]
                list(X=x[i,,drop=FALSE],y=if(is.matrix(y)) y[i,,drop=FALSE] else y[i],
                     weights=w[i],offset=offset[i])
            }
            reference <- fastglm(x,y,family,weights=w,offset=offset,method=0,
                                  firth=firth,tol=1e-10)
            fit <- fastglm_streaming(source,length(groups),family,method="qr",
                                     firth=firth,tol=1e-10)
            expect_true(fit$converged,info=paste(family$family,family$link,firth))
            expect_identical(fit$solver,"qr")
            expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=3e-7)
            expect_equal(fit$dispersion,reference$dispersion,tolerance=3e-7)
            expect_equal(unname(fit$cov.unscaled),unname(reference$cov.unscaled),tolerance=2e-8)
            expect_equal(reads,length(groups)*fit$passes)
            expect_equal(sort(fit$qr$pivot),seq_len(ncol(x)))
            if(firth) expect_lt(max(abs(fit$standardized.adjusted.score)),2e-6)
        }
    }
})

test_that("streaming QR resolves near collinearity without normal equations", {
    set.seed(91402)
    n <- 600L
    a <- rnorm(n); small <- rnorm(n)
    x <- cbind("(Intercept)"=1,a=a,b=a+1e-7*small)
    offset <- runif(n,-.1,.1)
    y <- .3+.6*a-.2*small+offset+rnorm(n,sd=.05)
    w <- rep(c(0,.5,1,2),length.out=n)
    reference <- fastglm(x,y,gaussian(),weights=w,offset=offset,method=0,tol=1e-10)
    for(block in c(1L,7L,97L,600L)) {
        groups <- split(seq_len(n),ceiling(seq_len(n)/block))
        reader <- function(k) {i<-groups[[k]];list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])}
        fit <- fastglm_streaming(reader,length(groups),gaussian(),method=0,tol=1e-10)
        expect_true(fit$converged)
        expect_equal(drop(x%*%coef(fit)),drop(x%*%coef(reference)),tolerance=2e-8)
        expect_equal(unname(fit$cov.unscaled),unname(reference$cov.unscaled),tolerance=2e-7)
        expect_true(all(is.finite(fit$se)))
    }
    reader <- function(k) list(X=x,y=y,weights=w,offset=offset)
    expect_error(fastglm_streaming(reader,1,gaussian(),method=2),"rank deficient|rank-deficient|factorization")
})

test_that("streaming QR reports rank and convergence failures explicitly", {
    x <- cbind(1,seq_len(40),seq_len(40));y<-seq(.3,1.3,length.out=40)
    reader <- function(k) {i<-((k-1)*10+1):(k*10);list(X=x[i,],y=y[i])}
    expect_error(fastglm_streaming(reader,4,method="qr"),"full column rank|rank deficient")
    expect_error(fastglm_streaming(reader,4,Gamma("log"),method="qr",firth=TRUE),"full column rank|rank deficient")
    set.seed(91403)
    x<-cbind(1,rnorm(120));y<-rbinom(120,1,plogis(.2+.5*x[,2]))
    reader<-function(k)list(X=x,y=y)
    expect_warning(fit<-fastglm_streaming(reader,1,binomial(),method="qr",firth=TRUE,maxit=1),"did not converge")
    expect_false(fit$converged)
    expect_identical(fit$convergence.reason,"iteration_limit")
    expect_equal(fit$iter,1)
    mu<-plogis(drop(x%*%coef(fit)))
    expect_equal(unname(fit$cov.unscaled),unname(solve(crossprod(x,mu*(1-mu)*x))),tolerance=1e-10)
})

test_that("streaming QR preserves safeguarded weighted Gamma fitting", {
    set.seed(81033)
    n<-240L;x<-cbind(1,runif(n,-.3,.3));w<-c(rep(0,60),rep(c(.125,.7,2),60))
    offset<-runif(n,-.05,.05);mu<-exp(drop(x%*%c(.6,.2))+offset)
    y<-rgamma(n,shape=.2,scale=mu/.2)
    reference<-fastglm(x,y,Gamma("log"),weights=w,offset=offset,method=0,firth=TRUE,tol=1e-10)
    reader<-function(k){i<-((k-1)*60+1):(k*60);list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])}
    for(start in list(NULL,c(-5,0))) {
        fit<-fastglm_streaming(reader,4,Gamma("log"),start=start,method="qr",firth=TRUE,tol=1e-10)
        expect_true(fit$converged)
        expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=2e-8)
        expect_equal(fit$dispersion,reference$dispersion,tolerance=2e-8)
        expect_lt(max(abs(fit$standardized.adjusted.score)),1e-7)
        expect_lt(abs(fit$standardized.dispersion.score),1e-7)
    }
})


test_that("streaming QR leaves automatic defaults and custom callbacks intact", {
    set.seed(91404)
    x<-cbind(1,rnorm(120));y<-.2+.4*x[,2]+rnorm(120,sd=.2)
    reader<-function(k)list(X=x,y=y)
    expect_identical(fastglm_streaming(reader,1,method="auto")$solver,"llt")
    custom<-gaussian()
    custom$variance<-function(mu) rep(2,length(mu))
    reference<-fastglm(x,y,custom,method=0,tol=1e-10)
    fit<-fastglm_streaming(reader,1,custom,method="qr",tol=1e-10)
    expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=1e-10)
    expect_equal(unname(fit$cov.unscaled),unname(reference$cov.unscaled),tolerance=1e-10)
    expect_error(fastglm:::streaming_qr_update(x,y,rep(Inf,120)),"finite non-negative")
    expect_error(fastglm:::streaming_qr_update(x,y,rep(-1,120)),"finite non-negative")
})
