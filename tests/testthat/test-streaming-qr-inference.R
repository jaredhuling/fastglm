test_that("streaming QR retains stable uncertainty for nearly redundant predictors", {
    set.seed(916)
    n <- 180L; gap <- 1e-8
    x <- rnorm(n)
    X <- cbind(intercept=1, x=x, near=x+gap*rnorm(n))
    Z <- cbind(intercept=1, x=x, difference=(X[,3]-X[,2])/gap)
    w <- rep(c(0,1,2), length.out=n)
    off <- runif(n,-.2,.2)
    y <- drop(Z%*%c(.4,.2,-.3))+off+rnorm(n)
    reader <- function(k) {
        i <- ((k-1L)*30L+1L):(k*30L)
        list(X=X[i,,drop=FALSE],y=y[i],weights=w[i],offset=off[i])
    }
    fit <- fastglm_streaming(reader,6L,gaussian(),method="qr",tol=1e-10)
    ref <- fastglm(Z,y,gaussian(),method="qr",weights=w,offset=off,tol=1e-10)
    expect_true(fit$converged)
    expect_equal(unname(predict(fit)),unname(predict(ref)),tolerance=2e-7)
    mu <- predict(fit)
    expect_equal(fit$deviance,sum(w*(y-mu)^2),tolerance=1e-13)
    expect_equal(residuals(fit,type="response"),y-mu)
    expect_lt(fastglm_diagnostics(fit)$standardized.score,1e-6)
    expect_equal(predict(fit,se.fit=TRUE)$se.fit,
                 unname(predict(ref,se.fit=TRUE)$se.fit),tolerance=2e-7)
    expect_equal(unname(predict(fit,newdata=X,offset=off,se.fit=TRUE)$se.fit),
                 unname(predict(ref,se.fit=TRUE)$se.fit),tolerance=2e-7)
    expect_equal(hatvalues(fit),unname(hatvalues(ref)),tolerance=2e-7)
    expect_equal(sum(hatvalues(fit)),3,tolerance=1e-8)
    expect_equal(fastglm_average(fit),fastglm_average(ref),tolerance=2e-7)
    contrast <- function(x) { x[,2:3] <- x[,2:3]+1; x }
    contrast.ref <- function(x) { x[,2] <- x[,2]+1; x }
    expect_equal(fastglm_average(fit,contrast=contrast),
                 fastglm_average(ref,contrast=contrast.ref),tolerance=2e-7)

    # Transform the well-conditioned reference covariance into X coordinates.
    transform <- matrix(c(1,0,0,0,1,0,0,-1/gap,1/gap),3,3)
    scale <- outer(c(1,1/gap,1/gap),c(1,1/gap,1/gap))
    for(type in c("HC0","HC1","HC2","HC3")) {
        expected <- transform%*%fastglm_vcov(ref,type=type)%*%t(transform)
        expect_equal(unname(fastglm_vcov(fit,type=type)/scale),
                     unname(expected/scale),tolerance=2e-7)
    }
    groups <- rep(seq_len(30),each=6)
    get.groups <- function(k) groups[((k-1L)*30L+1L):(k*30L)]
    expected <- transform%*%fastglm_vcov(ref,type="HC1",cluster=groups)%*%t(transform)
    expect_equal(unname(fastglm_vcov(fit,type="HC1",cluster=get.groups)/scale),
                 unname(expected/scale),tolerance=2e-7)
})

test_that("streaming QR prediction handles scale, permutation, and explicit covariance", {
    set.seed(917)
    n <- 120L
    X <- cbind(small=rnorm(n)*1e-4,intercept=1,large=rnorm(n)*1e4)
    off <- runif(n,-.1,.1)
    y <- rpois(n,exp(drop(X%*%c(2000,.2,-.00002))+off))
    reader <- function(k) {
        i <- ((k-1L)*20L+1L):(k*20L)
        list(X=X[i,,drop=FALSE],y=y[i],offset=off[i])
    }
    fit <- fastglm_streaming(reader,6L,poisson(),method="qr",tol=1e-10)
    ref <- fastglm(X,y,poisson(),method="qr",offset=off,tol=1e-10)
    expect_true(fit$converged)
    expect_equal(predict(fit,type="response",se.fit=TRUE),
                 predict(ref,type="response",se.fit=TRUE),ignore_attr=TRUE,tolerance=1e-7)
    pieces <- list()
    predict(fit,se.fit=TRUE,dispersion=2,
            sink=function(value,k) pieces[[k]] <<- value$se.fit)
    expect_equal(unlist(pieces,use.names=FALSE),
                 sqrt(2)*predict(ref,se.fit=TRUE)$se.fit,ignore_attr=TRUE,tolerance=1e-7)
    expect_equal(fastglm_average(fit),fastglm_average(fit,vcov.=vcov(fit)),tolerance=1e-8)
    expect_equal(fastglm_average(fit,vcov.=2*vcov(fit))$std.error,
                 sqrt(2)*fastglm_average(fit)$std.error,tolerance=1e-8)
})

test_that("QR averages block sparse newdata and preserve whole-design contrasts", {
    set.seed(918)
    X <- cbind(1,rnorm(120),rnorm(120))
    y <- rpois(120,exp(drop(X%*%c(.2,.3,-.1))))
    reader <- function(k) list(X=X,y=y)
    fit <- fastglm_streaming(reader,1L,poisson(),method="qr",tol=1e-10)
    ref <- fastglm(X,y,poisson(),method="qr",tol=1e-10)
    new <- cbind(1,seq(-1,2,length.out=17000),rep(c(0,1),8500))
    sparse <- as(new,"dgCMatrix")
    weights <- seq(.5,2,length.out=nrow(new))
    calls <- integer()
    contrast <- function(x) {
        calls <<- c(calls,nrow(x))
        x[,2] <- x[,2]-mean(x[,2])
        x
    }
    expected <- fastglm_average(ref,newdata=new,contrast=contrast,weights=weights)
    calls <- integer()
    sizes <- integer()
    original <- fastglm:::.stream_qr_transform
    local_mocked_bindings(.stream_qr_transform=function(x,qr) {
        sizes <<- c(sizes,nrow(x))
        original(x,qr)
    },.package="fastglm")
    actual <- fastglm_average(fit,newdata=sparse,contrast=contrast,weights=weights)
    expect_equal(actual,expected,tolerance=1e-8)
    expect_identical(calls,nrow(sparse))
    expect_gt(length(sizes),1L)
    expect_lt(max(sizes),nrow(sparse))
    expect_error(fastglm_average(fit,newdata=sparse,weights=c(weights,1)),
                 "invalid averaging weights")
})
