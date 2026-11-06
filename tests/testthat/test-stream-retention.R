# The source closure holds no rows, so saved size can grow with block size only
# when a fitted object retains block data.
stream_generator <- function(size,kind) {
    force(size);force(kind)
    fastglm_source(function(k) {
        set.seed(1000L+k)
        X <- cbind(`(Intercept)`=1,a=rnorm(size),b=rnorm(size))
        eta <- 0.3+0.4*X[,2]-0.3*X[,3]
        y <- switch(kind,binary=rbinom(size,1,plogis(eta)),count=rpois(size,exp(eta)),
            positive=rexp(size,exp(-eta))*rbinom(size,1,plogis(eta)),
            ordinal=ordered(cut(rlogis(size)+eta,c(-Inf,-0.5,0.5,Inf),labels=c("l","m","h"))))
        if(kind=="ordinal") X <- X[,-1L,drop=FALSE]
        list(X=X,y=y)
    },3L)
}

test_that("streamed fits do not retain source blocks", {
    fits <- list(
        glm=function(s) fastglm_streaming(stream_generator(s,"binary"),family=binomial()),
        nb=function(s) fastglm_nb(stream_generator(s,"count")),
        hurdle=function(s) fastglm_hurdle(stream_generator(s,"count")),
        hurdle.nb=function(s) fastglm_hurdle(stream_generator(s,"count"),dist="negbin"),
        twopart=function(s) fastglm_twopart(stream_generator(s,"positive")),
        ordinal=function(s) fastglm_ordinal_streaming(stream_generator(s,"ordinal")),
        partial=function(s) fastglm_ordinal_streaming(stream_generator(s,"ordinal"),nonparallel="b"))
    for(name in names(fits)) {
        small <- suppressWarnings(fits[[name]](200L))
        large <- suppressWarnings(fits[[name]](4000L))
        # One retained column of 3,800 extra rows would add 30,400 bytes. Smaller
        # differences come from byte compilation of closures between the fits.
        growth <- length(serialize(large,NULL))-length(serialize(small,NULL))
        expect_lt(growth,8*3800,label=paste(name,"saved-size growth"))
        expect_equal(nrow(large$fetch(1L)$X),4000L)
    }
})

test_that("streamed Gaussian initialization still reads a supplied start", {
    set.seed(123)
    x <- cbind(1,rnorm(60));y <- exp(0.5+0.2*x[,2])+rnorm(60,sd=0.05)
    chunks <- function(k) {i <- ((k-1L)*20L+1L):(k*20L);list(X=x[i,,drop=FALSE],y=y[i])}
    fit <- fastglm_streaming(chunks,3L,family=gaussian("log"),start=c(0.4,0.1))
    expect_true(fit$converged)
    expect_equal(unname(coef(fit)),unname(coef(glm(y~x-1,family=gaussian("log"),start=c(0.4,0.1)))),tolerance=1e-5)
})
