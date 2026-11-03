test_that("tiny Gamma responses do not distort source initialization or null models", {
    set.seed(81221)
    n <- 360L
    x <- cbind("(Intercept)"=1,matrix(rnorm(n*3,sd=.25),n))
    offset <- runif(n,-.1,.1)
    w <- rep(c(0,.15,.7,2.3),n/4)
    mu <- exp(drop(x%*%c(.4,.15,-.1,.1))+offset)
    y <- rgamma(n,shape=.08,scale=mu/.08)
    expect_lt(min(y[w>0]),.Machine$double.eps)
    dense <- fastglm(x,y,Gamma("log"),weights=w,offset=offset,
                     firth=TRUE,method=2,tol=1e-10)
    source <- function(k) {
        i <- ((k-1L)*90L+1L):(k*90L)
        list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])
    }
    streamed <- fastglm_streaming(source,4L,Gamma("log"),firth=TRUE,tol=1e-10)
    expect_true(dense$converged)
    expect_true(streamed$converged)
    expect_equal(unname(coef(streamed)),unname(coef(dense)),tolerance=1e-8)
    expect_equal(streamed$dispersion,dense$dispersion,tolerance=1e-8)
    # The intercept-only Gamma-log null model has an explicit score root.
    null.intercept <- log(sum(w*y*exp(-offset))/sum(w))
    null.mu <- exp(null.intercept+offset)
    null.dev <- sum(Gamma("log")$dev.resids(y,null.mu,w))
    expect_lt(abs(sum(w*(y/null.mu-1))),1e-10)
    expect_equal(dense$null.deviance,null.dev,tolerance=1e-10)
    expect_equal(streamed$null.deviance,null.dev,tolerance=1e-10)
    # The numerical reference starts at a data-based value to avoid the
    # response-wise initialization that caused the original null-model error.
    ref <- glm.fit(matrix(1,n,1),y,weights=w,offset=offset,family=Gamma("log"),
                   start=null.intercept,control=glm.control(epsilon=1e-12))
    expect_true(ref$converged)
    expect_equal(ref$deviance,null.dev,tolerance=1e-10)
})

test_that("Gamma null calculation is invariant to offset shifts and block boundaries", {
    y <- c(1e-100,.2,3,5,9)
    w <- c(1,.25,0,2,1)
    offset <- c(-.2,.1,2,-.3,.05)
    total <- fastglm:::.gamma_log_null_total(y,w,offset)
    expected <- exp(offset)*sum(w*y*exp(-offset))/sum(w)
    for(shift in c(-700,0,700)) {
        logtotal <- fastglm:::.gamma_log_null_total(y,w,offset+shift)
        expect_equal(exp(offset+shift+logtotal-log(sum(w))),expected,tolerance=1e-12)
    }
    combined <- -Inf
    for(i in list(1:2,3L,4:5))
        combined <- fastglm:::.gamma_log_null_total(y[i],w[i],offset[i],combined)
    expect_equal(combined,total,tolerance=1e-14)
})
