test_that("skewed weighted Gamma fits reach the finite adjusted-score root", {
    skip_if_not_installed("brglm2")
    set.seed(81033)
    n <- 240L
    x <- cbind("(Intercept)"=1, x=runif(n,-.3,.3))
    w <- c(rep(0,60L),rep(c(.125,.7,2),60L))
    offset <- runif(n,-.05,.05)
    mu <- exp(drop(x %*% c(.6,.2))+offset)
    y <- rgamma(n,shape=.2,scale=mu/.2)
    ref <- brglm2::brglmFit(x,y,weights=w,offset=offset,family=Gamma("log"),
        control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10,maxit=100))
    expect_true(ref$converged)
    check <- function(fit) {
        expect_true(fit$converged)
        expect_identical(fit$convergence.reason,"converged")
        expect_equal(unname(coef(fit)),unname(ref$coefficients),tolerance=2e-8)
        expect_equal(fit$dispersion,ref$dispersion,tolerance=2e-8)
        # The actual adjusted equations, rather than the accepted step size,
        # establish that all backends returned the same stationary estimate.
        fitted <- exp(drop(x %*% coef(fit))+offset)
        info <- crossprod(x,w*x)
        h <- w*rowSums((x %*% solve(info))*x)
        score <- drop(crossprod(x,w*(y/fitted-1)/fit$dispersion+h/2))
        expect_lt(max(abs(score)/sqrt(diag(info)/fit$dispersion)),1e-7)
        expect_equal(unname(fit$adjusted.score),unname(score),tolerance=1e-8)
        expect_lt(abs(fit$standardized.dispersion.score),1e-7)
        expect_equal(unname(fit$cov.unscaled),unname(solve(info)),tolerance=1e-10)
    }
    for (method in 0:5)
        check(fastglm(x,y,Gamma("log"),weights=w,offset=offset,
                      firth=TRUE,method=method,tol=1e-10))
    # An explicitly supplied poor start needs protection on its first step.
    for (method in c(0L,2L))
        check(fastglm(x,y,Gamma("log"),weights=w,offset=offset,start=c(-5,0),
                      firth=TRUE,method=method,tol=1e-10))
    for (method in 2:3) {
        check(fastglm(Matrix::Matrix(x,sparse=TRUE),y,Gamma("log"),weights=w,
                      offset=offset,firth=TRUE,method=method,tol=1e-10))
        check(fastglm(Matrix::Matrix(x,sparse=TRUE),y,Gamma("log"),weights=w,
                      offset=offset,start=c(-5,0),firth=TRUE,method=method,tol=1e-10))
        last <- 0L; reads <- 0L
        reader <- function(k) {
            if(k != 1L) stopifnot(k==last+1L)
            last <<- k; reads <<- reads+1L
            i <- ((k-1L)*60L+1L):(k*60L)
            list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])
        }
        fit <- fastglm_streaming(reader,4L,Gamma("log"),firth=TRUE,
                                 method=method,tol=1e-10)
        check(fit)
        expect_equal(reads,4L*fit$passes)
    }
    prepared <- fastglm_prepare(x,firth=TRUE,weights=w)
    check(fastglm_batch(prepared,matrix(y,ncol=1),family=Gamma("log"),offset=offset,
                        firth=TRUE,method=2,tol=1e-10)[[1L]])
    if(requireNamespace("bigmemory",quietly=TRUE)) {
        mapped <- bigmemory::as.big.matrix(x)
        check(fastglm(mapped,y,Gamma("log"),weights=w,
                      offset=offset,firth=TRUE,method=2,tol=1e-10))
        check(fastglm(mapped,y,Gamma("log"),weights=w,
                      offset=offset,start=c(-5,0),firth=TRUE,method=2,tol=1e-10))
    }
})

test_that("unfinished Gamma iterations report their remaining score", {
    set.seed(81217)
    x <- cbind(1,rnorm(120L))
    y <- rgamma(120L,shape=.4,scale=exp(.3+.2*x[,2])/.4)
    expect_warning(fit <- fastglm(x,y,Gamma("log"),firth=TRUE,method=2,maxit=1),
                   "did not converge")
    expect_false(fit$converged)
    expect_identical(fit$convergence.reason,"iteration_limit")
    expect_gt(max(abs(fit$standardized.adjusted.score)),1e-3)
    expect_true(all(is.finite(fit$adjusted.score)))
})
