# The observed score and information are compared with an independent R likelihood.
ordinal_reference_loglik <- function(b,x,counts,weights,offset,link) {
    q <- ncol(counts)-1L;p <- ncol(x)
    forward <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
    eta <- drop(x%*%b[q+seq_len(p)])+offset
    upper <- forward(outer(-eta,b[seq_len(q)],"+"))
    probability <- cbind(upper,1)-cbind(0,upper)
    sum(weights*counts*log(probability))
}

test_that("observed ordinal terms match numerical derivatives of the likelihood", {
    skip_if_not_installed("numDeriv")
    set.seed(123)
    n <- 60L;x <- cbind(a=rnorm(n),b=rnorm(n));K <- 4L
    codes <- sample.int(K,n,TRUE);weights <- rpois(n,2)+1;offset <- rnorm(n,sd=0.2)
    counts <- matrix(0,n,K);counts[cbind(seq_len(n),codes)] <- 1
    b <- c(-0.8,0.1,0.9,0.3,-0.2)
    for(link in c("logit","probit","cloglog")) {
        code <- match(link,c("logit","probit","cloglog"))-1L
        state <- fastglm:::ordinal_observed_evaluate(x,codes,matrix(numeric(),0L,0L),weights,offset,K,b,7L,code)
        f <- function(par) ordinal_reference_loglik(par,x,counts,weights,offset,link)
        expect_equal(state$loglik,f(b),tolerance=1e-10)
        expect_equal(state$score,numDeriv::grad(f,b),tolerance=1e-7)
        expect_equal(state$information,-numDeriv::hessian(f,b),tolerance=1e-5)
        expect_true(isSymmetric(state$information))
        grouped <- fastglm:::ordinal_observed_evaluate(x,integer(),counts*weights,rep(1,n),offset,K,b,7L,code)
        expect_equal(grouped$loglik,state$loglik,tolerance=1e-12)
        expect_equal(grouped$score,state$score,tolerance=1e-12)
        expect_equal(grouped$information,state$information,tolerance=1e-12)
    }
})

test_that("observed terms remain finite and accurate in the far tails", {
    x <- cbind(a=c(-30,-10,0,10,30,-30,30))
    codes <- c(1L,1L,2L,3L,3L,3L,1L)
    b <- c(-0.5,0.5,1)
    for(link in c("logit","probit","cloglog")) {
        code <- match(link,c("logit","probit","cloglog"))-1L
        state <- tryCatch(fastglm:::ordinal_observed_evaluate(x[1:5,,drop=FALSE],codes[1:5],matrix(numeric(),0L,0L),
            rep(1,5L),numeric(5L),3L,b,7L,code),error=function(e) NULL)
        expect_false(is.null(state))
        expect_true(is.finite(state$loglik));expect_true(all(is.finite(state$information)))
        expect_true(all(eigen(state$information,symmetric=TRUE,only.values=TRUE)$values>-1e-8))
    }
})

test_that("Newton and expected scoring certify the same maximum likelihood fit", {
    set.seed(123)
    n <- 800L;x <- cbind(a=rnorm(n),b=rnorm(n),c=rbinom(n,1,0.3))
    for(link in c("logit","probit","cloglog")) for(K in c(2L,4L,12L)) {
        forward <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
        inverse <- switch(link,logit=qlogis,probit=qnorm,cloglog=function(u) log(-log1p(-u)))
        eta <- drop(x%*%c(0.6,-0.4,0.5));u <- runif(n);y <- rep(K,n)
        cuts <- inverse(seq_len(K-1L)/K)
        for(j in rev(seq_len(K-1L))) y[u<=forward(cuts[j]-eta)] <- j
        y <- ordered(y,levels=seq_len(K))
        fit <- fastglm_ordinal(x,y,link=link,tol=1e-10)
        expect_true(fit$converged)
        expect_gt(fit$observed.iterations,0L)
        expect_lt(fit$scaled.score,1e-10);expect_lt(fit$predictor.step,1e-10)
        reference <- fastglm_ordinal(x,y,link=link,tol=1e-10,control=list(observed=FALSE))
        expect_identical(reference$observed.iterations,0L)
        expect_equal(coef(fit),coef(reference),tolerance=1e-8)
        expect_equal(vcov(fit),vcov(reference),tolerance=1e-8)
        expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(reference)),tolerance=1e-10)
    }
})

test_that("separated data report nonconvergence with or without Newton steps", {
    x <- cbind(a=seq(-3,3,length.out=30))
    y <- ordered(rep(1:3,each=10))
    for(link in c("logit","probit","cloglog")) {
        run <- function(observed) tryCatch(suppressWarnings(
            fastglm_ordinal(x,y,link=link,control=list(observed=observed))),error=function(e) conditionMessage(e))
        newton <- run(TRUE);reference <- run(FALSE)
        expect_type(newton,typeof(reference))
        if(is.list(reference)) expect_identical(newton$converged,reference$converged)
    }
})

test_that("Newton steps stop promptly when likelihood changes reach rounding size", {
    skip_on_cran()
    # Near a million rows, rounding in the likelihood sum exceeds a valid step's gain.
    set.seed(123)
    n <- 1000000L;p <- 10L
    x <- matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    latent <- drop(x%*%(0.6*(-1)^seq_len(p)/sqrt(seq_len(p))))+rlogis(n)
    y <- ordered(findInterval(latent,qlogis(c(0.2,0.4,0.6,0.8)))+1L)
    fit <- fastglm_ordinal(x,y)
    expect_true(fit$converged)
    expect_lte(fit$observed.iterations,10L)
})

test_that("observed threshold-specific terms match numerical derivatives", {
    skip_if_not_installed("numDeriv")
    set.seed(123)
    n <- 60L;K <- 4L;x <- cbind(a=rnorm(n));z <- cbind(b=rnorm(n)/4,c=rnorm(n)/4)
    codes <- sample.int(K,n,TRUE);weights <- rpois(n,2)+1;offset <- rnorm(n,sd=0.2)
    counts <- matrix(0,n,K);counts[cbind(seq_len(n),codes)] <- 1
    b <- c(-1,0.1,1,0.3,0.2,-0.1,0.25,-0.15,0.3,-0.05)
    likelihood <- function(par,link) {
        forward <- switch(link,logit=plogis,probit=pnorm,cloglog=function(u) -expm1(-exp(u)))
        location <- drop(x%*%par[4L])+offset
        upper <- sapply(1:3,function(s) forward(par[s]-location-drop(z%*%par[4L+(s-1L)*2L+1:2])))
        sum(weights*counts*log(cbind(upper,1)-cbind(0,upper)))
    }
    for(link in c("logit","probit","cloglog")) {
        code <- match(link,c("logit","probit","cloglog"))-1L
        state <- fastglm:::ordinal_partial_observed_evaluate(x,z,codes,matrix(numeric(),0L,0L),weights,offset,K,b,7L,code)
        f <- function(par) likelihood(par,link)
        expect_equal(state$loglik,f(b),tolerance=1e-10)
        expect_equal(state$score,numDeriv::grad(f,b),tolerance=1e-7)
        expect_equal(state$information,-numDeriv::hessian(f,b),tolerance=1e-5)
        grouped <- fastglm:::ordinal_partial_observed_evaluate(x,z,integer(),counts*weights,rep(1,n),offset,K,b,7L,code)
        expect_equal(grouped$information,state$information,tolerance=1e-12)
    }
})

test_that("threshold-specific Newton and expected scoring reach the same fit", {
    set.seed(123)
    n <- 1500L;x <- cbind(a=rnorm(n),b=rnorm(n),c=rbinom(n,1,0.3))
    for(link in c("logit","probit","cloglog")) {
        forward <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
        inverse <- switch(link,logit=qlogis,probit=qnorm,cloglog=function(u) log(-log1p(-u)))
        eta <- drop(x%*%c(0.5,-0.3,0.4));u <- runif(n);y <- rep(4L,n);cuts <- inverse(c(.25,.5,.75))
        for(j in 3:1) y[u<=forward(cuts[j]-eta-0.04*(j-2)*x[,2])] <- j
        y <- ordered(y,levels=1:4)
        fit <- fastglm_ordinal(x,y,link=link,nonparallel="b",tol=1e-10)
        reference <- fastglm_ordinal(x,y,link=link,nonparallel="b",tol=1e-10,control=list(observed=FALSE))
        expect_true(fit$converged);expect_true(reference$converged)
        expect_gt(fit$observed.iterations,0L);expect_identical(reference$observed.iterations,0L)
        expect_equal(coef(fit),coef(reference),tolerance=1e-8)
        expect_equal(vcov(fit),vcov(reference),tolerance=1e-8)
        expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(reference)),tolerance=1e-10)
    }
})
