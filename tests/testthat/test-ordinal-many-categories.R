many_category_data <- function(n,K,link="logit",seed=123L) {
    set.seed(seed)
    x <- cbind(a=rnorm(n,3,2),b=rnorm(n,-1,0.5),c=rbinom(n,1,0.4))
    inverse <- switch(link,logit=qlogis,probit=qnorm,cloglog=function(u) log(-log1p(-u)))
    latent <- drop(x%*%c(0.4,-0.6,0.5))+inverse(runif(n))
    list(x=x,y=ordered(ceiling(rank(latent,ties.method="first")*K/n),levels=seq_len(K)))
}

test_that("block factorization reproduces dense inverses and coordinate changes", {
    for(link in c("logit","probit","cloglog")) for(firth in c(FALSE,TRUE)) for(K in c(2L,3L,8L,60L)) {
        d <- many_category_data(300L,K,link)
        fit <- suppressWarnings(fastglm_ordinal(d$x,d$y,link=link,firth=firth,tol=1e-9))
        expect_true(fit$converged)
        w <- fit$working;q <- K-1L;p <- ncol(d$x);size <- q+p
        identity <- diag(size)
        expect_equal(unname(w$covariance%*%w$information),identity,tolerance=1e-8)
        # Dense pure-R reference for the structured coordinate transformation.
        T <- Ti <- diag(size)
        T[seq_len(q),q+seq_len(p)] <- matrix(w$center/w$scale,q,p,byrow=TRUE)
        T[q+seq_len(p),q+seq_len(p)] <- diag(1/w$scale,nrow=p)
        Ti[seq_len(q),q+seq_len(p)] <- matrix(-w$center,q,p,byrow=TRUE)
        Ti[q+seq_len(p),q+seq_len(p)] <- diag(w$scale,nrow=p)
        expect_equal(unname(fit$effective$covariance),unname(T%*%w$covariance%*%t(T)),tolerance=1e-9)
        expect_equal(unname(fit$effective$information),unname(crossprod(Ti,w$information%*%Ti)),tolerance=1e-9)
        expect_equal(unname(fit$effective$coefficients),unname(drop(T%*%w$coefficients)),tolerance=1e-10)
        expect_lt(max(abs(fit$effective$score-drop(crossprod(Ti,w$score)))),1e-7)
        condition <- 1/(max(colSums(abs(cov2cor(w$information))))*max(colSums(abs(solve(cov2cor(w$information))))))
        expect_equal(w$information.rcond,condition,tolerance=1e-6)
    }
})

test_that("one category per observation and threshold-only fits remain available", {
    d <- many_category_data(150L,150L)
    fit <- suppressWarnings(fastglm_ordinal(d$x,d$y,tol=1e-8))
    expect_true(fit$converged)
    expect_equal(dim(vcov(fit)),c(152L,152L))
    # Each slope score is zero at a solution of the ungrouped likelihood.
    expect_lt(max(abs(fit$effective$ordinary.score)),1e-4)
    only <- fastglm_ordinal(matrix(numeric(),60L,0L),many_category_data(60L,4L)$y)
    expect_true(only$converged)
    expect_equal(unname(plogis(only$alpha)),c(0.25,0.5,0.75),tolerance=1e-7)
})
