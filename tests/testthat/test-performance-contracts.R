test_that("final Cholesky covariance uses the reported mean without another coefficient update", {
    set.seed(7031)
    x <- cbind(1, matrix(rnorm(800), 200, 4))
    offset <- rnorm(200, sd=.15)
    weights <- rep(c(0, .5, 1, 2), 50)
    eta <- drop(x %*% c(.1, .3, -.2, .1, .2)) + offset
    families <- list(binomial(), poisson(), Gamma("log"), gaussian())
    response <- list(rbinom(200, 1, plogis(eta)), rpois(200, exp(eta)),
                     rgamma(200, shape=2, scale=exp(eta)/2), eta+rnorm(200))
    for (j in seq_along(families)) {
        fam <- families[[j]]
        for (method in c(2L, 3L)) {
            full <- suppressWarnings(fastglm(x, response[[j]], fam, weights=weights,
                offset=offset, method=method, maxit=1L))
            compact <- suppressWarnings(fastglm(x, response[[j]], fam, weights=weights,
                offset=offset, method=method, maxit=1L, inference="none"))
            w <- weights * fam$mu.eta(full$linear.predictors)^2 / fam$variance(full$fitted.values)
            expected <- solve(crossprod(x, w*x))
            expect_equal(unname(full$cov.unscaled), expected, tolerance=1e-10)
            expect_equal(coef(full), coef(compact), tolerance=1e-13)
            expect_equal(full$fitted.values, compact$fitted.values, tolerance=1e-13)
        }
    }
})

test_that("constant-column detection retains intercept and null-model accounting", {
    set.seed(7032)
    x <- cbind(2, rnorm(100), 0, c(rep(1, 99), 2), rep(-3, 100))
    expect_identical(fastglm:::.intercept_columns(x), c(TRUE, FALSE, FALSE, FALSE, TRUE))
    y <- rpois(100, 2)
    fit <- fastglm(x[,1:2], y, poisson())
    expect_true(fit$intercept)
    expect_equal(fit$null.deviance, sum(poisson()$dev.resids(y, mean(y), rep(1,100))))
})

test_that("Gamma dispersion grouping preserves the adjusted score for unequal prior weights", {
    skip_if_not_installed("brglm2")
    set.seed(7033)
    n <- 180L
    x <- cbind(1, rnorm(n), rnorm(n))
    mu <- exp(drop(x %*% c(.2, .2, -.1)))
    prior <- list(rep(1,n), rep(c(0,.5,1,2),length.out=n), runif(n,.25,2))
    for (shape in c(.2, 5, 250)) {
        y <- rgamma(n, shape=shape, scale=mu/shape)
        for (w in prior) {
            initial <- c(log(weighted.mean(y,w)),rep(0,ncol(x)-1L))
            ref <- brglm2::brglmFit(x,y,weights=w,start=initial,family=Gamma("log"),
                control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10,maxit=300))
            fit <- fastglm(x,y,Gamma("log"),weights=w,start=initial,firth=TRUE,method=2,tol=1e-10,maxit=300)
            expect_true(fit$converged)
            expect_equal(unname(coef(fit)),unname(coef(ref)),tolerance=1e-7)
            expect_equal(fit$dispersion,ref$dispersion,tolerance=1e-7)
            positive <- w[w>0]; phi <- fit$dispersion; a <- positive/phi
            expected <- sum(2*positive*(log(a)-digamma(a)))
            d2 <- sum(positive^2*(2*trigamma(a)-2/a))
            d3 <- sum(positive^3*(-2*psigamma(a,2)-2/a^2))
            score <- fit$deviance-expected+(fit$rank-2)*phi+d3/d2
            expect_lt(abs(score)/(1+fit$deviance),1e-9)
            sparse <- fastglm(as(x,"dgCMatrix"),y,Gamma("log"),weights=w,
                start=initial,firth=TRUE,method=2,tol=1e-10,maxit=300)
            expect_equal(unname(coef(sparse)),unname(coef(fit)),tolerance=1e-7)
            expect_equal(sparse$dispersion,fit$dispersion,tolerance=1e-7)
        }
    }
})

test_that("constant-weight Firth decompositions agree across solvers", {
    skip_if_not_installed("brglm2")
    set.seed(7034)
    x <- cbind(1, matrix(rnorm(240),120,2))
    eta <- drop(x %*% c(2,.15,-.1))
    families <- list(Gamma("log"), gaussian(), poisson("sqrt"))
    response <- list(rgamma(120,shape=2,scale=exp(eta)/2),eta+rnorm(120),rpois(120,eta^2))
    for (j in seq_along(families)) {
        reference <- brglm2::brglmFit(x,response[[j]],family=families[[j]],
            control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10,maxit=300))
        for (method in 0:5) {
            fit <- fastglm(x,response[[j]],families[[j]],method=method,
                           firth=TRUE,tol=1e-10,maxit=300)
            expect_true(fit$converged)
            expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=1e-7)
            expect_equal(fit$dispersion,reference$dispersion,tolerance=1e-7)
            w <- families[[j]]$mu.eta(fit$linear.predictors)^2 /
                 families[[j]]$variance(fit$fitted.values)
            expect_equal(unname(vcov(fit)),solve(crossprod(x,w*x))*fit$dispersion,tolerance=1e-9)
        }
    }
})
