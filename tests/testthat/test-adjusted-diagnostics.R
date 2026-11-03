test_that("Firth diagnostics report the adjusted equations at the returned estimate", {
    set.seed(81201)
    n <- 240L
    x <- cbind("(Intercept)"=1, x=rnorm(n), z=runif(n,-.4,.4))
    eta <- drop(x %*% c(.3,.25,-.1))
    w <- rep(c(0,.5,1,2),length.out=n)
    cases <- list(
        list(family=binomial(), y=rbinom(n,1,plogis(eta)),
             second=function(eta,mu) mu*(1-mu)*(1-2*mu)),
        list(family=binomial("probit"), y=rbinom(n,1,pnorm(eta)),
             second=function(eta,mu) -eta*dnorm(eta)),
        list(family=poisson(), y=rpois(n,exp(eta)),
             second=function(eta,mu) mu),
        list(family=Gamma("log"), y=rgamma(n,shape=2,scale=exp(eta)/2),
             second=function(eta,mu) mu),
        list(family=gaussian(), y=eta+rnorm(n),
             second=function(eta,mu) rep(0,length(mu))))
    for(case in cases) {
        weights <- if(case$family$family=="binomial") 2*w else w
        f <- fastglm(x,case$y,case$family,weights=weights,method=2,
                     firth=TRUE,tol=1e-10)
        d <- fastglm_diagnostics(f)
        expect_true(f$converged)
        expect_true(d$adjusted.score.available)
        mu <- f$fitted.values
        dm <- case$family$mu.eta(f$linear.predictors)
        variance <- case$family$variance(mu)
        working <- weights*dm^2/variance
        information <- crossprod(x,working*x)
        h <- working*rowSums((x %*% solve(information))*x)
        contribution <- weights*dm*(case$y-mu)/(f$dispersion*variance) +
            h*case$second(f$linear.predictors,mu)/(2*dm)
        expected <- drop(crossprod(x,contribution))
        expect_equal(d$adjusted.score,expected,tolerance=1e-8)
        expect_equal(d$standardized.adjusted.score,
                     max(abs(expected)/sqrt(diag(information)/f$dispersion)),
                     tolerance=1e-8)
        expect_lt(d$standardized.adjusted.score,1e-6)
        if(case$family$family %in% c("binomial","poisson")) {
            expect_true(is.na(d$adjusted.dispersion.score))
        } else {
            phi <- f$dispersion
            dev <- sum(case$family$dev.resids(case$y,mu,weights))
            if(case$family$family=="Gamma") {
                wp <- w[w>0]; a <- wp/phi
                expected.dev <- sum(2*wp*(log(a)-digamma(a)))
                d2 <- sum(wp^2*(2*trigamma(a)-2/a))
                d3 <- sum(wp^3*(-2*psigamma(a,2)-2/a^2))
                expected.phi <- (dev-expected.dev+(f$rank-2)*phi+d3/d2)/(2*phi^2)
                info.phi <- d2/(2*phi^4)
            } else {
                expected.phi <- (dev-(sum(w>0)-f$rank)*phi)/(2*phi^2)
                info.phi <- sum(w>0)/(2*phi^2)
            }
            expect_equal(d$adjusted.dispersion.score,expected.phi,tolerance=1e-8)
            expect_equal(d$standardized.dispersion.score,
                         expected.phi/sqrt(info.phi),tolerance=1e-8)
        }
    }
})

test_that("adjusted diagnostics retain coefficient units through automatic scaling", {
    set.seed(81202)
    x <- cbind("(Intercept)"=1,x=rnorm(160),z=runif(160,-1,1))
    y <- rgamma(160,shape=2,scale=exp(drop(x%*%c(.2,.3,-.2)))/2)
    scale <- c(1,1e-5,1e5)
    xx <- sweep(x,2,scale,"*")
    # One iteration leaves nonzero scores, so this checks their units rather
    # than comparing only values already close to numerical zero.
    expect_warning(a <- fastglm(x,y,Gamma("log"),firth=TRUE,method="auto",maxit=1),
                   "did not converge")
    expect_warning(b <- fastglm(xx,y,Gamma("log"),firth=TRUE,method="auto",maxit=1),
                   "did not converge")
    expect_equal(unname(b$adjusted.score),unname(a$adjusted.score)*scale,tolerance=1e-8)
    expect_equal(unname(b$adjusted.information.diagonal),
                 unname(a$adjusted.information.diagonal)*scale^2,tolerance=1e-8)
    expect_equal(fastglm_diagnostics(a)$standardized.adjusted.score,
                 fastglm_diagnostics(b)$standardized.adjusted.score,tolerance=1e-8)
    expect_false(fastglm_diagnostics(a)$converged)
    expect_identical(fastglm_diagnostics(a)$convergence.reason,"iteration_limit")
})

test_that("compact prepared fits retain adjusted diagnostics and older fits remain readable", {
    set.seed(81203)
    x <- cbind("(Intercept)"=1,x=rnorm(160)*1e3)
    y <- rgamma(160,shape=3,scale=exp(.2+.0003*x[,2])/3)
    full <- fastglm(x,y,Gamma("log"),firth=TRUE,method="auto",tol=1e-10)
    compact <- fastglm_batch(fastglm_prepare(x,firth=TRUE),matrix(y,ncol=1),
        family=Gamma("log"),method="auto",inference="none",retain.data=FALSE,
        tol=1e-10)[[1L]]
    d <- fastglm_diagnostics(compact)
    expect_true(d$adjusted.score.available)
    expect_false(d$ordinary.score.available)
    expect_true(is.na(d$standardized.score))
    expect_true(is.na(d$covariance.condition))
    expect_equal(d$adjusted.score,fastglm_diagnostics(full)$adjusted.score,tolerance=1e-7)
    expect_lt(d$standardized.adjusted.score,1e-6)
    old <- full
    old$adjusted.score <- old$standardized.adjusted.score <- NULL
    old$convergence.reason <- NULL
    legacy <- fastglm_diagnostics(old)
    expect_false(legacy$adjusted.score.available)
    expect_true(is.na(legacy$standardized.adjusted.score))
    expect_true(legacy$ordinary.score.available)
    expect_identical(legacy$convergence.reason,"converged")
    poisson <- fastglm(x,rpois(160,2),poisson(),firth=TRUE,method="auto",
                       inference="none",retain.data=FALSE)
    expect_false(fastglm_diagnostics(poisson)$adjusted.score.available)
    expect_true(is.na(fastglm_diagnostics(poisson)$standardized.adjusted.score))
})
