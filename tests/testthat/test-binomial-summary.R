test_that("one-pass binomial summaries preserve likelihood and weight conventions", {
    eta <- seq(-2,2,length.out=12)
    y <- rep(c(0,.25,.5,1),3)
    for(link in c("probit","cloglog")) {
        family <- binomial(link);code <- family_code(family)
        for(trials in list(rep(1,12),rep(c(4,8,12),4),8)) {
            weights <- if(any(trials>1)) rep(trials,length.out=12)*rep(c(0,1,2),4) else
                c(0,.25,.5,1,1.5,2,2.5,3,4,5,6,7)
            result <- binomial_firth_summary(eta,y,weights,trials,code)
            probability <- family$linkinv(eta);derivative <- family$mu.eta(eta)
            deviance <- sum(family$dev.resids(y,probability,weights))
            expect_equal(result$deviance,deviance,tolerance=1e-11)
            expect_equal(result$aic,family$aic(y,rep(trials,length.out=length(y)),probability,weights,deviance),tolerance=1e-11)
            expect_equal(result$working,weights*derivative^2/family$variance(probability),tolerance=1e-11)
            residual <- (y-probability)/derivative;residual[weights==0] <- 0
            expect_equal(result$residuals,residual,tolerance=1e-11)
        }
    }
})

test_that("summary residuals and likelihoods retain extreme-tail contributions", {
    for(link in c("probit","cloglog")) {
        eta <- if(link=="probit") c(-40,-12,-1,1,12,40,0) else c(-1000,-40,-1,1,5,1000,0)
        y <- c(0,0,.25,.75,1,1,0);weights <- c(1,2,3,4,2,1,0)
        code <- family_code(binomial(link))
        result <- binomial_firth_summary(eta,y,weights,rep(1,7),code)
        if(link=="probit") {
            lp <- pnorm(eta,log.p=TRUE);lq <- pnorm(eta,lower.tail=FALSE,log.p=TRUE)
        } else {
            t <- exp(eta);lp <- eta
            middle <- eta>=-36 & t<=log(2);upper <- t>log(2)
            lp[middle] <- log(-expm1(-t[middle]));lp[upper] <- log1p(-exp(-t[upper]));lq <- -t
        }
        positive <- weights>0;success <- y>0 & positive;failure <- y<1 & positive
        expected <- sum(2*weights[success]*y[success]*(log(y[success])-lp[success]))+
            sum(2*weights[failure]*(1-y[failure])*(log1p(-y[failure])-lq[failure]))
        expect_equal(result$deviance,expected,tolerance=1e-13)
        expect_true(all(is.finite(result$residuals)))
        expect_identical(result$residuals[7],0)
        moments <- binomial_firth_moments(eta,y,weights,code)
        expect_equal(result$residuals,.binomial_working_residual(moments,y,eta,code)*positive,tolerance=1e-13)
        expect_equal(result$working,moments$working,tolerance=1e-13)
        expect_equal(result$aic,.binomial_tail_aic(moments,y,rep(1,7),weights),tolerance=1e-13)
    }
})

test_that("derivative-only evaluation preserves tails and missing values", {
    eta <- c(-Inf,-1000,-40,-12,-1,0,1,5,12,40,1000,Inf,NA_real_,NaN)
    for(code in c(4L,5L)) {
        expected <- if(code==4L) dnorm(eta) else exp(eta-exp(eta))
        expected[is.infinite(eta)] <- 0
        result <- binomial_firth_derivative(eta,code)
        expect_equal(result,expected,tolerance=1e-14)
        expect_identical(is.na(result),is.na(expected))
        expect_identical(is.nan(result),is.nan(expected))
    }
    expect_error(binomial_firth_derivative(0,3L),"unsupported")
    expect_error(binomial_firth_summary(0,0,c(1,2),1,4L),"incompatible")
})

test_that("native reporting and final accounting use the returned predictor state", {
    set.seed(113102)
    x <- cbind(1,x=rnorm(60),z=runif(60,-1,1));offset <- sin(seq_len(60))/10
    weights <- rep(c(0,1,2),20);y <- rep(c(0,1,1,0,1),12)
    designs <- list(dense=x,sparse=as(x,"dgCMatrix"))
    if(requireNamespace("bigmemory",quietly=TRUE)) designs$mapped <- bigmemory::as.big.matrix(x)
    for(link in c("probit","cloglog")) for(design in designs) for(inference in c("full","none")) {
        for(maxit in c(1L,200L)) {
            raw <- suppressWarnings(fastglmPure(design,y,binomial(link),weights,offset,
                method="llt",firth=TRUE,inference=inference,maxit=maxit,tol=1e-10))
            complete <- .finish_glm(raw,offset)
            expect_equal(raw$deviance,complete$deviance,tolerance=1e-12)
            eta <- complete$linear.predictors
            lp <- if(link=="probit") pnorm(eta,log.p=TRUE) else log(-expm1(-exp(eta)))
            lq <- if(link=="probit") pnorm(eta,lower.tail=FALSE,log.p=TRUE) else -exp(eta)
            expected <- -2*sum(weights[y==1]*lp[y==1])-2*sum(weights[y==0]*lq[y==0])
            expect_equal(raw$deviance,expected,tolerance=1e-12)
        }
    }
})
