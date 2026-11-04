# Independent score/information calculations use log probabilities throughout.
tail_reference <- function(x,y,weights,offset,b,link) {
    eta <- drop(x%*%b)+offset
    if(link=="probit") {
        lp <- pnorm(eta,log.p=TRUE);lq <- pnorm(eta,lower.tail=FALSE,log.p=TRUE)
        ld <- dnorm(eta,log=TRUE);ratio <- -eta
    } else {
        t <- exp(eta);lq <- -t
        lp <- ifelse(t>log(2),log1p(-exp(-t)),log(-expm1(-t)))
        ld <- eta-t;ratio <- 1-t
    }
    working <- weights*exp(2*ld-lp-lq)
    u <- numeric(length(y));positive <- weights>0
    i <- positive & y>0;u[i] <- weights[i]*y[i]*exp(ld[i]-lp[i])
    i <- positive & y<1;u[i] <- u[i]-weights[i]*(1-y[i])*exp(ld[i]-lq[i])
    information <- crossprod(x,x*working);V <- solve(information)
    h <- working*rowSums((x%*%V)*x)
    list(score=drop(crossprod(x,u+.5*h*ratio)),ordinary=drop(crossprod(x,u)),
         information=information,covariance=V,working=working,h=h,lp=lp,lq=lq)
}

test_that("binomial tail moments retain small likelihood and score contributions", {
    for(link in c("probit","cloglog")) {
        eta <- if(link=="probit") c(-35,-10,-2,0,2,10,35) else c(-700,-40,-3,0,2,5,7)
        y <- c(0,1,.2,.6,.8,1,1);w <- c(1,0,2,3,1,2,1)
        code <- family_code(binomial(link))
        got <- binomial_firth_moments(eta,y,w,code)
        if(link=="probit") {
            lp <- pnorm(eta,log.p=TRUE);lq <- pnorm(eta,lower.tail=FALSE,log.p=TRUE);ld <- dnorm(eta,log=TRUE)
        } else {
            lp <- ifelse(exp(eta)>log(2),log1p(-exp(-exp(eta))),log(-expm1(-exp(eta))))
            lq <- -exp(eta);ld <- eta-exp(eta)
        }
        expect_equal(got$logp,lp,tolerance=1e-14)
        expect_equal(got$logq,lq,tolerance=1e-14)
        expect_equal(got$working,w*exp(2*ld-lp-lq),tolerance=1e-13)
        u <- numeric(length(y));take <- y>0 & w>0
        u[take] <- w[take]*y[take]*exp(ld[take]-lp[take]);take <- y<1 & w>0
        u[take] <- u[take]-w[take]*(1-y[take])*exp(ld[take]-lq[take])
        expect_equal(got$score,u,tolerance=1e-13)
        expect_true(all(is.finite(got$deviance)))
        expect_gt(got$deviance[6],0)
        expect_equal(got$deviance[6]/(-2*w[6]*lp[6]),1,tolerance=1e-13)
        expect_identical(got$score[2],0)
    }
})

test_that("separated noncanonical fits solve the unfloored adjusted equations", {
    set.seed(112101)
    x <- cbind("(Intercept)"=1,x=seq(-2,2,length.out=240),z=rnorm(240,sd=.25))
    y <- as.numeric(x[,2]>0);w <- rep(c(0,1,2,1),60);off <- cos(1:240)/8
    for(link in c("probit","cloglog")) {
        fits <- list()
        for(method in c("qr","qr_unpivoted","llt","ldlt","svd")) {
            fits[[method]] <- suppressWarnings(fastglm(x,y,binomial(link),weights=w,offset=off,
                firth=TRUE,method=method,tol=1e-10,maxit=500))
        }
        fits$sparse <- suppressWarnings(fastglm(as(x,"dgCMatrix"),y,binomial(link),weights=w,offset=off,
            firth=TRUE,method="llt",tol=1e-10,maxit=500))
        if(requireNamespace("bigmemory",quietly=TRUE)) fits$mapped <- suppressWarnings(fastglm(bigmemory::as.big.matrix(x),
            y,binomial(link),weights=w,offset=off,firth=TRUE,method="llt",tol=1e-10,maxit=500))
        for(size in c(24,60)) for(method in c("qr","llt")) {
            read <- local({chunk <- size;function(k) {
                i<-((k-1)*chunk+1):(k*chunk);list(X=x[i,],y=y[i],weights=w[i],offset=off[i])
            }})
            fits[[paste(size,method)]] <- suppressWarnings(fastglm_streaming(read,240/size,family=binomial(link),
                firth=TRUE,method=method,tol=1e-10,maxit=500))
        }
        ref <- fits$llt
        exact <- tail_reference(x,y,w,off,coef(ref),link)
        expect_lt(max(abs(exact$score/sqrt(diag(exact$information)))),1e-8)
        for(name in names(fits)) {
            fit <- fits[[name]]
            expect_true(fit$converged,info=paste(link,name))
            expect_equal(unname(coef(fit)),unname(coef(ref)),tolerance=1e-8,info=paste(link,name))
            expect_equal(unname(vcov(fit)),unname(exact$covariance),tolerance=1e-7)
            expect_equal(drop(fit$adjusted.score),unname(exact$score),tolerance=1e-8)
            expect_equal(as.numeric(logLik(fit)),sum(w[y==1]*exact$lp[y==1])+sum(w[y==0]*exact$lq[y==0]),tolerance=1e-10)
            for(type in c("working","deviance","pearson","response"))
                expect_equal(unname(residuals(fit,type=type)),unname(residuals(ref,type=type)),tolerance=1e-8)
            expect_equal(sum(residuals(fit)^2),deviance(fit),tolerance=1e-10)
            expect_equal(unname(hatvalues(fit)),unname(exact$h),tolerance=1e-8)
            expect_equal(unname(fastglm_diagnostics(fit)$score),unname(exact$ordinary),tolerance=1e-8)
        }
        ordinal <- fastglm_ordinal(-x[,-1],cbind(low=y,high=1-y),weights=w,offset=-off,
            link=link,firth=TRUE,tol=1e-10,maxit=500)
        expect_true(ordinal$converged)
        expect_equal(unname(coef(ordinal)),unname(coef(ref)),tolerance=1e-8)
        expect_equal(unname(vcov(ordinal)),unname(vcov(ref)),tolerance=1e-7,ignore_attr=TRUE)
    }
})

test_that("tail prediction derivatives and missing-row handling are consistent", {
    d <- data.frame(y=rep(c(0,1,0,1,1),12),x=seq(-1,1,length.out=60))
    d$x[3] <- NA_real_
    for(link in c("probit","cloglog")) {
        fit <- fastglm(y~x,d,binomial(link),firth=TRUE,na.action=na.exclude)
        for(type in c("working","deviance","pearson","response")) {
            r <- residuals(fit,type=type)
            expect_length(r,60);expect_true(is.na(r[3]))
        }
        eta <- if(link=="probit") c(-12,0,12) else c(-40,0,5)
        new <- data.frame(x=(eta-coef(fit)[1])/coef(fit)[2])
        scale <- predict(fit,new,type="link",se.fit=TRUE)$se.fit
        result <- predict(fit,new,type="response",se.fit=TRUE)$se.fit
        derivative <- if(link=="probit") dnorm(eta) else exp(eta-exp(eta))
        expect_equal(unname(result/scale/derivative),rep(1,3),tolerance=1e-11)
        older <- fit;older$stable.binomial <- NULL
        expect_equal(.glm_mu_eta(older,eta),older$family$mu.eta(eta))
    }
})

test_that("stable likelihoods retain grouped-trial and frequency conventions", {
    x <- cbind(1,x=seq(-1,1,length.out=40));y <- cbind(rep(c(0,1,4,8),10),rep(c(8,7,4,0),10))
    w <- rep(c(0,1,2,1),10)
    for(link in c("probit","cloglog")) {
        fit <- fastglm(x,y,binomial(link),weights=w,firth=TRUE)
        eta <- drop(x%*%coef(fit));prob <- binomial(link)$linkinv(eta)
        expect_equal(as.numeric(logLik(fit)),sum(w*dbinom(y[,1],rowSums(y),prob,log=TRUE)),tolerance=1e-10)
        compact <- fastglm(x,y,binomial(link),weights=w,firth=TRUE,retain.data=FALSE)
        for(type in c("working","deviance","pearson","response"))
            expect_equal(residuals(compact,type=type)[w>0],residuals(fit,type=type)[w>0],tolerance=1e-10)
        ml <- fastglm(x,y,binomial(link),weights=w)
        ref <- glm.fit(x,y,weights=w,family=binomial(link))
        expect_equal(coef(ml),coef(ref),tolerance=1e-7)
        expect_false(isTRUE(ml$stable.binomial))
    }
})
