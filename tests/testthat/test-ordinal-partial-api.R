partial_api_data <- function(link="logit",n=90L) {
    d <- data.frame(x=rep(seq(-1,1,length.out=n/3L),3L),
        g=factor(rep(c("a","b","c"),each=n/3L)),off=sin(seq_len(n))*.1,
        w=rep(c(1,2,0,3,1),length.out=n))
    shift <- rbind(a=c(0,0),b=c(.35,-.15),c=c(-.2,.25))[d$g,,drop=FALSE]
    eta <- cbind(-.7,.85)[rep(1L,n),,drop=FALSE]-.4*d$x-d$off-shift
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
    cumulative <- inverse(eta);p <- cbind(cumulative,1)-cbind(0,cumulative)
    counts <- floor(60*p)+1
    d$low <- counts[,1L];d$middle <- counts[,2L];d$high <- counts[,3L]
    d
}

test_that("partial ordinal formulas and matrix fits preserve model methods", {
    for(link in c("logit","probit","cloglog")) {
        d <- partial_api_data(link)
        fit <- fastglm_ordinal(cbind(low,middle,high)~x+g+offset(off),d,weights=w,
            nonparallel=~g,link=link,tol=1e-9)
        xx <- model.matrix(~x+g,d)[,-1L,drop=FALSE]
        reference <- fastglm_ordinal(xx,as.matrix(d[c("low","middle","high")]),
            weights=d$w,offset=d$off,nonparallel=c("gb","gc"),link=link,tol=1e-9)
        expect_s3_class(fit,"fastglm_ordinal_partial")
        expect_s3_class(fit,"fastglm_ordinal")
        expect_true(fit$converged)
        expect_equal(coef(fit),coef(reference),tolerance=1e-11)
        expect_equal(vcov(fit),vcov(reference),tolerance=1e-11)
        expect_equal(fit$common,1L)
        expect_equal(fit$nonparallel,2:3)
        expect_identical(fit$nonparallel.terms,"g")
        expect_equal(dim(fit$gamma),c(2L,2L))
        expect_named(coef(fit,"common"),"x")
        expect_length(coef(fit,"location"),5L)
        expect_length(coef(fit,"nonparallel"),4L)
        expect_equal(unname(coef(fit,"nonparallel")),as.numeric(fit$gamma))
        expect_equal(dim(vcov(fit)),c(7L,7L))
        expect_equal(dim(vcov(fit,"nonparallel")),c(4L,4L))
        expect_equal(nobs(fit),sum(rowSums(d[c("low","middle","high")])*d$w))
        expect_equal(model.matrix(fit),xx)
        expect_equal(weights(fit),d$w)
        expect_equal(fitted(fit),predict(fit))
        expect_equal(rowSums(predict(fit)),rep(1,nrow(d)),tolerance=1e-12,ignore_attr=TRUE)
        expect_equal(predict(fit,type="response"),predict(fit,type="probability"))
        eta <- predict(fit,type="link")
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
        expect_equal(unname(predict(fit,type="cumulative")),unname(inverse(eta)),tolerance=1e-12)
        expect_true(all(eta[,2L]>eta[,1L]))
        expect_equal(dim(predict(fit,d[FALSE,])),c(0L,3L))
        far.common <- d[1:2,];far.common$x <- 1e16
        expect_true(all(is.finite(predict(fit,far.common))))
        expect_equal(rowSums(predict(fit,far.common)),c(1,1),tolerance=1e-12,ignore_attr=TRUE)
        expect_true(is.ordered(predict(fit,type="class")))
        expect_equal(levels(predict(fit,type="class")),c("low","middle","high"))
        expect_s3_class(summary(fit),"summary.fastglm_ordinal_partial")
        expect_output(print(fit),"Threshold-specific slopes")
        expect_output(print(summary(fit)),"Common slopes")
        expect_equal(nrow(summary(fit)$nonparallel),4L)
        expect_equal(length(fastglm_diagnostics(fit)$score),7L)
        expect_gt(fastglm_diagnostics(fit)$minimum.gap,0)
        expect_equal(as.numeric(fit$working$T%*%fit$working$coefficients),unname(coef(fit)),tolerance=1e-11)
        expect_equal(fit$working$Ti%*%fit$working$T,diag(7),tolerance=1e-12)
        saturated <- sum(vapply(seq_len(nrow(d)),function(i) {
            yy <- unlist(d[i,c("low","middle","high")])*d$w[i]
            if(!sum(yy)) return(0)
            sum(yy*log(yy/sum(yy)))
        },numeric(1)))
        expect_equal(deviance(fit),2*(saturated-as.numeric(logLik(fit))),tolerance=1e-10)
        expect_equal(dim(confint(fit)),c(7L,2L))
        name <- names(coef(fit,"nonparallel"))[1L]
        expect_equal(unname(confint(fit,name)[1L,]),
            unname(coef(fit)[name])+qnorm(c(.025,.975))*sqrt(vcov(fit)[name,name]),tolerance=1e-12)
    }
})

test_that("nonparallel term selection keeps complete factors and interactions", {
    d <- partial_api_data()
    fit <- fastglm_ordinal(cbind(low,middle,high)~x*g+offset(off),d,nonparallel=~g:x,tol=1e-9)
    expect_true(fit$converged)
    expect_identical(fit$nonparallel.terms,"x:g")
    expect_identical(fit$design.columns[fit$nonparallel],c("x:gb","x:gc"))
    all <- fastglm_ordinal(cbind(low,middle,high)~x+g+offset(off),d,nonparallel=~.,tol=1e-9)
    expect_true(all$converged)
    expect_length(all$beta,0L)
    expect_equal(nrow(all$gamma),3L)
    expect_equal(nrow(vcov(all,"common")),0L)
    expect_output(print(summary(all)),"No coefficients")
    common <- fastglm_ordinal(cbind(low,middle,high)~x+g+offset(off),d,tol=1e-9)
    none <- update(common,nonparallel=~1)
    expect_identical(class(none),class(common))
    expect_equal(coef(none),coef(common))
    xx <- model.matrix(common)
    matrix.none <- fastglm_ordinal(xx,as.matrix(d[c("low","middle","high")]),
        offset=d$off,nonparallel=integer(),tol=1e-9)
    expect_identical(class(matrix.none),"fastglm_ordinal")
    expect_equal(coef(matrix.none),coef(common),tolerance=1e-10)
    expect_error(update(common,nonparallel=~I(x^2)),"complete terms")
    expect_error(update(common,nonparallel=~offset(off)),"offset")
    expect_error(update(common,nonparallel=~gb),"complete terms")
    expect_error(update(common,nonparallel=c("x")),"one-sided")
    expect_error(update(common,nonparallel=x~g),"one-sided")
    expect_error(fastglm_ordinal(xx,as.matrix(d[c("low","middle","high")]),nonparallel=~x),"column name")
    expect_error(fastglm_ordinal(xx,as.matrix(d[c("low","middle","high")]),nonparallel=c(1,1)),"distinct")
    expect_error(fastglm_ordinal(xx,as.matrix(d[c("low","middle","high")]),nonparallel=4),"existing")
    expect_error(fastglm_ordinal(xx,as.matrix(d[c("low","middle","high")]),nonparallel=TRUE),"column names")
})

test_that("partial predictions use the joint covariance and preserve missing rows", {
    skip_if_not_installed("numDeriv")
    d <- partial_api_data();d$x[5L] <- NA_real_;d$extra <- cos(seq_len(nrow(d)))/10
    fit <- fastglm_ordinal(cbind(low,middle,high)~x+g+offset(off),d,weights=w,offset=extra,
        nonparallel=~g,na.action=na.exclude,tol=1e-10)
    training <- predict(fit,se.fit=TRUE)
    expect_equal(dim(training$fit),c(90L,3L))
    expect_true(all(is.na(training$fit[5L,])))
    expect_true(all(is.na(training$se.fit[5L,])))
    new <- d[c(1,3,32,35,63,65),];x <- model.matrix(~x+g,new)[,-1L,drop=FALSE]
    off <- new$off+new$extra
    probability <- function(b,cumulative=FALSE,link=FALSE) {
        eta <- matrix(b[1:2],nrow(x),2,byrow=TRUE)-drop(x[,1L,drop=FALSE]%*%b[3L])-off-
            x[,2:3,drop=FALSE]%*%matrix(b[4:7],2,2)
        if(link) return(eta)
        F <- plogis(eta)
        if(cumulative) F else cbind(F,1)-cbind(0,F)
    }
    for(type in c("probability","cumulative","link")) {
        fun <- function(b) probability(b,cumulative=type=="cumulative",link=type=="link")
        J <- numDeriv::jacobian(function(b) as.numeric(fun(b)),coef(fit))
        expected.se <- matrix(sqrt(rowSums((J%*%vcov(fit))*J)),nrow(new))
        result <- predict(fit,new,se.fit=TRUE,type=type)
        expect_equal(unname(result$fit),unname(fun(coef(fit))),tolerance=1e-11)
        expect_equal(unname(result$se.fit),unname(expected.se),tolerance=1e-7)
    }
    expect_equal(as.numeric(predict(fit,new,offset=off,type="link")),as.numeric(probability(coef(fit),link=TRUE)),tolerance=1e-11)
    incomplete <- new;incomplete$x[2L] <- NA_real_
    expect_true(all(is.na(predict(fit,incomplete)[2L,])))
    expect_error(predict(fit,new,type="class",se.fit=TRUE),"class predictions")
    expect_error(predict(fit,transform(new,g="unknown")),"new level")
    expect_error(predict(fit,transform(new,x=Inf)),"finite or missing")
    expect_error(predict(fit,new,se.fit=NA),"TRUE or FALSE")
    expect_true(all(is.finite(confint(fit,parm=3L,method="profile"))))
    expect_equal(vcov(fit,type="HC0"),fastglm_vcov(fit))
    expect_error(confint(fit,parm=c(3,3)),"duplicated")
})

test_that("partial models retain grouped and expanded outcome semantics", {
    d <- partial_api_data();x <- model.matrix(~x+g,d)[,-1L,drop=FALSE]
    counts <- as.matrix(d[c("low","middle","high")])
    fit <- fastglm_ordinal(x,counts,weights=d$w,offset=d$off,nonparallel=2:3,tol=1e-9)
    effective <- counts*d$w
    rows <- rep(seq_len(nrow(d)),rowSums(effective))
    codes <- unlist(lapply(seq_len(nrow(d)),function(i)rep(seq_len(3L),effective[i,])))
    expanded <- fastglm_ordinal(x[rows,,drop=FALSE],ordered(codes,levels=1:3,labels=colnames(counts)),
        offset=d$off[rows],nonparallel=2:3,tol=1e-9)
    expect_equal(coef(fit),coef(expanded),tolerance=1e-8)
    expect_equal(vcov(fit),vcov(expanded),tolerance=1e-8)
    expect_equal(logLik(fit),logLik(expanded),tolerance=1e-10)
    positive <- d$w>0
    reduced <- fastglm_ordinal(x[positive,,drop=FALSE],counts[positive,,drop=FALSE],
        weights=d$w[positive],offset=d$off[positive],nonparallel=2:3,tol=1e-9)
    expect_equal(coef(fit),coef(reduced),tolerance=1e-10)
    expect_equal(predict(fit)[positive,,drop=FALSE],predict(reduced),tolerance=1e-10,ignore_attr=TRUE)
    expect_equal(length(residuals(fit)),nrow(d))
    expect_true(all(is.na(residuals(fit)[!positive])))
    expect_true(all(abs(residuals(fit)[positive])<=1))
    expect_equal(dim(residuals(fit,"cumulative")),c(nrow(d),2L))
    draws <- simulate(fit,nsim=2,seed=114811)
    expect_equal(rowSums(draws[[1L]]),rowSums(effective),ignore_attr=TRUE)
    expect_equal(dim(draws[[2L]]),dim(counts))
    if(requireNamespace("broom",quietly=TRUE)) {
        tidy <- broom::tidy(fit,conf.int=TRUE)
        expect_equal(nrow(tidy),7L)
        expect_equal(tidy$estimate,unname(coef(fit)))
        expect_equal(tidy$std.error,unname(sqrt(diag(vcov(fit)))))
        expect_equal(sum(tidy$component=="nonparallel"),4L)
        expect_true(all(is.na(tidy$p.value[tidy$component=="threshold"])))
        exponentiated <- broom::tidy(fit,exponentiate=TRUE)
        slopes <- tidy$component!="threshold"
        expect_equal(exponentiated$estimate[slopes],exp(tidy$estimate[slopes]))
        expect_equal(broom::glance(fit)$nobs,nobs(fit))
        augmented <- broom::augment(fit,se_fit=TRUE)
        expect_equal(unname(augmented$.fitted),unname(predict(fit)))
        expect_equal(augmented$.resid,unname(residuals(fit)))
    }
})

test_that("partial predictions reject crossing cumulative probabilities", {
    x <- matrix(seq(10,12,length.out=40),ncol=1,dimnames=list(NULL,"x"))
    eta <- cbind(-1,.8)[rep(1L,40),]-cbind((x[,1]-11)*.7,(x[,1]-11)*.15)
    P <- cbind(plogis(eta),1)-cbind(0,plogis(eta))
    counts <- floor(100*P)+1;colnames(counts) <- c("low","middle","high")
    fit <- fastglm_ordinal(x,counts,nonparallel="x",tol=1e-10)
    expect_true(fit$converged)
    expect_true(all(is.finite(predict(fit))))
    # The origin is outside the fitted covariate range; valid training
    # probabilities do not require ordered thresholds evaluated at the origin.
    expect_lt(diff(unname(fit$alpha)),0)
    unscaled <- fastglm_ordinal(x,counts,nonparallel="x",control=list(standardize=FALSE),tol=1e-10)
    expect_true(unscaled$converged)
    expect_equal(coef(unscaled),coef(fit),tolerance=1e-8)
    expect_equal(vcov(unscaled),vcov(fit),tolerance=1e-8)
    origin <- matrix(0,ncol=1,dimnames=list(NULL,"x"))
    expect_error(predict(fit,origin),"row\\(s\\): 1")
    invalid <- matrix(c(11,0,11),ncol=1,dimnames=list(NULL,"x"))
    expect_error(predict(fit,invalid),"row\\(s\\): 2")
    expect_error(predict(fit,invalid,se.fit=TRUE),"not ordered")
    zero.row <- fastglm_ordinal(rbind(x,origin),rbind(counts,c(1,1,1)),
        weights=c(rep(1,nrow(x)),0),nonparallel="x",tol=1e-10)
    expect_equal(coef(zero.row),coef(fit),tolerance=1e-10)
    expect_error(predict(zero.row),"row\\(s\\): 41")
    expect_equal(predict(zero.row,x),predict(fit),tolerance=1e-10)
    simulated <- simulate(zero.row,nsim=2,seed=321)
    expect_equal(simulated[[1L]][41L,],setNames(c(0L,0L,0L),colnames(counts)))
    expect_equal(rowSums(simulated[[1L]]),c(rowSums(counts),0),ignore_attr=TRUE)
    expect_equal(simulated[[1L]][seq_len(40L),],simulate(fit,seed=321)[[1L]])
})

test_that("partial fitting rejects unsupported scales and retains failures", {
    d <- partial_api_data();x <- model.matrix(~x+g,d)[,-1L,drop=FALSE]
    counts <- as.matrix(d[c("low","middle","high")])
    empty <- counts;empty[,2L] <- 0
    expect_error(fastglm_ordinal(x,empty,nonparallel=2:3),"every declared category")
    empty <- counts;empty[,1L] <- 0
    expect_error(fastglm_ordinal(x,empty,nonparallel=2:3),"every declared category")
    expect_error(fastglm_ordinal(x,counts,nonparallel=2:3,control=list(score.tol=NULL)),"control")
    expect_error(fastglm_ordinal(cbind(x,x),counts,nonparallel=1),"distinct")
    expect_error(fastglm_ordinal(cbind(intercept=1,x),counts,nonparallel=1),"constant|full rank")
    alias <- cbind(x,alias=x[,1L]);expect_error(fastglm_ordinal(alias,counts,nonparallel=2:3),"full rank")
    badnames <- x;colnames(badnames)[1L] <- "nonparallel:gb:low|middle"
    expect_error(fastglm_ordinal(badnames,counts,nonparallel=2:3),"conflict")
    expect_warning(failed <- fastglm_ordinal(x,counts,nonparallel=2:3,maxit=1),"did not converge")
    expect_false(failed$converged)
    expect_error(vcov(failed),"converged")
    expect_error(predict(failed,se.fit=TRUE),"converged")
    expect_true(all(is.na(summary(failed)$coefficients[,2L])))
    expect_true(all(is.finite(predict(failed))))
    fitted <- fastglm_ordinal(cbind(low,middle,high)~x+g,d,nonparallel=~g,tol=1e-9)
    expect_equal(coef(update(fitted,nonparallel=~1)),coef(fastglm_ordinal(cbind(low,middle,high)~x+g,d,tol=1e-9)))
    expect_true(is.call(update(fitted,link="probit",evaluate=FALSE)))
})

test_that("partial bias reduction preserves grouping and affine predictor recoding", {
    for(link in c("logit","probit","cloglog")) {
        d <- partial_api_data(link,n=30L);x <- model.matrix(~x+g,d)[,-1L,drop=FALSE]
        counts <- as.matrix(d[c("low","middle","high")])
        fit <- fastglm_ordinal(x,counts,weights=d$w,offset=d$off,nonparallel=2:3,
            firth=TRUE,link=link,tol=1e-9,maxit=500)
        formula <- fastglm_ordinal(cbind(low,middle,high)~x+g+offset(off),d,weights=w,
            nonparallel=~g,firth=TRUE,link=link,tol=1e-9,maxit=500)
        expect_true(fit$converged);expect_true(fit$firth)
        expect_identical(fit$method,"meanBR")
        expect_equal(coef(fit),coef(formula),tolerance=1e-10)
        shift <- c(3,-2,.5);scale <- c(-2,3,.4)
        changed <- sweep(sweep(x,2L,shift,"-"),2L,scale,"/")
        recoded <- fastglm_ordinal(changed,counts,weights=d$w,offset=d$off,nonparallel=2:3,
            firth=TRUE,link=link,tol=1e-9,maxit=500)
        map <- diag(7);map[1:2,3] <- -shift[1];map[3,3] <- scale[1]
        for(j in 1:2) {
            at <- 3+(j-1)*2+1:2
            map[j,at] <- -shift[2:3];map[at,at] <- diag(scale[2:3])
        }
        expect_true(recoded$converged)
        expect_equal(unname(coef(recoded)),drop(map%*%coef(fit)),tolerance=1e-8)
        expect_equal(unname(vcov(recoded)),unname(map%*%vcov(fit)%*%t(map)),tolerance=1e-8,ignore_attr=TRUE)
        expect_equal(predict(recoded),predict(fit),tolerance=1e-10)
        expect_equal(residuals(recoded),residuals(fit),tolerance=1e-10)
        effective <- counts*d$w;rows <- rep(seq_len(nrow(d)),rowSums(effective))
        codes <- unlist(lapply(seq_len(nrow(d)),function(i)rep(1:3,effective[i,])))
        expanded <- fastglm_ordinal(x[rows,,drop=FALSE],ordered(codes,levels=1:3,labels=colnames(counts)),
            offset=d$off[rows],nonparallel=2:3,firth=TRUE,link=link,tol=1e-9,maxit=500)
        expect_true(expanded$converged)
        expect_equal(coef(fit),coef(expanded),tolerance=1e-8)
        expect_equal(vcov(fit),vcov(expanded),tolerance=1e-8)
        expect_equal(logLik(fit),logLik(expanded),tolerance=1e-10)
    }
})

test_that("partial bias reduction solves the published wine example", {
    counts <- rbind(c(4,9,5,0,0),c(1,7,8,2,0),c(0,5,8,3,2),c(0,1,5,7,5))
    colnames(counts) <- as.character(1:5)
    x <- cbind(contact=c(0,1,0,1),temperature=c(0,0,1,1))
    fit <- fastglm_ordinal(x,counts,nonparallel="temperature",firth=TRUE,tol=1e-9,maxit=1000)
    expect_true(fit$converged)
    expect_equal(unname(coef(fit)),c(-1.194970681,1.054511572,3.497848021,5.196343297,
        1.397393080,2.620809623,2.050016870,2.648218933,2.961184939),tolerance=1e-7)
    expect_output(print(summary(fit)),"mean bias reduction")
    expect_lt(fastglm_diagnostics(fit)$standardized.score,1e-9)
    expect_lt(fastglm_diagnostics(fit)$remaining.predictor.step,1e-9)
    expect_true(all(is.finite(vcov(fit))))
})

test_that("ordinal Pearson residuals use cumulative binomial variance for every link", {
    d <- MASS::housing
    for(link in c("logit","probit","cloglog")) for(partial in c(FALSE,TRUE)) {
        fit <- fastglm_ordinal(Sat~Infl+Cont,d,weights=Freq,
            nonparallel=if(partial) ~Cont else NULL,link=link)
        cumulative <- predict(fit,type="cumulative")
        observed <- outer(as.integer(d$Sat),seq_len(ncol(cumulative)),"<=")
        expected <- (observed-cumulative)*sqrt(d$Freq)/sqrt(cumulative*(1-cumulative))
        expected[d$Freq==0,] <- NA_real_
        expect_equal(residuals(fit,type="pearson"),expected,tolerance=1e-12,ignore_attr=TRUE)
    }
})

test_that("binary partial fits reduce to the common-slope ordinal model", {
    x <- cbind(x=seq(-1,1,length.out=24L),z=rep(c(0,1),12L))
    counts <- cbind(low=round(30*plogis(.3-.4*x[,1]+.2*x[,2]))+1,
        high=round(30*plogis(-.3+.4*x[,1]-.2*x[,2]))+1)
    for(link in c("logit","probit","cloglog")) for(firth in c(FALSE,TRUE)) {
        fit <- fastglm_ordinal(x,counts,nonparallel="z",link=link,firth=firth,tol=1e-10)
        reference <- fastglm_ordinal(x,counts,link=link,firth=firth,tol=1e-10)
        expect_true(fit$converged)
        expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=1e-9)
        expect_equal(unname(vcov(fit)),unname(vcov(reference)),tolerance=1e-9)
        expect_equal(predict(fit),predict(reference),tolerance=1e-10)
        expect_equal(predict(fit,x[1L,,drop=FALSE]),predict(reference,x[1L,,drop=FALSE]),tolerance=1e-10)
        expect_equal(dim(predict(fit,x[1L,,drop=FALSE],type="link")),c(1L,1L))
    }
})

test_that("partial predictions preserve small terms when large predictor effects cancel", {
    x <- rbind(c(-1e17,1,1e17,-1e17),c(1e17,1,-1e17,1e17))
    colnames(x) <- c("large.common","small.common","varying.a","varying.b")
    alpha <- c(-1.25,.875);beta <- c(.1,.375)
    gamma <- matrix(c(.1,0,.2,.1),2L,2L)
    offset <- c(.25,-.125)
    expected.eta <- matrix(alpha,2L,2L,byrow=TRUE)-beta[2L]-offset
    for(link in c("logit","probit","cloglog")) {
        object <- structure(list(alpha=alpha,beta=beta,gamma=gamma,
            common=1:2,nonparallel=3:4,coefficients=c(alpha,beta,gamma),
            working=list(alpha=alpha,beta=beta,gamma=gamma,center=rep(0,4L),scale=rep(1,4L)),
            levels=c("low","middle","high"),design.columns=colnames(x),
            block.size=1L,link=link),class=c("fastglm_ordinal_partial","fastglm_ordinal"))
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
        expected.cumulative <- inverse(expected.eta)
        expected.probability <- cbind(expected.cumulative,1)-cbind(0,expected.cumulative)
        expect_equal(predict(object,x,type="link",offset=offset),expected.eta,tolerance=1e-14,ignore_attr=TRUE)
        expect_equal(predict(object,x,type="cumulative",offset=offset),expected.cumulative,tolerance=1e-14,ignore_attr=TRUE)
        probability <- predict(object,x,offset=offset)
        expect_equal(probability,expected.probability,tolerance=1e-14,ignore_attr=TRUE)
        expect_equal(rowSums(probability),c(1,1),tolerance=1e-14,ignore_attr=TRUE)
    }
    # In binary floating-point, .4 - .3 - .1 equals 2^-55. Computing
    # differences between gamma columns before multiplying by 1e17 loses
    # part of this quantity, despite accurate individual predictors.
    alpha <- c(-1.25,5.875);gamma <- matrix(c(.1,0,.4,.3),2L,2L)
    expected.eta <- matrix(alpha,2L,2L,byrow=TRUE)-beta[2L]-offset
    expected.eta[,2L] <- expected.eta[,2L]-c(1,-1)*(2^-55*1e17)
    object$alpha <- object$working$alpha <- alpha
    object$gamma <- object$working$gamma <- gamma
    object$coefficients <- c(alpha,beta,gamma)
    for(link in c("logit","probit","cloglog")) {
        object$link <- link
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
        expected.cumulative <- inverse(expected.eta)
        expected.probability <- cbind(expected.cumulative,1)-cbind(0,expected.cumulative)
        expect_equal(predict(object,x,type="link",offset=offset),expected.eta,tolerance=1e-14,ignore_attr=TRUE)
        probability <- predict(object,x,offset=offset)
        expect_equal(probability,expected.probability,tolerance=1e-14,ignore_attr=TRUE)
        expect_equal(rowSums(probability),c(1,1),tolerance=1e-14,ignore_attr=TRUE)
    }
})
