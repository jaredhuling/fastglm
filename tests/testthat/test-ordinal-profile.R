# Independent ordinal likelihood used only to check constrained optimization.
ordinal_profile_reference <- function(x,counts,weights,offset,link,fixed,value,start) {
    q <- ncol(counts)-1L;p <- ncol(x)
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
    encode <- function(b) c(b[1L],log(diff(b[seq_len(q)])),b[q+seq_len(p)])
    decode <- function(z) c(cumsum(c(z[1L],exp(z[seq_len(q)[-1L]]))),z[q+seq_len(p)])
    theta <- encode(start);free <- setdiff(seq_along(theta),fixed)
    objective <- function(free.theta) {
        z <- theta;z[free] <- free.theta;z[fixed] <- value;b <- decode(z)
        F <- inverse(outer(-drop(x%*%b[q+seq_len(p)])-offset,b[seq_len(q)],"+"))
        P <- cbind(F,1)-cbind(0,F)
        if(any(P<=0)) return(Inf)
        -sum(weights*counts*log(P))
    }
    answer <- nlminb(theta[free],objective,control=list(iter.max=1000L,eval.max=3000L,rel.tol=1e-12))
    z <- theta;z[free] <- answer$par;z[fixed] <- value
    list(coefficients=decode(z),loglik=-answer$objective,convergence=answer$convergence)
}

test_that("ordinal likelihood-ratio tests agree with nested ML fits", {
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,link=link,tol=1e-10)
        reduced <- fastglm_ordinal(Sat~Infl+Cont,MASS::housing,weights=Freq,link=link,tol=1e-10)
        result <- .ordinal_profile_test(fit,grep("^Type",names(coef(fit))))
        expect_true(result$available)
        expect_equal(unname(result$statistic),2*as.numeric(logLik(fit)-logLik(reduced)),tolerance=1e-8)
        expect_equal(unname(result$parameter),3)
        expect_equal(unname(result$loglik["restricted"]),as.numeric(logLik(reduced)),tolerance=1e-10)
        expect_output(print(result),"Ordinal likelihood ratio test")
        expect_true(as.data.frame(result)$available)
        expect_identical(as.data.frame(result)$null.estimation,"ML")
    }
})

test_that("ordinal constrained fits match independent likelihood maximization", {
    set.seed(113501)
    x <- cbind(x=runif(16,-1,1),z=rnorm(16,sd=.3));offset <- sin(x[,1])/5
    counts <- matrix(rpois(64,4)+1,16,4);weights <- rep(c(0,1,2,1),4)
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(x,counts,weights=weights,offset=offset,link=link,tol=1e-10)
        setup <- .ordinal_profile_setup(fit,list())
        L <- diag(5)[4,,drop=FALSE]
        result <- .ordinal_constrained_fit(setup,L,.35)
        reference <- ordinal_profile_reference(x,counts,weights,offset,link,4,.35,coef(fit))
        expect_true(result$converged)
        expect_equal(unname(result$coefficients[4]),.35,tolerance=1e-12)
        expect_equal(result$value,reference$loglik,tolerance=1e-9)
        expect_equal(unname(result$coefficients),unname(reference$coefficients),tolerance=2e-5)
        expect_lt(result$score,1e-7);expect_lt(result$predictor.step,1e-7)
        all.fixed <- .ordinal_profile_test(fit,seq_along(coef(fit)),coef(fit))
        expect_true(all.fixed$available);expect_lt(unname(all.fixed$statistic),1e-10)
    }
})

test_that("a finite ordinal score null remains usable under separated alternatives", {
    x <- matrix(seq(-2,2,length.out=90),ncol=1,dimnames=list(NULL,"x"))
    y <- ordered(findInterval(x[,1],c(-.45,.6))+1,levels=1:3)
    fit <- fastglm_ordinal(x,y,firth=TRUE,tol=1e-10)
    expect_true(fit$converged)
    result <- fastglm_score_test(fit,terms="x")
    expect_true(result$available)
    null <- fastglm_ordinal(matrix(numeric(),90,0),y,tol=1e-10)
    expected <- ordinal_native_evaluate(x,as.integer(y),matrix(numeric(),0,0),rep(1,90),numeric(90),
        3L,c(coef(null),0))
    statistic <- drop(crossprod(expected$score,expected$cov.unscaled%*%expected$score))
    expect_equal(unname(result$statistic),statistic,tolerance=1e-8)
    expect_equal(unname(result$diagnostics$restricted$coefficients[1:2]),unname(coef(null)),tolerance=1e-8)
})

test_that("binary threshold profile intervals invert the binomial likelihood ratio", {
    counts <- matrix(c(3,11),1,dimnames=list(NULL,c("low","high")))
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(matrix(numeric(),1,0),counts,link=link,tol=1e-10)
        result <- confint(fit,method="profile")
        objective <- function(p) 2*sum(counts*log(c(3,11)/14/c(p,1-p)))-qchisq(.95,1)
        p <- c(uniroot(objective,c(1e-12,3/14),tol=1e-12)$root,
            uniroot(objective,c(3/14,1-1e-12),tol=1e-12)$root)
        reference <- binomial(link)$linkfun(p)
        expect_equal(as.numeric(result),reference,tolerance=2e-6)
        diagnostics <- attr(result,"diagnostics")
        expect_true(all(diagnostics$converged));expect_lt(max(abs(diagnostics$lr.error)),2e-6)
        expect_true(all(diff(as.numeric(result))>0))
        narrow <- confint(fit,method="profile",level=.8)
        expect_gt(narrow[1,1],result[1,1]);expect_lt(narrow[1,2],result[1,2])
    }
})

test_that("binary adjusted-score tests agree with scalar hand calculations", {
    counts <- matrix(c(3,11),1,dimnames=list(NULL,c("low","high")));theta <- .3
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(matrix(numeric(),1,0),counts,link=link,firth=TRUE,tol=1e-10)
        family <- binomial(link);p <- family$linkinv(theta);d <- family$mu.eta(theta)
        ratio <- switch(link,logit=1-2*p,probit=-theta,cloglog=1-exp(theta))
        information <- 14*d^2/(p*(1-p));ordinary <- d*(3-14*p)/(p*(1-p))
        for(estimation in c("ML","meanBR")) {
            result <- fastglm_score_test(fit,contrast=setNames(1,names(coef(fit))),null=theta,null.estimation=estimation)
            expected <- (ordinary+if(estimation=="meanBR") ratio/2 else 0)^2/information
            expect_true(result$available)
            expect_equal(unname(result$statistic),expected,tolerance=1e-12)
            expect_identical(result$null.estimation,estimation)
        }
    }
})

test_that("ordinal score tests re-estimate full-model adjusted nuisance equations", {
    set.seed(113502)
    x <- cbind(x=runif(18,-1,1),z=rnorm(18,sd=.4));offset <- sin(x[,1])/4
    counts <- matrix(rpois(54,2)+1,18,3);weights <- rep(c(0,1,2),6)
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(x,counts,weights=weights,offset=offset,link=link,firth=TRUE,tol=1e-10)
        result <- fastglm_score_test(fit,terms="x",null.estimation="meanBR")
        expect_true(result$available)
        b <- result$diagnostics$restricted$coefficients
        evaluated <- ordinal_native_evaluate(x,integer(),counts,weights,offset,3L,b,firth=TRUE,link=.ordinal_link_code(link))
        expect_lt(max(abs(evaluated$score[-3]/sqrt(diag(evaluated$information)[-3]))),1e-7)
        expected <- drop(crossprod(evaluated$score,evaluated$cov.unscaled%*%evaluated$score))
        expect_equal(unname(result$statistic),expected,tolerance=1e-8)
        # A smaller model has a different bias adjustment. Its solution must
        # not be substituted for the full-model nuisance equations.
        smaller <- fastglm_ordinal(x[,"z",drop=FALSE],counts,weights=weights,offset=offset,
            link=link,firth=TRUE,tol=1e-10)
        inserted <- c(coef(smaller)[1:2],0,coef(smaller)[3])
        reduced <- ordinal_native_evaluate(x,integer(),counts,weights,offset,3L,inserted,
            firth=TRUE,link=.ordinal_link_code(link))
        expect_gt(max(abs(reduced$score[-3])),1e-4)
        expect_output(print(result),"adjusted-score test")
    }
})

test_that("ordinal constraints respect grouping, offsets, and linear recoding", {
    set.seed(113503)
    x <- cbind(x=runif(12,-1,1),z=rnorm(12,sd=.4));offset <- cos(x[,1])/4
    counts <- matrix(rpois(36,3)+1,12,3);weights <- rep(c(0,1,2),4)
    rows <- rep(seq_len(12),rowSums(counts)*weights)
    y <- ordered(unlist(lapply(seq_len(12),function(i) rep(1:3,counts[i,]*weights[i]))),levels=1:3)
    A <- matrix(c(2,.3,-.4,.8),2);moved <- sweep(x%*%A,2,c(30,-20),"+")
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(x,counts,weights=weights,offset=offset,link=link,tol=1e-10)
        expanded <- fastglm_ordinal(x[rows,],y,offset=offset[rows],link=link,tol=1e-10)
        recoded <- fastglm_ordinal(moved,counts,weights=weights,offset=offset,link=link,tol=1e-10)
        a <- confint(fit,parm="x",method="profile");b <- confint(expanded,parm="x",method="profile")
        expect_equal(as.numeric(a),as.numeric(b),tolerance=1e-7)
        L <- matrix(c(1,-2),1,dimnames=list("x minus twice z"=NULL,c("x","z")))
        Lnew <- L%*%A;colnames(Lnew) <- names(recoded$beta)
        for(estimation in c("ML","meanBR")) {
            aa <- fastglm_score_test(fit,contrast=L,null=.2,null.estimation=estimation)
            bb <- fastglm_score_test(expanded,contrast=L,null=.2,null.estimation=estimation)
            cc <- fastglm_score_test(recoded,contrast=Lnew,null=.2,null.estimation=estimation)
            expect_true(aa$available);expect_true(bb$available);expect_true(cc$available)
            expect_equal(unname(aa$statistic),unname(bb$statistic),tolerance=1e-8)
            expect_equal(unname(aa$statistic),unname(cc$statistic),tolerance=1e-7)
        }
    }
})

test_that("joint threshold and slope score constraints preserve their hypotheses", {
    set.seed(113504)
    x <- cbind(x=rnorm(22,sd=.5),z=runif(22,-1,1));counts <- matrix(rpois(88,4)+1,22,4)
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(x,counts,link=link,tol=1e-10)
        L <- rbind(c(-1,1,0,0,0),c(0,0,0,1,-.4))
        colnames(L) <- names(coef(fit));null <- c(.7,.15)
        A <- matrix(c(2,-.3,.4,1.2),2)
        for(estimation in c("ML","meanBR")) {
            result <- fastglm_score_test(fit,contrast=L,null=null,null.estimation=estimation)
            changed <- fastglm_score_test(fit,contrast=A%*%L,null=drop(A%*%null),null.estimation=estimation)
            expect_true(result$available);expect_true(changed$available)
            expect_equal(drop(L%*%result$diagnostics$restricted$coefficients),null,tolerance=1e-11)
            expect_equal(unname(result$statistic),unname(changed$statistic),tolerance=1e-8)
        }
    }
})

test_that("ordinal profile and score failures remain explicit", {
    fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq)
    br <- update(fit,firth=TRUE)
    expect_error(confint(br,method="profile"),"maximum-likelihood")
    expect_error(.ordinal_profile_test(br,"InflHigh"),"maximum-likelihood")
    expect_equal(confint(fit),confint(fit,method="wald"))
    expect_error(confint(fit,method="profile",parm=c("InflHigh","InflHigh")),"distinct")
    expect_error(confint(fit,method="profile",control=list(unknown=1)),"control")
    expect_error(fastglm_score_test(fit,control=list(maxit=NULL)),"positive finite")
    expect_error(fastglm_score_test(fit,contrast=rbind(c(InflHigh=1),c(InflHigh=2))),"independent")
    expect_warning(failed <- fastglm_score_test(fit,terms="Type",control=list(maxit=1)),"iteration limit")
    expect_false(failed$available);expect_true(is.na(failed$p.value))
    expect_warning(failed <- .ordinal_profile_test(fit,1:2,c(1,-1)),"not strictly ordered")
    expect_false(failed$available);expect_true(is.na(failed$p.value))
    gap <- setNames(c(-1,1),names(coef(fit))[1:2])
    expect_warning(failed <- fastglm_score_test(fit,contrast=gap,null=-1),"no ordered starting thresholds")
    expect_false(failed$available);expect_true(is.na(failed$p.value))
    expect_warning(interval <- confint(fit,parm="InflHigh",method="profile",control=list(max.root=1)),"endpoints failed")
    expect_true(anyNA(interval));expect_true(any(!attr(interval,"diagnostics")$converged))
    bad <- fit;bad$converged <- FALSE
    expect_error(confint(bad,method="profile"),"converged")
    stream <- fit;class(stream) <- c("fastglm_ordinal_stream",class(fit))
    expect_error(fastglm_score_test(stream),"retained dense")
    empty <- suppressWarnings(fastglm_ordinal(matrix(numeric(),1,0),matrix(c(3,0,5),1),firth=TRUE))
    expect_error(fastglm_score_test(empty,contrast=c(1,0)),"every declared category")
})
