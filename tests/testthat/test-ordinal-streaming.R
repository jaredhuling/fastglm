ordinal_test_source <- function(x,y,weights=NULL,offset=NULL,size=17L,resettable=FALSE) {
    rows <- split(seq_len(nrow(x)),ceiling(seq_len(nrow(x))/size))
    position <- 0L
    fastglm_source(function(k) {
        if(resettable) {stopifnot(k==position+1L);position <<- k}
        i <- rows[[k]]
        list(X=x[i,,drop=FALSE],y=if(is.matrix(y)) y[i,,drop=FALSE] else y[i],
            weights=if(!is.null(weights)) weights[i],offset=if(!is.null(offset)) offset[i])
    },length(rows),reset=if(resettable) function() position <<- 0L)
}

test_that("ordinal source fits reproduce dense likelihood and adjusted equations", {
    set.seed(114001)
    x <- cbind(exposure=rnorm(100),modifier=runif(100,-1,1))
    y <- ordered(cut(.5*x[,1]+rlogis(100),c(-Inf,-.7,.4,Inf)),labels=c("low","middle","high"))
    weights <- rep(c(0,1,2,3),25);offset <- .15*sin(seq_len(100))
    for(link in c("logit","probit","cloglog")) for(firth in c(FALSE,TRUE)) {
        ref <- fastglm_ordinal(x,y,weights=weights,offset=offset,link=link,firth=firth,tol=1e-9)
        for(size in c(9L,23L)) {
            fit <- fastglm_ordinal_streaming(ordinal_test_source(x,y,weights,offset,size,TRUE),
                link=link,firth=firth,tol=1e-9,block.size=7L)
            expect_true(fit$converged)
            expect_equal(coef(fit),coef(ref),tolerance=2e-7)
            expect_equal(vcov(fit),vcov(ref),tolerance=2e-7)
            expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(ref)),tolerance=1e-10)
            expect_equal(deviance(fit),deviance(ref),tolerance=1e-10)
            expect_equal(fit$effective$score,ref$effective$score,tolerance=1e-7)
            expect_equal(nobs(fit),nobs(ref));expect_equal(fit$nobs.rows,ref$nobs.rows)
            expect_null(fit$x);expect_null(fit$y);expect_null(fit$category$totals)
            expect_identical(fit$state$maximum.block.rows,size)
            expect_gt(fit$state$passes,3L)
            if(!firth) expect_error(confint(fit,parm="exposure",method="profile"),"retained dense")
        }
    }
})

test_that("source predictions and residuals agree and support block callbacks", {
    d <- MASS::housing;x <- model.matrix(~Infl+Type+Cont,d)[,-1,drop=FALSE]
    counts <- cbind(low=d$Freq*(d$Sat=="Low"),middle=d$Freq*(d$Sat=="Medium"),high=d$Freq*(d$Sat=="High"))
    weights <- rep(c(0,1,2),length.out=nrow(x));offset <- cos(seq_len(nrow(x)))/5
    fit <- fastglm_ordinal_streaming(ordinal_test_source(x,counts,weights,offset,13L),firth=TRUE,tol=1e-10)
    ref <- fastglm_ordinal(x,counts,weights=weights,offset=offset,firth=TRUE,tol=1e-10)
    expect_equal(coef(fit),coef(ref),tolerance=1e-8)
    expect_equal(deviance(fit),deviance(ref),tolerance=1e-10)
    for(type in c("probability","cumulative","class","link")) {
        expect_equal(predict(fit,type=type),predict(ref,type=type),tolerance=1e-8)
        if(type!="class") {
            a <- predict(fit,type=type,se.fit=TRUE);b <- predict(ref,type=type,se.fit=TRUE)
            expect_equal(a$fit,b$fit,tolerance=1e-8);expect_equal(a$se.fit,b$se.fit,tolerance=1e-8)
        }
    }
    for(type in c("probability","cumulative","pearson")) expect_equal(residuals(fit,type),residuals(ref,type),tolerance=1e-8)
    expect_equal(weights(fit),weights)
    blocks <- list()
    result <- predict(fit,se.fit=TRUE,sink=function(value,k) blocks[[k]] <<- value)
    expect_identical(result,fit$source$n_chunks)
    expect_equal(do.call(rbind,lapply(blocks,`[[`,"fit")),predict(ref),tolerance=1e-8)
    blocks <- list();residuals(fit,sink=function(value,k) blocks[[k]] <<- value)
    expect_equal(do.call(c,blocks),residuals(ref),tolerance=1e-8)
    expect_output(print(summary(fit)),"Location coefficients")
    expect_output(print(fastglm_diagnostics(fit)),"converged")
    restored <- unserialize(serialize(fit,NULL))
    expect_equal(predict(restored),predict(fit))
    expect_error(model.matrix(fit),"do not retain")
    expect_error(simulate(fit),"not implemented")
    expect_error(predict(fit,sink=1),"function")
    # This weighting pattern removes one declared category globally; the
    # empty-category covariance restriction applies to both representations.
    expect_error(fastglm_vcov(fit),"every declared category")
    expect_error(fastglm_vcov(ref),"every declared category")
    expect_error(vcov(fit,type="HC0"),"every declared category")
    expect_error(fastglm_bootstrap(fit,R=2),"not yet available")
    expect_error(fastglm_average(fit),"source")
    expect_error(fastglm_ordinal_assess(fit),"source")
    expect_error(fastglm_po_test(fit),"source")
    expect_error(fastglm_marginal_effects(fit,"InflMedium"),"source")
    if(requireNamespace("broom",quietly=TRUE)) {
        expect_error(broom::augment(fit),"do not retain")
        expect_equal(broom::augment(fit,newdata=x[1:8,,drop=FALSE]),
            broom::augment(ref,newdata=x[1:8,,drop=FALSE]),tolerance=1e-8)
        expect_equal(broom::glance(fit),broom::glance(ref),tolerance=1e-8)
    }
})

test_that("ordinal formula sources preserve columns, offsets, and category order", {
    d <- MASS::housing;d$off <- sin(seq_len(nrow(d)))/10
    rows <- split(seq_len(nrow(d)),ceiling(seq_len(nrow(d))/11))
    source <- fastglm_source(function(k) d[rows[[k]],],length(rows),
        formula=Sat~Infl+Type+Cont+offset(off),prototype=d,weights="Freq")
    fit <- fastglm_ordinal_streaming(source,firth=TRUE,link="probit",tol=1e-10)
    ref <- fastglm_ordinal(Sat~Infl+Type+Cont+offset(off),d,weights=Freq,firth=TRUE,link="probit",tol=1e-10)
    expect_equal(coef(fit),coef(ref),tolerance=1e-8)
    expect_equal(predict(fit),unname(predict(ref)),tolerance=1e-8,ignore_attr=TRUE)
    expect_equal(predict(fit,d[1:8,]),predict(ref,d[1:8,]),tolerance=1e-8)
    expect_equal(fit$design.columns,ref$design.columns)
    expect_equal(fit$design.assign,ref$design.assign)
    expect_error(update(fit,.~.-Type),"source schema")
    updated <- update(fit,link="cloglog")
    expect_equal(coef(updated),coef(update(ref,link="cloglog")),tolerance=1e-8)
    expect_error(fastglm_marginal_effects(fit,"off"),"supply 'newdata'")
})

test_that("global ordinal category reduction permits locally empty and zero blocks", {
    x <- matrix(seq(-1,1,length.out=12),ncol=1,dimnames=list(NULL,"x"))
    counts <- cbind(low=rep(c(3,2,1),4),middle=0,high=rep(c(1,2,4),4))
    w <- c(rep(0,4),rep(1,8))
    expect_warning(fit <- fastglm_ordinal_streaming(ordinal_test_source(x,counts,w,size=4),firth=TRUE),"empty interior")
    expect_warning(ref <- fastglm_ordinal(x,counts,weights=w,firth=TRUE),"empty interior")
    expect_equal(coef(fit),coef(ref),tolerance=1e-7)
    expect_equal(predict(fit),predict(ref),tolerance=1e-7)
    expect_equal(vcov(fit),vcov(ref),tolerance=1e-7)
    expect_equal(residuals(fit),residuals(ref),tolerance=1e-7)
    none <- cbind(low=0,middle=counts[,1],high=counts[,3])
    expect_error(fastglm_ordinal_streaming(ordinal_test_source(x,none,size=3)),"empty endpoint")
    a <- fastglm_ordinal_streaming(ordinal_test_source(x,none,size=3),firth=TRUE)
    b <- fastglm_ordinal(x,none,firth=TRUE)
    expect_equal(coef(a),coef(b),tolerance=1e-7)
})

test_that("ordinal source rank and schema checks use all blocks", {
    x <- cbind(x=rep(c(-1,0,1),each=4),z=rep(c(-1,1),6));y <- ordered(rep(1:3,4),levels=1:3)
    a <- fastglm_ordinal_streaming(ordinal_test_source(x,y,size=2),firth=TRUE)
    b <- fastglm_ordinal(x,y,firth=TRUE)
    expect_equal(coef(a),coef(b),tolerance=1e-7)
    expect_error(fastglm_ordinal_streaming(ordinal_test_source(cbind(x,alias=x[,1]),y,size=2)),"full rank")
    expect_error(fastglm_ordinal_streaming(ordinal_test_source(cbind(intercept=1,x),y,size=2)),"constant")
    bad <- function(k) list(X=if(k==1) x[1:6,] else `colnames<-`(x[7:12,],c("z","x")),y=y[1:6])
    expect_error(fastglm_ordinal_streaming(bad,2),"columns changed")
    bad <- function(k) list(X=x[1:6,],y=if(k==1) y[1:6] else ordered(y[1:6],levels=3:1))
    expect_error(fastglm_ordinal_streaming(bad,2),"factor levels changed")
    expect_error(fastglm_ordinal_streaming(function(k) list(X=x,y=as.integer(y)),1),"supply 'levels'")
    expect_error(fastglm_ordinal_streaming(ordinal_test_source(x,y,weights=rep(.5,12)),firth=TRUE),"integer frequency")
    zero <- matrix(numeric(),12,0)
    fit <- fastglm_ordinal_streaming(ordinal_test_source(zero,y,size=2),firth=TRUE)
    ref <- fastglm_ordinal(zero,y,firth=TRUE)
    expect_equal(coef(fit),coef(ref),tolerance=1e-8)
})

test_that("binary source bias reduction retains accurate separated-tail equations", {
    set.seed(112101)
    x <- cbind(x=seq(-2,2,length.out=240),z=rnorm(240,sd=.25))
    y <- cbind(low=as.numeric(x[,1]>0),high=as.numeric(x[,1]<=0))
    w <- rep(c(0,1,2,1),60);off <- -cos(seq_len(240))/8
    for(link in c("probit","cloglog")) {
        ref <- fastglm_ordinal(-x,y,weights=w,offset=off,link=link,firth=TRUE,tol=1e-9,maxit=500)
        fit <- fastglm_ordinal_streaming(ordinal_test_source(-x,y,w,off,31L),
            link=link,firth=TRUE,tol=1e-9,maxit=500)
        expect_true(fit$converged)
        expect_equal(coef(fit),coef(ref),tolerance=1e-7)
        expect_equal(vcov(fit),vcov(ref),tolerance=1e-7)
        expect_lt(fit$scaled.score,1e-8);expect_lt(fit$predictor.step,1e-8)
    }
})

test_that("ordinal sources preserve recoding and explicit nonconvergence", {
    set.seed(114002)
    x <- cbind(x=rnorm(80),z=runif(80,-1,1));y <- ordered(rep(1:4,20),levels=1:4)
    x <- sweep(sweep(x,2,c(1e-4,1e3),"*"),2,c(1e5,-1e7),"+")
    ref <- fastglm_ordinal(x,y,firth=TRUE,tol=1e-9)
    fit <- fastglm_ordinal_streaming(ordinal_test_source(x,y,size=13),firth=TRUE,tol=1e-9)
    expect_true(fit$converged)
    expect_equal(predict(fit),predict(ref),tolerance=2e-7)
    expect_equal(predict(fit,se.fit=TRUE)$se.fit,predict(ref,se.fit=TRUE)$se.fit,tolerance=2e-7)
    expect_warning(failed <- fastglm_ordinal_streaming(ordinal_test_source(x,y,size=13),firth=TRUE,maxit=1),"did not converge")
    expect_false(failed$converged)
    expect_error(vcov(failed),"converged fit")
    calls <- 0L
    changing <- function(k) {
        calls <<- calls+1L;i <- seq_len(if(calls<=2L) 20L else 21L)
        list(X=x[i,,drop=FALSE],y=y[i])
    }
    expect_error(fastglm_ordinal_streaming(changing,2,firth=TRUE),"size changed")
})
