partial_native_fixture <- function() {
    x <- cbind(c(-.7,-.2,.3,.9,-.4,.5,.1,.8),c(.2,.8,-.3,.4,.6,-.5,.9,-.8))
    z <- cbind(c(0,1,0,1,0,1,0,1),c(-.4,.3,.2,-.1,.5,-.2,.1,-.3))
    counts <- rbind(c(3,2,1,2),c(1,4,3,1),c(2,1,2,3),c(1,2,3,4),
        c(4,2,3,1),c(2,4,1,3),c(3,1,4,2),c(1,3,2,4))
    list(x=x,z=z,counts=counts,weights=c(1,2,0,1,3,1,2,1),
        offset=c(.1,-.2,0,.1,-.1,.2,0,-.1),
        b=c(-1,.1,1.2,.2,-.15,.1,.12,-.05,.08,.15,-.1))
}

partial_native_probability <- function(b,x,z,offset,link) {
    q <- (length(b)-ncol(x))/(1+ncol(z));p <- ncol(x);pn <- ncol(z)
    location <- if(p) drop(x%*%b[q+seq_len(p)]) else numeric(nrow(x))
    eta <- outer(-location-offset,b[seq_len(q)],"+")
    if(pn) for(j in seq_len(q)) eta[,j] <- eta[,j]-drop(z%*%b[q+p+(j-1L)*pn+seq_len(pn)])
    G <- switch(link,logit=plogis,probit=pnorm,cloglog=function(eta)-expm1(-exp(eta)))
    F <- G(eta)
    cbind(F,1)-cbind(0,F)
}

test_that("nonparallel native score and information match independent derivatives", {
    skip_if_not_installed("numDeriv")
    d <- partial_native_fixture();m <- rowSums(d$counts)*d$weights
    for(link in c("logit","probit","cloglog")) {
        pr <- function(b) partial_native_probability(b,d$x,d$z,d$offset,link)
        P <- pr(d$b)
        ll <- function(b) sum(d$counts*d$weights*log(pr(b)))
        score <- numDeriv::grad(ll,d$b)
        information <- -numDeriv::hessian(function(b)sum(P*m*log(pr(b))),d$b)
        got <- fastglm:::ordinal_partial_native_evaluate(d$x,d$z,integer(),d$counts,
            d$weights,d$offset,4L,d$b,FALSE,3L,link=match(link,c("logit","probit","cloglog"))-1L)
        expect_equal(got$loglik,ll(d$b),tolerance=1e-12)
        expect_equal(drop(got$score),score,tolerance=2e-7)
        expect_equal(got$information,information,tolerance=2e-6)
        expect_equal(got$cov.unscaled,solve(information),tolerance=2e-6)
        expect_gt(got$minimum.gap,0)
        expect_true(all(got$adjustment==0))
    }
})

test_that("nonparallel mean adjustment matches cumulative-probability Hessians", {
    skip_if_not_installed("numDeriv")
    d <- partial_native_fixture();q <- ncol(d$counts)-1L
    for(link in c("logit","probit","cloglog")) {
        got <- fastglm:::ordinal_partial_native_evaluate(d$x,d$z,integer(),d$counts,
            d$weights,d$offset,4L,d$b,TRUE,3L,link=match(link,c("logit","probit","cloglog"))-1L)
        independent <- numeric(length(d$b))
        for(i in which(d$weights>0)) {
            pr <- function(b) drop(partial_native_probability(b,d$x[i,,drop=FALSE],
                d$z[i,,drop=FALSE],d$offset[i],link))
            J <- numDeriv::jacobian(pr,d$b)
            correction <- vapply(seq_len(q),function(j) {
                H <- numDeriv::hessian(function(b)sum(pr(b)[seq_len(j)]),d$b)
                .5*sum(d$counts[i,])*d$weights[i]*sum(got$cov.unscaled*t(H))
            },numeric(1))
            independent <- independent+drop(crossprod(J,diff(c(0,correction,0))/pr(d$b)))
        }
        expect_equal(drop(got$adjustment),independent,tolerance=2e-6)
        expect_equal(drop(got$score),drop(got$ordinary.score)+independent,tolerance=2e-6)
    }
})

test_that("native nonparallel kernel reduces to the optimized common-slope equations", {
    d <- partial_native_fixture();zero <- matrix(numeric(),nrow(d$x),0L)
    b <- d$b[1:5]
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        a <- fastglm:::ordinal_partial_native_evaluate(d$x,zero,integer(),d$counts,
            d$weights,d$offset,4L,b,firth,2L,link=link)
        ref <- fastglm:::ordinal_native_evaluate(d$x,integer(),d$counts,d$weights,
            d$offset,4L,b,firth,2L,link=link)
        for(name in c("loglik","score","ordinary.score","adjustment","information","cov.unscaled","step"))
            expect_equal(a[[name]],ref[[name]],tolerance=1e-11)
    }
})

test_that("row-specific ordering permits unordered baseline thresholds and ignores zero rows", {
    x <- matrix(numeric(),6L,0L);z <- matrix(c(2,3,4,2.5,3.5,-100),ncol=1)
    counts <- matrix(rep(c(2,3,4),6),6,3,byrow=TRUE);w <- c(rep(1,5),0)
    b <- c(1,0,.4,-.4)
    a <- fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,w,numeric(6),3L,b)
    expect_gt(a$minimum.gap,0)
    expect_equal(a$minimum.gap,.6,tolerance=1e-12)
    ref <- fastglm:::ordinal_partial_native_evaluate(x[1:5,,drop=FALSE],z[1:5,,drop=FALSE],
        integer(),counts[1:5,,drop=FALSE],rep(1,5),numeric(5),3L,b)
    expect_equal(a$information,ref$information,tolerance=1e-12)
    expect_equal(a$score,ref$score,tolerance=1e-12)
    expect_error(fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
        rep(1,6),numeric(6),3L,b),"not strictly ordered")
    expect_error(fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
        w,numeric(6),3L,b[-1]),"invalid nonparallel")
})

test_that("native fitting preserves frequencies and gives explicit nonconvergence", {
    x <- matrix(c(-1,-.5,.2,.8,1.1,-.3),ncol=1)
    z <- matrix(c(0,1,0,1,0,1),ncol=1)
    counts <- rbind(c(7,5,2),c(4,6,3),c(3,5,6),c(2,7,6),c(1,4,8),c(3,6,5))
    w <- c(1,2,0,1,3,1);off <- c(.1,-.1,0,.15,-.2,0)
    ii <- rep(seq_len(nrow(x)),rowSums(counts*w))
    y <- unlist(lapply(seq_len(nrow(x)),function(i)rep(1:3,counts[i,]*w[i])))
    start <- c(-.5,.8,0,0,0)
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        a <- fastglm:::ordinal_partial_native_fit(x,z,integer(),counts,w,off,3L,start,
            firth,score_tol=1e-9,step_tol=1e-9,maxit=500L,link=link)
        b <- fastglm:::ordinal_partial_native_fit(x[ii,,drop=FALSE],z[ii,,drop=FALSE],
            as.integer(y),matrix(numeric(),0L,0L),rep(1,length(ii)),off[ii],3L,start,
            firth,score_tol=1e-9,step_tol=1e-9,maxit=500L,link=link)
        expect_true(a$converged);expect_true(b$converged)
        expect_equal(a$coefficients,b$coefficients,tolerance=1e-8)
        expect_equal(a$cov.unscaled,b$cov.unscaled,tolerance=1e-8)
        expect_equal(a$loglik,b$loglik,tolerance=1e-10)
        expect_lt(a$scaled.score,1e-9);expect_lt(a$predictor.step,1e-9)
        expect_true(all(a$history[,4]>0))
        if(!firth) expect_true(all(diff(a$history[,1])>=-1e-10))
    }
    stopped <- fastglm:::ordinal_partial_native_fit(x,z,integer(),counts,w,off,3L,start,maxit=1L)
    expect_false(stopped$converged)
    expect_identical(stopped$convergence.reason,"iteration_limit")
})

test_that("binary nonparallel slopes reproduce ordinary ordinal coordinates", {
    x <- matrix(c(-1,-.2,.4,1,.7,-.6),ncol=1)
    z <- matrix(c(0,1,0,1,0,1),ncol=1)
    counts <- rbind(c(4,2),c(3,4),c(2,4),c(1,5),c(2,5),c(4,3))
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        a <- fastglm:::ordinal_partial_native_fit(x,z,integer(),counts,rep(1,6),numeric(6),
            2L,c(0,0,0),firth,score_tol=1e-10,step_tol=1e-10,maxit=500L,link=link)
        b <- fastglm:::ordinal_native_fit(cbind(x,z),integer(),counts,rep(1,6),numeric(6),
            2L,c(0,0,0),firth,score_tol=1e-10,step_tol=1e-10,maxit=500L,link=link)
        expect_true(a$converged);expect_true(b$converged)
        expect_equal(a$coefficients,b$coefficients,tolerance=1e-8)
        expect_equal(a$cov.unscaled,b$cov.unscaled,tolerance=1e-8)
    }
})

test_that("partial-model conditional category boundaries do not report convergence", {
    # All declared categories are observed globally, but the saturated first
    # predictor group has no middle-category outcomes and reaches a zero gap.
    x <- matrix(numeric(),2L,0L);z <- matrix(0:1,ncol=1)
    counts <- rbind(c(8,0,5),c(3,7,4))
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        fit <- fastglm:::ordinal_partial_native_fit(x,z,integer(),counts,c(1,1),numeric(2),
            3L,c(-1,1,0,0),firth,maxit=80L,link=link)
        expect_false(fit$converged)
        expect_true(fit$minimum.gap>0)
        expect_true(fit$scaled.score>=1e-8 || fit$predictor.step>=1e-8)
    }
})

test_that("nonparallel native tail evaluations retain finite log probabilities", {
    x <- matrix(numeric(),5L,0L);counts <- matrix(1,5L,3L)
    logdifference <- function(a,b) a+log(-expm1(b-a))
    for(link in 0:2) {
        z <- matrix(c(-40,-1,0,1,40),ncol=1)
        if(link==0L) z[c(1,5),1] <- c(-500,500)
        b <- c(-1,1,1,1)
        eta <- outer(-z[,1],b[1:2],"+")
        logs <- switch(link+1L,
            list(lower=plogis(eta,log.p=TRUE),upper=plogis(eta,lower.tail=FALSE,log.p=TRUE)),
            list(lower=pnorm(eta,log.p=TRUE),upper=pnorm(eta,lower.tail=FALSE,log.p=TRUE)),
            list(lower=log(-expm1(-exp(eta))),upper=-exp(eta)))
        middle <- ifelse(eta[,2]<=0,logdifference(logs$lower[,2],logs$lower[,1]),
            logdifference(logs$upper[,1],logs$upper[,2]))
        expected <- sum(cbind(logs$lower[,1],middle,logs$upper[,2]))
        value <- fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
            rep(1,5),numeric(5),3L,b,TRUE,link=link)
        expect_true(is.finite(value$loglik))
        expect_true(all(is.finite(value$score)))
        expect_true(all(is.finite(value$adjustment)))
        expect_lt(abs(value$loglik/expected-1),1e-12)
    }
})

test_that("native domain gaps cancel large common location shifts directly", {
    x <- matrix(numeric(),6L,0L)
    z <- matrix(c(-1,0,1,2,1,1e17),ncol=1)
    counts <- rbind(c(4,3,2),c(2,4,3),c(3,2,4),c(2,3,4),c(0,0,1),c(0,0,1))
    offset <- c(0,0,0,0,1e16,0);b <- c(-.5,.5,.1,.1)
    for(firth in c(FALSE,TRUE)) {
        value <- fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,rep(1,6),
            offset,3L,b,firth,link=0L)
        ref <- fastglm:::ordinal_partial_native_evaluate(x[1:4,,drop=FALSE],z[1:4,,drop=FALSE],
            integer(),counts[1:4,,drop=FALSE],rep(1,4),numeric(4),3L,b,firth,link=0L)
        expect_equal(value$minimum.gap,1,tolerance=1e-14)
        expect_equal(value$loglik,ref$loglik,tolerance=1e-12)
        expect_equal(value$score,ref$score,tolerance=1e-12)
        expect_equal(value$information,ref$information,tolerance=1e-12)
    }
    # These very large probit/cloglog evaluations can exceed the tail kernel's
    # numerical range, but their structural prediction domain is still ordered.
    for(link in 1:2) {
        result <- tryCatch(fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
            rep(1,6),offset,3L,b,link=link),error=conditionMessage)
        if(is.character(result)) {
            expect_match(result,"non-finite")
            expect_false(grepl("not strictly ordered",result,fixed=TRUE))
        } else expect_equal(result$minimum.gap,1,tolerance=1e-14)
    }
})

test_that("native predictors preserve thresholds when offsets and varying slopes cancel", {
    # The last row has a very large varying-slope contribution canceled by a
    # known offset. Information remains identifiable after diagonal scaling.
    x <- matrix(numeric(),5L,0L);z <- matrix(c(-1,0,1,2,1e17),ncol=1)
    counts <- matrix(1,5L,3L);offset <- c(0,0,0,0,-1.25e16)
    b <- c(-.5,.5,.125,.125)
    eta <- outer(-c(-1,0,1,2,0)*.125,b[1:2],"+")
    for(link in 0:2) {
        G <- switch(link+1L,plogis,pnorm,function(t)-expm1(-exp(t)))
        F <- G(eta);P <- cbind(F,1)-cbind(0,F)
        value <- fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,rep(1,5),
            offset,3L,b,FALSE,link=link)
        expect_equal(value$loglik,sum(log(P)),tolerance=1e-12)
        expect_equal(value$minimum.gap,1,tolerance=1e-14)
        expect_true(all(is.finite(value$score)))
    }
})

test_that("native domain gaps preserve original slope coefficient bits", {
    # Tiny positive frequency keeps the extreme row's information moderate.
    # Exact binary arithmetic gives .4 - .3 - .1 = 2^-55, so rounding the
    # coefficient difference before multiplying by 1e17 changes this gap.
    z <- rbind(c(-1,0),c(0,1),c(1,0),c(0,-1),c(1e17,-1e17))
    x <- matrix(numeric(),5L,0L);counts <- matrix(1,5L,3L)
    weights <- c(rep(1,4),1e-34);offset <- c(numeric(4),-1e16)
    b <- c(-2.25,1.875,.1,0,.4,.3)
    eta <- cbind(b[1]-z[,1]*.1,b[2]-z[,1]*.4-z[,2]*.3)
    # Constants independently evaluated as exact rational values of the
    # input binary64 numbers, then rounded once to binary64.
    eta[5,] <- c(-2.8051115123125783,-1.4556690738754696)
    for(link in 0:2) {
        logs <- switch(link+1L,
            list(lower=plogis(eta,log.p=TRUE),upper=plogis(eta,lower.tail=FALSE,log.p=TRUE)),
            list(lower=pnorm(eta,log.p=TRUE),upper=pnorm(eta,lower.tail=FALSE,log.p=TRUE)),
            list(lower=log(-expm1(-exp(eta))),upper=-exp(eta)))
        middle <- logs$upper[,1]+log(-expm1(logs$upper[,2]-logs$upper[,1]))
        logP <- cbind(logs$lower[,1],middle,logs$upper[,2])
        value <- fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
            weights,offset,3L,b,FALSE,link=link)
        expect_equal(value$minimum.gap,1.3494424384371086,tolerance=1e-14)
        expect_equal(value$loglik,sum(weights*rowSums(logP)),tolerance=1e-12)
        expect_true(all(is.finite(value$score)))
    }
})

test_that("unresolvable cloglog information is rejected despite valid row gaps", {
    z <- rbind(c(-1,0),c(0,1),c(1,0),c(0,-1),c(1e17,-1e17))
    x <- matrix(numeric(),5L,0L);counts <- matrix(1,5L,3L)
    # The high second threshold leaves virtually no information about the
    # sum of its two slopes. An independent category-gradient calculation
    # bounds the scaled reciprocal condition number below 5e-97.
    expect_error(fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
        c(rep(1,4),1e-34),c(numeric(4),-1e16),3L,
        c(-1.25,5.875,.1,0,.4,.3),FALSE,link=2L),
        "ordinal information is (not positive definite|numerically singular)")
})
