test_that("cumulative-link maximum likelihood matches MASS and retains the link", {
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,link=link,tol=1e-10)
        ref <- MASS::polr(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,
            method=if(link=="logit") "logistic" else link,Hess=TRUE,control=list(reltol=1e-12))
        expect_true(fit$converged)
        expect_identical(fit$link,link)
        expect_equal(unname(fit$beta),unname(coef(ref)),tolerance=2e-5)
        expect_equal(unname(fit$alpha),unname(ref$zeta),tolerance=2e-5)
        expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(ref)),tolerance=1e-9)
        expect_equal(unname(predict(fit)),unname(predict(ref,type="probs")),tolerance=2e-6)
        assessment <- fastglm_ordinal_assess(fit,proportional.odds=TRUE)
        expect_equal(assessment$scores$log_loss,-as.numeric(logLik(fit))/sum(MASS::housing$Freq),tolerance=1e-12)
        expect_true(all(assessment$proportional.odds$converged))
        expect_true(fastglm_po_test(fit)$available)
        expect_identical(update(fit,firth=TRUE)$link,link)
    }
    expect_error(fastglm_ordinal(Sat~Infl,MASS::housing,link="invalid"),"arg")
})

test_that("binary cumulative-link bias reduction agrees with binary GLMs", {
    set.seed(111410)
    x <- cbind(x=rnorm(120),z=runif(120,-1,1)); offset <- .2*x[,1]
    counts <- cbind(low=rpois(120,3)+1,high=rpois(120,5)+1)
    for(link in c("probit","cloglog")) for(firth in c(FALSE,TRUE)) {
        fit <- fastglm_ordinal(x,counts,offset=offset,link=link,firth=firth,tol=1e-10)
        ref <- fastglm(cbind("(Intercept)"=1,-x),counts,binomial(link),offset=-offset,firth=firth,tol=1e-10)
        expect_true(fit$converged); expect_true(ref$converged)
        expect_equal(unname(coef(fit)),unname(coef(ref)),tolerance=1e-7)
        expect_equal(unname(predict(fit)[,1]),unname(predict(ref,type="response")),tolerance=1e-8)
        expect_equal(unname(vcov(fit)),unname(vcov(ref)),tolerance=1e-7,ignore_attr=TRUE)
    }
})

test_that("link-specific scores, information, and bias adjustments match derivatives", {
    skip_if_not_installed("numDeriv")
    set.seed(111411)
    x <- cbind(x=runif(14,-.5,.5),z=rnorm(14)/3); offset <- sin(x[,1])/5
    counts <- matrix(rpois(56,3)+1,14,4); weights <- rep(c(1,2),7)
    b <- c(-.8,.1,1,.3,-.2); q <- 3L; p <- 2L
    for(link in c("probit","cloglog")) {
        inverse <- if(link=="probit") pnorm else function(x) -expm1(-exp(x))
        cumulative <- function(b) inverse(outer(-drop(x%*%b[4:5])-offset,b[1:3],"+"))
        probability <- function(b) {F <- cumulative(b); cbind(F,1)-cbind(0,F)}
        P <- probability(b); m <- weights*rowSums(counts)
        U <- numDeriv::grad(function(b) sum(weights*counts*log(probability(b))),b)
        I <- -numDeriv::hessian(function(b) sum(m*P*log(probability(b))),b)
        ordinary <- ordinal_native_evaluate(x,integer(),counts,weights,offset,4,b,link=.ordinal_link_code(link))
        adjusted <- ordinal_native_evaluate(x,integer(),counts,weights,offset,4,b,firth=TRUE,link=.ordinal_link_code(link))
        expect_equal(drop(ordinary$score),U,tolerance=2e-7)
        expect_equal(ordinary$information,I,tolerance=2e-6)
        inv <- solve(I); A <- numeric(5)
        for(i in seq_len(nrow(x))) {
            corrections <- vapply(seq_len(q),function(j) {
                H <- numDeriv::hessian(function(b) cumulative(b)[i,j],b)
                .5*m[i]*sum(inv*H)
            },numeric(1))
            J <- numDeriv::jacobian(function(b) probability(b)[i,],b)
            A <- A+drop(crossprod(J,diff(c(0,corrections,0))/P[i,]))
        }
        expect_equal(drop(adjusted$adjustment),A,tolerance=3e-6)
        sandwich <- .ordinal_sandwich_block(x,b[1:3],b[4:5],offset,integer(),counts,weights,link=link)
        observed <- -numDeriv::hessian(function(b) sum(weights*counts*log(probability(b))),b)
        expect_equal(sandwich$information,observed,tolerance=3e-6)
        expect_equal(sandwich$score,U,tolerance=2e-7)
    }
})

test_that("alternative links preserve grouping, prediction derivatives, and refit settings", {
    set.seed(111412)
    d <- data.frame(x=runif(24,-.5,.5),z=rnorm(24)/4)
    counts <- matrix(rpois(96,3)+1,24,4,dimnames=list(NULL,c("low","medium","high","top")))
    x <- as.matrix(d); offset <- d$x/5
    rows <- rep(seq_len(24),rowSums(counts))
    y <- unlist(lapply(seq_len(24),function(i) rep(1:4,counts[i,])))
    for(link in c("probit","cloglog")) {
        fit <- fastglm_ordinal(x,counts,offset=offset,link=link,firth=TRUE,tol=1e-10)
        expanded <- fastglm_ordinal(x[rows,],ordered(y,levels=1:4),offset=offset[rows],link=link,firth=TRUE,tol=1e-10)
        expect_equal(unname(coef(fit)),unname(coef(expanded)),tolerance=1e-8)
        expect_equal(unname(vcov(fit)),unname(vcov(expanded)),tolerance=1e-8,ignore_attr=TRUE)
        expect_equal(unname(fastglm_vcov(fit,type="HC0")),unname(fastglm_vcov(expanded,type="HC0")),
            tolerance=1e-8,ignore_attr=TRUE)
        prepared <- fastglm_ordinal_prepare(x,offset=offset,link=link,firth=TRUE)
        prepared <- unserialize(serialize(prepared,NULL))
        batch <- fastglm_ordinal_batch(prepared,list(counts,counts),tol=1e-10)
        expect_identical(batch[[1]]$link,link)
        expect_equal(coef(batch[[2]]),coef(fit),tolerance=1e-9)
        expect_true(batch[[2]]$preparation.reused)
        refit <- .ordinal_bootstrap_refit(fit,.ordinal_bootstrap_input(fit))
        expect_identical(refit$link,link)
        expect_equal(coef(refit),coef(fit),tolerance=1e-9)
        h <- 1e-5; high <- low <- x; high[,1] <- high[,1]+h; low[,1] <- low[,1]-h
        numerical <- colMeans((predict(fit,high,offset=offset)-predict(fit,low,offset=offset))/(2*h))
        marginal <- fastglm_marginal_effects(fit,"x")
        expect_equal(marginal$estimate,unname(numerical),tolerance=1e-7)
        expect_true(all(is.finite(marginal$std.error)))
        score <- fastglm_po_test(fit,by.term=TRUE,null.estimation="meanBR")
        expect_true(all(as.data.frame(score)$available))
        for(term in c("x","z")) expect_equal(score$tests[[term]]$statistic,
            fastglm_po_test(fit,terms=term,null.estimation="meanBR")$statistic,tolerance=1e-9)
    }
})

test_that("alternative-link tails retain probabilities and finite log losses", {
    tiny <- .ordinal_link_kernel(5,0,"cloglog")$log.probability[1,1]
    expect_gt(-tiny,0)
    expect_equal(tiny/log1p(-exp(-exp(5))),1,tolerance=1e-14)
    for(link in c("probit","cloglog")) {
        kernel <- .ordinal_link_kernel(c(-1,0,1),c(-40,0,40),link)
        expect_equal(rowSums(kernel$probability),rep(1,3),tolerance=1e-14)
        expect_true(all(kernel$probability>=0))
        expect_true(all(is.finite(kernel$log.probability)))
        counts <- matrix(c(0,8,2),1,dimnames=list(NULL,c("low","middle","high")))
        fit <- fastglm_ordinal(matrix(numeric(),1,0),counts,link=link,firth=TRUE)
        expect_true(fit$converged); expect_true(all(is.finite(coef(fit))))
        expect_true(all(predict(fit)>0))
    }
})

test_that("alternative-link common-slope tests match an unrestricted numerical score", {
    skip_if_not_installed("numDeriv")
    set.seed(111413)
    x <- cbind(x=runif(35,-.6,.6),z=rnorm(35)/3); offset <- .2*x[,1]
    counts <- matrix(rpois(140,4)+1,35,4)
    for(link in c("probit","cloglog")) for(estimation in c("ML","meanBR")) {
        fit <- fastglm_ordinal(x,counts,offset=offset,link=link,
            firth=estimation=="meanBR",tol=1e-10,maxit=500)
        null <- c(coef(fit),0,0)
        inverse <- if(link=="probit") pnorm else function(z) -expm1(-exp(z))
        probability <- function(b) {
            eta <- outer(-drop(x%*%b[4:5])-offset,b[1:3],"+")-outer(x[,2],c(0,b[6:7]))
            F <- inverse(eta); cbind(F,1)-cbind(0,F)
        }
        P <- probability(null); expected <- P*rowSums(counts)
        U <- numDeriv::grad(function(step) sum(counts*log(probability(null+step))),numeric(7))
        I <- -numDeriv::hessian(function(step) sum(expected*log(probability(null+step))),numeric(7),
            method.args=list(eps=.01,d=.01))
        C <- I[6:7,1:5]%*%solve(I[1:5,1:5])
        efficient <- U[6:7]-drop(C%*%U[1:5])
        J <- I[6:7,6:7]-C%*%I[1:5,6:7]
        reference <- drop(crossprod(efficient,solve(J,efficient)))
        result <- fastglm_po_test(fit,terms="z",null.estimation=estimation)
        expect_true(result$available)
        expect_equal(result$statistic,reference,tolerance=1e-5)
    }
})

test_that("alternative-link marginal standard errors match numerical delta derivatives", {
    skip_if_not_installed("numDeriv")
    set.seed(111414)
    x <- cbind(x=runif(30,-.5,.5),z=rnorm(30)/3)
    counts <- matrix(rpois(120,3)+1,30,4)
    for(link in c("probit","cloglog")) {
        fit <- fastglm_ordinal(x,counts,link=link,firth=TRUE)
        average.derivative <- function(b) {
            eta <- outer(-drop(x%*%b[4:5]),b[1:3],"+")
            density <- if(link=="probit") dnorm(eta) else exp(eta-exp(eta))
            colMeans(-b[4]*(cbind(density,0)-cbind(0,density)))
        }
        J <- numDeriv::jacobian(average.derivative,coef(fit))
        reference <- sqrt(diag(J%*%vcov(fit)%*%t(J)))
        result <- fastglm_marginal_effects(fit,"x")
        expect_equal(result$estimate,unname(average.derivative(coef(fit))),tolerance=1e-9)
        expect_equal(result$std.error,unname(reference),tolerance=1e-7)
    }
})

test_that("an entirely zero design is rejected by the Firth solver", {
    x <- matrix(0,20,1)
    y <- rep(c(0,1),10)
    expect_error(fastglm(x,y,binomial("cloglog"),firth=TRUE,method="qr"),"cannot find valid starting values")
})

test_that("complementary-log-log tail scores avoid subtraction of large log probabilities", {
    location <- c(-40,0,1000)
    kernel <- .ordinal_link_kernel(c(-.1,.3),location,"cloglog",scores=TRUE)
    expect_equal(kernel$right[,2],exp(.3-location),tolerance=1e-14)
    expect_equal(kernel$right[3,1],1/expm1(.4),tolerance=1e-12)
    expect_equal(kernel$left[3,2],1/(-expm1(-.4)),tolerance=1e-12)
    expect_true(all(is.finite(kernel$translation)))
    counts <- rbind(c(0,1),c(1,1))
    evaluation <- ordinal_native_evaluate(matrix(numeric(),2,0),integer(),counts,
        c(1,1),c(-40,0),2,0,link=2L)
    expect_equal(drop(evaluation$score),-exp(40)+1/expm1(1)-1,tolerance=1e-14)
    sandwich <- .ordinal_sandwich_block(matrix(numeric(),2,0),0,numeric(),c(-40,0),
        integer(),counts,c(1,1),link="cloglog")
    expect_equal(drop(sandwich$information),exp(40),tolerance=1e-14)
    # A narrow middle category has large opposing threshold derivatives but
    # finite curvature in the common location direction.
    h <- 1e-7
    middle <- .ordinal_sandwich_block(matrix(1,1,1),c(0,h),0,0,
        integer(),matrix(c(0,1,0),1,3),1,link="cloglog")
    expect_equal(middle$information[3,3],1+h/2,tolerance=1e-10)
})
