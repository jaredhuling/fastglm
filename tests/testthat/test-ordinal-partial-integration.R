test_that("nonparallel averages and joint term tests use the full covariance", {
    skip_if_not_installed("numDeriv")
    d <- MASS::housing
    fit <- fastglm_ordinal(Sat~Infl+Type+Cont,d,weights=Freq,nonparallel=~Cont,tol=1e-10)
    expect_true(fit$converged)
    probabilities <- predict(fit)
    average <- fastglm_average(fit,weights=d$Freq)
    expect_equal(average$estimate,colSums(probabilities*d$Freq)/sum(d$Freq),ignore_attr=TRUE)
    supplied <- fastglm_average(fit,weights=d$Freq,vcov.=vcov(fit))
    expect_equal(supplied,average,tolerance=1e-10)
    q <- length(fit$alpha);pc <- length(fit$beta);pn <- length(fit$nonparallel)
    mean.probability <- function(b) {
        common <- if(pc) drop(fit$x[,fit$common,drop=FALSE]%*%b[q+seq_len(pc)]) else numeric(nrow(fit$x))
        gamma <- matrix(b[q+pc+seq_len(q*pn)],pn,q)
        eta <- matrix(b[seq_len(q)],nrow(fit$x),q,byrow=TRUE)-common-fit$offset-
            fit$x[,fit$nonparallel,drop=FALSE]%*%gamma
        F <- plogis(eta)
        P <- cbind(F,1)-cbind(0,F)
        colSums(P*d$Freq)/sum(d$Freq)
    }
    J <- numDeriv::jacobian(mean.probability,coef(fit))
    expect_equal(average$std.error,sqrt(diag(J%*%vcov(fit)%*%t(J))),tolerance=1e-7,ignore_attr=TRUE)
    test <- fastglm_wald_test(fit,terms="Cont")
    indices <- q+pc+seq_len(q*pn)
    expected <- drop(crossprod(coef(fit)[indices],solve(vcov(fit)[indices,indices],coef(fit)[indices])))
    expect_equal(test$statistic,expected,tolerance=1e-9)
    expect_equal(test$df,q*pn)
    expect_equal(fastglm_wald_test(fit,terms="Type")$df,3L)
    direction <- sign(diff(fit$gamma[1L,]))[1L]
    expect_error(fastglm_average(fit,contrast=function(x) {
        x[,fit$nonparallel[1L]] <- direction*1e8;x
    }),"not ordered")
})

test_that("nonparallel assessment uses the fitted joint category distribution", {
    d <- MASS::housing
    for(link in c("logit","probit","cloglog")) {
        fit <- fastglm_ordinal(Sat~Infl+Cont,d,weights=Freq,nonparallel=~Cont,link=link)
        result <- fastglm_ordinal_assess(fit,bins=5)
        P <- predict(fit);F <- predict(fit,type="cumulative")
        y <- as.integer(d$Sat);w <- d$Freq
        expected.log <- -sum(w*log(P[cbind(seq_along(y),y)]))/sum(w)
        expected.rps <- sum(w*rowSums((outer(y,seq_len(ncol(F)),"<=")-F)^2))/sum(w)
        expect_equal(result$scores$log_loss,expected.log,tolerance=1e-10)
        expect_equal(result$scores$rps,expected.rps,tolerance=1e-10)
        expect_error(fastglm_ordinal_assess(fit,po.test="score"),"nonparallel")
        expect_error(fastglm_po_test(fit),"nonparallel")
        expect_error(fastglm_score_test(fit),"nonparallel")
        expect_error(fastglm_profile_test(fit,1L),"nonparallel")
        expect_error(fastglm_vcov(fit),"nonparallel")
        expect_error(fastglm_bootstrap(fit,R=2),"nonparallel")
        expect_error(fastglm_marginal_effects(fit,"Infl"),"nonparallel")
    }
})
