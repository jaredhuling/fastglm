test_that("hypothesis tests have compact printing and numeric table extraction", {
    fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,firth=TRUE)
    fields <- c("alpha","beta","working","x","y","weights","offset")
    original <- serialize(fit[fields],NULL)
    direct <- fastglm_po_test(fit)
    nested <- fastglm_ordinal_assess(fit,po.test="score")$proportional.odds.test
    expect_s3_class(direct,"htest")
    expect_equal(direct$statistic,nested$statistic,tolerance=1e-12)
    table <- as.data.frame(direct)
    expect_named(table,c("statistic","df","p.value","null.estimation","available"))
    expect_identical(table$p.value,direct$p.value)
    expect_true(is.numeric(table$statistic)); expect_equal(nrow(table),1L)
    printed <- capture.output(value <- print(direct,digits=4))
    expect_identical(value,direct)
    expect_true(any(grepl("Statistic.*df.*p-value",printed)))
    expect_true(any(grepl("8.729",printed,fixed=TRUE)))
    expect_false(any(grepl("^\\$|^\\[1\\]",printed)))
    expect_identical(serialize(fit[fields],NULL),original)
    if(requireNamespace("broom",quietly=TRUE)) expect_identical(broom::tidy(direct),table)
    expect_error(print(direct,digits=0),"digits")
    direct$p.value <- 1e-30
    expect_output(print(direct),"<\\s*1e-04")
    profile.fit <- fastglm(cbind(1,x=seq(-1,1,length.out=40)),rep(c(0,1),20),family=binomial(),firth=TRUE)
    profile <- fastglm_profile_test(profile.fit,"x")
    expect_s3_class(profile,"htest")
    expect_identical(as.data.frame(profile)$p.value,profile$p.value)
    expect_output(print(profile),"Statistic.*df.*p-value")
    limits <- confint(profile.fit,"x",method="profile")
    saved <- serialize(limits,NULL)
    display <- capture.output(returned <- print(limits,digits=4))
    expect_identical(returned,limits)
    expect_identical(serialize(limits,NULL),saved)
    expect_true(is.matrix(limits))
    expect_true(all(is.finite(as.numeric(limits))))
    expect_true(all(attr(limits,"diagnostics")$converged))
    expect_true(any(grepl("Profile confidence intervals",display,fixed=TRUE)))
    expect_false(any(grepl("^attr\\(|^\\$|^\\[1\\]",display)))
    limits[1L,1L] <- NA_real_
    expect_output(print(limits),"Unavailable limits: 1")
    expect_error(print(limits,digits=0),"digits")
})

test_that("separation has a certificate and explicit bias-reduced inference", {
    x <- cbind(x=seq(-1,1,length.out=60),rare=rep(c(0,1),c(57,3)))
    y <- ordered(c(rep(1:3,19),rep(3,3)),levels=1:3)
    fit <- fastglm_ordinal(x,y,firth=TRUE)
    ml <- fastglm_po_test(fit)
    br <- fastglm_po_test(fit,null.estimation="meanBR")
    expect_false(ml$available); expect_identical(ml$reason,"null_separation")
    expect_equal(ml$null.iterations,0L)
    expect_identical(ml$separation$column,"rare")
    expect_output(print(ml),"no finite solution")
    expect_true(br$available); expect_identical(br$null.estimation,"meanBR")
    expect_true(is.finite(br$p.value))
    expect_identical(br$separation,ml$separation)
    expect_identical(as.data.frame(ml,details=TRUE)$reason,"null_separation")
    expect_equal(fastglm_ordinal_assess(fit,po.test="score",po.estimation="meanBR")$
        proportional.odds.test$statistic,br$statistic,tolerance=1e-12)
    reverse <- fastglm_ordinal(-x,ordered(4-as.integer(y),levels=1:3),firth=TRUE)
    expect_identical(fastglm_po_test(reverse)$reason,"null_separation")
    expect_equal(fastglm_po_test(reverse,null.estimation="meanBR")$statistic,br$statistic,tolerance=1e-8)
    xx <- x; xx[,2] <- 0
    expect_identical(fastglm_po_test(fit,newdata=xx,y=y,null.estimation="meanBR")$reason,"rank_deficient_design")
    expect_identical(fastglm_po_test(fit,y=ordered(ifelse(y==2,1,as.integer(y)),levels=1:3),
        null.estimation="meanBR")$reason,"empty_categories")
    # A mixed endpoint group does not provide this sufficient certificate.
    mixed <- y; mixed[60] <- 2
    expect_null(fastglm_po_test(fit,y=mixed)$separation)
})

test_that("the bias-reduced root retains the ordinary nuisance-score projection", {
    skip_if_not_installed("numDeriv")
    set.seed(111301)
    x <- cbind(x=runif(45,-1,1)); offset <- .3*x[,1]
    counts <- cbind(low=rep(2,45),middle=rep(3,45),high=rep(4,45))
    counts[1:10,] <- c(1,2,7)
    fit <- fastglm_ordinal(x,counts,offset=offset,firth=TRUE,tol=1e-10,maxit=500)
    null <- c(fit$alpha,fit$beta,0)
    probability <- function(b) {
        eta <- outer(-drop(x%*%b[3])-offset,b[1:2],"+")-outer(x[,1],c(0,b[4]))
        F <- plogis(eta); cbind(F,1)-cbind(0,F)
    }
    P <- probability(null)
    U <- numDeriv::grad(function(b) sum(counts*log(probability(null+b))),numeric(4))
    I <- -numDeriv::hessian(function(b) sum(rowSums(counts)*P*log(probability(null+b))),numeric(4),
        method.args=list(eps=.01,d=.01))
    C <- I[4,1:3,drop=FALSE]%*%solve(I[1:3,1:3])
    e <- U[4]-drop(C%*%U[1:3]); J <- I[4,4]-drop(C%*%I[1:3,4])
    result <- fastglm_po_test(fit,null.estimation="meanBR")
    expect_true(result$available)
    expect_equal(result$statistic,e^2/J,tolerance=1e-5)
    expect_gt(sum(U[1:3]^2),1e-8)
    rows <- rep(seq_len(nrow(x)),rowSums(counts))
    y <- unlist(lapply(seq_len(nrow(x)),function(i) rep(1:3,counts[i,])))
    expanded <- fastglm_po_test(fit,newdata=x[rows,,drop=FALSE],y=y,offset=offset[rows],null.estimation="meanBR")
    expect_equal(expanded$statistic,result$statistic,tolerance=1e-8)
})

test_that("model and bootstrap displays share tables without changing results", {
    set.seed(111302)
    d <- data.frame(x=runif(100,-1,1),y=rpois(100,2),positive=rgamma(100,2))
    models <- list(fastglm(y~x,d,family=poisson()),fastglm_nb(cbind(1,x=d$x),d$y),
        fastglm_hurdle(y~x,data=d),fastglm_zi(y~x,data=d),
        fastglm_truncated(y~x,data=d[d$y>0,]),fastglm_twopart(ifelse(y>0,positive,0)~x,data=d),
        fastglm_ordinal(ordered(1+y%%3)~x,d))
    for(model in models) {
        fields <- c("coefficients","cov.unscaled","fitted.values","weights")
        before <- serialize(model[fields],NULL)
        expect_output(print(model,digits=3),"Estimate")
        expect_output(print(summary(model),digits=3),"Estimate.*Std. Error")
        expect_identical(serialize(model[fields],NULL),before)
    }
    expect_s3_class(summary(models[[1]]),"summary.glm")
    boot <- fastglm_bootstrap(models[[1]],R=3,seed=111303)
    expect_output(print(boot,digits=3),"Estimates and confidence intervals")
})
