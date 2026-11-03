test_that("ordinal simulation preserves categorical frequencies and RNG state", {
    set.seed(92701)
    x <- matrix(rnorm(100),ncol=1,dimnames=list(NULL,"x"))
    y <- ordered(cut(.3*x[,1]+rlogis(100),c(-Inf,-.4,.6,Inf)),
                 labels=c("low","middle","high"))
    fit <- fastglm_ordinal(x,y,firth=TRUE)
    state <- .Random.seed
    draws <- simulate(fit,nsim=3,seed=19)
    expect_identical(.Random.seed,state)
    expect_equal(draws,simulate(fit,nsim=3,seed=19))
    expect_s3_class(draws,"data.frame")
    expect_equal(dim(draws),c(100L,3L))
    expect_true(all(vapply(draws,is.ordered,logical(1))))
    expect_identical(levels(draws[[1]]),fit$levels)
    expect_error(simulate(fit,nsim=1.5),"maxit|integer")
    expect_error(simulate(fit,seed=NA_real_),"seed")

    counts <- rbind(c(4,6,2),c(0,0,0),c(1,3,4))
    grouped <- fastglm_ordinal(matrix(numeric(),3,0),counts,weights=c(.5,2,1),firth=TRUE)
    draws <- simulate(grouped,nsim=2,seed=20)
    expect_type(draws,"list")
    expect_false(is.data.frame(draws))
    for(z in draws) {
        expect_equal(rowSums(z),c(6,0,8))
        expect_true(all(z>=0 & z==floor(z)))
        expect_identical(colnames(z),grouped$levels)
    }
    # Integer frequency weights represent repeated individuals, not one draw
    # replicated after simulation.
    weighted <- fastglm_ordinal(matrix(numeric(),3,0),ordered(1:3),weights=c(2,3,4),firth=TRUE)
    expect_equal(rowSums(simulate(weighted,seed=21)[[1]]),c(2,3,4))
    expect_warning(boundary <- fastglm_ordinal(matrix(numeric(),1,0),matrix(c(3,0,5),1),firth=TRUE),
                   "empty interior")
    expect_identical(unname(simulate(boundary,seed=22)[[1]][,2]),0L)
    huge <- fastglm_ordinal(matrix(numeric(),1,0),matrix(rep(1e9,3),1),firth=TRUE)
    expect_error(simulate(huge),"integer.max")
})

test_that("ordinal multinomial draws have the specified totals and moments", {
    probability <- matrix(c(.2,.3,.5,1,0,0,NA,NA,NA),3,3,byrow=TRUE)
    set.seed(92702)
    counts <- fastglm:::.ordinal_draw_counts(probability,c(100000,10,0))
    expect_equal(rowSums(counts),c(100000,10,0))
    expect_equal(unname(counts[2,]),c(10L,0L,0L))
    target <- 100000*probability[1,]
    sd <- sqrt(100000*probability[1,]*(1-probability[1,]))
    expect_true(all(abs(counts[1,]-target)<6*sd))
    expect_error(fastglm:::.ordinal_draw_counts(matrix(c(.2,.3),1),1),"summing to one")
    expect_error(fastglm:::.ordinal_draw_counts(matrix(c(.2,.8),1),1.5),"integer row")
})

test_that("probability-scale residuals agree with independent categorical formulas", {
    set.seed(92703)
    x <- matrix(rnorm(150),ncol=1,dimnames=list(NULL,"x"))
    y <- ordered(findInterval(.4*x[,1]+rlogis(150),c(-.5,.6))+1,levels=1:3)
    fit <- fastglm_ordinal(x,y,weights=rep(c(0,1,2),50),firth=TRUE)
    pr <- predict(fit); code <- as.integer(y)
    expected <- vapply(seq_along(y),function(i)
        sum(pr[i,seq_len(code[i]-1L)])-sum(pr[i,seq_len(3)>code[i]]),numeric(1))
    expected[fit$weights==0] <- NA_real_
    expect_equal(unname(residuals(fit)),expected,tolerance=1e-12)
    F <- cbind(pr[,1],pr[,1]+pr[,2])
    cumulative <- outer(code,1:2,"<=")-F
    cumulative[fit$weights==0,] <- NA_real_
    expect_equal(unname(residuals(fit,type="cumulative")),unname(cumulative),tolerance=1e-12)
    pearson <- sqrt(fit$weights)*cumulative/sqrt(F*(1-F))
    expect_equal(unname(residuals(fit,type="pearson")),unname(pearson),tolerance=1e-12)
    expect_true(all(abs(residuals(fit))<=1,na.rm=TRUE))

    binary <- fastglm_ordinal(x,ordered(code>1),firth=TRUE)
    expect_equal(unname(residuals(binary)),as.numeric(code>1)-predict(binary)[,2],
                 tolerance=1e-12,ignore_attr=TRUE)
    reversed <- fastglm_ordinal(x,ordered(4-code,levels=1:3),weights=fit$weights,firth=TRUE)
    expect_equal(unname(residuals(reversed)),-unname(residuals(fit)),tolerance=1e-8)
    counts <- rbind(c(4,2,1),c(2,3,4),c(0,0,0))
    grouped <- fastglm_ordinal(matrix(numeric(),3,0),counts,firth=TRUE)
    pr <- predict(grouped)
    scale <- cbind(0,pr[,1],pr[,1]+pr[,2])+cbind(pr[,1],pr[,1]+pr[,2],1)-1
    expected <- rowSums(scale*counts)/rowSums(counts); expected[3] <- NA_real_
    expect_equal(unname(residuals(grouped)),expected,tolerance=1e-12)
    # Direct upper tails preserve cumulative count discrepancies when a
    # category has probability close enough to one to lose relative precision.
    large <- fastglm_ordinal(matrix(numeric(),1,0),matrix(c(1e16,1,1),1),firth=TRUE)
    upper <- plogis(-large$alpha)
    target <- sqrt(large$frequency)*(upper-c(2,1)/large$frequency)/
        sqrt(plogis(large$alpha)*upper)
    expect_equal(as.numeric(residuals(large,type="pearson")),unname(target),tolerance=1e-10)
})

test_that("ordinal workflow methods restore excluded formula rows", {
    set.seed(92704)
    d <- data.frame(x=rnorm(100),y=ordered(rep(1:3,length.out=100)))
    d$x[2] <- NA; d$y[5] <- NA
    fit <- fastglm_ordinal(y~x,d,na.action=na.exclude,firth=TRUE)
    r <- residuals(fit)
    expect_length(r,100L)
    expect_true(all(is.na(r[c(2,5)])))
    expect_equal(dim(residuals(fit,type="cumulative")),c(100L,2L))
    simulation <- simulate(fit,nsim=2,seed=92704)
    expect_equal(dim(simulation),c(100L,2L))
    expect_true(all(is.na(simulation[c(2,5),])))
    expect_identical(levels(simulation[[1]]),fit$levels)
    if(requireNamespace("broom",quietly=TRUE)) {
        augmented <- broom::augment(fit,data=d,se_fit=TRUE)
        expect_equal(nrow(augmented),100L)
        expect_equal(unclass(augmented$.fitted),predict(fit))
        expect_equal(augmented$.resid,unname(residuals(fit)))
        expect_equal(unclass(augmented$.se.fit),predict(fit,se.fit=TRUE)$se.fit)
        expect_true(all(is.na(augmented$.class[c(2,5)])))
        expect_equal(nrow(broom::augment(fit)),100L)
        new <- broom::augment(fit,newdata=d[c(1,3),])
        expect_false(".resid" %in% names(new))
        expect_equal(unclass(new$.fitted),predict(fit,newdata=d[c(1,3),]))
        expect_error(broom::augment(fit,data=d,newdata=d),"either data or newdata")
    }
})

test_that("ordinal tidy output distinguishes odds ratios and thresholds", {
    skip_if_not_installed("broom")
    set.seed(92705)
    x <- matrix(rnorm(180),ncol=1,dimnames=list(NULL,"x"))
    y <- ordered(findInterval(.5*x[,1]+rlogis(180),c(-.4,.7))+1,levels=1:3)
    fit <- fastglm_ordinal(x,y,firth=TRUE)
    tidy <- broom::tidy(fit,conf.int=TRUE)
    odds <- broom::tidy(fit,conf.int=TRUE,exponentiate=TRUE)
    expect_equal(tidy$estimate,unname(coef(fit)))
    expect_identical(tidy$component,c("threshold","threshold","location"))
    expect_true(all(is.na(tidy$p.value[1:2])))
    expect_equal(odds$estimate[1:2],tidy$estimate[1:2])
    expect_equal(odds$estimate[3],exp(tidy$estimate[3]))
    expect_equal(odds$conf.low[3],exp(tidy$conf.low[3]))
    expect_equal(odds$std.error,tidy$std.error)
    expect_equal(nrow(broom::tidy(fit,component="location")),1L)
    supplied <- 2*vcov(fit)
    expect_equal(broom::tidy(fit,vcov.=supplied)$std.error,sqrt(2)*tidy$std.error)
    expect_equal(summary(fit,vcov.=supplied)$location[,2],sqrt(2)*summary(fit)$location[,2])
    expect_error(summary(fit,vcov.=diag(2)),"full ordinal covariance")
    glance <- broom::glance(fit)
    expect_equal(glance$nobs,nobs(fit))
    expect_equal(glance$logLik,as.numeric(logLik(fit)))
    expect_equal(glance$estimation,"meanBR")
    expect_true(glance$converged)
    expect_warning(incomplete <- fastglm_ordinal(x,y,maxit=1L),"did not converge")
    expect_true(all(is.na(broom::tidy(incomplete)$std.error)))
    expect_false(broom::glance(incomplete)$converged)
    expect_error(simulate(incomplete),"converged fit")
})
