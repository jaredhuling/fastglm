test_that("ordinal score inference agrees with numerical likelihood derivatives", {
    skip_if_not_installed("numDeriv")
    set.seed(111201)
    x <- cbind(a=runif(100,-1,1),b=rnorm(100)); off <- .3*sin(x[,1])
    eta <- outer(-drop(x%*%c(.6,-.4))-off,c(-1,.1,1.3),"+")
    F <- plogis(eta); y <- ordered(1L+rowSums(runif(100)>F),levels=1:4)
    fit <- fastglm_ordinal(x,y,offset=off,firth=TRUE)
    ml <- fastglm_ordinal(x,y,offset=off,tol=1e-10,maxit=500)
    null <- c(ml$alpha,ml$beta,numeric(2)); q <- 3L; p <- 2L
    probability <- function(b) {
        deviation <- c(0,b[-seq_len(q+p)])
        eta <- outer(-drop(x%*%b[q+seq_len(p)])-off,b[seq_len(q)],"+")-outer(x[,2],deviation)
        F <- plogis(eta); cbind(F,1)-cbind(0,F)
    }
    P <- probability(null)
    U <- numDeriv::grad(function(step) sum(log(probability(null+step)[cbind(1:100,as.integer(y))])),numeric(length(null)))
    I <- -numDeriv::hessian(function(step) sum(P*log(probability(null+step))),numeric(length(null)),
        method.args=list(eps=.01,d=.01))
    a <- seq_len(q+p); b <- seq.int(q+p+1L,length(null))
    C <- I[b,a,drop=FALSE]%*%solve(I[a,a]); e <- U[b]-drop(C%*%U[a])
    J <- I[b,b]-C%*%I[a,b,drop=FALSE]
    expected <- drop(crossprod(e,solve(J,e)))
    test <- fastglm_ordinal_assess(fit,po.test="score",po.terms="b")$proportional.odds.test
    expect_true(test$available); expect_equal(test$df,2)
    expect_equal(test$statistic,expected,tolerance=1e-5)
    expect_identical(test$null.estimation,"ML")
    from.ml <- fastglm_ordinal_assess(ml,po.test="score",po.terms="b")$proportional.odds.test
    expect_equal(test$statistic,from.ml$statistic,tolerance=1e-12)
})

test_that("ordinal score tests preserve frequency grouping, offsets and outcome reversal", {
    set.seed(111202)
    x <- cbind(a=runif(25,-1,1),b=rnorm(25)); off <- .4*x[,1]
    F <- plogis(outer(-drop(x%*%c(.5,-.2))-off,c(-1,.2,1.1),"+"))
    P <- cbind(F,1)-cbind(0,F)
    counts <- t(vapply(seq_len(nrow(x)),function(i) drop(rmultinom(1,12,P[i,])),numeric(4)))
    colnames(counts) <- c("low","middle","high","top")
    fit <- fastglm_ordinal(x,counts/2,weights=rep(2,25),offset=off,firth=TRUE)
    grouped <- fastglm_ordinal_assess(fit,po.test="score")
    index <- rep(seq_len(nrow(x)),rowSums(counts))
    y <- unlist(lapply(seq_len(nrow(x)),function(i) rep(1:4,counts[i,])))
    expanded <- fastglm_ordinal_assess(fit,newdata=x[index,],y=y,offset=off[index],po.test="score")
    expect_true(grouped$proportional.odds.test$available)
    expect_equal(grouped$proportional.odds.test$frequency,300)
    expect_equal(grouped$proportional.odds.test$statistic,expanded$proportional.odds.test$statistic,tolerance=1e-8)
    one <- fastglm_ordinal(x[index,],ordered(y,levels=1:4),offset=off[index],firth=TRUE)
    individual <- fastglm_ordinal_assess(one,po.test="score")$proportional.odds.test
    doubled <- fastglm_ordinal_assess(one,y=ordered(y,levels=1:4),weights=rep(2,length(y)),po.test="score")$proportional.odds.test
    expect_equal(doubled$statistic,2*individual$statistic,tolerance=1e-8)
    reverse <- fastglm_ordinal(-x,counts[,4:1],weights=rep(1,25),offset=-off,firth=TRUE)
    reversed <- fastglm_ordinal_assess(reverse,po.test="score")$proportional.odds.test
    expect_equal(reversed$statistic,individual$statistic,tolerance=1e-8)
    fit$block.size <- 3L
    blocked <- fastglm_ordinal_assess(fit,po.test="score")$proportional.odds.test
    expect_equal(blocked$statistic,individual$statistic,tolerance=1e-8)
    scaled <- fastglm_ordinal(sweep(sweep(x,2,c(.001,100),"*"),2,c(40,-20),"+"),counts,offset=off)
    expect_equal(fastglm_ordinal_assess(scaled,po.test="score")$proportional.odds.test$statistic,
        individual$statistic,tolerance=1e-7)
})

test_that("formula terms select all their design columns for score tests", {
    set.seed(111203)
    d <- data.frame(x=runif(350,-1,1),g=factor(sample(letters[1:3],350,TRUE)),off=rnorm(350)/3)
    d$y <- ordered(1L+rowSums(runif(350)>plogis(outer(-.5*d$x-d$off,c(-1,0,1),"+"))),levels=1:4)
    fit <- fastglm_ordinal(y~x*g+offset(off),d,firth=TRUE,model=FALSE)
    a <- fastglm_ordinal_assess(fit,po.test="score",po.terms="g")$proportional.odds.test
    expect_true(a$available); expect_equal(a$df,4); expect_identical(a$columns,c("gb","gc"))
    b <- fastglm_ordinal_assess(fit,po.test="score",po.terms="x:g")$proportional.odds.test
    expect_true(b$available); expect_equal(b$df,4)
    prepared <- fastglm_ordinal_prepare(~x*g+offset(off),d)
    prepared <- unserialize(serialize(prepared,NULL))
    prep.fit <- fastglm_ordinal(prepared,d$y,firth=TRUE)
    expect_equal(fastglm_ordinal_assess(prep.fit,po.test="score",po.terms="g")$proportional.odds.test$statistic,
        a$statistic,tolerance=1e-10)
    matrix.fit <- fastglm_ordinal(fit$x,d$y,offset=d$off,firth=TRUE)
    matrix.test <- fastglm_ordinal_assess(matrix.fit,po.test="score",po.terms=c("gb","gc"))$proportional.odds.test
    expect_equal(matrix.test$statistic,a$statistic,tolerance=1e-10)
    supplied <- fastglm_ordinal_assess(fit,newdata=d,y=d$y,po.test="score",po.terms="g")$proportional.odds.test
    expect_equal(supplied$statistic,a$statistic,tolerance=1e-10)
    expect_error(fastglm_ordinal_assess(fit,po.test="score",po.terms="gb"),"unknown formula term")
    expect_error(fastglm_ordinal_assess(fit,po.terms="g"),"require po.test")
})

test_that("score tests report unsupported samples without changing predictions or exclusions", {
    set.seed(111204)
    x <- cbind(x=runif(140,-1,1)); y <- ordered(rep(1:3,length.out=140),levels=1:3)
    fit <- fastglm_ordinal(x,y,firth=TRUE)
    plain <- fastglm_ordinal_assess(fit)
    full <- fastglm_ordinal_assess(fit,po.test="score")
    expect_identical(full$scores,plain$scores); expect_identical(full$calibration,plain$calibration)
    expect_null(plain$proportional.odds.test); expect_null(full$proportional.odds)
    sparse <- fastglm_ordinal_assess(fit,y=ordered(ifelse(y==2,1,as.integer(y)),levels=1:3),po.test="score")
    expect_false(sparse$proportional.odds.test$available)
    expect_identical(sparse$proportional.odds.test$reason,"empty_categories")
    expect_identical(sparse$proportional.odds.test$empty.categories,"2")
    expect_true(is.na(sparse$proportional.odds.test$p.value))
    limited <- fastglm_ordinal_assess(fit,po.test="score",po.test.control=list(maxit=1))
    expect_false(limited$proportional.odds.test$available)
    expect_identical(limited$proportional.odds.test$reason,"null_fit_not_converged")
    yy <- as.integer(y); yy[1] <- NA; xx <- x; xx[2,1] <- NA
    omitted <- fastglm_ordinal_assess(fit,newdata=xx,y=yy,po.test="score")
    subset <- fastglm_ordinal_assess(fit,newdata=x[-(1:2),,drop=FALSE],y=y[-(1:2)],po.test="score")
    expect_equal(omitted$proportional.odds.test$statistic,subset$proportional.odds.test$statistic,tolerance=1e-10)
    expect_equal(omitted$proportional.odds.test$observations,138)
    deficient <- fastglm_ordinal_assess(fit,newdata=matrix(0,140,1,dimnames=list(NULL,"x")),y=y,po.test="score")
    expect_identical(deficient$proportional.odds.test$reason,"rank_deficient_design")
    expect_output(print(full),"Expected-information")
    expect_output(print(sparse),"Unavailable")
    expect_error(fastglm_ordinal_assess(fit,po.test="score",po.test.control=list(maxit=0)),"positive")
    expect_error(fastglm_ordinal_assess(fit,po.test="score",po.test.control=list(foo=1)),"permits only")
    binary <- fastglm_ordinal(x,ordered(rep(1:2,70)),firth=TRUE)
    expect_identical(fastglm_ordinal_assess(binary,po.test="score")$proportional.odds.test$reason,"no_slope_contrasts")
})
