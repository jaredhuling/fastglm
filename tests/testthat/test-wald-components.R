test_that("count term tests use complete component terms and joint covariance", {
    set.seed(111701)
    d <- data.frame(x=rnorm(360),g=factor(rep(letters[1:3],120)))
    d$y <- rpois(360,exp(.7+.3*d$x)) * rbinom(360,1,.65)
    for(fit in list(fastglm_hurdle(y~x*g|g,d,x=FALSE,model=FALSE),
        fastglm_zi(y~x*g|g,d,x=FALSE,model=FALSE,em.maxit=500,em.tol=1e-10))) {
        expect_true(fit$converged)
        b <- coef(fit); V <- vcov(fit)
        result <- fastglm_wald_test(fit,terms="g")
        index <- match(c("count_gb","count_gc","zero_gb","zero_gc"),names(b))
        expect_true(result$available)
        expect_equal(result$statistic,drop(crossprod(b[index],solve(V[index,index],b[index]))),tolerance=1e-9)
        expect_equal(result$df,4)
        expect_equal(fastglm_wald_test(fit,terms="x:g",component="count")$df,2)
        expect_equal(fastglm_wald_test(fit,terms="x")$df,1)
        expect_equal(fastglm_wald_test(fit,terms="g",component="zero")$df,2)
        expect_error(fastglm_wald_test(fit,terms="x",component="zero"),"unknown")
        L <- c(count_gb=1,zero_gb=-1)
        i <- match(names(L),names(b)); delta <- sum(L*b[i])-.1
        contrast <- fastglm_wald_test(fit,contrast=L,null=.1)
        expect_equal(contrast$statistic,delta^2/drop(L%*%V[i,i]%*%L),tolerance=1e-10)
        expect_equal(fastglm_wald_test(fit,contrast=L,null=.1,vcov.=V[rev(names(b)),rev(names(b))])$statistic,
            contrast$statistic,tolerance=1e-10)
        expect_error(fastglm_wald_test(fit,contrast=L,component="count"),"outside")
        expect_error(fastglm_wald_test(fit,component="binary"),"two-part")
        expect_output(print(result),"Component")
        if(inherits(fit,"fastglm_zi")) expect_output(print(result),"Structural-zero")
    }
})

test_that("two-part Wald contrasts retain cluster cross-component covariance", {
    set.seed(111702)
    d <- data.frame(x=rnorm(300),g=factor(rep(letters[1:3],100)))
    d$y <- rgamma(300,2,scale=exp(.3+.2*d$x)/2)*rbinom(300,1,.6)
    fit <- fastglm_twopart(y~x+g|x,d,tol=1e-11)
    fit$x <- fit$model <- NULL
    expect_true(fastglm_wald_test(fit,terms="g",component="positive")$available)
    expect_equal(fastglm_wald_test(fit,terms="x",component="binary")$statistic,
        fastglm_wald_test(fit,terms="x",component="zero")$statistic)
    # Robust covariance requires retained data; the compact fit above checks
    # that model-based term selection needs neither design nor model frame.
    fit <- fastglm_twopart(y~x+g|x,d,tol=1e-11)
    V <- fastglm_vcov(fit,cluster=rep(1:30,each=10),type="HC1")
    L <- c(count_x=1,zero_x=-1); i <- names(L)
    expect_gt(abs(V["count_x","zero_x"]),1e-8)
    test <- fastglm_wald_test(fit,contrast=L,vcov.=function(object) V)
    expect_true(test$available)
    expect_equal(test$statistic,sum(L*coef(fit)[i])^2/drop(L%*%V[i,i]%*%L),tolerance=1e-10)
    expect_equal(test$covariance,matrix(drop(L%*%V[i,i]%*%L),1,dimnames=list("Contrast 1","Contrast 1")))
    adjusted <- fastglm_twopart(y~x,d,firth=TRUE)
    expect_true(fastglm_wald_test(adjusted,terms="x")$available)
    expect_identical(fastglm_wald_test(adjusted,terms="x",vcov.=fastglm_vcov)$reason,"covariance_unavailable")
})

test_that("component terms survive sparse and streamed model preparation", {
    set.seed(111703)
    d <- data.frame(x=rnorm(240),g=factor(rep(letters[1:3],80)))
    d$y <- rpois(240,exp(.6+.2*d$x))*rbinom(240,1,.6)
    reads <- 0L
    source <- fastglm_source(function(k) {
        reads <<- reads+1L; d[((k-1)*60+1):(k*60),]
    },4,formula=y~x+g|g,prototype=d)
    dense <- fastglm_hurdle(y~x+g|g,d,tol=1e-11)
    stream <- fastglm_hurdle(source,tol=1e-11)
    old <- reads
    expect_equal(fastglm_wald_test(stream,terms="g")$statistic,
        fastglm_wald_test(dense,terms="g")$statistic,tolerance=1e-7)
    expect_identical(reads,old)
    no.intercept <- fastglm_twopart(y~0+g|0+g,d)
    expect_error(fastglm_wald_test(no.intercept,terms="(Intercept)"),"unknown")
    x <- model.matrix(~x+g,d); z <- model.matrix(~g,d)
    sparse <- fastglm_hurdle(as(x,"dgCMatrix"),response=d$y,z=as(z,"dgCMatrix"),tol=1e-11)
    expect_equal(fastglm_wald_test(sparse,terms=c("gb","gc"))$statistic,
        fastglm_wald_test(dense,terms="g")$statistic,tolerance=1e-7)
    d$y <- rgamma(240,2)*rbinom(240,1,.6)
    stream <- fastglm_twopart(source,tol=1e-11)
    dense <- fastglm_twopart(y~x+g|g,d,tol=1e-11)
    old <- reads
    expect_equal(fastglm_wald_test(stream,terms="g",component="positive")$statistic,
        fastglm_wald_test(dense,terms="g",component="positive")$statistic,tolerance=1e-7)
    expect_identical(reads,old)
})

test_that("truncated and NB tests use regression covariance and reject size boundaries", {
    set.seed(111704)
    d <- data.frame(x=rnorm(400));d$y <- rnbinom(400,mu=exp(.6+.3*d$x),size=.8)
    fits <- list(fastglm_truncated(y~x,data=subset(d,y>0)),
        fastglm_truncated(y~x,data=subset(d,y>0),dist="negbin",tol=1e-11,outer.tol=1e-8,outer.maxit=500),
        fastglm_nb(model.matrix(~x,d),d$y,outer.tol=1e-9))
    for(fit in fits) {
        out <- fastglm_wald_test(fit,terms="x")
        expect_true(out$available)
        expect_equal(out$statistic,coef(fit)[["x"]]^2/vcov(fit)["x","x"],tolerance=1e-10)
        expect_error(fastglm_wald_test(fit,component="count"),"two-component")
        if(.count_has_theta(fit)) {
            expect_error(fastglm_wald_test(fit,terms="x",vcov.=vcov(fit,full=TRUE)),"one row per coefficient")
            fit$theta.boundary <- TRUE
            expect_identical(fastglm_wald_test(fit,terms="x")$reason,"theta_boundary")
        }
        fit$converged <- FALSE
        expect_identical(fastglm_wald_test(fit,terms="x")$reason,"fit_not_converged")
    }
})

test_that("aliased two-part terms preserve component estimability", {
    set.seed(111705)
    x <- cbind("(Intercept)"=1,x=rnorm(240));x <- cbind(x,duplicate=x[,2])
    y <- rgamma(240,2)*rbinom(240,1,.6)
    fit <- fastglm_twopart(y~x+duplicate,data=data.frame(y,x=x[,2],duplicate=x[,3]),method="auto",tol=1e-11)
    reduced <- fastglm_twopart(x[,1:2],response=y,tol=1e-11)
    expect_true(anyNA(coef(fit)))
    expect_identical(fastglm_wald_test(fit,terms="x",component="positive")$reason,"nonestimable_contrast")
    out <- fastglm_wald_test(fit,contrast=c(count_x=1,count_duplicate=1))
    expect_true(out$available)
    expect_equal(out$statistic,fastglm_wald_test(reduced,terms="x",component="positive")$statistic,tolerance=1e-6)
})
