test_that("bounded noncanonical scoring recovers the grouped housing solution", {
    d <- MASS::housing
    x <- model.matrix(~Infl+Type+Cont,d)
    y <- cbind(d$Freq*(d$Sat=="Low"),d$Freq*(d$Sat!="Low"))
    for(link in c("probit","cloglog")) {
        reference <- fastglm(x,y,binomial(link),firth=TRUE,method="qr",start=numeric(ncol(x)),tol=1e-11)
        expect_true(reference$converged)
        for(method in c("qr","llt","ldlt")) {
            fit <- fastglm(x,y,binomial(link),firth=TRUE,method=method,tol=1e-10)
            expect_true(fit$converged,info=paste(link,method))
            expect_equal(coef(fit),coef(reference),tolerance=1e-8)
            expect_lt(max(abs(fit$standardized.adjusted.score)),1e-7)
            if(link=="cloglog") expect_gt(fit$step.reductions,0)
        }
        if(requireNamespace("brglm2",quietly=TRUE)) {
            ref <- brglm2::brglmFit(x,y,family=binomial(link),start=numeric(ncol(x)),
                control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10,maxit=500))
            expect_true(ref$converged)
            expect_equal(unname(coef(reference)),unname(coef(ref)),tolerance=1e-7)
        }
    }
    expect_output(print(fastglm_diagnostics(fit)),"Reduced scoring steps")
})

test_that("sparse, mapped, and streamed safeguards solve the same adjusted equations", {
    d <- MASS::housing;x <- model.matrix(~Infl+Type+Cont,d)
    y <- cbind(d$Freq*(d$Sat=="Low"),d$Freq*(d$Sat!="Low"))
    offset <- sin(seq_len(nrow(x)))/10
    w <- rep(c(0,1,2),length.out=nrow(x))
    designs <- list(sparse=as(x,"dgCMatrix"))
    if(requireNamespace("bigmemory",quietly=TRUE)) designs$mapped <- bigmemory::as.big.matrix(x)
    for(link in c("probit","cloglog")) {
        reference <- fastglm(x,y,binomial(link),firth=TRUE,weights=w,offset=offset,
            method="qr",start=numeric(ncol(x)),tol=1e-11)
        for(design in designs) for(method in c("llt","ldlt")) {
            fit <- fastglm(design,y,binomial(link),firth=TRUE,method=method,
                weights=w,offset=offset,tol=1e-10,maxit=200)
            expect_true(fit$converged)
            expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=1e-7)
        }
        for(size in c(9,24)) for(method in c("qr","llt","ldlt")) {
            position <- 0L
            source <- fastglm_source(function(k) {
                stopifnot(k==position+1L);position <<- k
                i <- ((k-1)*size+1):(k*size)
                list(X=x[i,],y=y[i,],weights=w[i],offset=offset[i])
            },nrow(x)/size,reset=function() position <<- 0L)
            fit <- fastglm_streaming(source,family=binomial(link),firth=TRUE,method=method,tol=1e-10,maxit=200)
            expect_true(fit$converged,info=paste(link,method,size))
            expect_equal(unname(coef(fit)),unname(coef(reference)),tolerance=1e-7)
            expect_lt(max(abs(fit$standardized.adjusted.score)),1e-7)
        }
    }
})

test_that("predictor step control respects grouping, recoding, and zero weights", {
    set.seed(111706)
    x <- cbind("(Intercept)"=1,x=seq(-2,2,length.out=48),z=rnorm(48,sd=.3))
    y <- cbind(rep(c(0,1,4,7),12),rep(c(8,7,4,1),12))
    w <- rep(c(0,1,2),16);off <- sin(x[,2])/5
    rows <- rep(seq_len(nrow(x)),w*rowSums(y))
    expanded <- unlist(lapply(seq_len(nrow(x)),function(i) rep(c(rep(1,y[i,1]),rep(0,y[i,2])),w[i])))
    transform <- matrix(c(1,0,0,0,2,.3,0,-.5,1),3)
    for(link in c("probit","cloglog")) {
        fit <- fastglm(x,y,binomial(link),weights=w,offset=off,firth=TRUE,tol=1e-11)
        recoded <- fastglm(x%*%transform,y,binomial(link),weights=w,offset=off,firth=TRUE,tol=1e-11)
        expect_true(fit$converged);expect_true(recoded$converged)
        expect_equal(drop(transform%*%coef(recoded)),unname(coef(fit)),tolerance=1e-8)
        duplicated <- fastglm(x[rows,],expanded,binomial(link),offset=off[rows],firth=TRUE,tol=1e-11)
        expect_equal(coef(duplicated),coef(fit),tolerance=1e-8)
        omit <- fastglm(x[w>0,],y[w>0,],binomial(link),weights=w[w>0],offset=off[w>0],firth=TRUE,tol=1e-11)
        expect_equal(coef(omit),coef(fit),tolerance=1e-8)
    }
})

test_that("a capped tiny coefficient update is not declared convergence", {
    x <- cbind(intercept=rep(1e10,20),x=seq(-1,1,length.out=20)*1e10)
    for(link in c("probit","cloglog")) {
        fit <- suppressWarnings(fastglm(x,rep(1,20),binomial(link),firth=TRUE,
            start=c(-2e-10,0),method="qr",tol=1e-8,maxit=1))
        expect_gt(fit$step.reductions,0)
        expect_false(fit$converged)
    }
})
