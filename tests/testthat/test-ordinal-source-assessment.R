ordinal_assessment_source_fixture <- function(size=6L) {
    n <- 36L;i <- seq_len(n)
    x <- cbind(x=seq(-1,1,length.out=n),z=sin(i/3))
    counts <- cbind(low=4+i%%3,middle=5+i%%4,high=3+i%%2)
    weights <- rep(c(1,2,1),length.out=n);weights[1:6] <- 0
    offset <- cos(i)/10
    rows <- split(i,ceiling(i/size))
    source <- fastglm_source(function(k) {
        at <- rows[[k]]
        list(X=x[at,,drop=FALSE],y=counts[at,,drop=FALSE],weights=weights[at],offset=offset[at])
    },length(rows))
    list(x=x,counts=counts,weights=weights,offset=offset,rows=rows,source=source)
}

test_that("source assessment preserves dense losses, calibration, and row frequencies", {
    a <- ordinal_assessment_source_fixture()
    for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) for(partial in c(FALSE,TRUE)) {
        varying <- if(partial) "z" else NULL
        dense <- fastglm_ordinal(a$x,a$counts,weights=a$weights,offset=a$offset,
            nonparallel=varying,link=link,firth=br,tol=1e-10)
        source <- fastglm_ordinal_streaming(a$source,nonparallel=varying,link=link,firth=br,tol=1e-10)
        expect_true(source$converged)
        reference <- fastglm_ordinal_assess(dense,bins=5)
        before <- source$state$passes
        compact <- fastglm_ordinal_assess(source,bins=5)
        expect_identical(source$state$passes,before+1L)
        expect_null(compact$rows)
        expect_equal(compact$scores,reference$scores,tolerance=1e-9)
        expect_equal(compact$calibration,reference$calibration,tolerance=1e-9)
        expect_equal(compact$exclusions,reference$exclusions)
        expect_identical(compact$evaluation,"training source")
        expect_equal(compact$source$rows,nrow(a$x))
        delivered <- list();indices <- integer()
        retained <- fastglm_ordinal_assess(source,bins=5,keep.rows=TRUE,sink=function(value,k) {
            delivered[[k]] <<- value;indices <<- c(indices,k)
        })
        expect_equal(retained$rows,reference$rows,tolerance=1e-9)
        expect_equal(do.call(rbind,delivered),retained$rows,tolerance=1e-12,ignore_attr=TRUE)
        expect_identical(indices,seq_along(a$rows))
        expect_equal(lapply(delivered,`[[`,"row"),lapply(a$rows,as.numeric),ignore_attr=TRUE)
        expect_output(print(compact),"Source blocks")
        expect_null(source$x);expect_null(source$y)
        expect_null(fastglm_ordinal_assess(dense,keep.rows=FALSE)$rows)
    }
})

test_that("source assessment agrees with independent individual-category scores", {
    a <- ordinal_assessment_source_fixture(5L)
    fit <- fastglm_ordinal_streaming(a$source,nonparallel="z",firth=TRUE,tol=1e-10)
    result <- fastglm_ordinal_assess(fit,bins=4)
    probability <- predict(fit);cumulative <- predict(fit,type="cumulative")
    effective <- a$counts*a$weights;N <- sum(effective)
    logloss <- -sum((effective/N)*log(probability))
    loss <- 0
    for(k in seq_len(ncol(probability))) for(j in seq_len(ncol(cumulative)))
        loss <- loss+sum((effective[,k]/N)*(cumulative[,j]-as.numeric(k<=j))^2)
    expect_equal(result$scores$log_loss,logloss,tolerance=1e-12)
    expect_equal(result$scores$rps,loss,tolerance=1e-12)
    expect_equal(result$scores$rps_normalized,loss/(ncol(probability)-1L),tolerance=1e-12)
    expect_equal(tapply(result$calibration$frequency,result$calibration$threshold,sum),
        rep(N,ncol(cumulative)),ignore_attr=TRUE)
    expect_true(all(is.na(result$calibration$predicted[result$calibration$frequency==0])))
    other <- ordinal_assessment_source_fixture(11L)
    split <- fastglm_ordinal_streaming(other$source,nonparallel="z",firth=TRUE,tol=1e-10)
    expect_equal(fastglm_ordinal_assess(split,bins=4)$scores,result$scores,tolerance=1e-10)
    expect_equal(fastglm_ordinal_assess(split,bins=4)$calibration,result$calibration,tolerance=1e-10)
})

test_that("formula source assessment and common-slope tests use recorded data", {
    a <- ordinal_assessment_source_fixture()
    data <- data.frame(a$x,w=a$weights,off=a$offset,g=factor(rep(letters[1:3],12)))
    data$counts <- a$counts
    source <- fastglm_source(function(k) data[a$rows[[k]],],length(a$rows),
        formula=counts~x+g+offset(off),weights="w",prototype=data)
    fit <- fastglm_ordinal_streaming(source,nonparallel=~g,firth=TRUE,tol=1e-10)
    dense <- fastglm_ordinal(counts~x+g+offset(off),data,weights=w,nonparallel=~g,firth=TRUE,tol=1e-10)
    result <- fastglm_ordinal_assess(fit,po.test="score",po.terms="g")
    reference <- fastglm_ordinal_assess(dense,po.test="score",po.terms="g")
    expect_equal(result$scores,reference$scores,tolerance=1e-9)
    expect_equal(result$proportional.odds.test$statistic,reference$proportional.odds.test$statistic,tolerance=1e-8)
    supplied <- fastglm_ordinal_assess(fit,newdata=data,y=data$counts,weights=data$w)
    expect_equal(supplied$scores,reference$scores,tolerance=1e-9)
    expect_equal(nrow(supplied$rows),nrow(data))
    expect_error(fastglm_ordinal_assess(fit,weights=a$weights),"recorded block weights and offsets")
    expect_error(fastglm_ordinal_assess(fit,offset=0),"recorded block weights and offsets")
    expect_error(fastglm_ordinal_assess(fit,keep.rows=NA),"keep.rows")
    expect_error(fastglm_ordinal_assess(fit,sink=1),"function")
    expect_error(fastglm_ordinal_assess(dense,sink=function(...) NULL),"original fitted source")
    data$x[7L] <- NA_real_
    expect_error(fastglm_ordinal_assess(fit),"missing|finite|NA")
})

test_that("source summary counters support more than two billion input rows", {
    # Exercise only summary accumulation. A zero-column matrix represents a
    # large block row count without allocating observations; the dense block
    # assessment is mocked, rather than attempting a billion-row fit in a test.
    block <- list(X=matrix(numeric(),1100000000L,0L))
    assessment <- list(scores=data.frame(frequency=0,log_loss=NA_real_,rps=NA_real_,rps_normalized=NA_real_),
        calibration=data.frame(frequency=0,predicted=NA_real_,observed=NA_real_),
        exclusions=data.frame(reason="zero_frequency",rows=1100000000L),
        rows=data.frame(row=numeric()))
    testthat::local_mocked_bindings(fastglm_ordinal_assess=function(...) assessment,.package="fastglm")
    object <- structure(list(fetch=function(k) block,source=list(n_chunks=3L),
        levels=c("low","high"),link="logit",category=list(grouped=FALSE)),
        class=c("fastglm_ordinal_stream","fastglm_ordinal"))
    result <- fastglm:::.ordinal_stream_assessment(object,2L,"none",NULL,list(),"ML",FALSE,NULL,NULL)
    expect_equal(result$exclusions$rows,3300000000)
    expect_equal(result$source$rows,3300000000)
    expect_true(is.finite(result$exclusions$rows))
    expect_null(result$rows)
})

test_that("source assessment errors identify the original block", {
    a <- ordinal_assessment_source_fixture()
    fit <- fastglm_ordinal_streaming(a$source,nonparallel="z",firth=TRUE)
    fit$working$gamma[1L,2L] <- fit$working$gamma[1L,1L]+100
    expect_error(fastglm_ordinal_assess(fit),"assessment block 2:.*row")
})

test_that("coded source assessment preserves category scale and extreme tails", {
    x <- matrix(numeric(),3L,0L)
    y <- ordered(c("low","middle","high"),levels=c("low","middle","high"))
    source <- fastglm_source(function(k) list(X=x[k,,drop=FALSE],y=y[k],weights=c(2,0,3)[k]),3L)
    fit <- suppressWarnings(fastglm_ordinal_streaming(source,firth=TRUE))
    dense <- suppressWarnings(fastglm_ordinal(x,y,weights=c(2,0,3),firth=TRUE))
    result <- fastglm_ordinal_assess(fit,keep.rows=TRUE)
    expect_equal(result$scores,fastglm_ordinal_assess(dense)$scores,tolerance=1e-12)
    expect_identical(result$levels,levels(y))
    expect_equal(result$rows$exclusion,c("","zero_frequency",""))
    expect_error(fastglm_ordinal_assess(fit,proportional.odds=TRUE),"explicit evaluation data")
    full <- fastglm_ordinal_streaming(fastglm_source(function(k)
        list(X=x[k,,drop=FALSE],y=y[k]),3L),firth=TRUE)
    full$working$alpha <- full$working$alpha+1000
    extreme <- fastglm_ordinal_assess(full,keep.rows=TRUE)
    expect_true(is.finite(extreme$scores$log_loss))
    expect_gt(extreme$rows$log_loss[3L],900)
    # A structural zero at positive frequency must remain an infinite loss.
    full$working$alpha[1L] <- full$working$alpha[2L]
    expect_identical(fastglm_ordinal_assess(full)$scores$log_loss,Inf)
})
