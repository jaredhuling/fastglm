partial_source_fixture <- function() {
    set.seed(154001)
    n <- 60L
    x <- cbind(exposure=runif(n,-1,1),modifier=runif(n,-1,1),group=rep(0:1,n/2))
    counts <- matrix(sample(2:8,n*4,TRUE),n,4,
        dimnames=list(NULL,c("low","moderate","high","very high")))
    list(x=x,counts=counts,weights=c(rep(0,5),rep(c(1,2,1,3,1),11)),
        offset=sin(seq_len(n))/10)
}

partial_test_source <- function(d,size,resettable=FALSE) {
    blocks <- split(seq_len(nrow(d$x)),ceiling(seq_len(nrow(d$x))/size))
    position <- 0L
    fastglm_source(function(k) {
        if(resettable) {stopifnot(k==position+1L);position <<- k}
        at <- blocks[[k]]
        list(X=d$x[at,,drop=FALSE],y=if(is.matrix(d$counts)) d$counts[at,,drop=FALSE] else d$counts[at],
            weights=d$weights[at],offset=d$offset[at])
    },length(blocks),reset=if(resettable) function() position <<- 0L)
}

test_that("partial source evaluations use globally combined information", {
    d <- partial_source_fixture();b <- c(-1,0,1,.15,.1,-.1,.05,.02,-.03,.08)
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        ref <- fastglm:::ordinal_partial_native_evaluate(d$x[,1,drop=FALSE],
            d$x[,2:3,drop=FALSE],integer(),d$counts,d$weights,d$offset,4L,b,firth,link=link)
        for(size in c(1L,13L)) {
            rows <- split(seq_len(nrow(d$x)),ceiling(seq_len(nrow(d$x))/size))
            fetch <- function(k) {
                at <- rows[[k]]
                list(X=d$x[at,1,drop=FALSE],Z=d$x[at,2:3,drop=FALSE],codes=integer(),
                    counts=d$counts[at,,drop=FALSE],weights=d$weights[at],offset=d$offset[at])
            }
            got <- fastglm:::ordinal_partial_stream_native_evaluate(fetch,length(rows),
                1L,2L,4L,b,firth,link=link)
            for(key in c("loglik","ordinary.score","adjustment","score","information",
                         "cov.unscaled","step","minimum.gap","predictor.step"))
                expect_equal(got[[key]],ref[[key]],tolerance=2e-12)
        }
    }
})

test_that("partial source fits match dense fits across links and block partitions", {
    d <- partial_source_fixture()
    for(link in c("logit","probit","cloglog")) for(firth in c(FALSE,TRUE)) {
        ref <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
            link=link,firth=firth,nonparallel=c("modifier","group"),tol=1e-9)
        for(size in c(5L,17L)) {
            fit <- fastglm_ordinal_streaming(partial_test_source(d,size,TRUE),
                link=link,firth=firth,nonparallel=c("modifier","group"),tol=1e-9,block.size=3L)
            expect_true(fit$converged)
            expect_s3_class(fit,"fastglm_ordinal_partial")
            expect_equal(coef(fit),coef(ref),tolerance=2e-7)
            expect_equal(vcov(fit),vcov(ref),tolerance=2e-7)
            expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(ref)),tolerance=1e-10)
            expect_equal(deviance(fit),deviance(ref),tolerance=1e-10)
            expect_equal(fit$effective$score,ref$effective$score,tolerance=1e-7)
            expect_equal(fit$effective$adjustment,ref$effective$adjustment,tolerance=1e-7)
            expect_equal(fit$minimum.gap,ref$minimum.gap,tolerance=1e-8)
            expect_null(fit$x);expect_null(fit$y);expect_null(fit$category$totals)
            expect_identical(fit$maximum.block.rows,size)
            expect_gt(fit$source.passes,3L)
        }
    }
})

test_that("partial source predictions and residuals retain parameter blocks", {
    d <- partial_source_fixture()
    fit <- fastglm_ordinal_streaming(partial_test_source(d,13L),nonparallel=2:3,firth=TRUE)
    ref <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,nonparallel=2:3,firth=TRUE)
    for(type in c("probability","cumulative","class","link")) {
        expect_equal(predict(fit,type=type),predict(ref,type=type),tolerance=1e-7)
        if(type!="class") expect_equal(predict(fit,type=type,se.fit=TRUE),
            predict(ref,type=type,se.fit=TRUE),tolerance=1e-7)
    }
    for(type in c("probability","cumulative","pearson"))
        expect_equal(residuals(fit,type),residuals(ref,type),tolerance=1e-7)
    expect_equal(weights(fit),d$weights)
    blocks <- list()
    count <- predict(fit,se.fit=TRUE,sink=function(value,k) blocks[[k]] <<- value)
    expect_identical(count,fit$source$n_chunks)
    expect_equal(do.call(rbind,lapply(blocks,`[[`,"fit")),predict(ref),tolerance=1e-7)
    expect_equal(predict(unserialize(serialize(fit,NULL))),predict(fit))
    expect_error(model.matrix(fit),"do not retain")
})

test_that("partial formula sources preserve complete term selection and offsets", {
    d <- partial_source_fixture()
    data <- data.frame(exposure=d$x[,1],g=factor(rep(letters[1:3],20)),off=d$offset)
    data$counts <- d$counts;data$frequency <- d$weights
    blocks <- split(seq_len(nrow(data)),ceiling(seq_len(nrow(data))/7))
    source <- fastglm_source(function(k) data[blocks[[k]],],length(blocks),
        formula=counts~exposure+g+offset(off),prototype=data,weights="frequency")
    fit <- fastglm_ordinal_streaming(source,nonparallel=~g,firth=TRUE,link="probit")
    ref <- fastglm_ordinal(counts~exposure+g+offset(off),data,weights=frequency,
        nonparallel=~g,firth=TRUE,link="probit")
    expect_equal(coef(fit),coef(ref),tolerance=1e-7)
    expect_equal(fit$nonparallel.terms,"g")
    expect_equal(rownames(fit$gamma),c("gb","gc"))
    expect_equal(predict(fit,data[1:4,],se.fit=TRUE),predict(ref,data[1:4,],se.fit=TRUE),tolerance=1e-7)
    expect_error(fastglm_ordinal_streaming(source,nonparallel="gb"),"one-sided")
    expect_error(fastglm_ordinal_streaming(source,nonparallel=~abs(exposure)),"complete terms")
})

test_that("partial sources keep global category and reread schema checks", {
    d <- partial_source_fixture();d$counts[,2] <- 0
    expect_error(fastglm_ordinal_streaming(partial_test_source(d,9L),nonparallel=2L),
        "every declared category")
    d <- partial_source_fixture();reads <- 0L
    changing <- function(k) {
        reads <<- reads+1L
        x <- d$x
        if(reads>1L) colnames(x)[2L] <- "changed"
        list(X=x,y=d$counts)
    }
    expect_error(fastglm_ordinal_streaming(changing,1L,nonparallel=2L),"columns changed")
    expect_warning(failed <- fastglm_ordinal_streaming(partial_test_source(d,11L),
        nonparallel=2L,firth=TRUE,maxit=1L),"did not converge")
    expect_false(failed$converged)
    expect_error(vcov(failed),"converged fit")
})

test_that("fully varying source fits retain coded and binary response paths", {
    set.seed(154005);n <- 96L
    # Repeated predictor rows retain every category at each design point, so
    # this workflow check exercises an interior fit. A sparse-row boundary
    # case is retained separately below.
    x <- cbind(a=runif(n/4,-1,1),b=runif(n/4,-1,1))
    d <- list(x=x[rep(seq_len(n/4),each=4),,drop=FALSE],
        weights=rep(c(0,1,2,1),n/4),offset=rep(cos(seq_len(n/4))/10,each=4))
    for(K in c(2L,4L)) {
        d$counts <- ordered(rep(seq_len(K),length.out=n),levels=seq_len(K))
        # Use a weight pattern independent of the category code so every
        # declared category remains globally represented.
        d$weights <- rep(c(0,1,2,1,1),length.out=n)
        for(link in c("logit","probit","cloglog")) {
            fit <- fastglm_ordinal_streaming(partial_test_source(d,7L,TRUE),
                nonparallel=1:2,link=link,firth=TRUE)
            ref <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
                nonparallel=1:2,link=link,firth=TRUE)
            expect_true(fit$converged)
            expect_length(fit$beta,0L)
            expect_equal(coef(fit),coef(ref),tolerance=1e-7)
            expect_equal(vcov(fit),vcov(ref),tolerance=1e-7)
            expect_equal(predict(fit,type="link",se.fit=TRUE),
                predict(ref,type="link",se.fit=TRUE),tolerance=1e-7)
        }
    }
})

test_that("coded partial sources preserve explicit near-boundary failure", {
    set.seed(154005);n <- 96L
    d <- list(x=cbind(a=runif(n,-1,1),b=runif(n,-1,1)),
        counts=ordered(rep(1:4,length.out=n),levels=1:4),
        weights=rep(c(0,1,2,1,1),length.out=n),offset=cos(seq_len(n))/10)
    expect_warning(fit <- fastglm_ordinal_streaming(partial_test_source(d,24L),
        nonparallel=1:2,firth=TRUE,maxit=30L),"did not converge")
    expect_warning(ref <- fastglm_ordinal(d$x,d$counts,weights=d$weights,offset=d$offset,
        nonparallel=1:2,firth=TRUE,maxit=30L),"did not converge")
    expect_false(fit$converged);expect_false(ref$converged)
    expect_lt(fit$minimum.gap,1e-3);expect_lt(ref$minimum.gap,1e-3)
    expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(ref)),tolerance=1e-8)
    expect_error(vcov(fit),"converged fit")
})
