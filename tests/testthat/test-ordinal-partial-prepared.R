test_that("prepared partial fits and batches match direct fits for every link", {
    set.seed(154002);n <- 60L
    x <- cbind(exposure=runif(n,-1,1),modifier=runif(n,-1,1))
    counts <- matrix(sample(2:8,n*3,TRUE),n,3,dimnames=list(NULL,c("low","middle","high")))
    weights <- rep(c(0,1,2),20);offset <- sin(seq_len(n))/10
    for(link in c("logit","probit","cloglog")) for(firth in c(FALSE,TRUE)) {
        design <- fastglm_ordinal_prepare(x,weights=weights,offset=offset,
            nonparallel="modifier",firth=firth,link=link)
        fits <- fastglm_ordinal_batch(design,list(first=counts,second=counts[,3:1]))
        expect_false(fits[[1]]$preparation.reused)
        expect_true(fits[[2]]$preparation.reused)
        expect_identical(design$cache$builds,1L)
        for(j in 1:2) {
            response <- if(j==1L) counts else counts[,3:1]
            ref <- fastglm_ordinal(x,response,weights=weights,offset=offset,
                nonparallel="modifier",firth=firth,link=link)
            expect_equal(coef(fits[[j]]),coef(ref),tolerance=1e-11)
            expect_equal(vcov(fits[[j]]),vcov(ref),tolerance=1e-11)
            expect_equal(predict(fits[[j]],se.fit=TRUE),predict(ref,se.fit=TRUE),tolerance=1e-11)
        }
        restored <- unserialize(serialize(design,NULL))
        again <- fastglm_ordinal(restored,counts)
        expect_true(again$preparation.reused)
        expect_equal(coef(again),coef(fits[[1]]),tolerance=1e-11)
    }
})

test_that("partial prepared setup keys include selection, design, and row frequencies", {
    set.seed(154003);n <- 48L
    x <- cbind(a=runif(n,-1,1),b=runif(n,-1,1))
    counts <- matrix(sample(2:6,n*3,TRUE),n,3)
    design <- fastglm_ordinal_prepare(x,nonparallel="b",firth=TRUE)
    original <- fastglm_ordinal(design,counts)
    alternative <- counts[,3:1]
    expect_true(fastglm_ordinal(design,alternative)$preparation.reused)
    expect_false(fastglm_ordinal(design,counts,nonparallel="a")$preparation.reused)
    expect_false(fastglm_ordinal(design,counts,nonparallel=NULL)$preparation.reused)
    expect_true(fastglm_ordinal(design,counts,nonparallel=NULL)$preparation.reused)
    expect_false(fastglm_ordinal(design,counts)$preparation.reused)
    expect_identical(design$cache$builds,4L)
    design$weights[1] <- 2
    expect_false(fastglm_ordinal(design,counts)$preparation.reused)
    design$x[1,1] <- design$x[1,1]+.1
    changed <- fastglm_ordinal(design,counts)
    expect_false(changed$preparation.reused)
    expect_equal(coef(changed),coef(fastglm_ordinal(design$x,counts,weights=design$weights,
        nonparallel="b",firth=TRUE)),tolerance=1e-11)
    valid <- design$x;builds <- design$cache$builds
    design$x[,2] <- design$x[,1]
    expect_error(fastglm_ordinal(design,counts),"full rank")
    expect_identical(design$cache$builds,builds)
    design$x <- valid
    expect_true(fastglm_ordinal(design,counts)$preparation.reused)
    expect_equal(original$x,x)
})

test_that("prepared partial formulas retain complete term blocks and prediction coding", {
    set.seed(154004);n <- 90L
    data <- data.frame(exposure=runif(n,-1,1),g=factor(rep(letters[1:3],30)),
        off=sin(seq_len(n))/10,frequency=rep(c(1,2,1),30))
    data$counts <- matrix(sample(2:8,n*3,TRUE),n,3,
        dimnames=list(NULL,c("low","middle","high")))
    design <- fastglm_ordinal_prepare(~exposure+g+offset(off),data,
        weights=frequency,nonparallel=~g,firth=TRUE)
    expect_equal(design$nonparallel.terms,"g")
    expect_equal(design$nonparallel.columns,2:3)
    fit <- fastglm_ordinal(design,data$counts)
    ref <- fastglm_ordinal(counts~exposure+g+offset(off),data,weights=frequency,
        nonparallel=~g,firth=TRUE)
    expect_equal(coef(fit),coef(ref),tolerance=1e-11)
    expect_equal(fit$nonparallel.terms,"g")
    expect_equal(predict(fit,data[1:5,],se.fit=TRUE),predict(ref,data[1:5,],se.fit=TRUE),tolerance=1e-11)
    expect_error(fastglm_ordinal(design,data$counts,nonparallel="gb"),"one-sided")
    expect_error(fastglm_ordinal_prepare(~exposure+g,data,nonparallel=~abs(exposure)),"complete terms")
    common <- fastglm_ordinal(design,data$counts,nonparallel=NULL)
    expect_false(inherits(common,"fastglm_ordinal_partial"))
    expect_equal(coef(common),coef(fastglm_ordinal(counts~exposure+g+offset(off),
        data,weights=frequency,firth=TRUE)),tolerance=1e-11)
    replacement <- rep(0,n)
    changed <- fastglm_ordinal(design,data$counts,offset=replacement)
    expect_error(predict(changed,data[1:5,]),"explicit prediction offset")
})
