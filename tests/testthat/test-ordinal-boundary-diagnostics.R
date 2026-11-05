ordinal_boundary_diagnostic_fixture <- function() {
    x <- cbind(a=seq(-.8,.9,length.out=12),b=sin(seq_len(12)))
    rownames(x) <- paste0("case",seq_len(nrow(x)))
    y <- cbind(low=rep(c(5,3,4),4),middle=rep(c(2,4,3),4),
        high=rep(c(4,3,6),4),top=rep(c(3,5,4),4))
    list(x=x,y=y,weights=rep(c(1,2),6),offset=cos(seq_len(12))/8)
}

test_that("explicit ordinal row diagnostics agree with fitted probabilities and stored scores", {
    d <- ordinal_boundary_diagnostic_fixture()
    for(link in c("logit","probit","cloglog")) for(partial in c(FALSE,TRUE)) {
        fit <- fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,
            link=link,nonparallel=if(partial) 1L else NULL)
        result <- .ordinal_boundary_diagnostics(fit,4L)
        expect_true(result$available)
        p <- predict(fit)
        expect_equal(result$minimum.probability,min(p),tolerance=1e-12)
        expect_equal(result$minimum.log.probability,log(min(p)),tolerance=1e-12)
        expect_equal(result$raw.score.maximum,max(abs(fit$effective$score)))
        expect_identical(result$standardized.score,fit$scaled.score)
        expect_identical(result$information.rcond,fit$information.rcond)
        expect_equal(result$rows.examined,nrow(d$x))
        expect_equal(result$frequency.examined,sum(d$y*d$weights))
        expect_equal(result$source.reads,0L)
        tab <- result$smallest.probabilities
        expect_lte(nrow(tab),4L)
        expect_lte(nrow(result$smallest.gaps),4L)
        expect_equal(tab$probability,p[cbind(tab$row,tab$category.index)],tolerance=1e-12)
        expect_equal(tab$observed.frequency,d$y[cbind(tab$row,tab$category.index)]*d$weights[tab$row])
        expect_equal(tab$row.name,rownames(d$x)[tab$row])
        expect_equal(tab$predictors[[1L]],d$x[tab$row[1L],])
        expect_equal(tab$offset,d$offset[tab$row])
        expect_true(all(is.na(tab$block)))
        expect_match(result$raw.score.coordinates,"reported predictor units")
        expect_match(result$standardized.score.coordinates,"centered/scaled")
    }
})

test_that("source row diagnostics make one explicit pass and retain bounded rows", {
    d <- ordinal_boundary_diagnostic_fixture();reads <- 0L
    source <- fastglm_source(function(k) {
        reads <<- reads+1L;i <- ((k-1L)*3L+1L):(k*3L)
        list(X=d$x[i,,drop=FALSE],y=d$y[i,,drop=FALSE],
            weights=d$weights[i],offset=d$offset[i])
    },4L)
    fit <- fastglm_ordinal_streaming(source,nonparallel="a",firth=TRUE)
    before <- reads
    ordinary <- fastglm_diagnostics(fit)
    expect_equal(reads,before)
    result <- .ordinal_boundary_diagnostics(fit,3L)
    expect_true(result$available)
    expect_equal(reads-before,4L)
    expect_equal(result$source.reads,4L)
    expect_equal(result$source.passes,1L)
    expect_equal(result$rows.examined,12L)
    tab <- result$smallest.probabilities
    expect_lte(nrow(tab),3L)
    expect_lte(nrow(result$smallest.gaps),3L)
    expect_equal(tab$block,ceiling(tab$row/3))
    expect_equal(tab$row.in.block,(tab$row-1)%%3+1)
    expect_equal(tab$predictors[[1L]],d$x[tab$row[1L],])
    dense <- fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,nonparallel=1L,firth=TRUE)
    dense.result <- .ordinal_boundary_diagnostics(dense,3L)
    expect_equal(result$minimum.probability,dense.result$minimum.probability,tolerance=1e-10)
    expect_equal(tab$row,dense.result$smallest.probabilities$row)
    expect_equal(tab$category,dense.result$smallest.probabilities$category)
    expect_identical(ordinary$score.type,"mean-bias-reduced")
})

test_that("boundary rows exclude zero effective frequency and respect category reduction", {
    d <- ordinal_boundary_diagnostic_fixture()
    d$x <- rbind(d$x,c(1e100,-1e100));d$y <- rbind(d$y,rep(1e-300,4))
    d$weights <- c(d$weights,1e-300);d$offset <- c(d$offset,1e100)
    fit <- fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,nonparallel=1L)
    result <- .ordinal_boundary_diagnostics(fit)
    expect_true(result$available)
    expect_equal(result$rows.examined,12L)
    expect_false(any(result$smallest.probabilities$row==13L))
    expect_false(any(result$smallest.gaps$row==13L))
    d <- ordinal_boundary_diagnostic_fixture();d$y[,2L] <- 0
    fit <- suppressWarnings(fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset))
    result <- .ordinal_boundary_diagnostics(fit)
    expect_true(result$available)
    expect_equal(result$collapsed.categories,"middle")
    expect_false(any(result$smallest.probabilities$category=="middle"))
    expect_equal(result$minimum.probability,min(predict(fit)[,-2L]),tolerance=1e-12)
})

test_that("coded rows and binary models retain meaningful boundary details", {
    d <- ordinal_boundary_diagnostic_fixture()
    rows <- rep(seq_len(nrow(d$x)),rowSums(d$y*d$weights))
    y <- unlist(lapply(seq_len(nrow(d$x)),function(i) rep(seq_len(4),d$y[i,]*d$weights[i])))
    fit <- fastglm_ordinal(d$x[rows,,drop=FALSE],ordered(y,levels=1:4),
        offset=d$offset[rows],nonparallel=1L)
    result <- .ordinal_boundary_diagnostics(fit,2L)
    expect_true(result$available)
    tab <- result$smallest.probabilities
    expect_equal(tab$observed.frequency,as.numeric(y[tab$row]==tab$category.index))
    expect_equal(tab$row.frequency,rep(1,nrow(tab)))
    binary <- fastglm_ordinal(d$x,d$y[,c(1L,4L)],offset=d$offset)
    binary.result <- .ordinal_boundary_diagnostics(binary)
    expect_true(binary.result$available)
    expect_true(is.na(binary.result$minimum.gap))
    expect_equal(nrow(binary.result$smallest.gaps),0L)
})

test_that("boundary diagnostics identify the retained sparse failure without claiming root absence", {
    set.seed(154005);n <- 96L
    x <- cbind(a=runif(n,-1,1),b=runif(n,-1,1))
    fit <- suppressWarnings(fastglm_ordinal(x,ordered(rep(1:4,length.out=n),levels=1:4),
        weights=rep(c(0,1,2,1,1),length.out=n),offset=cos(seq_len(n))/10,
        nonparallel=1:2,firth=TRUE,maxit=30))
    result <- .ordinal_boundary_diagnostics(fit,3L)
    expect_false(fit$converged)
    expect_true(result$available)
    expect_equal(result$smallest.gaps$row[1L],12L)
    expect_equal(result$smallest.gaps$category[1L],"2")
    expect_equal(result$smallest.gaps$observed.frequency[1L],0)
    expect_equal(result$minimum.gap,fit$minimum.gap,tolerance=1e-11)
    expect_gt(result$raw.score.maximum,fit$scaled.score)
    expect_match(result$interpretation,"not a test for root existence")
})

test_that("unavailable row diagnostics fail explicitly without losing stored convergence information", {
    d <- ordinal_boundary_diagnostic_fixture();fit <- fastglm_ordinal(d$x,d$y)
    for(value in list(0,-1,NA_real_,Inf,1.5,c(1,2),"five"))
        expect_error(.ordinal_boundary_diagnostics(fit,value),"positive integer")
    absent <- fit;absent$x <- NULL
    result <- .ordinal_boundary_diagnostics(absent)
    expect_false(result$available)
    expect_match(result$reason,"retained design")
    expect_equal(result$raw.score.maximum,max(abs(fit$effective$score)))
    unreadable <- fit;class(unreadable) <- c("fastglm_ordinal_stream",class(fit))
    unreadable$source <- list(n_chunks=1L)
    unreadable$fetch <- function(k) stop("source intentionally unavailable")
    result <- .ordinal_boundary_diagnostics(unreadable)
    expect_false(result$available)
    expect_match(result$reason,"source intentionally unavailable")
    expect_equal(result$source.reads,1L)
    expect_null(result$smallest.probabilities)
})

test_that("boundary diagnostics preserve logarithmic tails and compensated gaps", {
    d <- ordinal_boundary_diagnostic_fixture()
    fit <- fastglm_ordinal(d$x,d$y)
    # Exercise the diagnostic kernel at a constructed strictly ordered state;
    # this is not a claim that these modified coefficients solve the equations.
    fit$working$alpha <- c(1,2,3)
    fit$working$beta <- c(1e16,0)
    result <- .ordinal_boundary_diagnostics(fit,2L)
    expect_true(result$available)
    expect_equal(result$minimum.probability,0)
    expect_true(is.finite(result$minimum.log.probability))
    expect_lt(result$minimum.log.probability,log(.Machine$double.xmin))
    expect_equal(result$minimum.gap,1)
    expect_true(all(result$smallest.gaps$gap==1))
    expect_match(result$interpretation,"floating-point underflow")
})
