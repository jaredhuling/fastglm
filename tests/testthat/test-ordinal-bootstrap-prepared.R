test_that("prepared parametric refits retain independent-fit results and metadata", {
    set.seed(8301);n <- 45L
    x <- cbind(a=rnorm(n),b=runif(n,-1,1))
    attr(x,"description") <- "retained input metadata"
    counts <- matrix(sample(1:6,n*4L,TRUE),n,4L,dimnames=list(NULL,letters[1:4]))
    counts[1L,] <- 0
    w <- rep(c(.5,1,2),length.out=n);off <- runif(n,-.3,.3)
    for(br in c(FALSE,TRUE)) for(standardize in c(FALSE,TRUE)) {
        fit <- fastglm_ordinal(x,counts/w,weights=w,offset=off,firth=br,
                              control=list(standardize=standardize))
        input <- fastglm:::.ordinal_bootstrap_input(fit)
        cache <- new.env(parent=emptyenv())
        for(j in 1:2) {
            yy <- fastglm:::.ordinal_simulate_counts(fit)
            direct <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy)
            reused <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,cache=cache)
            expect_identical(reused,direct)
            expect_identical(cache$builds,1L)
            expect_false(reused$preparation.reused)
            expect_null(reused$prepared)
        }
        # NULL fitting arguments use the same public defaults on both paths.
        reset <- list(tol=NULL,maxit=NULL,block.size=NULL,control=NULL)
        direct <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,control=reset)
        reused <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,control=reset,cache=cache)
        expect_identical(reused,direct)
    }
})

test_that("parametric preparation checks totals and keeps failures local to refits", {
    set.seed(8302);n <- 40L
    x <- cbind(a=rnorm(n));counts <- matrix(sample(1:4,n*3L,TRUE),n,3L)
    fit <- fastglm_ordinal(x,counts,firth=TRUE)
    input <- fastglm:::.ordinal_bootstrap_input(fit);cache <- new.env(parent=emptyenv())
    yy <- fastglm:::.ordinal_simulate_counts(fit)
    a <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,cache=cache)
    yy[2L,1L] <- yy[2L,1L]+1
    b <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,cache=cache)
    expect_identical(cache$builds,2L)
    expect_identical(b,fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy))
    expect_warning(failed <- fastglm_bootstrap(fit,R=3L,type="parametric",seed=831,
        control=list(control=list(standardize=NA))),"uncertainty summaries are unavailable")
    expect_identical(failed$successful,0L)
    expect_true(all(failed$diagnostics$convergence.reason=="fit_error"))
    expect_true(all(grepl("optimizer flag",failed$diagnostics$error)))
    expect_true(all(is.na(failed$replicates)))
    expect_warning(incomplete <- fastglm_bootstrap(fit,R=3L,type="parametric",seed=831,
        control=list(maxit=1L)),"uncertainty summaries are unavailable")
    expect_true(all(incomplete$diagnostics$statistic.ok))
    expect_true(all(!incomplete$diagnostics$converged))
    expect_true(all(is.finite(incomplete$replicates)))
    expect_true(all(is.na(incomplete$std.error)))
})

test_that("parametric reuse preserves draws and formula statistic metadata", {
    set.seed(8303);n <- 90L
    data <- data.frame(x=rnorm(n),g=factor(rep(letters[1:3],30L)),w=sample(0:3,n,TRUE),
                       off=runif(n,-.2,.2),extra=runif(n,-.1,.1))
    data$y <- ordered(cut(.4*data$x+rlogis(n),c(-Inf,-.7,.8,Inf)))
    data$x[4L] <- NA_real_
    fit <- fastglm_ordinal(y~splines::ns(x,df=3)+g+offset(off),data,weights=w,
                          offset=extra,firth=TRUE,na.action=na.exclude)
    newdata <- data[c(1L,3L,8L),,drop=FALSE]
    statistic <- function(f) c(as.numeric(predict(f,newdata=newdata)),
                               frequency=nobs(f),preparation=as.numeric(f$preparation.reused))
    before <- .Random.seed
    result <- fastglm_bootstrap(fit,R=4L,type="parametric",statistic=statistic,seed=832)
    expect_identical(.Random.seed,before)
    input <- fastglm:::.ordinal_bootstrap_input(fit)
    cache <- new.env(parent=emptyenv())
    for(j in seq_len(result$R)) {
        set.seed(result$seeds[j]);yy <- fastglm:::.ordinal_simulate_counts(fit)
        direct <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy)
        reused <- fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,cache=cache)
        expect_identical(reused,direct)
        expect_identical(unname(result$replicates[j,]),unname(statistic(direct)))
        expect_identical(direct$na.action,NULL)
        expect_equal(model.response(direct$model),yy)
    }
    expect_true(result$inference.available)
    expect_false(any(result$diagnostics$boundary.changed))
})

test_that("parametric boundary diagnostics are preserved during setup reuse", {
    x <- matrix(numeric(),1L,0L)
    expect_warning(fit <- fastglm_ordinal(x,matrix(c(5,0,7),1L,
        dimnames=list(NULL,c("a","b","c"))),firth=TRUE),"collapsed")
    input <- fastglm:::.ordinal_bootstrap_input(fit);cache <- new.env(parent=emptyenv())
    for(yy in list(matrix(c(3,0,9),1L),matrix(c(0,0,12),1L))) {
        colnames(yy) <- fit$levels
        direct <- suppressWarnings(fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy))
        reused <- suppressWarnings(fastglm:::.ordinal_bootstrap_refit(fit,input,counts=yy,cache=cache))
        expect_identical(reused,direct)
        expect_identical(cache$builds,1L)
    }
    expect_warning(result <- fastglm_bootstrap(fit,R=3L,type="parametric",seed=833),
                   "original fit has empty interior categories")
    expect_false(result$inference.available)
    expect_true(all(is.finite(result$replicates)))
    expect_true(all(is.na(result$std.error)))
})

test_that("parametric bootstrap workers retain their own setup across assigned refits", {
    skip_on_cran()
    set.seed(8304);n <- 60L
    x <- cbind(a=rnorm(n));counts <- matrix(sample(1:7,n*3L,TRUE),n,3L)
    fit <- fastglm_ordinal(x,counts,firth=TRUE)
    # Observe private cache persistence without exposing it in public results.
    # A marker set by one statistic must survive the next fit in that process.
    statistic <- function(f) {
        shared <- get0("cache",envir=parent.frame(),inherits=TRUE)
        seen <- is.environment(shared) && isTRUE(shared$test.seen)
        builds <- if(is.environment(shared) && !is.null(shared$builds)) shared$builds else 0L
        if(is.environment(shared)) shared$test.seen <- TRUE
        c(previous=as.numeric(seen),builds=builds)
    }
    serial <- fastglm_bootstrap(fit,R=6L,type="parametric",statistic=statistic,seed=834)
    parallel <- fastglm_bootstrap(fit,R=6L,type="parametric",statistic=statistic,seed=834,workers=2L)
    expect_identical(unname(serial$replicates[,"previous"]),c(0,rep(1,5)))
    expect_identical(unname(parallel$replicates[,"previous"]),rep(c(0,1,1),2L))
    expect_equal(serial$replicates[,"builds"],rep(1,6))
    expect_equal(parallel$replicates[,"builds"],rep(1,6))
})
