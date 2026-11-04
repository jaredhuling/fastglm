test_that("certified saturated logit boundaries fail before iterative fitting", {
    x <- matrix(c(0,1),ncol=1,dimnames=list(NULL,"group"))
    counts <- rbind(c(low=6,middle=0,high=4),c(low=3,middle=5,high=2))
    offset <- c(.2,-.1)
    expected <- "no mean-bias-reduced solution.*middle"
    expect_error(fastglm_ordinal(x,counts,nonparallel="group",firth=TRUE,
        offset=offset),expected)
    prepared <- fastglm_ordinal_prepare(x,nonparallel="group",firth=TRUE,offset=offset)
    expect_error(fastglm_ordinal(prepared,counts),expected)
    reads <- 0L
    source <- fastglm_source(function(k) {
        reads <<- reads+1L
        list(X=x[k,,drop=FALSE],y=counts[k,,drop=FALSE],offset=offset[k])
    },2L)
    expect_error(fastglm_ordinal_streaming(source,nonparallel="group",firth=TRUE),expected)
    expect_lte(reads,6L)
    positive <- counts;positive[1L,"middle"] <- 1
    fit <- fastglm_ordinal(x,positive,nonparallel="group",firth=TRUE,offset=offset,tol=1e-10)
    target <- positive;target[,c("low","high")] <- target[,c("low","high")]+.5
    target <- target/rowSums(target)
    expect_true(fit$converged)
    expect_equal(predict(fit),target,tolerance=1e-9,ignore_attr=TRUE)
})
