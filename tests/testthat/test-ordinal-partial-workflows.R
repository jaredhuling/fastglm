test_that("partial prediction summaries agree across dense, prepared, and source fits", {
    i <- seq_len(18L)
    x <- cbind(exposure=seq(-1,1,length.out=length(i)),group=sin(i))
    counts <- cbind(low=4+i%%3,middle=3+i%%4,high=5+i%%2)
    offset <- cos(i)/10
    blocks <- split(i,ceiling(i/6))
    source <- fastglm_source(function(k) {
        at <- blocks[[k]]
        list(X=x[at,,drop=FALSE],y=counts[at,,drop=FALSE],offset=offset[at])
    },length(blocks))
    for(link in c("logit","probit","cloglog")) {
        dense <- fastglm_ordinal(x,counts,offset=offset,nonparallel="group",
            link=link,firth=TRUE,tol=1e-10)
        prepared <- fastglm_ordinal_prepare(x,nonparallel="group",link=link,firth=TRUE)
        retained <- fastglm_ordinal(prepared,counts,offset=offset,tol=1e-10)
        streamed <- fastglm_ordinal_streaming(source,nonparallel="group",
            link=link,firth=TRUE,tol=1e-10)
        expect_true(dense$converged && retained$converged && streamed$converged)
        expect_equal(fastglm_vcov(retained),fastglm_vcov(dense),tolerance=1e-9)
        # Explicit matrix prediction data use zero offsets unless the method
        # receives a replacement offset; both representations use that rule.
        expect_equal(fastglm_average(streamed,newdata=x,weights=rowSums(counts)),
            fastglm_average(dense,newdata=x,weights=rowSums(counts)),tolerance=1e-8)
        for(variable in colnames(x)) {
            expected <- fastglm_marginal_effects(retained,variable,newdata=x,
                weights=rowSums(counts),offset=offset)
            actual <- fastglm_marginal_effects(streamed,variable,newdata=x,
                weights=rowSums(counts),offset=offset)
            expect_equal(actual,expected,tolerance=1e-8)
            expect_equal(fastglm_marginal_effects(streamed,variable,newdata=x,
                weights=rowSums(counts),offset=offset,vcov.=vcov(streamed)),actual,
                tolerance=1e-9)
        }
    }
})
