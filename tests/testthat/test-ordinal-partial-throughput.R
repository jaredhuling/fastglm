test_that("partial source searches reuse their computed scoring-step bound", {
    i <- seq_len(30L)
    x <- cbind(common=sin(i),varying=cos(i*.7))
    counts <- cbind(low=4+i%%3,middle=5+i%%4,high=3+i%%2)
    source <- fastglm_source(function(k) {
        at <- seq.int((k-1L)*10L+1L,k*10L)
        list(X=x[at,,drop=FALSE],y=counts[at,,drop=FALSE])
    },3L)
    for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) {
        fit <- fastglm_ordinal_streaming(source,nonparallel="varying",link=link,firth=br)
        expect_true(fit$converged)
        expect_equal(fit$fallbacks,0L)
        # Three preprocessing passes check frequencies, scaling, and rank.
        # Each evaluation requires information and
        # predictor diagnostics, plus the global adjustment pass for meanBR.
        # A first scoring search needs only its row-domain feasibility pass.
        bound <- 3L+(2L+as.integer(br))*fit$evaluations+fit$iterations-1L
        expect_lte(fit$source.passes,bound)
    }
})
