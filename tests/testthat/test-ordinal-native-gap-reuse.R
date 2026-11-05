test_that("partial row evaluations preserve tiny gaps across links and row blocks", {
    x<-matrix(c(-.7,-.2,.3,.9,-.4,.5,.1,.8),ncol=1)
    z<-cbind(c(0,1,0,1,0,1,0,1),c(-.4,.3,.2,-.1,.5,-.2,.1,-.3))
    alpha<-c(-1,-.4,-.4+2^-20,.5,1.1)
    b<-c(alpha,.2,rep(c(.125,-.0625),length(alpha)))
    counts<-matrix(rep(1:3,length.out=nrow(x)*6L),nrow(x),6L)
    weights<-c(1,2,0,1,3,1,2,1);offset<-c(.1,-.2,0,.1,-.1,.2,0,-.1)
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        small<-fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
            weights,offset,6L,b,firth,block_size=1L,link=link)
        large<-fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
            weights,offset,6L,b,firth,block_size=2048L,link=link)
        expect_equal(small,large,tolerance=1e-13)
        expect_equal(small$minimum.gap,2^-20,tolerance=1e-13)
        expect_true(is.finite(small$loglik))
        expect_true(all(is.finite(small$score)))
        expect_true(all(is.finite(small$adjustment)))
        expect_true(all(is.finite(small$information)))
    }
})

test_that("tiny positive row gaps retain the large-covariate cancellation invariant", {
    x<-matrix(numeric(),5L,0L)
    z<-rbind(c(-1,0),c(0,1),c(1,0),c(0,-1),c(1e17,-1e17))
    # For the extreme row, exact binary .4-.3-.1 equals 2^-55.
    # The direct compensated gap therefore remains 2^-20, although premature
    # coefficient subtraction or separate predictor rounding can destroy it.
    b<-c(-2.25,-2.25+2^-55*1e17+2^-20,.1,0,.4,.3)
    counts<-matrix(1,5L,3L);weights<-c(rep(1,4),1e-34)
    for(link in 0:2) for(firth in c(FALSE,TRUE)) {
        value<-fastglm:::ordinal_partial_native_evaluate(x,z,integer(),counts,
            weights,c(numeric(4),-1e16),3L,b,firth,link=link)
        expect_lt(abs(value$minimum.gap-2^-20),5e-15)
        expect_true(is.finite(value$loglik))
        expect_true(all(is.finite(value$score)))
        expect_true(all(is.finite(value$adjustment)))
        expect_true(all(is.finite(value$cov.unscaled)))
    }
})
