# The R implementations are retained as references for the compiled blocks.
inference_data <- function(link,seed=123L,n=250L) {
    set.seed(seed)
    x <- cbind(a=rnorm(n),b=rbinom(n,1,0.4),c=rnorm(n,2,3))
    inverse <- switch(link,logit=qlogis,probit=qnorm,cloglog=function(u) log(-log1p(-u)))
    latent <- drop(x%*%c(0.5,-0.4,0.1))+inverse(runif(n))
    y <- ordered(findInterval(latent,quantile(latent,c(0.2,0.45,0.8)))+1L)
    list(x=x,y=y,weights=rpois(n,2)+1,offset=rnorm(n,sd=0.3))
}

test_that("native sandwich blocks match the R reference", {
    for(link in c("logit","probit")) for(independent in c(TRUE,FALSE)) for(grouped in c(FALSE,TRUE)) {
        d <- inference_data(link)
        fit <- fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,link=link,tol=1e-9)
        w <- fit$working;xx <- fastglm:::.ordinal_prediction_scale(d$x,w$center,w$scale)
        counts <- if(grouped) {z <- matrix(0,nrow(d$x),4L);z[cbind(seq_len(nrow(d$x)),as.integer(d$y))] <- 2;z} else numeric()
        codes <- as.integer(d$y)
        native <- fastglm:::.ordinal_sandwich_block(xx,w$alpha,w$beta,d$offset,codes,counts,d$weights,7,independent,link)
        reference <- fastglm:::.ordinal_sandwich_block_r(xx,w$alpha,w$beta,d$offset,codes,counts,d$weights,7,independent,link)
        expect_equal(native$information,reference$information,tolerance=1e-9)
        expect_equal(native$meat,reference$meat,tolerance=1e-9)
        expect_equal(native$score,unname(reference$score),tolerance=1e-7)
        if(independent) expect_null(native$rows) else expect_equal(unname(native$rows),unname(reference$rows),tolerance=1e-9)
    }
})

test_that("native common-slope moments match the R reference", {
    for(link in c("logit","probit","cloglog")) {
        d <- inference_data(link)
        fit <- fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,link=link,tol=1e-9)
        response <- fastglm:::.ordinal_assessment_response(d$y,d$weights,fit$levels,nrow(d$x))
        for(selected in list(1L,c(1L,3L),1:3)) for(block in c(13L,2048L)) {
            native <- fastglm:::.ordinal_po_score_moments(d$x,response,d$offset,fit,selected,block)
            reference <- fastglm:::.ordinal_po_score_moments_r(d$x,response,d$offset,fit,selected,block)
            expect_equal(native$score,reference$score,tolerance=1e-8)
            expect_equal(native$information,reference$information,tolerance=1e-9)
            expect_identical(native$nuisance,reference$nuisance)
        }
    }
})
