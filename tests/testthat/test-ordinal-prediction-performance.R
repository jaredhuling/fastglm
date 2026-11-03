.ordinal_prediction_reference <- function(object,x,offset) {
    xx <- if(ncol(x)) sweep(sweep(x,2,object$working$center,"-"),2,object$working$scale,"/") else x
    location <- fastglm:::streaming_qr_eta(xx,object$working$beta,offset)
    a <- object$working$alpha; q <- length(a); eta <- outer(-location,a,"+")
    F <- plogis(eta); S <- plogis(-eta)
    probability <- matrix(0,nrow(x),q+1L)
    probability[,1L] <- F[,1L]; probability[,q+1L] <- S[,q]
    softplus <- function(z) pmax(z,0)+log1p(exp(-abs(z)))
    if(q>1L) for(j in 2:q)
        probability[,j] <- exp(-softplus(-eta[,j])+log(-expm1(a[j-1L]-a[j]))-softplus(eta[,j-1L]))
    list(x=xx,probability=probability,cumulative=F,survival=S,derivative=F*S)
}

test_that("ordinal prediction retains subnormal interior probabilities", {
    fit <- fastglm_ordinal(matrix(numeric(),1,0),matrix(c(2,3,5,7),1),firth=TRUE)
    offset <- fit$working$alpha[2]+c(-745,-740,-710,-40,0,40,710,740,745)
    x <- matrix(numeric(),length(offset),0)
    expected <- .ordinal_prediction_reference(fit,x,offset)
    probability <- predict(fit,x,offset=offset)
    interior <- expected$probability[,2L]
    positive <- interior>0
    expect_true(any(interior[positive]<.Machine$double.xmin))
    expect_equal(probability[positive,2L]/interior[positive],rep(1,sum(positive)),tolerance=2e-13)
    expect_identical(unname(probability[,c(1,4)]),expected$probability[,c(1,4)])
    expect_equal(unname(predict(fit,x,offset=offset,type="cumulative")),unname(expected$cumulative),tolerance=0)
    # Distinct but very close thresholds retain their small category mass.
    fit$working$alpha[] <- c(-1e-12,0,1e-12)
    offset <- c(-740,-40,0,40,740)
    x <- matrix(numeric(),length(offset),0)
    expected <- .ordinal_prediction_reference(fit,x,offset)$probability
    actual <- predict(fit,x,offset=offset)
    positive <- expected[,2L]>0
    expect_equal(actual[positive,2L]/expected[positive,2L],rep(1,sum(positive)),tolerance=2e-13)
    expect_gt(actual[3,2],0)
})

test_that("selective ordinal prediction preserves full helper and uncertainty results", {
    set.seed(97801)
    x <- cbind(a=1e9+rnorm(250,sd=.4),b=rnorm(250)*1e-7)
    location <- .7*(x[,1]-1e9)-.3*x[,2]/1e-7
    y <- ordered(findInterval(location+rlogis(250),c(-1,0,1))+1L,levels=1:4)
    fit <- fastglm_ordinal(x,y,firth=TRUE)
    new <- x[1:19,,drop=FALSE]; offset <- seq(-.2,.2,length.out=nrow(new))
    expected <- .ordinal_prediction_reference(fit,new,offset)
    actual <- fastglm:::.ordinal_probability_block(fit,new,offset)
    for(name in names(expected)) expect_equal(actual[[name]],expected[[name]],tolerance=2e-14,info=name)
    predicted <- predict(fit,new,offset=offset,type="cumulative",se.fit=TRUE)
    q <- length(fit$working$alpha); p <- ncol(new)
    se <- matrix(0,nrow(new),q)
    for(j in seq_len(q)) {
        gradient <- cbind(matrix(0,nrow(new),q),-expected$x)*expected$derivative[,j]
        gradient[,j] <- expected$derivative[,j]
        se[,j] <- sqrt(rowSums((gradient%*%fit$working$covariance)*gradient))
    }
    expect_equal(unname(predicted$fit),unname(expected$cumulative),tolerance=0)
    expect_equal(unname(predicted$se.fit),se,tolerance=2e-14)
    empty <- fastglm:::.ordinal_probability_block(fit,new[FALSE,,drop=FALSE],numeric())
    expect_identical(dim(empty$probability),c(0L,4L))
    expect_identical(dim(empty$derivative),c(0L,3L))
})

test_that("binary and reduced-category ordinal classes agree across blocks", {
    binary <- fastglm_ordinal(matrix(numeric(),1,0),matrix(c(12,17),1),firth=TRUE)
    x0 <- matrix(numeric(),5,0); off <- c(-40,-1,0,1,40)
    probability <- predict(binary,x0,offset=off,se.fit=TRUE)
    expect_equal(rowSums(probability$fit),rep(1,5),tolerance=1e-15)
    expect_true(all(is.finite(probability$se.fit)))
    expect_equal(predict(binary,x0,offset=off,type="cumulative",se.fit=TRUE)$se.fit[,1],
                 probability$se.fit[,1],tolerance=1e-15)
    set.seed(97802)
    x <- cbind(a=rnorm(120),b=rnorm(120))
    counts <- matrix(0,120,5);counts[,c(1,3,5)] <- matrix(rpois(360,3)+1,120)
    expect_warning(fit <- fastglm_ordinal(x,counts,firth=TRUE),"empty interior")
    new <- x[rep(seq_len(120),3),c("b","a"),drop=FALSE]
    new[9,1] <- NA_real_; off <- rep(.1,nrow(new));off[17] <- NA_real_
    expected <- predict(fit,new,offset=off)
    classes <- ordered(fit$levels[max.col(expected,ties.method="first")],levels=fit$levels)
    for(block in c(7L,113L)) {
        fit$block.size <- block
        expect_identical(unname(predict(fit,new,offset=off,type="class")),unname(classes))
        expect_equal(predict(fit,new,offset=off),expected,tolerance=2e-14)
        if(requireNamespace("Matrix",quietly=TRUE))
            expect_equal(predict(fit,Matrix::Matrix(new,sparse=TRUE),offset=off),expected,tolerance=2e-14)
    }
    fit$prediction.offset.required <- TRUE
    expect_error(predict(fit,new),"explicit prediction offset")
    expect_equal(predict(fit,new,offset=off),expected,tolerance=2e-14)
    expect_equal(dim(predict(fit)),c(120L,5L))
})
