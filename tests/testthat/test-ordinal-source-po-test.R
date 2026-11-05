source_po_fixture <- function(link="logit") {
    d <- expand.grid(x=seq(-1,1,length.out=8),g=factor(c("a","b","c")))
    d$z <- sin(seq_len(nrow(d)));d$off <- cos(seq_len(nrow(d)))/10
    X <- model.matrix(~x+g+z+x:g,d)[,-1,drop=FALSE]
    slopes <- matrix(c(.07,.03,.05,.04,-.02,.09,.02,.03,-.03,.01,
        .1,-.02,.01,-.01,.03),5L,3L)
    eta <- matrix(c(-1,.15,1.25),nrow(d),3L,byrow=TRUE)-.12*d$x-d$off-X[,-1L]%*%slopes
    F <- switch(link,logit=plogis(eta),probit=pnorm(eta),cloglog=-expm1(-exp(eta)))
    d[c("a","b","c","d")] <- 1+floor(32*(cbind(F,1)-cbind(0,F)))
    d$w <- rep(c(1,2,0,1),length.out=nrow(d));d
}

source_po_reader <- function(d,parts,formula=FALSE) {
    if(formula) return(fastglm_source(function(k)d[parts[[k]],,drop=FALSE],length(parts),
        formula=cbind(a,b,c,d)~x+g+z+x:g+offset(off),prototype=d,weights="w"))
    X <- as.matrix(d[c("x","z")]);y <- as.matrix(d[c("a","b","c","d")])
    fastglm_source(function(k) {
        i <- parts[[k]]
        list(X=X[i,,drop=FALSE],y=y[i,,drop=FALSE],weights=d$w[i],offset=d$off[i])
    },length(parts))
}

test_that("original-source common and partial tests agree with dense tests for every link", {
    for(link in c("logit","probit","cloglog")) {
        d <- source_po_fixture(link);parts <- split(seq_len(nrow(d)),rep(1:4,each=6))
        source <- source_po_reader(d,parts,TRUE)
        formula <- cbind(a,b,c,d)~x+g+z+x:g+offset(off)
        for(partial in c(FALSE,TRUE)) {
            selector <- if(partial) ~g+z+x:g else NULL
            streamed <- fastglm_ordinal_streaming(source,link=link,nonparallel=selector,tol=1e-10)
            dense <- fastglm_ordinal(formula,d,weights=w,link=link,nonparallel=selector,tol=1e-10)
            for(estimation in c("ML","meanBR")) {
                a <- fastglm_po_test(streamed,c("g","x:g"),null.estimation=estimation,by.term=TRUE)
                b <- fastglm_po_test(dense,c("g","x:g"),null.estimation=estimation,by.term=TRUE)
                expect_true(all(vapply(a$tests,function(t)t$available,logical(1))))
                expect_equal(as.data.frame(a)$statistic,as.data.frame(b)$statistic,tolerance=3e-7)
                expect_equal(as.data.frame(a)$df,as.data.frame(b)$df)
                expect_equal(a$tests$Joint$frequency,sum(as.matrix(d[c("a","b","c","d")])*d$w))
                expect_equal(a$tests$Joint$observations,sum(d$w>0))
                expect_equal(a$tests$g$columns,c("gb","gc"))
                if(partial) {
                    expect_equal(a$tests$g$null.coefficients,b$tests$g$null.coefficients,tolerance=2e-7)
                    expect_setequal(a$tests$g$diagnostics$null$nonparallel,c(4L,5L,6L))
                }
            }
            if(partial) {
                a <- fastglm_po_test(streamed,c("g","x:g"),method="lr",by.term=TRUE)
                b <- fastglm_po_test(dense,c("g","x:g"),method="lr",by.term=TRUE)
                expect_true(all(vapply(a$tests,function(t)t$available,logical(1))))
                expect_equal(as.data.frame(a)$statistic,as.data.frame(b)$statistic,tolerance=2e-7)
                expect_true(a$tests$g$diagnostics$full$converged)
                expect_true(a$tests$g$diagnostics$null$converged)
                expect_equal(fastglm_po_test(streamed,"x")$reason,"no_slope_contrasts")
                expect_equal(fastglm_po_test(streamed,"x")$df,0L)
            }
        }
    }
})

test_that("source tests preserve grouping, partition, recoding and offset conventions", {
    for(link in c("logit","probit","cloglog")) {
        d <- source_po_fixture(link);X <- as.matrix(d[c("x","z")])
        counts <- as.matrix(d[c("a","b","c","d")]);effective <- counts*d$w
        expanded <- which(effective>0,arr.ind=TRUE)
        expanded <- expanded[rep(seq_len(nrow(expanded)),effective[expanded]),,drop=FALSE]
        ii <- expanded[,1];yy <- expanded[,2]
        code.parts <- split(seq_along(ii),ceiling(seq_along(ii)/113))
        codes <- fastglm_source(function(k) {
            at <- code.parts[[k]]
            list(X=X[ii[at],,drop=FALSE],y=yy[at],offset=d$off[ii[at]])
        },length(code.parts))
        grouped <- fastglm_ordinal_streaming(source_po_reader(d,split(seq_len(nrow(d)),rep(1:4,6))),
            link=link,nonparallel="z")
        individual <- fastglm_ordinal_streaming(codes,levels=colnames(counts),link=link,nonparallel="z")
        for(estimation in c("ML","meanBR")) {
            a <- fastglm_po_test(grouped,null.estimation=estimation)
            b <- fastglm_po_test(individual,null.estimation=estimation)
            expect_true(a$available);expect_true(b$available)
            expect_equal(a$statistic,b$statistic,tolerance=3e-7)
            expect_equal(a$frequency,b$frequency)
        }
        recoded <- d;recoded$x <- 3*d$x+4;recoded$z <- -2*d$z+1;recoded$off <- d$off+2
        affine <- fastglm_ordinal_streaming(source_po_reader(recoded,list(1:9,10:24)),link=link,nonparallel="z")
        expect_equal(fastglm_po_test(affine)$statistic,fastglm_po_test(grouped)$statistic,tolerance=3e-7)
        replicated <- d;replicated$w <- d$w*2
        twice <- fastglm_ordinal_streaming(source_po_reader(replicated,list(1:24)),link=link,nonparallel="z")
        expect_equal(fastglm_po_test(twice)$statistic,2*fastglm_po_test(grouped)$statistic,tolerance=3e-7)
        fractional <- d;fractional$w <- fractional$w/2
        fractional[c("a","b","c","d")] <- 2*fractional[c("a","b","c","d")]
        half <- fastglm_ordinal_streaming(source_po_reader(fractional,list(1:24)),link=link,nonparallel="z")
        expect_equal(fastglm_po_test(half,null.estimation="meanBR")$statistic,
            fastglm_po_test(grouped,null.estimation="meanBR")$statistic,tolerance=3e-7)
    }
})

test_that("source partial ordinary scores match independent probability derivatives", {
    skip_if_not_installed("numDeriv")
    for(link in c("logit","probit","cloglog")) {
        d <- source_po_fixture(link);X <- as.matrix(d[c("x","z")])
        y <- as.matrix(d[c("a","b","c","d")]);w <- d$w
        fit <- fastglm_ordinal_streaming(source_po_reader(d,list(1:5,6:17,18:24)),link=link,nonparallel="z")
        out <- fastglm_po_test(fit,null.estimation="meanBR")
        b <- out$null.coefficients;q <- 3L
        probability <- function(b) {
            eta <- matrix(b[1:3],nrow(X),q,byrow=TRUE)-X[,1]*b[4]-d$off-outer(X[,2],b[5:7])
            F <- switch(link,logit=plogis(eta),probit=pnorm(eta),cloglog=-expm1(-exp(eta)))
            cbind(F,1)-cbind(0,F)
        }
        pr <- probability(b);J <- numDeriv::jacobian(function(b)as.vector(probability(b)),b)
        U <- drop(crossprod(J,as.vector(y*w/pr)))
        I <- crossprod(J,J*as.vector(rowSums(y)*w/pr))
        L <- out$contrast;Q <- qr.Q(qr(t(L)),complete=TRUE);B <- Q[,(nrow(L)+1L):ncol(L),drop=FALSE]
        e <- U-I%*%B%*%solve(crossprod(B,I%*%B),crossprod(B,U))
        expected <- drop(crossprod(e,solve(I,e)))
        expect_true(out$available)
        expect_equal(out$statistic,expected,tolerance=2e-6)
        expect_gt(max(abs(crossprod(B,U))),1e-4)
    }
})

test_that("source tests report unsupported data and unsuccessful fits explicitly", {
    d <- source_po_fixture();source <- source_po_reader(d,list(1:8,9:16,17:24))
    fit <- fastglm_ordinal_streaming(source,nonparallel="z")
    expect_error(fastglm_po_test(fit,weights=1),"recorded weights and offsets")
    expect_error(fastglm_po_test(fit,offset=0),"recorded weights and offsets")
    expect_error(fastglm_po_test(fit,method="lr",null.estimation="meanBR"),"require null.estimation")
    expect_error(fastglm_po_test(fit,"missing"),"unknown")
    expect_error(fastglm_po_test(fit,control=list(maxit=0)),"positive")
    exhausted <- fastglm_po_test(fit,control=list(maxit=1))
    expect_false(exhausted$available);expect_equal(exhausted$reason,"null_fit_not_converged")
    expect_false(exhausted$diagnostics$null$converged)
    exhausted.lr <- fastglm_po_test(fit,control=list(maxit=1),method="lr")
    expect_equal(exhausted.lr$reason,"full_fit_not_converged")
    expect_false(exhausted.lr$diagnostics$full$converged)
    original <- fit;original$converged <- FALSE
    expect_equal(fastglm_po_test(original)$statistic,fastglm_po_test(fit)$statistic)
    expect_true(fastglm_po_test(original,method="lr")$available)
    reduced <- fastglm_ordinal_streaming(source,nonparallel="z",firth=TRUE)
    expect_equal(fastglm_po_test(reduced,method="lr")$statistic,
        fastglm_po_test(fit,method="lr")$statistic,tolerance=1e-8)
    mixed <- fastglm_po_test(fit,c("x","z"),by.term=TRUE)
    expect_equal(mixed$tests$x$reason,"no_slope_contrasts")
    expect_equal(mixed$tests$x$df,0L)
    expect_equal(mixed$tests$Joint$statistic,mixed$tests$z$statistic)
    fetch <- fit$fetch
    empty <- fit;empty$fetch <- function(k) {
        z <- fetch(k);z$counts[,2] <- 0;z$totals <- rowSums(z$counts)*z$weights;z
    }
    expect_equal(fastglm_po_test(empty)$reason,"empty_categories")
    expect_equal(fastglm_po_test(empty)$empty.categories,"b")
    zero <- fit;zero$fetch <- function(k) {z <- fetch(k);z$weights[] <- 0;z$totals[] <- 0;z}
    expect_equal(fastglm_po_test(zero)$reason,"no_usable_outcomes")
    rank <- fit;rank$fetch <- function(k) {z <- fetch(k);z$X[,2] <- z$X[,1];z}
    expect_equal(fastglm_po_test(rank)$reason,"rank_deficient_design")
    broken <- fit;broken$fetch <- function(k)stop("reader unavailable")
    expect_equal(fastglm_po_test(broken)$reason,"source_read_error")
    expect_match(fastglm_po_test(broken)$detail,"reader unavailable")
    binary <- source_po_reader(d,list(1:24));read <- binary$callback
    binary$callback <- function(k) {z <- read(k);z$y <- z$y[,c(1,4)];z}
    binary <- fastglm_ordinal_streaming(binary,nonparallel="z")
    expect_equal(fastglm_po_test(binary)$reason,"no_slope_contrasts")
    expect_equal(fastglm_po_test(binary)$df,0L)
})

test_that("source separation certificates combine all blocks and ignore zero frequency", {
    X <- cbind(x=c(0,0,1,1),z=c(-1,1,-1,1))
    counts <- rbind(c(6,3,2),c(2,4,3),c(0,0,8),c(0,0,9));colnames(counts) <- c("low","mid","high")
    reader <- fastglm_source(function(k)list(X=X[k,,drop=FALSE],y=counts[k,,drop=FALSE]),4)
    fit <- fastglm_ordinal_streaming(reader,firth=TRUE)
    out <- fastglm_po_test(fit)
    expect_equal(out$reason,"null_separation");expect_equal(out$separation$column,"x")
    expect_equal(out$separation$value,1);expect_equal(out$separation$category,3L)
    expect_true(fastglm_po_test(fit,null.estimation="meanBR")$available)
    # Two values in each block may still be three values globally. The state
    # must discard the candidate after observing the third positive value.
    scan.object <- list(design.columns="x",levels=c("a","b","c"),source=list(n_chunks=3L),
        fetch=function(k)list(X=matrix(c(k-1,k),2,1),counts=matrix(c(2,0,0,0,0,2),2,3,byrow=TRUE),
            weights=c(1,1),totals=c(2,2)))
    expect_null(fastglm:::.ordinal_stream_po_scan(scan.object)$separation)
    scan.object$fetch <- function(k)list(X=matrix(c(0,1,k+10),3,1),
        counts=rbind(c(2,1,1),c(0,0,2),c(1,1,1)),weights=c(1,1,0),totals=c(4,2,0))
    expect_equal(fastglm:::.ordinal_stream_po_scan(scan.object)$separation$value,1)
})

test_that("source tests avoid retained-data paths and use existing presentation methods", {
    d <- source_po_fixture();fit <- fastglm_ordinal_streaming(source_po_reader(d,list(1:12,13:24)),nonparallel="z")
    testthat::local_mocked_bindings(
        .ordinal_stream_collect=function(...)stop("unexpected source collection"),
        .ordinal_prediction_design=function(...)stop("unexpected retained design"),
        .ordinal_fit_matrix=function(...)stop("unexpected dense fit"),.package="fastglm")
    out <- fastglm_po_test(fit);lr <- fastglm_po_test(fit,method="lr")
    expect_true(out$available);expect_true(lr$available)
    expect_null(fit$x);expect_null(fit$y)
    expect_s3_class(out,"fastglm_test");expect_s3_class(out,"htest")
    expect_output(print(out),"p-value")
    expect_equal(as.data.frame(out)$statistic,out$statistic)
    expect_equal(as.data.frame(lr)$statistic,lr$statistic)
})

test_that("source common tests retain information failure details", {
    d <- source_po_fixture();fit <- fastglm_ordinal_streaming(source_po_reader(d,list(1:24)))
    testthat::local_mocked_bindings(.ordinal_po_score_statistic=function(...)
        stop("expected information is numerically singular"),.package="fastglm")
    out <- fastglm_po_test(fit)
    expect_false(out$available);expect_equal(out$reason,"information_unavailable")
    expect_match(out$detail,"expected information is numerically singular")
    expect_true(out$null.converged)
})
