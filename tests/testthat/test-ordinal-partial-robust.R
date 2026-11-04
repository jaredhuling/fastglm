partial_robust_fixture <- function(link="logit",n=30L) {
    x <- cbind(x=seq(-1,1,length.out=n),z=sin(seq_len(n)))
    offset <- .1*cos(seq_len(n))
    eta <- cbind(-.8,.9)[rep(1L,n),]-.35*x[,1L]-offset-
        x[,2L]%o%c(.15,-.05)
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
    F <- inverse(eta);counts <- floor(24*(cbind(F,1)-cbind(0,F)))+1
    colnames(counts) <- c("low","middle","high")
    list(x=x,counts=counts,offset=offset,weights=rep(c(0,1,2),length.out=n),
        cluster=rep(1:6,length.out=n))
}

partial_robust_reference <- function(b,x,offset,q,common,nonparallel,link) {
    pc <- length(common);pn <- length(nonparallel)
    location <- if(pc) drop(x[,common,drop=FALSE]%*%b[q+seq_len(pc)]) else numeric(nrow(x))
    gamma <- matrix(b[q+pc+seq_len(q*pn)],pn,q)
    eta <- matrix(b[seq_len(q)],nrow(x),q,byrow=TRUE)-location-offset-
        x[,nonparallel,drop=FALSE]%*%gamma
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(x)-expm1(-exp(x)))
    F <- inverse(eta)
    log(cbind(F,1)-cbind(0,F))
}

test_that("partial observed information and score meat match independent derivatives", {
    skip_if_not_installed("numDeriv")
    a <- partial_robust_fixture();x <- a$x;counts <- a$counts;w <- a$weights
    q <- 2L;b <- c(-.8,.9,.35,.15,-.05);d <- length(b)
    for(link in c("logit","probit","cloglog")) {
        object <- structure(list(alpha=b[1:2],beta=b[3L],common=1L,nonparallel=2L,
            coefficients=b,link=link,working=list(alpha=b[1:2],beta=b[3L],
            gamma=matrix(b[4:5],1L),center=c(0,0),scale=c(1,1))),
            class=c("fastglm_ordinal_partial","fastglm_ordinal"))
        logprob <- function(b) partial_robust_reference(b,x,a$offset,q,1L,2L,link)
        likelihood <- function(b) sum((counts*w)*logprob(b))
        H <- -numDeriv::hessian(likelihood,b);U <- numDeriv::grad(likelihood,b)
        M <- matrix(0,d,d);rows <- matrix(0,nrow(x),d)
        for(i in seq_len(nrow(x))) {
            score <- numDeriv::jacobian(function(b) logprob(b)[i,],b)
            M <- M+crossprod(score,score*(w[i]*counts[i,]))
            rows[i,] <- colSums(score*(w[i]*counts[i,]))
        }
        block <- fastglm:::.ordinal_partial_sandwich_block(object,x,a$offset,integer(),counts,w)
        cluster <- fastglm:::.ordinal_partial_sandwich_block(object,x,a$offset,integer(),counts,w,independent=FALSE)
        expect_equal(block$information,H,tolerance=3e-7)
        expect_equal(block$score,U,tolerance=3e-7,ignore_attr=TRUE)
        expect_equal(block$meat,M,tolerance=3e-7)
        expect_equal(cluster$rows,rows,tolerance=3e-7)
        N <- sum(counts*w)
        scaled <- fastglm:::.ordinal_partial_sandwich_block(object,x,a$offset,integer(),counts,w,frequency=N)
        expect_equal(scaled$information,block$information/N,tolerance=1e-12)
        expect_equal(scaled$meat,block$meat/N,tolerance=1e-12)
    }
})

test_that("partial sandwiches preserve frequency expansion and whole-cluster scores", {
    for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) {
        a <- partial_robust_fixture(link);effective <- a$counts*a$weights
        ii <- rep(seq_len(nrow(a$x)),rowSums(effective))
        yy <- unlist(lapply(seq_len(nrow(a$x)),function(i) rep(1:3,effective[i,])))
        fit <- fastglm_ordinal(a$x,a$counts,weights=a$weights,offset=a$offset,
            nonparallel="z",link=link,firth=br,tol=1e-10)
        expanded <- fastglm_ordinal(a$x[ii,,drop=FALSE],ordered(yy,levels=1:3),
            offset=a$offset[ii],nonparallel="z",link=link,firth=br,tol=1e-10)
        expect_true(fit$converged);expect_true(expanded$converged)
        for(type in c("HC0","HC1")) {
            V <- fastglm_vcov(fit,type,block.size=4L)
            expect_equal(c(V),c(fastglm_vcov(expanded,type,block.size=29L)),tolerance=2e-8)
            clustered <- fastglm_vcov(fit,type,cluster=a$cluster,block.size=4L)
            expect_equal(c(clustered),c(fastglm_vcov(expanded,type,cluster=a$cluster[ii],block.size=29L)),tolerance=2e-8)
            expect_equal(attr(clustered,"clusters"),4L)
        }
        V <- fastglm_vcov(fit)
        expect_equal(vcov(fit,type="HC0"),V)
        expect_equal(c(vcov(fit,component="nonparallel",type="HC0")),c(V[4:5,4:5]))
        expect_match(attr(V,"bread"),"observed negative score Jacobian")
        expect_equal(c(fastglm_vcov(fit,"HC1")),c(V)*fit$frequency/(fit$frequency-fit$rank),tolerance=1e-12)
        if(br) expect_match(attr(V,"covariance.scope"),"first-order ordinary-score")
        original <- fastglm_vcov(fit,cluster=a$cluster,cadjust=FALSE)
        corrected <- fastglm_vcov(fit,"HC1",cluster=a$cluster)
        expect_equal(c(corrected),c(original)*(4/3)*(fit$frequency-1)/(fit$frequency-fit$rank),tolerance=1e-12)
        expect_equal(summary(fit,vcov.=V)$coefficients[,2L],sqrt(diag(V)),ignore_attr=TRUE)
        shifted <- sweep(a$x,2L,c(3,-2),"+")
        changed <- sweep(shifted,2L,c(.01,-20),"*")
        recoded <- fastglm_ordinal(changed,a$counts,weights=a$weights,offset=a$offset,
            nonparallel="z",link=link,firth=br,tol=1e-10)
        T <- diag(5L);T[1:2,3L] <- 3;T[3L,3L] <- 100
        T[1L,4L] <- T[2L,5L] <- -2;T[4L,4L] <- T[5L,5L] <- -.05
        expect_equal(c(fastglm_vcov(recoded)),c(T%*%V%*%t(T)),tolerance=2e-8)
    }
})

test_that("partial fitted sandwiches use observed bread at ML and meanBR estimates", {
    skip_if_not_installed("numDeriv")
    for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) {
        a <- partial_robust_fixture(link)
        f <- fastglm_ordinal(a$x,a$counts,weights=a$weights,offset=a$offset,
            nonparallel="z",link=link,firth=br,tol=1e-10)
        logprob <- function(b) partial_robust_reference(b,a$x,a$offset,2L,1L,2L,link)
        H <- -numDeriv::hessian(function(b)sum((a$counts*a$weights)*logprob(b)),coef(f))
        M <- matrix(0,5L,5L);rows <- matrix(0,nrow(a$x),5L)
        for(i in seq_len(nrow(a$x))) {
            S <- numDeriv::jacobian(function(b)logprob(b)[i,],coef(f))
            M <- M+crossprod(S,S*(a$counts[i,]*a$weights[i]))
            rows[i,] <- colSums(S*(a$counts[i,]*a$weights[i]))
        }
        B <- solve(H)
        expect_equal(c(fastglm_vcov(f)),c(B%*%M%*%B),tolerance=3e-7)
        cluster <- rowsum(rows,a$cluster,reorder=FALSE)
        expect_equal(c(fastglm_vcov(f,cluster=a$cluster,cadjust=FALSE)),
            c(B%*%crossprod(cluster)%*%B),tolerance=3e-7)
    }
})

test_that("partial covariance rejects invalid requests and aligns formula cluster IDs", {
    a <- partial_robust_fixture();d <- data.frame(a$x,a$counts,w=a$weights,off=a$offset)
    d$x[2L] <- NA;groups <- a$cluster;groups[d$w==0] <- NA
    f <- fastglm_ordinal(cbind(low,middle,high)~x+z+offset(off),d,weights=w,
        nonparallel=~z,na.action=na.exclude)
    expect_equal(fastglm_vcov(f,cluster=groups),fastglm_vcov(f,cluster=groups[-2L]))
    expect_error(vcov(f,cluster=groups),"require type")
    expect_error(fastglm_vcov(f,"HC2"),"HC0.*HC1")
    expect_error(fastglm_vcov(f,cluster=rep(1,nrow(f$x))),"at least two")
    expect_error(fastglm_vcov(f,cluster=rep(NA,nrow(f$x))),"positive-frequency")
    expect_error(fastglm_vcov(f,block.size=0),"block.size")
    expect_error(fastglm_vcov(f,cadjust=NA),"cadjust")
    f$converged <- FALSE;expect_error(fastglm_vcov(f),"converged")
})

test_that("partial observed curvature uses stable tails and refuses unresolved subtraction", {
    object <- function(alpha,link) {
        q <- length(alpha)
        structure(list(alpha=alpha,beta=numeric(),common=integer(),nonparallel=1L,
            coefficients=c(alpha,numeric(q)),link=link,
            working=list(alpha=alpha,beta=numeric(),gamma=matrix(0,1L,q),center=0,scale=1)),
            class=c("fastglm_ordinal_partial","fastglm_ordinal"))
    }
    evaluate <- function(alpha,link,counts) fastglm:::.ordinal_partial_sandwich_block(
        object(alpha,link),matrix(0,1L,1L),0,integer(),matrix(counts,1L),1)
    # Independent 90-digit continued-fraction reference (NIST DLMF 7.9.1),
    # reproduced by benchmarks/ordinal-partial-sandwich/normal-tail-reference.py.
    tail <- evaluate(-40,"probit",c(1,0))
    expect_equal(tail$score[1L],40.024968847207263723,tolerance=1e-11)
    expect_equal(tail$information[1L,1L],.9993773316214086112,tolerance=2e-8)
    mirrored <- evaluate(40,"probit",c(0,1))
    expect_equal(mirrored$score[1L],-tail$score[1L],tolerance=1e-12)
    expect_equal(mirrored$information,tail$information,tolerance=1e-12)
    expect_error(evaluate(-1e6,"probit",c(1,0)),"curvature is numerically unresolved")
    expect_error(evaluate(c(10,10+1e-12),"probit",c(0,1,0)),"curvature is numerically unresolved")
    cloglog <- evaluate(c(1,800),"cloglog",c(0,1,0))
    expect_equal(cloglog$score,c(-exp(1),0,0,0),tolerance=1e-12)
    expect_equal(cloglog$information[1L,1L],exp(1),tolerance=1e-12)
    expect_true(all(is.finite(cloglog$information)))
    low <- evaluate(c(-800,-799),"cloglog",c(0,1,0))
    expect_true(all(is.finite(low$information)))
    expect_equal(sum(low$score[1:2]),1,tolerance=1e-12)
    gap <- 1e-8;logit <- evaluate(c(0,gap),"logit",c(0,1,0))
    expect_equal(logit$information[1:2,1:2]*gap^2,matrix(c(1,-1,-1,1),2L),tolerance=1e-12)
})
