partial_boundary_fixture <- function() {
    list(z=rbind(c(.2,-.4),c(1.1,.3),c(-.5,.8)),
        counts=rbind(c(4,0,3,2),c(2,5,0,4),c(1,3,2,5)),
        weights=c(2,1,3),offset=c(.13,-.21,.08))
}

partial_boundary_certificate <- function(d,blocks=list(seq_len(nrow(d$z))),common=0L,
                                         link="logit",firth=TRUE) {
    state <- .ordinal_partial_boundary_start(common,ncol(d$z),ncol(d$counts),firth,link)
    for(i in blocks) .ordinal_partial_boundary_add(state,d$z[i,,drop=FALSE],integer(),
        d$counts[i,,drop=FALSE],d$weights[i],d$offset[i])
    .ordinal_partial_boundary_finish(state)
}

test_that("saturated logit boundary certification respects patterns and frequency blocks", {
    d <- partial_boundary_fixture()
    certificate <- partial_boundary_certificate(d)
    expect_equal(certificate$patterns,3L)
    expect_equal(certificate$empty.pattern.categories,unname(matrix(c(1,2,2,3),2L)),ignore_attr=TRUE)
    expect_match(certificate$message,"no mean-bias-reduced solution")
    blocked <- partial_boundary_certificate(d,list(1L,2L,3L))
    expect_equal(blocked,certificate)
    transformed <- d
    transformed$z <- sweep(d$z%*%matrix(c(2,.3,-.4,.8),2L),2L,c(1e4,-2e4),"+")
    expect_equal(partial_boundary_certificate(transformed),certificate)
    # A coded expansion has the same pooled frequency counts and certificate.
    rows <- rep(1:3,rowSums(d$counts*d$weights))
    y <- unlist(lapply(1:3,function(i) rep(1:4,d$counts[i,]*d$weights[i])))
    state <- .ordinal_partial_boundary_start(0L,2L,4L,TRUE,"logit")
    for(at in split(seq_along(rows),ceiling(seq_along(rows)/7L)))
        .ordinal_partial_boundary_add(state,d$z[rows[at],,drop=FALSE],as.integer(y[at]),
            matrix(numeric(),0L,0L),rep(1,length(at)),d$offset[rows[at]])
    expect_equal(.ordinal_partial_boundary_finish(state),certificate)
    # Exact zero effective frequency must be ignored, including multiplication
    # underflow. Such a row cannot create an additional predictor pattern.
    zero <- d;zero$z <- rbind(d$z,c(9,9));zero$counts <- rbind(d$counts,rep(1,4))
    zero$weights <- c(d$weights,0);zero$offset <- c(d$offset,20)
    expect_equal(partial_boundary_certificate(zero),certificate)
    zero$counts[4L,] <- 1e-300;zero$weights[4L] <- 1e-300
    expect_equal(partial_boundary_certificate(zero),certificate)
})

test_that("the boundary certificate declines models outside its proved scope", {
    d <- partial_boundary_fixture()
    expect_null(partial_boundary_certificate(d,link="probit"))
    expect_null(partial_boundary_certificate(d,link="cloglog"))
    expect_null(partial_boundary_certificate(d,firth=FALSE))
    expect_null(partial_boundary_certificate(d,common=1L))
    extra <- d;extra$z <- rbind(d$z,c(.4,.9));extra$counts <- rbind(d$counts,c(1,1,1,1))
    extra$weights <- c(d$weights,1);extra$offset <- c(d$offset,0)
    expect_null(partial_boundary_certificate(extra))
    # Tiny offset differences are real differences, not approximate duplicates.
    changed <- d;changed$z <- rbind(d$z,d$z[1L,]);changed$counts <- rbind(d$counts,d$counts[1L,])
    changed$weights <- c(d$weights,1);changed$offset <- c(d$offset,d$offset[1L]+2^-52)
    expect_null(partial_boundary_certificate(changed))
    # Centering before scaling prevents false full-rank classification after a
    # large location shift of exactly affinely dependent columns.
    singular <- d;singular$z <- cbind(c(0,2,4),1e16+c(0,2,4))
    expect_null(partial_boundary_certificate(singular))
    positive <- d;positive$counts[positive$counts==0] <- 1
    expect_null(partial_boundary_certificate(positive))
    endpoint <- d;endpoint$counts <- rbind(c(0,2,3,4),c(2,3,4,0),c(1,2,3,4))
    expect_null(partial_boundary_certificate(endpoint))
    binary <- d;binary$counts <- d$counts[,c(1,4)]
    expect_null(partial_boundary_certificate(binary))
})

test_that("saturated logit adjusted equations equal endpoint-half-count scores", {
    d <- partial_boundary_fixture();q <- 3L
    b <- c(-.9,.15,1.2,.1,-.15,.2,.05,-.1,.12)
    native <- fastglm:::ordinal_partial_native_evaluate(matrix(numeric(),3L,0L),d$z,
        integer(),d$counts,d$weights,d$offset,4L,b,TRUE)
    eta <- matrix(b[1:3],3L,3L,byrow=TRUE)-d$z%*%matrix(b[4:9],2L,3L)-d$offset
    F <- plogis(eta);P <- cbind(F,1)-cbind(0,F)
    pseudo <- d$counts*d$weights
    pseudo[,c(1L,4L)] <- pseudo[,c(1L,4L)]+.5
    reference <- numeric(9L)
    for(i in 1:3) for(j in 1:3) {
        direction <- numeric(9L);direction[j] <- 1
        direction[q+(j-1L)*2L+1:2] <- -d$z[i,]
        score <- F[i,j]*(1-F[i,j])*(pseudo[i,j]/P[i,j]-pseudo[i,j+1L]/P[i,j+1L])
        reference <- reference+direction*score
    }
    expect_equal(drop(native$score),reference,tolerance=1e-11)
    target <- pseudo/rowSums(pseudo)
    expect_equal(target[cbind(c(1L,2L),c(2L,3L))],c(0,0))
    cumulative <- t(apply(target,1L,cumsum))[,1:3]
    eta.boundary <- qlogis(cumulative)
    expect_equal(eta.boundary[1L,1L],eta.boundary[1L,2L])
    expect_equal(eta.boundary[2L,2L],eta.boundary[2L,3L])
    mapped <- solve(cbind(1,-d$z),eta.boundary+d$offset)
    expect_true(all(is.finite(mapped)))
    expect_equal(cbind(1,-d$z)%*%mapped-d$offset,eta.boundary,tolerance=1e-12,ignore_attr=TRUE)
})

test_that("saturated logit models with positive intermediate counts retain their exact interior root", {
    d <- partial_boundary_fixture();d$counts[d$counts==0] <- 1
    colnames(d$z) <- c("a","b")
    fit <- fastglm_ordinal(d$z,d$counts,weights=d$weights,offset=d$offset,
        nonparallel=1:2,firth=TRUE,tol=1e-10)
    pseudo <- d$counts*d$weights;pseudo[,c(1L,4L)] <- pseudo[,c(1L,4L)]+.5
    expected <- pseudo/rowSums(pseudo)
    expect_true(fit$converged)
    expect_equal(predict(fit),expected,tolerance=1e-10,ignore_attr=TRUE)
})

test_that("maximum likelihood at the ordering boundary stops promptly with an explanation", {
    set.seed(321)
    n <- 1000L;p <- 10L
    x <- matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    alpha <- qlogis(seq_len(4L)/5L);eta <- drop(x%*%(0.6*(-1)^seq_len(p)/sqrt(seq_len(p))))
    # A design whose slopes change quickly across thresholds, relative to their gaps.
    u <- runif(n);code <- rep(5L,n)
    for(j in 4:1) code[u<=plogis(alpha[j]-eta-0.4*(j-2.5)*(x[,2]-x[,3]))] <- j
    y <- ordered(code,levels=seq_len(5L))
    expect_warning(fit <- fastglm_ordinal(x,y,nonparallel=c("x2","x3"),maxit=500L),"ordering boundary")
    expect_false(fit$converged)
    expect_identical(fit$convergence.reason,"ordering_boundary")
    expect_lt(fit$iterations,200L)
    expect_lt(fit$minimum.gap,1e-8)
})
