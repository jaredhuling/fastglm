test_that("ordinal pair and cluster bootstraps preserve row frequency semantics", {
    set.seed(7001)
    n <- 70L
    x <- cbind(a=rnorm(n),b=runif(n,-1,1))
    offset <- runif(n,-.3,.3)
    y <- ordered(cut(.4*x[,1]-.3*x[,2]+offset+rlogis(n),
                     c(-Inf,-.6,.7,Inf)),labels=c("low","middle","high"))
    weights <- sample(0:3,n,TRUE)
    fit <- fastglm_ordinal(x,y,weights=weights,offset=offset,firth=TRUE)
    before <- .Random.seed
    pairs <- fastglm_bootstrap(fit,R=4L,seed=701,type="pairs")
    expect_identical(.Random.seed,before)
    expect_true(pairs$inference.available)
    expect_identical(pairs$sampling.unit,"stored rows")
    set.seed(pairs$seeds[1L]); i <- sample.int(n,n,replace=TRUE)
    direct <- fastglm_ordinal(x[i,,drop=FALSE],y[i],weights=weights[i],
                              offset=offset[i],firth=TRUE)
    expect_equal(pairs$replicates[1L,],coef(direct),tolerance=1e-9)
    expect_equal(pairs$diagnostics$frequency[1L],sum(weights[i]))
    expect_equal(pairs$std.error,apply(pairs$replicates,2L,sd))
    expect_equal(unname(pairs$conf.int),unname(t(apply(pairs$replicates,2L,quantile,
        probs=c(.025,.975),names=FALSE))))

    group <- rep(seq_len(14L),each=5L)
    clusters <- fastglm_bootstrap(fit,R=4L,seed=702,type="cluster",cluster=group)
    units <- split(seq_len(n),match(group,unique(group)))
    set.seed(clusters$seeds[1L])
    i <- unlist(units[sample.int(length(units),length(units),replace=TRUE)],use.names=FALSE)
    direct <- fastglm_ordinal(x[i,,drop=FALSE],y[i],weights=weights[i],
                              offset=offset[i],firth=TRUE)
    expect_equal(clusters$replicates[1L,],coef(direct),tolerance=1e-9)
    expect_equal(clusters$diagnostics$frequency[1L],sum(weights[i]))
    expect_error(fastglm_bootstrap(fit,R=2L,type="cluster",cluster=rep(1,n)),"at least two")
    expect_error(fastglm_bootstrap(fit,R=2L,type="pairs",cluster=group),"only for type")
})

test_that("ordinal grouped pair resampling copies counts and weights together", {
    set.seed(7002)
    x <- cbind(a=rnorm(18L)); offset <- runif(18L,-.2,.2)
    effective <- matrix(sample(1:9,54L,TRUE),18L,3L,
                        dimnames=list(NULL,c("a","b","c")))
    weights <- rep(c(.5,2,1),6L)
    raw <- effective/weights
    fit <- fastglm_ordinal(x,raw,weights=weights,offset=offset,firth=TRUE)
    result <- fastglm_bootstrap(fit,R=3L,seed=703)
    set.seed(result$seeds[1L]); i <- sample.int(nrow(x),nrow(x),replace=TRUE)
    direct <- fastglm_ordinal(x[i,,drop=FALSE],raw[i,,drop=FALSE],weights=weights[i],
                              offset=offset[i],firth=TRUE)
    expect_equal(result$replicates[1L,],coef(direct),tolerance=1e-9)
    expect_equal(result$diagnostics$frequency[1L],sum(effective[i,,drop=FALSE]))
    expect_true(result$inference.available)
})

test_that("ordinal parametric refits use effective count totals with unit weights", {
    set.seed(7003)
    x <- cbind(a=rnorm(20L)); offset <- runif(20L,-.4,.4)
    counts <- matrix(sample(2:7,60L,TRUE),20L,3L,
                     dimnames=list(NULL,c("a","b","c")))
    weights <- rep(c(.5,2),10L)
    counts[1L,] <- 0
    fit <- fastglm_ordinal(x,counts/weights,weights=weights,offset=offset,firth=TRUE)
    result <- fastglm_bootstrap(fit,R=3L,seed=704,type="parametric")
    set.seed(result$seeds[1L]); simulated <- fastglm:::.ordinal_simulate_counts(fit)
    direct <- fastglm_ordinal(x,simulated,offset=offset,levels=fit$levels,firth=TRUE)
    expect_equal(rowSums(simulated),rowSums(counts))
    expect_true(all(simulated==round(simulated)))
    expect_identical(colnames(simulated),fit$levels)
    expect_equal(result$replicates[1L,],coef(direct),tolerance=1e-9)
    expect_equal(result$diagnostics$frequency,rep(sum(counts),3L))
    expect_true(result$inference.available)

    individual <- ordered(rep(c("a","b","c"),20L))
    xx <- cbind(a=rep(seq(-1,1,length.out=20L),3L))
    ww <- rep(1:3,20L)
    f <- fastglm_ordinal(xx,individual,weights=ww,firth=TRUE)
    b <- fastglm_bootstrap(f,R=3L,seed=705,type="parametric")
    set.seed(b$seeds[1L]); yy <- fastglm:::.ordinal_simulate_counts(f)
    manual <- fastglm_ordinal(xx,yy,levels=f$levels,firth=TRUE)
    expect_equal(rowSums(yy),as.numeric(ww))
    expect_equal(b$replicates[1L,],coef(manual),tolerance=1e-9)
    expect_equal(b$diagnostics$frequency,rep(sum(ww),3L))
    expect_false(any(b$diagnostics$boundary.changed))
})

test_that("ordinal bootstrap preserves formula prediction metadata and offsets", {
    set.seed(7004)
    n <- 120L
    d <- data.frame(x=rnorm(n),g=factor(rep(c("a","b"),n/2)),
                    w=sample(0:3,n,TRUE),off=runif(n,-.2,.2),extra=runif(n,-.2,.2))
    d$y <- ordered(cut(.4*d$x+.3*(d$g=="b")+d$off+d$extra+rlogis(n),
                       c(-Inf,-.8,.7,Inf)))
    fit <- fastglm_ordinal(y~splines::ns(x,df=3)+g+offset(off),data=d,
                           weights=w,offset=extra,firth=TRUE,model=FALSE)
    nd <- d[c(3L,15L,22L),,drop=FALSE]
    statistic <- function(f) as.numeric(predict(f,newdata=nd))
    result <- fastglm_bootstrap(fit,R=3L,statistic=statistic,seed=706)
    set.seed(result$seeds[1L]); i <- sample.int(n,n,replace=TRUE)
    direct <- fastglm_ordinal(fit$x[i,,drop=FALSE],fit$y[i],weights=fit$weights[i],
                              offset=fit$offset[i],firth=TRUE)
    tt <- delete.response(fit$terms)
    frame <- model.frame(tt,nd,xlev=fit$xlevels)
    xx <- model.matrix(tt,frame,contrasts.arg=fit$contrasts)
    xx <- xx[,attr(xx,"assign")!=0L,drop=FALSE]
    expected <- predict(direct,newdata=xx,offset=nd$off+nd$extra)
    expect_equal(unname(result$replicates[1L,]),as.numeric(expected),tolerance=1e-9)
    expect_true(result$inference.available)
})

test_that("ordinal bootstrap retains boundary point replicates without regular intervals", {
    x <- matrix(numeric(),2L,0L)
    counts <- rbind(c(10,0,2),c(2,1,10));colnames(counts) <- c("a","b","c")
    fit <- fastglm_ordinal(x,counts,firth=TRUE)
    expect_warning(result <- fastglm_bootstrap(fit,R=12L,seed=707),"uncertainty summaries are unavailable")
    expect_true(any(result$diagnostics$boundary.changed))
    expect_true(any(result$diagnostics$boundary))
    expect_true(all(result$diagnostics$converged))
    expect_true(all(result$diagnostics$statistic.ok))
    expect_true(all(is.finite(result$replicates)))
    expect_true(all(is.na(result$std.error)))
    expect_true(all(is.na(result$conf.int)))
    expect_identical(result$successful,12L)
    expect_true(all(vapply(result$diagnostics$empty.categories[result$diagnostics$boundary],
                           identical,logical(1),"b")))

    expect_warning(boundary <- fastglm_ordinal(matrix(numeric(),1L,0L),
        matrix(c(3,0,6),1L,dimnames=list(NULL,c("a","b","c"))),firth=TRUE),"collapsed")
    expect_warning(b <- fastglm_bootstrap(boundary,R=2L,seed=708),"original fit has empty")
    expect_true(all(is.finite(b$replicates)))
    expect_false(any(b$diagnostics$boundary.changed))
    expect_false(b$inference.available)
    expect_identical(b$original.empty.categories,"b")
    expect_warning(p <- fastglm_bootstrap(boundary,R=2L,seed=709,
        statistic=function(f)as.numeric(predict(f))),"original fit has empty")
    expect_equal(p$replicates[,2L],rep(0,2L))
    expect_true(all(is.na(p$std.error)))
})

test_that("only meanBR parametric bootstrap allows empty endpoints", {
    x <- matrix(numeric(),1L,0L)
    counts <- matrix(c(0,20,20),1L,dimnames=list(NULL,c("a","b","c")))
    fit <- fastglm_ordinal(x,counts,firth=TRUE)
    result <- fastglm_bootstrap(fit,R=29L,type="parametric",seed=110304)
    expect_true(result$inference.available)
    expect_true(any(lengths(result$diagnostics$empty.end.categories)>0L))
    expect_true(all(lengths(result$diagnostics$collapsed.categories)==0L))
    expect_true(all(is.finite(result$std.error)))
    expect_true(all(is.finite(result$conf.int)))
    expect_warning(pairs <- fastglm_bootstrap(fit,R=2L,type="pairs",seed=110304),
                   "original fit has empty categories")
    expect_false(pairs$inference.available)
    counts[1,1] <- 1
    ml <- fastglm_ordinal(x,counts)
    expect_warning(boot.ml <- fastglm_bootstrap(ml,R=29L,type="parametric",seed=110304),
                   "uncertainty summaries are unavailable")
    expect_false(boot.ml$inference.available)
    expect_true(any(!boot.ml$diagnostics$converged))
})

test_that("ordinal ML endpoint failures and nonconvergence remain visible", {
    x <- matrix(numeric(),2L,0L)
    counts <- rbind(c(0,4,2),c(2,4,0));colnames(counts) <- c("a","b","c")
    fit <- fastglm_ordinal(x,counts)
    expect_warning(result <- fastglm_bootstrap(fit,R=8L,seed=710),"uncertainty summaries are unavailable")
    failed <- !result$diagnostics$converged
    expect_true(any(failed))
    expect_true(all(result$diagnostics$boundary.changed[failed]))
    expect_true(all(result$diagnostics$has.empty.categories[failed]))
    expect_true(all(grepl("empty endpoint",result$diagnostics$error[failed])))
    expect_true(all(is.na(result$replicates[failed,,drop=FALSE])))
    expect_true(all(is.na(result$std.error)))

    set.seed(7005);xx <- cbind(a=rnorm(80L))
    yy <- ordered(cut(.8*xx[,1]+rlogis(80L),c(-Inf,-.5,.8,Inf)))
    f <- fastglm_ordinal(xx,yy,firth=TRUE)
    expect_warning(b <- fastglm_bootstrap(f,R=3L,seed=711,control=list(maxit=1L)),"uncertainty summaries are unavailable")
    expect_true(all(!b$diagnostics$converged))
    expect_true(all(b$diagnostics$statistic.ok))
    expect_true(all(is.finite(b$replicates)))
    expect_true(all(b$diagnostics$convergence.reason=="iteration_limit"))
    expect_true(all(is.na(b$conf.int)))
    expect_error(fastglm_bootstrap(f,R=2L,control=list(weights=rep(1,80L))),"may change only")
    expect_error(fastglm_bootstrap(f,R=2L,control=list(firth=FALSE)),"may change only")
})

test_that("ordinal bootstrap seeds are reproducible across worker scheduling", {
    skip_on_cran()
    old.kind <- RNGkind()
    on.exit(do.call(RNGkind,as.list(old.kind)))
    RNGkind("L'Ecuyer-CMRG","Box-Muller","Rejection")
    set.seed(7006)
    x <- cbind(a=rnorm(60L))
    counts <- matrix(sample(1:6,180L,TRUE),60L,3L)
    fit <- fastglm_ordinal(x,counts,firth=TRUE)
    before <- .Random.seed
    serial <- fastglm_bootstrap(fit,R=4L,type="parametric",seed=712,workers=1L)
    parallel <- fastglm_bootstrap(fit,R=4L,type="parametric",seed=712,workers=2L)
    expect_identical(.Random.seed,before)
    expect_identical(serial$seeds,parallel$seeds)
    expect_identical(serial$replicates,parallel$replicates)
    expect_identical(serial$diagnostics,parallel$diagnostics)
    expect_identical(serial$std.error,parallel$std.error)
    expect_identical(serial$rng.kind,RNGkind())
})

test_that("ordinal bootstrap aligns omitted rows and preserves exact cluster labels", {
    set.seed(7007)
    d <- data.frame(x=rnorm(90),y=ordered(rep(1:3,30)),w=1)
    d$x[c(2,5)] <- NA
    d$w[8] <- 0
    fit <- fastglm_ordinal(y~x,d,weights=w,na.action=na.exclude,firth=TRUE)
    group <- rep(1:15,each=6)
    group[c(2,5)] <- NA
    full <- fastglm_bootstrap(fit,R=3,type="cluster",cluster=group,seed=713)
    retained <- fastglm_bootstrap(fit,R=3,type="cluster",cluster=group[-c(2,5)],seed=713)
    expect_identical(full$replicates,retained$replicates)
    expect_identical(full$diagnostics,retained$diagnostics)
    group[8] <- NA
    expect_error(fastglm_bootstrap(fit,R=2,type="cluster",cluster=group),"retained sampling rows")

    x <- cbind(x=seq(-1,1,length.out=8))
    counts <- matrix(sample(1:6,24,TRUE),8,3)
    grouped <- fastglm_ordinal(x,counts,firth=TRUE)
    close <- rep(c(1,1+1e-15),each=4)
    labels <- rep(c("first","second"),each=4)
    numeric <- fastglm_bootstrap(grouped,R=3,type="cluster",cluster=close,seed=714)
    text <- fastglm_bootstrap(grouped,R=3,type="cluster",cluster=labels,seed=714)
    expect_identical(numeric$replicates,text$replicates)
    expect_identical(numeric$diagnostics,text$diagnostics)
})
