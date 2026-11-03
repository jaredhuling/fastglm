test_that("prepared ordinal fits reuse setup and agree with direct fits", {
    set.seed(8101)
    n <- 100L;x <- cbind(a=rnorm(n),b=runif(n,-1,1))
    y <- ordered(cut(.5*x[,1]+rlogis(n),c(-Inf,-.7,.6,Inf)))
    y2 <- ordered(cut(.2*x[,2]+rlogis(n),c(-Inf,-.7,.6,Inf)))
    w <- sample(0:3,n,TRUE);off <- runif(n,-.2,.2)
    for(br in c(FALSE,TRUE)) {
        design <- fastglm_ordinal_prepare(x,weights=w,offset=off,firth=br)
        fits <- fastglm_ordinal_batch(design,list(first=y,second=y2),tol=1e-10)
        expect_identical(names(fits),c("first","second"))
        expect_false(fits[[1]]$preparation.reused)
        expect_true(fits[[2]]$preparation.reused)
        expect_identical(design$cache$builds,1L)
        for(j in 1:2) {
            direct <- fastglm_ordinal(x,list(y,y2)[[j]],weights=w,offset=off,firth=br,tol=1e-10)
            expect_equal(coef(fits[[j]]),coef(direct),tolerance=1e-11)
            expect_equal(vcov(fits[[j]]),vcov(direct),tolerance=1e-11)
            expect_equal(predict(fits[[j]]),predict(direct),tolerance=1e-11)
            expect_identical(fits[[j]]$iterations,direct$iterations)
            expect_equal(fits[[j]]$effective$score,direct$effective$score,tolerance=1e-11)
        }
        again <- fastglm_ordinal(design,y,tol=1e-10)
        expect_true(again$preparation.reused)
        expect_equal(coef(again),coef(fits[[1]]),tolerance=1e-11)
    }
})

test_that("ordinal setup keys use effective totals and not response allocations", {
    set.seed(8102);n <- 45L
    x <- cbind(a=rnorm(n));w <- sample(2:5,n,TRUE)
    y <- ordered(rep(c("a","b","c"),length.out=n))
    design <- fastglm_ordinal_prepare(x,weights=w,levels=levels(y),firth=TRUE)
    first <- fastglm_ordinal(design,y)
    counts <- matrix(0,n,3L,dimnames=list(NULL,levels(y)))
    counts[,1L] <- 1;counts[,2L] <- 1;counts[,3L] <- w-2
    second <- fastglm_ordinal(design,counts,weights=rep(1,n))
    expect_true(second$preparation.reused)
    expect_identical(design$cache$builds,1L)
    direct <- fastglm_ordinal(x,counts,firth=TRUE)
    expect_equal(coef(second),coef(direct),tolerance=1e-11)
    expect_equal(second$working$center,first$working$center)
    weights <- rep(c(.5,2,1),length.out=n)
    equivalent <- fastglm_ordinal(design,counts/weights,weights=weights)
    expect_true(equivalent$preparation.reused)
    expect_equal(coef(equivalent),coef(second),tolerance=1e-11)
    counts[,1L] <- counts[,1L]+1
    changed <- fastglm_ordinal(design,counts,weights=rep(1,n))
    expect_false(changed$preparation.reused)
    expect_identical(design$cache$builds,2L)
    expect_equal(coef(changed),coef(fastglm_ordinal(x,counts,firth=TRUE)),tolerance=1e-11)
})

test_that("ordinal caches invalidate safely after mutable inputs and failed setup", {
    set.seed(8103);n <- 70L
    x <- cbind(a=rnorm(n),b=rnorm(n));y <- ordered(rep(c("a","b","c"),length.out=n))
    design <- fastglm_ordinal_prepare(x,firth=TRUE)
    original <- fastglm_ordinal(design,y)
    design$x[1L,1L] <- design$x[1L,1L]+1
    changed <- fastglm_ordinal(design,y)
    expect_false(changed$preparation.reused)
    expect_equal(coef(changed),coef(fastglm_ordinal(design$x,y,firth=TRUE)),tolerance=1e-11)
    expect_identical(design$cache$builds,2L)
    design$weights[1L] <- 3
    weighted <- fastglm_ordinal(design,y)
    expect_false(weighted$preparation.reused)
    expect_equal(coef(weighted),coef(fastglm_ordinal(design$x,y,weights=design$weights,firth=TRUE)),tolerance=1e-11)
    expect_identical(design$cache$builds,3L)
    off <- seq(-.3,.3,length.out=n)
    offset.fit <- fastglm_ordinal(design,y,offset=off)
    expect_true(offset.fit$preparation.reused)
    expect_identical(design$cache$builds,3L)
    unscaled <- fastglm_ordinal(design,y,control=list(standardize=FALSE))
    expect_false(unscaled$preparation.reused)
    expect_equal(coef(unscaled),coef(weighted),tolerance=1e-7)
    expect_identical(design$cache$builds,4L)
    fastglm_ordinal(design,y)
    valid <- design$x;builds <- design$cache$builds
    design$x[,2L] <- design$x[,1L]
    expect_error(fastglm_ordinal(design,y),"full rank")
    expect_error(fastglm_ordinal(design,y),"full rank")
    expect_identical(design$cache$builds,builds)
    design$x <- valid
    restored <- fastglm_ordinal(design,y)
    expect_true(restored$preparation.reused)
    expect_equal(coef(restored),coef(weighted),tolerance=1e-11)
    design$x[1L,1L] <- NaN
    expect_error(fastglm_ordinal(design,y),"finite dense")
    expect_error(fastglm_ordinal(design,y),"finite dense")
    expect_identical(design$cache$builds,builds)
    expect_equal(original$x,x)
})

test_that("prepared formulas retain prediction coding and explicit offset semantics", {
    set.seed(8104);n <- 90L
    data <- data.frame(x=rnorm(n),g=factor(rep(c("a","b","c"),30L)),
                       w=sample(1:3,n,TRUE),off=runif(n,-.2,.2),extra=runif(n,-.1,.1))
    data$y <- ordered(cut(.5*data$x+rlogis(n),c(-Inf,-.6,.9,Inf)))
    design <- fastglm_ordinal_prepare(~splines::ns(x,df=3)+g+offset(off),data,
                                      weights=w,offset=extra,firth=TRUE)
    fit <- fastglm_ordinal(design,data$y)
    ordinary <- update(fit,firth=FALSE)
    expect_equal(coef(ordinary),coef(fastglm_ordinal(design,data$y,firth=FALSE)),tolerance=1e-11)
    expect_error(update(fit,~.),"prepar")
    direct <- fastglm_ordinal(y~splines::ns(x,df=3)+g+offset(off),data,
                              weights=w,offset=extra,firth=TRUE)
    newdata <- data[c(2L,21L,35L),,drop=FALSE]
    expect_equal(coef(fit),coef(direct),tolerance=1e-11)
    expect_equal(predict(fit,newdata=newdata,se.fit=TRUE),
                 predict(direct,newdata=newdata,se.fit=TRUE),tolerance=1e-11)
    replacement <- seq(-.2,.2,length.out=n)
    changed <- fastglm_ordinal(design,data$y,offset=replacement)
    expect_true(changed$preparation.reused)
    expect_error(predict(changed,newdata=newdata),"explicit prediction offset")
    matrix.fit <- fastglm_ordinal(design$x,data$y,weights=data$w,offset=replacement,firth=TRUE)
    expect_equal(predict(changed,newdata=newdata,offset=rep(0,3L)),
                 predict(matrix.fit,newdata=design$x[c(2L,21L,35L),,drop=FALSE],offset=rep(0,3L)))
    design$offset <- replacement
    modified <- fastglm_ordinal(design,data$y)
    expect_error(predict(modified,newdata=newdata),"explicit prediction offset")
    refitted <- fastglm:::.ordinal_bootstrap_refit(modified,fastglm:::.ordinal_bootstrap_input(modified))
    expect_error(predict(refitted,newdata=newdata),"explicit prediction offset")
    expect_equal(predict(refitted,newdata=newdata,offset=rep(0,3L)),
                 predict(modified,newdata=newdata,offset=rep(0,3L)),tolerance=1e-9)
    expect_error(fastglm_ordinal_prepare(y~x,data),"one-sided")
    data$x[1L] <- NA_real_
    expect_error(fastglm_ordinal_prepare(~x,data),"missing")
})

test_that("ordinal prepared caches serialize and handle zero rows and integer matrices", {
    set.seed(8105);n <- 60L
    x <- cbind(a=sample(1:5,n,TRUE),b=sample(1:7,n,TRUE))
    y <- ordered(rep(c("a","b","c"),20L));weights <- rep(1,n);weights[1L] <- 0
    design <- fastglm_ordinal_prepare(x,weights=weights,firth=TRUE)
    fit <- fastglm_ordinal(design,y)
    restored <- unserialize(serialize(design,NULL))
    again <- fastglm_ordinal(restored,y)
    expect_true(again$preparation.reused)
    expect_equal(coef(again),coef(fit),tolerance=1e-11)
    expect_identical(restored$cache$builds,1L)
    restored$x[1L,] <- 1e200
    huge <- fastglm_ordinal(restored,y)
    expect_equal(coef(huge),coef(fit),tolerance=1e-11)
    expect_false(huge$preparation.reused)
    expect_true(fastglm_ordinal(restored,y)$preparation.reused)
    expect_true(is.integer(x))
    empty <- fastglm_ordinal_prepare(matrix(numeric(),1L,0L),firth=TRUE)
    a <- fastglm_ordinal(empty,matrix(c(2,3,4),1L))
    expect_warning(b <- fastglm_ordinal(empty,matrix(c(4,0,5),1L)),"collapsed")
    expect_true(b$preparation.reused)
    expect_equal(as.numeric(predict(a)),c(2.5,3,4.5)/10,tolerance=1e-8)
    expect_equal(as.numeric(predict(b)),c(4.5,0,5.5)/10,tolerance=1e-8)
})

test_that("ordinal batches support response-specific fitting vectors and category orders", {
    set.seed(8106);n <- 65L
    x <- cbind(a=rnorm(n))
    y <- list(one=sample(1:3,n,TRUE),two=sample(1:4,n,TRUE))
    weights <- cbind(sample(1:2,n,TRUE),sample(1:3,n,TRUE))
    offset <- cbind(runif(n,-.2,.2),runif(n,-.4,.4))
    orders <- list(c("low","middle","high"),letters[1:4])
    design <- fastglm_ordinal_prepare(x,firth=TRUE)
    fits <- fastglm_ordinal_batch(design,y,weights=weights,offset=offset,levels=orders)
    expect_error(update(fits[[1L]],firth=FALSE),"refit")
    expect_identical(design$cache$builds,2L)
    for(j in 1:2) {
        direct <- fastglm_ordinal(x,y[[j]],weights=weights[,j],offset=offset[,j],levels=orders[[j]],firth=TRUE)
        expect_equal(coef(fits[[j]]),coef(direct),tolerance=1e-11)
        expect_identical(fits[[j]]$levels,orders[[j]])
    }
    expect_error(fastglm_ordinal_batch(design,matrix(1,n,3)),"list of ordinal responses")
    expect_error(fastglm_ordinal_batch(design,y,weights=matrix(1,n,1)),"one column per response")
    expect_error(fastglm_ordinal_batch(design,y,levels=list(letters[1:3])),"one element per response")
    expect_identical(fastglm_ordinal_batch(design,list()),list())
})
