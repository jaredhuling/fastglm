test_that("ordinal formula and matrix fits have consistent model methods", {
    skip_if_not_installed("MASS")
    set.seed(92601)
    d <- data.frame(x=rnorm(240),g=factor(rep(c("a","b"),120)),
                    off=runif(240,-.2,.2),w=rep(c(1,1,2),80))
    latent <- .5*d$x-.3*(d$g=="b")+d$off+rlogis(240)
    d$y <- ordered(cut(latent,c(-Inf,-.5,.7,Inf),labels=c("low","middle","high")))
    fit <- fastglm_ordinal(y~x+g+offset(off),d,weights=w,tol=1e-10)
    ref <- MASS::polr(y~x+g+offset(off),d,weights=w,Hess=TRUE,
                     control=list(reltol=1e-12,maxit=1000))
    expect_true(fit$converged)
    expect_equal(unname(coef(fit,component="location")),unname(coef(ref)),tolerance=2e-6)
    expect_equal(unname(fit$alpha),unname(ref$zeta),tolerance=2e-6)
    expect_equal(unname(predict(fit)),unname(predict(ref,type="probs")),tolerance=1e-6)
    expect_equal(as.numeric(logLik(fit)),as.numeric(logLik(ref)),tolerance=1e-10)
    expect_equal(nobs(fit),sum(d$w))
    expect_equal(weights(fit),d$w)
    expect_equal(model.frame(fit)$y,d$y)
    expect_equal(dim(model.matrix(fit)),c(240L,2L))
    expect_s3_class(summary(fit),"summary.fastglm_ordinal")
    expect_output(print(summary(fit)),"Location coefficients")
    expect_output(print(fit),"Thresholds")
    expect_equal(fitted(fit),predict(fit))
    expect_equal(deviance(fit),-2*as.numeric(logLik(fit)))
    expect_equal(dim(vcov(fit)),c(4L,4L))
    expect_equal(unname(coef(fit,component="threshold")),unname(fit$alpha))
    expect_equal(confint(fit,parm="x")[1,],
        unname(fit$beta["x"])+qnorm(c(.025,.975))*sqrt(vcov(fit)["x","x"]),
        ignore_attr=TRUE)
    matrix.fit <- fastglm_ordinal(model.matrix(fit),d$y,weights=d$w,
                                  offset=d$off,firth=TRUE,tol=1e-10)
    formula.fit <- fastglm_ordinal(y~x+g+offset(off),d,weights=w,firth=TRUE,tol=1e-10)
    expect_equal(coef(matrix.fit),coef(formula.fit),tolerance=1e-12)
    expect_equal(vcov(matrix.fit),vcov(formula.fit),tolerance=1e-12)
    expect_equal(coef(update(fit,firth=TRUE)),coef(formula.fit),tolerance=1e-12)
    expect_true(update(fit,.~.-g)$converged)
    expect_true(is.call(update(fit,firth=TRUE,evaluate=FALSE)))
    diagnostics <- fastglm_diagnostics(formula.fit)
    expect_true(diagnostics$converged)
    expect_lt(diagnostics$remaining.predictor.step,1e-10)
})

test_that("ordinal prediction preserves offsets, missing rows, and declared order", {
    set.seed(92602)
    d <- data.frame(x=rnorm(180),g=factor(rep(c("a","b"),90)),
                    off=runif(180,-.2,.2),extra=runif(180,-.1,.1))
    d$y <- ordered(cut(.4*d$x+d$off+d$extra+rlogis(180),
                       c(-Inf,-.5,.7,Inf),labels=c("low","middle","high")))
    d$y[2] <- NA; d$x[5] <- NA
    fit <- fastglm_ordinal(y~x+g+offset(off),d,offset=extra,
                           na.action=na.exclude,firth=TRUE,tol=1e-10)
    prediction <- predict(fit,se.fit=TRUE)
    expect_equal(dim(prediction$fit),c(180L,3L))
    expect_true(all(is.na(prediction$fit[c(2,5),])))
    expect_true(all(is.na(prediction$se.fit[c(2,5),])))
    expect_true(all(is.finite(prediction$fit[-c(2,5),])))
    new <- d[c(1,3,4,6),]
    new$off <- new$off+.2; new$extra <- new$extra-.1
    x <- model.matrix(~x+g,new)[,-1,drop=FALSE]
    location <- drop(x%*%fit$beta)+new$off+new$extra
    expect_equal(unname(predict(fit,new,type="link")),unname(location),tolerance=1e-12)
    expected <- plogis(outer(-location,fit$alpha,"+"))
    expect_equal(unname(predict(fit,new,type="cumulative")),unname(expected),tolerance=1e-12)
    classes <- predict(fit,new,type="class")
    expect_true(is.ordered(classes))
    expect_identical(levels(classes),levels(d$y))
    expect_equal(dim(predict(fit,d[FALSE,])),c(0L,3L))
    expect_equal(predict(fit,new,type="response"),predict(fit,new,type="probability"))
    expect_error(predict(fit,new,type="class",se.fit=TRUE),"class predictions")
    unretained <- fastglm_ordinal(y~x+g,d,model=FALSE,firth=TRUE)
    expect_error(model.frame(unretained),"not retained")
    expect_error(fastglm_ordinal(matrix(1:6,3,2),factor(c("a","b","c"))),"ordered")
    expect_error(fastglm_ordinal(as(matrix(1:6,3,2),"dgCMatrix"),ordered(c("a","b","c"))),"dense")
    expect_error(predict(fit,transform(new,g="unknown")),"new level")
})

test_that("ordinal probability uncertainty and averages agree with numerical derivatives", {
    set.seed(92603)
    X <- cbind(x=rnorm(240,mean=3,sd=.4),z=rnorm(240,sd=2))
    off <- runif(240,-.2,.2)
    y <- ordered(cut(.3*X[,1]-.2*X[,2]+off+rlogis(240),
                     c(-Inf,.2,1.2,Inf),labels=c("low","middle","high")))
    fit <- fastglm_ordinal(X,y,offset=off,firth=TRUE,tol=1e-11)
    new <- X[1:7,,drop=FALSE]; new.off <- off[1:7]
    b <- coef(fit); q <- length(fit$alpha); v <- vcov(fit)
    probability <- function(b,x=new,offset=new.off) {
        cumulative <- plogis(outer(-drop(x%*%b[-seq_len(q)])-offset,b[seq_len(q)],"+"))
        cbind(cumulative,1)-cbind(0,cumulative)
    }
    derivative <- function(fun,b) {
        out <- matrix(0,length(fun(b)),length(b))
        for(j in seq_along(b)) {
            h <- 1e-5*(1+abs(b[j])); upper <- lower <- b
            upper[j] <- upper[j]+h; lower[j] <- lower[j]-h
            out[,j] <- (fun(upper)-fun(lower))/(2*h)
        }
        out
    }
    g <- derivative(function(b)as.numeric(probability(b)),b)
    expected.se <- matrix(sqrt(rowSums((g%*%v)*g)),nrow(new))
    pred <- predict(fit,newdata=new,offset=new.off,se.fit=TRUE)
    expect_equal(unname(pred$fit),unname(probability(b)),tolerance=1e-12)
    expect_equal(unname(pred$se.fit),unname(expected.se),tolerance=1e-8)
    average.weights <- seq(.5,2,length.out=nrow(new))
    # Newdata without an explicit offset uses zero, as in the matrix API.
    average.fun <- function(b,x=new) drop(crossprod(average.weights,probability(b,x,rep(0,nrow(x)))))/sum(average.weights)
    average <- fastglm_average(fit,newdata=new,weights=average.weights)
    g <- derivative(average.fun,b)
    expect_equal(average$estimate,unname(average.fun(b)),tolerance=1e-12)
    expect_equal(average$std.error,sqrt(rowSums((g%*%v)*g)),tolerance=1e-8)
    expect_equal(fastglm_average(fit,newdata=new,weights=average.weights,vcov.=vcov(fit)),
                 average,tolerance=1e-10)
    changed <- new; changed[,1] <- changed[,1]+.4
    contrast <- function(x) {x[,1] <- x[,1]+.4;x}
    contrast.fun <- function(b) average.fun(b,changed)-average.fun(b)
    g <- derivative(contrast.fun,b)
    difference <- fastglm_average(fit,newdata=new,weights=average.weights,contrast=contrast)
    expect_equal(difference$estimate,unname(contrast.fun(b)),tolerance=1e-12)
    expect_equal(difference$std.error,sqrt(rowSums((g%*%v)*g)),tolerance=1e-8)
    expect_equal(sum(difference$estimate),0,tolerance=1e-12)
    cumulative <- fastglm_average(fit,newdata=new,weights=average.weights,type="cumulative")
    expect_equal(cumulative$estimate,cumsum(average$estimate)[seq_len(q)],tolerance=1e-12)
    expect_error(fastglm_average(fit,newdata=new,weights=-average.weights),"averaging weights")
})

test_that("ordinal grouped deviance and nonconvergence are explicit", {
    counts <- matrix(c(4,3,5,2,6,4),2,3,byrow=TRUE,
                     dimnames=list(NULL,c("low","middle","high")))
    X <- matrix(c(-.5,.5),ncol=1,dimnames=list(NULL,"x"))
    fit <- fastglm_ordinal(X,counts,firth=TRUE)
    saturated <- sum(counts*log(counts/rowSums(counts)))
    expect_equal(deviance(fit),2*(saturated-as.numeric(logLik(fit))),tolerance=1e-12)
    expect_equal(nobs(fit),sum(counts))
    x <- matrix(seq(-2,2,length.out=60),ncol=1,dimnames=list(NULL,"x"))
    y <- ordered(findInterval(x[,1],c(-.4,.7))+1,levels=1:3)
    expect_warning(incomplete <- fastglm_ordinal(x,y,maxit=1),"did not converge")
    expect_false(incomplete$converged)
    expect_error(vcov(incomplete),"converged fit")
    expect_error(predict(incomplete,se.fit=TRUE),"converged fit")
    expect_error(fastglm_average(incomplete),"converged fit")
    expect_true(all(is.na(summary(incomplete)$location[,"Std. Error"])))
    expect_true(all(is.finite(predict(incomplete))))
})
