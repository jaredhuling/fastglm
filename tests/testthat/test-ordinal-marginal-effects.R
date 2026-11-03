test_that("marginal effects and their uncertainty agree with independent derivatives", {
    skip_if_not_installed("numDeriv")
    set.seed(110301)
    d <- data.frame(x=rnorm(220,2,.4),z=rnorm(220),off=runif(220,-.2,.2))
    d$y <- ordered(cut(.4*d$x-.1*d$x^2+.2*d$x*d$z+d$off+rlogis(220),
                       c(-Inf,-.4,.8,Inf)))
    fit <- fastglm_ordinal(y~x+I(x^2)+x:z+offset(off),d,firth=TRUE,tol=1e-10)
    new <- d[1:13,]; w <- seq_len(nrow(new))
    b <- coef(fit); q <- length(fit$alpha)
    # Independent probability calculation on the original coefficient scale.
    probability <- function(b,x) {
        location <- b["x"]*x+b["I(x^2)"]*x^2+b["x:z"]*x*new$z+new$off
        F <- plogis(outer(-location,b[seq_len(q)],"+"))
        cbind(F,1)-cbind(0,F)
    }
    effect <- function(b) {
        h <- 1e-4
        value <- (probability(b,new$x+h)-probability(b,new$x-h))/(2*h)
        colSums(w*value)/sum(w)
    }
    result <- fastglm_marginal_effects(fit,"x",newdata=new,weights=w)
    G <- numDeriv::jacobian(effect,b)
    expect_lt(max(abs(result$estimate-effect(b))),1e-9)
    expect_equal(result$std.error,sqrt(rowSums((G%*%vcov(fit))*G)),tolerance=2e-7)
    expect_equal(sum(result$estimate),0,tolerance=1e-14)
    cumulative <- fastglm_marginal_effects(fit,"x",newdata=new,weights=w,type="cumulative")
    expect_equal(cumulative$estimate,head(cumsum(result$estimate),-1),tolerance=1e-12)
    expect_equal(fastglm_marginal_effects(fit,"x",new,weights=w,vcov.=vcov(fit)),
                 result,tolerance=1e-9)
    expect_equal(fastglm_marginal_effects(fit,"x",new,weights=w,vcov.=4*vcov(fit))$std.error,
                 2*result$std.error,tolerance=1e-9)
    V <- fastglm_vcov(fit,type="HC0")
    expect_equal(fastglm_marginal_effects(fit,"x",new,weights=w,vcov.=V)$std.error,
                 sqrt(rowSums((G%*%V)*G)),tolerance=2e-7)
})

test_that("formula effects retain prediction bases, offsets, and block invariance", {
    set.seed(110302)
    d <- data.frame(x=runif(240,.5,3),g=factor(rep(c("a","b"),120)))
    d$y <- ordered(cut(log(d$x)+rlogis(240),c(-Inf,.2,1.4,Inf)))
    fit <- fastglm_ordinal(y~splines::ns(x,3)*g+offset(x/10),d,firth=TRUE)
    new <- d[1:21,]
    result <- fastglm_marginal_effects(fit,"x",new)
    high <- low <- new; high$x <- high$x+1e-4; low$x <- low$x-1e-4
    reference <- colMeans((predict(fit,high,offset=new$x/10)-
                           predict(fit,low,offset=new$x/10))/2e-4)
    expect_equal(result$estimate,unname(reference),tolerance=1e-8)
    fit$block.size <- 3L
    expect_equal(fastglm_marginal_effects(fit,"x",new),result,tolerance=1e-12)
    expect_error(fastglm_marginal_effects(fit,"x"),"original formula predictors")
    expect_error(fastglm_marginal_effects(fit,"g",new),"continuous numeric")
    transformed <- fastglm_ordinal(y~log(x),d,firth=TRUE)
    expect_error(fastglm_marginal_effects(transformed,"x"),"original formula predictors")
    expect_true(all(is.finite(fastglm_marginal_effects(transformed,"x",d)$std.error)))
    bare <- fastglm_ordinal(y~x,d,firth=TRUE,model=FALSE)
    expect_error(fastglm_marginal_effects(bare,"x"),"original formula predictors")
    expect_true(all(is.finite(fastglm_marginal_effects(bare,"x",d)$estimate)))
    retained <- fastglm_ordinal(y~x,d,firth=TRUE)
    expect_equal(fastglm_marginal_effects(retained,"x"),
                 fastglm_marginal_effects(retained,"x",d),tolerance=1e-12)
    centered <- fastglm_ordinal(y~I(x-mean(x)),d,firth=TRUE)
    expect_error(fastglm_marginal_effects(centered,"x",d),"rowwise formula terms")
    frozen <- fastglm_ordinal(y~scale(x),d,firth=TRUE)
    expect_equal(fastglm_marginal_effects(frozen,"x",d)$estimate,
                 fastglm_marginal_effects(retained,"x",d)$estimate,tolerance=1e-8)
})

test_that("matrix marginal effects preserve frequencies and reduced categories", {
    X <- matrix(seq(-1,1,length.out=12),ncol=1,dimnames=list(NULL,"x"))
    counts <- cbind(low=rep(4,12),middle=seq_len(12),high=rev(seq_len(12)))
    fit <- fastglm_ordinal(X,counts,firth=TRUE)
    w <- rowSums(counts)
    result <- fastglm_marginal_effects(fit,"x",weights=w)
    index <- rep(seq_len(nrow(X)),w)
    expect_equal(fastglm_marginal_effects(fit,"x",X[index,,drop=FALSE]),result,tolerance=1e-12)
    h <- 1e-4
    reference <- colSums(w*(predict(fit,X+h)-predict(fit,X-h)))/(2*h*sum(w))
    expect_equal(result$estimate,unname(reference),tolerance=1e-8)
    expect_warning(reduced <- fastglm_ordinal(X,cbind(low=counts[,1],empty=0,counts[,2:3]),firth=TRUE),
                   "collapsed")
    out <- fastglm_marginal_effects(reduced,"x",weights=w)
    expect_equal(out$estimate[c(1,3,4)],result$estimate,tolerance=1e-10)
    expect_equal(out$estimate[2],0)
    expect_true(is.na(out$std.error[2]))
    expect_identical(attr(out,"covariance.scope"),"reduced category model")
    new <- X; new[1,] <- NA
    expect_error(fastglm_marginal_effects(fit,"x",new),"complete finite")
    expect_true(all(is.finite(fastglm_marginal_effects(fit,"x",new,weights=c(0,w[-1]))$estimate)))
    expect_error(fastglm_marginal_effects(fit,"unknown"),"location-design column")
    expect_error(fastglm_marginal_effects(fit,"x",weights=rep(0,12)),"positive finite total")
    expect_error(fastglm_marginal_effects(fit,"x",eps=0),"positive finite")
    expect_error(fastglm_marginal_effects(fit,"x",level=1),"strictly between")
})
