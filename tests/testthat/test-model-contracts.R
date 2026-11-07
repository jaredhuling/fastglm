test_that("pivoted QR gives stable fitted values with aliases", {
    for(seed in c(1,17,31)) {
        set.seed(seed); xx <- matrix(rnorm(200*3),200,3)
        x <- cbind(.1*xx[,1],1,xx[,2],xx[,1],xx[,3])
        y <- rbinom(200,1,plogis(.2+xx[,2]))
        reference <- glm.fit(x,y,family=binomial())
        for(method in list(0,4,"auto")) {
            f <- fastglm(x,y,binomial(),method=method)
            expect_equal(f$fitted.values,reference$fitted.values,tolerance=1e-7)
            expect_equal(sum(is.na(coef(f))),1L)
            expect_equal(nrow(summary(f)$coefficients),4L)
            expect_equal(predict(f),f$linear.predictors)
        }
        expect_warning(f <- fastglm(x,y,binomial(),method=2),"pivoted QR")
        expect_equal(f$fitted.values,reference$fitted.values,tolerance=1e-7)
        expect_equal(sum(is.na(coef(f))),1L)
        expect_error(fastglm_fit(x,y,family=binomial(),singular.ok=FALSE),"singular")
    }
})

test_that("zero weights and binomial trials have consistent model accounting", {
    set.seed(45); x <- cbind(1,rnorm(150)); y <- rnorm(150); w <- rep(c(0,1,2),50)
    f <- fastglm(x,y,weights=w); g <- glm.fit(x,y,weights=w)
    expect_equal(f$df.residual,g$df.residual)
    expect_equal(f$dispersion,sum(w*(y-g$fitted.values)^2)/g$df.residual,tolerance=1e-6)
    trials <- sample(2:30,150,TRUE); yy <- rbinom(150,trials,plogis(x[,2]))
    response <- cbind(yy,trials-yy)
    f <- fastglm(x,response,binomial(),weights=w)
    g <- glm.fit(x,response,family=binomial(),weights=w)
    expect_equal(f$null.deviance,g$null.deviance,tolerance=1e-8)
    expect_equal(f$aic,g$aic,tolerance=1e-8)
    ff <- fastglm_fit(x,response,family=binomial(),weights=w)
    expect_equal(ff$aic,f$aic)
})

test_that("sparse Cholesky rejects nearly singular information", {
    set.seed(46)
    z <- rnorm(100)
    x <- as(cbind(1,z,z+1e-8*rnorm(100)),"dgCMatrix")
    y <- rnorm(100)
    for(method in c(2,3))
        expect_error(fastglm(x,y,method=method),"Cholesky")
})

test_that("formula predictions preserve offsets, factors, terms, and NA actions", {
    set.seed(125); d <- data.frame(x=rnorm(200),g=factor(rep(letters[1:4],50)),e=runif(200,1,3))
    d$y <- rpois(200,exp(.2+.3*d$x)*d$e); d$x[5] <- NA
    f <- fastglm(y~x+g+offset(log(e)),d,poisson(),na.action=na.exclude)
    g <- glm(y~x+g+offset(log(e)),d,family=poisson(),na.action=na.exclude)
    for(type in c("link","response","terms")) {
        a <- predict(f,type=type,se.fit=TRUE); b <- predict(g,type=type,se.fit=TRUE)
        expect_equal(a$fit,b$fit,tolerance=1e-6,ignore_attr=TRUE)
        expect_equal(a$se.fit,b$se.fit,tolerance=1e-5,ignore_attr=TRUE)
    }
    nd <- d[c(2,8,20),]
    expect_equal(predict(f,nd,type="response"),predict(g,nd,type="response"),tolerance=1e-6,ignore_attr=TRUE)
    expect_equal(hatvalues(f),hatvalues(g),tolerance=1e-5,ignore_attr=TRUE)
    expect_equal(f$null.deviance,g$null.deviance,tolerance=1e-6)
    expect_equal(nobs(f),nobs(g))
})

test_that("custom family kernels and exact NB theta retain their semantics", {
    set.seed(9); x <- cbind(1,rnorm(100)); y <- rnorm(100)
    fam <- gaussian(); fam$linkfun <- function(mu) mu/2
    fam$linkinv <- function(eta) 2*eta; fam$mu.eta <- function(eta) rep(2,length(eta))
    f <- fastglm(x,y,fam); g <- glm.fit(x,y,family=fam)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-8)
    expect_equal(fastglm:::family_code(fam),-1L)
    skip_if_not_installed("MASS")
    fam <- MASS::negative.binomial(1.23456789123)
    expect_equal(unname(fastglm:::family_params(fam)[1]),1.23456789123)
    f <- fastglm(x,rnbinom(100,mu=2,size=1.23456789123),fam)
    expect_identical(f$dispersion,1)
})

test_that("big.matrix views respect row and column offsets", {
    skip_if_not_installed("bigmemory")
    set.seed(82); x <- cbind(rnorm(250),1,matrix(rnorm(750),250,3))
    b <- bigmemory::as.big.matrix(x)
    sub <- bigmemory::sub.big.matrix(b,firstRow=51,lastRow=200,firstCol=2,lastCol=5)
    y <- rpois(150,2)
    f <- fastglm(sub,y,poisson(),method=2); g <- fastglm(x[51:200,2:5],y,poisson(),method=2)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-9)
    expect_equal(drop(sub%*%rep(1,4)),rowSums(x[51:200,2:5]),ignore_attr=TRUE)
    expect_equal(drop(rep(1,150)%*%sub),colSums(x[51:200,2:5]),ignore_attr=TRUE)
})

test_that("mean bias reduction includes the adjusted Gamma dispersion score", {
    skip_if_not_installed("brglm2")
    set.seed(62); x <- cbind(1,rnorm(50)); y <- rgamma(50,shape=.2,scale=exp(x%*%c(.2,.5))/.2)
    ref <- brglm2::brglmFit(x,y,family=Gamma("log"),control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10))
    f <- fastglm(x,y,Gamma("log"),firth=TRUE,tol=1e-10)
    expect_equal(unname(coef(f)),unname(coef(ref)),tolerance=1e-7)
    expect_equal(f$dispersion,ref$dispersion,tolerance=1e-7)
    sp <- fastglm(as(x,"dgCMatrix"),y,Gamma("log"),firth=TRUE,method=2,tol=1e-10)
    expect_equal(unname(coef(sp)),unname(coef(f)),tolerance=1e-7)
})

test_that("formula models can be updated and have tidy coefficient summaries", {
    set.seed(603);d<-data.frame(y=rnorm(120),x=rnorm(120),z=rnorm(120))
    f<-fastglm(formula=y~x+z,data=d)
    u<-update(f,.~.-z)
    expect_equal(unname(coef(u)),unname(coef(glm(y~x,data=d))),tolerance=1e-9)
    skip_if_not_installed("broom")
    expect_equal(broom::tidy(f)$estimate,unname(coef(f)))
    expect_equal(broom::glance(f)$nobs,120L)
})

test_that("binomial simulation preserves numeric response trial denominators", {
    set.seed(663);x<-cbind(1,rnorm(100));trials<-rep(c(5,10),50)
    y<-rbinom(100,trials,plogis(.1+.3*x[,2]))/trials
    f<-fastglm(x,y,binomial(),weights=trials)
    set.seed(664);expected<-rbinom(100,trials,f$fitted.values)/trials
    expect_equal(simulate(f,seed=664)[[1L]],expected)
    two<-fastglm(x,cbind(y*trials,(1-y)*trials),binomial())
    expect_equal(simulate(two,seed=664)[[1L]],expected)
    f$prior.weights[1]<-.5
    expect_error(simulate(f,seed=664),"integer trial weights")
})
