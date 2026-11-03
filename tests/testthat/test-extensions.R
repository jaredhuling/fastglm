test_that("count offsets apply only to their component and NB uncertainty is joint", {
    skip_if_not_installed("pscl")
    set.seed(544); d <- data.frame(x=rnorm(250),e=runif(250,.8,1.2))
    d$y <- rnbinom(250,mu=exp(.3+.8*d$x)*d$e,size=.4)
    d$y[runif(250)<.2] <- 0
    f <- fastglm_hurdle(y~x+offset(log(e))|x,d,dist="negbin",outer.tol=1e-9,outer.maxit=150)
    g <- pscl::hurdle(y~x+offset(log(e))|x,d,dist="negbin")
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-4)
    expect_equal(unname(vcov(f)),unname(vcov(g)),tolerance=3e-4)
    expect_equal(f$SE.theta,unname(g$theta*g$SE.logtheta),tolerance=3e-4)
    expect_equal(predict(f),predict(g),tolerance=1e-4,ignore_attr=TRUE)
    prob <- predict(f,type="prob",at=0:500)
    expect_equal(unname(rowSums(prob)),rep(1,nrow(d)),tolerance=1e-5)
    expect_true(all(is.finite(predict(f,d[1:4,],se.fit=TRUE)$se.fit)))
    expect_equal(ncol(vcov(f,full=TRUE)),length(coef(f))+1L)
    expect_equal(dim(simulate(f,nsim=2,seed=4)),c(250L,2L))
})

test_that("truncated and two-part models predict the modeled conditional means", {
    set.seed(41); x <- cbind(1,rnorm(400)); mu <- exp(.5+.3*x[,2])
    y <- qpois(runif(400)*(-expm1(-mu)),mu,lower.tail=FALSE)
    f <- fastglm_truncated(x,y)
    expect_true(f$converged)
    expect_true(all(predict(f)>=1))
    expect_equal(rowSums(predict(f,type="prob",at=1:50)),rep(1,400),tolerance=1e-8)
    expect_true(all(is.finite(predict(f,se.fit=TRUE)$se.fit)))
    d <- data.frame(x=x[,2],y=rbinom(400,1,.6)*rgamma(400,shape=2,scale=mu/2))
    t <- fastglm_twopart(y~x,d)
    expect_true(t$converged)
    expect_equal(predict(t),(1-predict(t,type="zero"))*predict(t,type="count"))
    expect_true(all(predict(t,se.fit=TRUE)$se.fit>0))
    expect_equal(nobs(t),400L)
})

test_that("prepared Gaussian batches reuse the factor and match individual fits", {
    set.seed(79); x <- cbind(1,matrix(rnorm(600),200,3)); y <- matrix(rnorm(600),200,3)
    design <- fastglm_prepare(x)
    fits <- fastglm_batch(design,y)
    expect_true(exists("factor",design$cache,inherits=FALSE))
    for(j in 1:3) {
        f <- fastglm(x,y[,j],method=2)
        expect_equal(unname(coef(fits[[j]])),unname(coef(f)),tolerance=1e-8)
        expect_equal(unname(vcov(fits[[j]])),unname(vcov(f)),tolerance=1e-8)
    }
})

test_that("streaming initialization, inference, and chunk partitions are consistent", {
    skip_if_not_installed("sandwich")
    set.seed(32); n <- 240; x <- cbind(1,rnorm(n)); mu <- exp(.2+.4*x[,2]); y <- rgamma(n,shape=2,scale=mu/2)
    callback <- function(k) { i <- ((k-1)*60+1):(k*60); list(X=x[i,,drop=FALSE],y=y[i]) }
    f <- fastglm_streaming(callback,4,Gamma("log"),tol=1e-10)
    g <- fastglm(x,y,Gamma("log"),method=2,tol=1e-10)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-7)
    expect_equal(predict(f,type="response"),g$fitted.values,tolerance=1e-7)
    expect_equal(as.numeric(logLik(f)),as.numeric(logLik(g)),tolerance=1e-7)
    expect_equal(fastglm_vcov(f,type="HC0"),fastglm_vcov(g,type="HC0"),tolerance=1e-7)
    group <- rep(1:12,each=20)
    cfun <- function(k) group[((k-1)*60+1):(k*60)]
    expect_equal(fastglm_vcov(f,type="HC1",cluster=cfun),fastglm_vcov(g,type="HC1",cluster=group),tolerance=1e-7)
    avg <- fastglm_average(f); expect_equal(avg$estimate,mean(g$fitted.values),tolerance=1e-7)
    chunks <- list(); predict(f,sink=function(z,k) chunks[[k]] <<- z)
    expect_length(chunks,4L)
    expect_error(fastglm_streaming(function(k) list(X=x,y=rep(2,n)),1,binomial()),"0 <= y <= 1")
    bad <- function(k) { if(k==2) stop("database disconnected"); callback(k) }
    expect_error(fastglm_streaming(bad,4,poisson()),"database disconnected")
})

test_that("formula sources retain factors and reject changed chunk schemas", {
    set.seed(24); d <- data.frame(y=rnorm(90),x=rnorm(90),g=factor(rep(c("a","b","c"),each=30)))
    source <- fastglm_source(function(k) d[((k-1)*30+1):(k*30),],3,
                             formula=y~x+g,prototype=d)
    f <- fastglm_streaming(source); g <- glm(y~x+g,data=d)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-8)
    cb <- function(k) { x <- cbind(1,d$x[1:30]); colnames(x) <- if(k==1) c("int","x") else c("x","int"); list(X=x,y=d$y[1:30]) }
    expect_error(fastglm_streaming(cb,2),"columns changed")
})

test_that("compact fits skip covariance and diagnostics recover small scores", {
    set.seed(341); x <- cbind(1,matrix(rnorm(300),100,3)); y <- rpois(100,2)
    full <- fastglm(x,y,poisson(),method="auto",tol=1e-12)
    small <- fastglm(x,y,poisson(),method="auto",inference="none",retain.data=FALSE,tol=1e-12)
    expect_equal(coef(small),coef(full))
    expect_null(small$cov.unscaled); expect_null(small$x)
    expect_error(vcov(small),"not retained")
    expect_lt(fastglm_diagnostics(full)$standardized.score,1e-6)
    scaled <- fastglm(sweep(x,2,c(1,1e-8,1e8,2),"*"),y,poisson(),method="auto",tol=1e-12)
    expect_equal(scaled$fitted.values,full$fitted.values,tolerance=1e-8)
})

test_that("streaming grouped-binomial trial totals and Gamma adjustment agree", {
    set.seed(351); x <- cbind(1,rnorm(160)); size <- sample(2:15,160,TRUE)
    yy <- rbinom(160,size,plogis(x[,2])); y <- cbind(yy,size-yy)
    cb <- function(k) { i<-((k-1)*40+1):(k*40); list(X=x[i,,drop=FALSE],y=y[i,,drop=FALSE]) }
    f <- fastglm_streaming(cb,4,binomial(),tol=1e-10)
    g <- fastglm(x,y,binomial(),method=2,tol=1e-10)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-8)
    expect_equal(f$aic,g$aic,tolerance=1e-8)
    y <- rgamma(160,shape=.8,scale=exp(.2+.3*x[,2])/.8)
    cb <- function(k) { i<-((k-1)*40+1):(k*40); list(X=x[i,,drop=FALSE],y=y[i]) }
    f <- fastglm_streaming(cb,4,Gamma("log"),firth=TRUE,tol=1e-10)
    g <- fastglm(x,y,Gamma("log"),firth=TRUE,tol=1e-10)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-7)
    expect_equal(f$dispersion,g$dispersion,tolerance=1e-7)
})

test_that("theta profile endpoints satisfy the likelihood-ratio equation", {
    set.seed(352); x<-cbind(1,rnorm(250)); y<-rnbinom(250,mu=exp(.5+.3*x[,2]),size=1.5)
    f<-fastglm_nb(x,y,outer.tol=1e-10)
    ci<-fastglm_confint_theta(f)
    expect_lt(ci[1],f$theta);expect_gt(ci[2],f$theta)
    for(theta in unname(ci)) {
        conditional <- fastglm(x,y,negbin(theta),method="auto",tol=1e-10)
        expect_equal(2*(as.numeric(logLik(f))-as.numeric(logLik(conditional))),qchisq(.95,1),tolerance=1e-4)
    }
})

test_that("two-part marginal uncertainty agrees with its analytic gradient", {
    set.seed(654); d<-data.frame(x=rnorm(400));d$y<-rbinom(400,1,.6)*rgamma(400,shape=2,scale=exp(.2+.3*d$x)/2)
    f<-fastglm_twopart(y~x,data=d)
    g0<-glm(I(y>0)~x,data=d,family=binomial())
    g1<-glm(y~x,data=d,subset=y>0,family=Gamma("log"),control=list(epsilon=1e-12))
    expect_equal(unname(coef(f,"zero")),unname(coef(g0)),tolerance=1e-6)
    expect_equal(unname(coef(f,"count")),unname(coef(g1)),tolerance=1e-5)
    nd<-d[1:4,]; xx<-model.matrix(~x,nd)
    pi<-1-predict(f,nd,type="zero");mu<-predict(f,nd,type="count")
    grad<-cbind(xx*(pi*mu),xx*(mu*pi*(1-pi)))
    expected<-sqrt(rowSums((grad%*%vcov(f))*grad))
    expect_equal(unname(predict(f,nd,se.fit=TRUE)$se.fit),unname(expected),tolerance=1e-6)
})

test_that("database sources preserve a common factor schema", {
    skip_if_not_installed("DBI");skip_if_not_installed("RSQLite")
    set.seed(655); d<-data.frame(id=1:100,y=rnorm(100),x=rnorm(100),g=factor(rep(c("a","b"),50)))
    con<-DBI::dbConnect(RSQLite::SQLite(),":memory:");on.exit(DBI::dbDisconnect(con))
    DBI::dbWriteTable(con,"observations",d)
    source<-fastglm_source_dbi(con,c("SELECT * FROM observations WHERE id <= 50 ORDER BY id",
                                   "SELECT * FROM observations WHERE id > 50 ORDER BY id"),y~x+g,prototype=d)
    f<-fastglm_streaming(source); g<-glm(y~x+g,data=d)
    expect_equal(unname(coef(f)),unname(coef(g)),tolerance=1e-8)
})

test_that("rank-deficient predictions identify nonestimable rows", {
    set.seed(656);x<-cbind(1,rnorm(100));x<-cbind(x,2*x[,2]);y<-rnorm(100)
    f<-fastglm(x,y)
    nd<-x[1:2,];nd[2,3]<-nd[2,3]+1
    expect_warning(pred<-predict(f,nd,se.fit=TRUE),"not estimable")
    expect_true(is.na(pred$fit[2]));expect_true(is.na(pred$se.fit[2]))
    expect_equal(pred$fit[1],f$linear.predictors[1],tolerance=1e-8)
})

test_that("streaming weighted continuous-family likelihoods use global dispersion", {
    set.seed(659);x<-cbind(1,rnorm(160));w<-rep(c(0,.5,1,2),40)
    for(fam in list(gaussian(),Gamma("log"),inverse.gaussian("log"))) {
        y<-if(fam$family=="gaussian") rnorm(160) else rgamma(160,shape=2,scale=exp(.1+.2*x[,2])/2)
        cb<-function(k) {i<-((k-1)*40+1):(k*40);list(X=x[i,,drop=FALSE],y=y[i],weights=w[i])}
        f<-fastglm_streaming(cb,4,fam,tol=1e-11)
        g<-fastglm(x,y,fam,weights=w,method=2,tol=1e-11)
        expect_equal(f$aic,g$aic,tolerance=1e-7)
        dropped<-fastglm(x[w>0,,drop=FALSE],y[w>0],fam,weights=w[w>0],method=2,tol=1e-11)
        expect_equal(g$aic,dropped$aic,tolerance=1e-7)
    }
})

test_that("streaming summaries use stored sufficient summaries without rereading", {
    set.seed(660);x<-cbind(1,rnorm(120));y<-rnorm(120)
    cb<-function(k){i<-((k-1)*40+1):(k*40);list(X=x[i,,drop=FALSE],y=y[i])}
    f<-fastglm_streaming(cb,3);reads<-f$state$reads
    s<-summary(f)
    expect_equal(f$state$reads,reads)
    expect_equal(s$coefficients[,1],coef(f))
    expect_equal(fitted(f),predict(f,type="response"))
})

test_that("compact inference preserves standard errors without inventing correlations", {
    set.seed(661);x<-cbind(1,rnorm(120));y<-rnorm(120)
    full<-fastglm(x,y)
    diagonal<-fastglm(x,y,inference="diagonal")
    none<-fastglm(x,y,inference="none")
    expect_equal(summary(diagonal)$coefficients,summary(full)$coefficients)
    expect_null(summary(diagonal)$cov.unscaled)
    expect_equal(summary(diagonal,dispersion=2)$coefficients[,2],
                 summary(full,dispersion=2)$coefficients[,2])
    expect_error(summary(diagonal,correlation=TRUE),"full covariance")
    expect_error(fastglm_vcov(diagonal),"not retained")
    expect_true(is.na(fastglm_diagnostics(none)$covariance.condition))
})

test_that("truncated starting coefficients are checked before native fitting", {
    x<-cbind(1,seq_len(20)/20);y<-rep(1:4,5)
    expect_error(fastglm_truncated(x,y,start=0),"design columns")
    expect_error(fastglm_truncated(x,y,start=c(0,Inf)),"finite")
    f<-fastglm_truncated(x,y,dist="negbin",theta=2)
    expect_error(fastglm_confint_theta(f),"estimate theta")
})

test_that("zero-weight clusters do not enter the cluster-count correction", {
    set.seed(662);x<-cbind(1,rnorm(120));y<-rnorm(120)
    groups<-rep(1:12,each=10);w<-as.numeric(groups>2)
    full<-fastglm(x,y,weights=w)
    reduced<-fastglm(x[w>0,,drop=FALSE],y[w>0])
    expect_equal(fastglm_vcov(full,type="HC1",cluster=groups),
                 fastglm_vcov(reduced,type="HC1",cluster=groups[w>0]))
})
