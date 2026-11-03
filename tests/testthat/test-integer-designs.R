test_that("integer dense designs match double designs in ordinary and Firth GLMs", {
    set.seed(941);x<-cbind(1L,rep(0:2,80));xd<-x;storage.mode(xd)<-"double"
    colnames(x)<-colnames(xd)<-c("intercept","group")
    before<-x;w<-rep(c(0,1,2,1),60);off<-rep(c(-.1,0,.1),80)
    responses<-list(gaussian=rnorm(240),binomial=rbinom(240,1,.4),Gamma=rgamma(240,2))
    families<-list(gaussian=gaussian(),binomial=binomial(),Gamma=Gamma("log"))
    for(nm in names(families)) for(firth in c(FALSE,TRUE)) for(method in c("auto","qr")) {
        f<-fastglm(x,responses[[nm]],families[[nm]],weights=w,offset=off,firth=firth,method=method)
        g<-fastglm(xd,responses[[nm]],families[[nm]],weights=w,offset=off,firth=firth,method=method)
        expect_equal(coef(f),coef(g),tolerance=1e-10)
        expect_equal(vcov(f),vcov(g),tolerance=1e-10)
        expect_type(f$x,"double")
    }
    expect_identical(x,before)
    bad<-x;bad[1,1]<-NA_integer_
    expect_error(fastglm(bad,responses$gaussian),"finite")
    expect_error(fastglm(matrix("1",240,1),responses$gaussian),"finite numeric")
})

test_that("integer designs work in negative-binomial and two-component paths", {
    set.seed(942);x<-cbind(1L,rep(0:2,100));z<-cbind(1L,rep(0:1,150))
    xd<-x;zd<-z;storage.mode(xd)<-storage.mode(zd)<-"double"
    y<-rnbinom(300,mu=exp(.3+.1*x[,2]),size=.8)
    pos<-y>0;continuous<-rgamma(300,2)*(y>0)
    fits<-list(nb=fastglm_nb(x,y),truncated=fastglm_truncated(x[pos,,drop=FALSE],y[pos],dist="negbin"),
        hurdle=fastglm_hurdle(x,response=y,z=z,dist="negbin"),
        twopart=fastglm_twopart(x,response=continuous,z=z,firth=TRUE))
    refs<-list(nb=fastglm_nb(xd,y),truncated=fastglm_truncated(xd[pos,,drop=FALSE],y[pos],dist="negbin"),
        hurdle=fastglm_hurdle(xd,response=y,z=zd,dist="negbin"),
        twopart=fastglm_twopart(xd,response=continuous,z=zd,firth=TRUE))
    for(nm in names(fits)) {
        expect_equal(coef(fits[[nm]]),coef(refs[[nm]]),tolerance=1e-9)
        expect_equal(vcov(fits[[nm]]),vcov(refs[[nm]]),tolerance=1e-9)
        expect_equal(as.numeric(logLik(fits[[nm]])),as.numeric(logLik(refs[[nm]])),tolerance=1e-9)
    }
    expect_type(x,"integer");expect_type(z,"integer")
})

test_that("integer source blocks normalize locally without changing the source", {
    set.seed(943);x<-cbind(1L,rep(0:1,100));y<-rnbinom(200,mu=2,size=1)
    callback<-function(k) {i<-seq.int((k-1L)*50L+1L,k*50L);list(X=x[i,,drop=FALSE],y=y[i])}
    source<-fastglm_source(callback,4L)
    streamed<-fastglm_streaming(source, family=poisson())
    dense<-fastglm(x,y,poisson(),method=2L)
    expect_equal(unname(coef(streamed)),unname(coef(dense)),tolerance=1e-7)
    nb<-fastglm_nb(source)
    expect_equal(unname(coef(nb)),unname(coef(fastglm_nb(x,y))),tolerance=1e-6)
    expect_type(callback(1L)$X,"integer")
    expect_type(streamed$fetch(1L)$X,"double")
})
