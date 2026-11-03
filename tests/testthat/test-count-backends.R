test_that("estimated-size NB agrees across dense, sparse, and streamed designs", {
    set.seed(741);n<-360L;x<-cbind(1,rnorm(n),rbinom(n,1,.15))
    colnames(x)<-c("(Intercept)","x","z")
    y<-rnbinom(n,mu=exp(.2+.5*x[,2]-.3*x[,3]),size=.7)
    w<-rep(c(0,1,1,2),length.out=n);off<-rnorm(n,0,.1)
    dense<-fastglm_nb(x,y,weights=w,offset=off,tol=1e-10,outer.tol=1e-9)
    sparse<-fastglm_nb(as(x,"dgCMatrix"),y,weights=w,offset=off,tol=1e-10,outer.tol=1e-9)
    expect_s4_class(sparse$x,"dgCMatrix")
    for(size in c(60L,120L)) {
        cb<-function(k){i<-seq.int((k-1L)*size+1L,k*size);list(X=x[i,,drop=FALSE],y=y[i],weights=w[i],offset=off[i])}
        stream<-fastglm_nb(fastglm_source(cb,n/size),tol=1e-10,outer.tol=1e-9)
        for(f in list(sparse,stream)) {
            expect_true(f$converged)
            expect_equal(unname(coef(f)),unname(coef(dense)),tolerance=1e-7)
            expect_equal(f$theta,dense$theta,tolerance=1e-6)
            expect_equal(as.numeric(logLik(f)),as.numeric(logLik(dense)),tolerance=1e-7)
            expect_equal(unname(vcov(f)),unname(vcov(dense)),tolerance=1e-7)
            expect_equal(f$vcov.full,dense$vcov.full,tolerance=1e-6)
            expect_equal(f$SE.theta,dense$SE.theta,tolerance=1e-6)
            expect_equal(nobs(f),sum(w>0))
        }
        expect_null(stream$x)
        expect_equal(predict(stream,type="response"),dense$fitted.values,tolerance=1e-7)
    }
})

test_that("sparse estimated-size NB preserves non-log link likelihoods", {
    set.seed(742);x<-cbind(1,runif(300,-.3,.3));y<-rnbinom(300,mu=2+x[,2],size=1.2)
    for(link in c("sqrt","identity")) {
        dense<-fastglm_nb(x,y,link=link,tol=1e-10,outer.tol=1e-8)
        sparse<-fastglm_nb(as(x,"dgCMatrix"),y,link=link,tol=1e-10,outer.tol=1e-8)
        expect_equal(unname(coef(sparse)),unname(coef(dense)),tolerance=1e-6)
        expect_equal(sparse$theta,dense$theta,tolerance=1e-6)
        expect_equal(sparse$vcov.full,dense$vcov.full,tolerance=1e-6,ignore_attr=TRUE)
        expect_equal(sparse$SE.theta,dense$SE.theta,tolerance=1e-6)
    }
})

test_that("hurdle likelihood and joint covariance agree across representations", {
    set.seed(745);n<-400L;d<-data.frame(x=rnorm(n),z=rbinom(n,1,.2))
    d$y<-rnbinom(n,mu=exp(.5+.4*d$x),size=.8)*rbinom(n,1,.65)
    x<-model.matrix(~x+z,d);z<-model.matrix(~x,d)
    for(dist in c("poisson","negbin")) {
        dense<-fastglm_hurdle(y~x+z|x,d,dist=dist,tol=1e-10,outer.tol=1e-8,outer.maxit=100)
        sparse<-fastglm_hurdle(as(x,"dgCMatrix"),response=d$y,z=as(z,"dgCMatrix"),
                              dist=dist,tol=1e-10,outer.tol=1e-8,outer.maxit=100)
        cb<-function(k){i<-seq.int((k-1L)*100L+1L,k*100L);list(X=x[i,],Z=z[i,],y=d$y[i])}
        stream<-fastglm_hurdle(fastglm_source(cb,4),dist=dist,tol=1e-10,outer.tol=1e-8,outer.maxit=100)
        expect_s4_class(sparse$x$count,"dgCMatrix")
        for(f in list(sparse,stream)) {
            expect_true(f$converged)
            expect_equal(coef(f),coef(dense),tolerance=1e-6)
            expect_equal(as.numeric(logLik(f)),as.numeric(logLik(dense)),tolerance=1e-7)
            expect_equal(vcov(f,full=TRUE),vcov(dense,full=TRUE),tolerance=2e-6)
        }
        expect_null(stream$x)
        expect_null(stream$y)
    }
})

test_that("source hurdle fits preserve separate offsets and zero-only blocks", {
    set.seed(746);n<-400L;d<-data.frame(x=rnorm(n),e=exp(rnorm(n,0,.1)),f=exp(rnorm(n,0,.1)))
    d$y<-rpois(n,exp(.4+.3*d$x))*rbinom(n,1,.6);d$w<-rep(c(0,1,2,1),length.out=n)
    d$y[1:100]<-0;d$w[101:200]<-0
    callback<-function(k) d[seq.int((k-1L)*100L+1L,k*100L),]
    source<-fastglm_source(callback,4,formula=y~x+offset(log(e))|x+offset(log(f)),prototype=d,weights="w")
    stream<-fastglm_hurdle(source,tol=1e-10)
    dense<-fastglm_hurdle(y~x+offset(log(e))|x+offset(log(f)),d,weights=w,tol=1e-10)
    expect_equal(coef(stream),coef(dense),tolerance=1e-7)
    expect_equal(vcov(stream),vcov(dense),tolerance=1e-7)
    expect_equal(stream$nobs,sum(d$w>0))
})

test_that("two-part fits agree when positive rows are supplied in blocks", {
    set.seed(747);n<-360L;d<-data.frame(x=rnorm(n),e=exp(rnorm(n,0,.1)))
    d$y<-rgamma(n,shape=1.3,scale=exp(.5+.3*d$x)/1.3)*rbinom(n,1,.6)
    d$y[1:60]<-0;x<-model.matrix(~x,d)
    source<-fastglm_source(function(k)d[seq.int((k-1L)*60L+1L,k*60L),],6,
                          formula=y~x+offset(log(e))|x,prototype=d)
    for(firth in c(FALSE,TRUE)) {
        dense<-fastglm_twopart(y~x+offset(log(e))|x,d,tol=1e-10,firth=firth)
        sparse<-fastglm_twopart(as(x,"dgCMatrix"),response=d$y,offset=log(d$e),tol=1e-10,firth=firth)
        stream<-fastglm_twopart(source,tol=1e-10,firth=firth)
        for(f in list(sparse,stream)) {
            expect_true(f$converged)
            expect_equal(coef(f),coef(dense),tolerance=1e-7)
            expect_equal(vcov(f),vcov(dense),tolerance=1e-7)
            expect_equal(as.numeric(logLik(f)),as.numeric(logLik(dense)),tolerance=1e-7)
        }
        expect_null(stream$x)
        expect_equal(stream$positive$dispersion,dense$positive$dispersion,tolerance=1e-7)
    }
})

test_that("count block readers reject changed columns and invalid responses", {
    x<-cbind(1,seq_len(30));y<-rep(0:2,10)
    bad<-fastglm_source(function(k)list(X=x[,seq_len(k),drop=FALSE],y=y),2)
    expect_error(fastglm_nb(bad),"columns changed")
    expect_error(fastglm_hurdle(x,response=y+.1),"invalid nonnegative response")
    expect_error(fastglm_twopart(x,response=-y),"invalid nonnegative response")
    expect_error(fastglm_nb(x,y+.1),"nonnegative integers")
    source<-fastglm_source(function(k)list(X=x,y=y),1)
    expect_error(fastglm_nb(source,y=y),"must supply its own")
    reader<-fastglm:::.count_backend_reader(x,y)
    reader$fetch<-function(k)stop("reader failed")
    expect_error(fastglm:::.count_backend_joint(reader,c(0,0),1),"reader failed")
})

test_that("positive-component passes reset sequential readers and visit skipped blocks", {
    set.seed(751);n<-240L;d<-data.frame(x=rnorm(n))
    d$y<-rgamma(n,2)*rbinom(n,1,.6);d$y[c(1:60,181:240)]<-0
    position<-0L;resets<-0L
    reset<-function(){position<<-0L;resets<<-resets+1L}
    cb<-function(k){
        if(k!=position+1L) stop("reader was not reset or a block was skipped")
        position<<-k;d[seq.int((k-1L)*60L+1L,k*60L),]
    }
    source<-fastglm_source(cb,4,reset=reset,formula=y~x,prototype=d)
    stream<-fastglm_twopart(source,tol=1e-10)
    dense<-fastglm_twopart(y~x,d,tol=1e-10)
    expect_equal(coef(stream),coef(dense),tolerance=1e-7)
    expect_gt(resets,1L)
    expect_equal(position,4L)
})

test_that("single-formula source offsets retain component meanings", {
    set.seed(752);n<-240L;d<-data.frame(x=rnorm(n),e=exp(rnorm(n,0,.2)),extra=rnorm(n,0,.1))
    d$y<-rgamma(n,2)*rbinom(n,1,.6)
    cb<-function(k)d[seq.int((k-1L)*60L+1L,k*60L),]
    source<-fastglm_source(cb,4,formula=y~x+offset(log(e)),prototype=d,offset="extra")
    stream<-fastglm_twopart(source,tol=1e-10)
    dense<-fastglm_twopart(y~x+offset(log(e)),d,offset=extra,tol=1e-10)
    expect_equal(coef(stream),coef(dense),tolerance=1e-7)
    expect_equal(predict(stream,newdata=d[1:5,]),predict(dense,newdata=d[1:5,]),tolerance=1e-7)
})

test_that("large-theta score retains its analytic Poisson-limit expansion", {
    y<-c(0,1,3,7);mu<-c(.5,2,2,5);theta<-1e10
    limit<-(y-(y-mu)^2)/2
    expect_equal(fastglm:::.count_backend_theta_score(y,mu,theta)*theta^2,limit,tolerance=1e-8)
})

test_that("analytic NB information matches an independent likelihood Hessian", {
    skip_if_not_installed("numDeriv")
    set.seed(753);x<-cbind(1,runif(100,-.2,.2));w<-rep(c(0,1,.5,2),25);off<-rnorm(100,0,.02)
    for(link in c("log","sqrt","identity")) {
        beta<-c(.6,.2);theta<-1.3
        y<-rnbinom(100,mu=make.link(link)$linkinv(drop(x%*%beta)+off),size=theta)
        reader<-fastglm:::.count_backend_reader(x,y,weights=w,offset=off)
        names(beta)<-reader$columns$count
        information<-fastglm:::.count_backend_nb_information(reader,beta,theta,link)
        objective<-function(par) -sum(w*dnbinom(y,
            mu=make.link(link)$linkinv(drop(x%*%par[1:2])+off),size=exp(par[3]),log=TRUE))
        expected<-numDeriv::hessian(objective,c(beta,log(theta)))
        expect_equal(unname(information),expected,tolerance=1e-6)
    }
})

test_that("NB Poisson-boundary fits keep coefficients without a regular theta SE", {
    set.seed(754);x<-matrix(1,200,1);y<-rep(0:2,length.out=200)
    dense<-fastglm_nb(x,y)
    sparse<-fastglm_nb(as(x,"dgCMatrix"),y)
    for(f in list(dense,sparse)) {
        expect_true(f$theta.boundary)
        expect_true(all(is.finite(coef(f))))
        expect_true(is.na(f$SE.theta))
        expect_true(all(is.na(f$vcov.full["theta",])))
        expect_equal(as.numeric(logLik(f)),sum(dnbinom(y,mu=f$fitted.values,size=f$theta,log=TRUE)),tolerance=1e-10)
    }
})

test_that("rare-event Poisson limits are not mistaken for an interior theta root", {
    x<-matrix(1,1000,1);y<-c(1,rep(0,999))
    source<-fastglm_source(function(k) {
        i<-seq.int((k-1L)*100L+1L,k*100L)
        list(X=x[i,,drop=FALSE],y=y[i])
    },10)
    fits<-list(fastglm_nb(x,y),fastglm_nb(as(x,"dgCMatrix"),y),fastglm_nb(source))
    for(f in fits) {
        expect_true(f$converged)
        expect_true(f$theta.boundary)
        expect_true(is.na(f$SE.theta))
        expect_equal(unname(coef(f)),log(mean(y)),tolerance=1e-6)
    }
})

test_that("unfinished truncated NB fits report their returned parameter values", {
    set.seed(119);x<-cbind(1,rnorm(300))
    y<-rnbinom(300,mu=exp(1+.3*x[,2]),size=.4);keep<-y>0
    x<-x[keep,,drop=FALSE];y<-y[keep]
    f<-suppressWarnings(fastglm_truncated(x,y,dist="negbin",outer.maxit=1))
    mu<-exp(drop(x%*%coef(f)))
    q<--expm1(-f$theta*log1p(mu/f$theta))
    expect_false(f$converged)
    expect_equal(f$fitted.values,mu/q,tolerance=1e-10)
    expect_equal(as.numeric(logLik(f)),sum(dnbinom(y,mu=mu,size=f$theta,log=TRUE)-log(q)),tolerance=1e-10)
    if(requireNamespace("numDeriv",quietly=TRUE)) {
        objective<-function(par) {
            mm<-exp(drop(x%*%par[1:2]));th<-exp(par[3])
            -sum(dnbinom(y,mu=mm,size=th,log=TRUE)-log(-expm1(-th*log1p(mm/th))))
        }
        information<-numDeriv::hessian(objective,c(coef(f),log(f$theta)))
        expected<-solve(information)*outer(c(1,1,f$theta),c(1,1,f$theta))
        expect_equal(unname(vcov(f,full=TRUE)),expected,tolerance=1e-5)
    }
})
