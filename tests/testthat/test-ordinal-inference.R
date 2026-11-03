test_that("ordinal observed information and frequency meat agree with likelihood derivatives", {
    skip_if_not_installed("numDeriv")
    x<-cbind(x=c(-1,.2,.8,1.5,-.5,.4),z=c(.2,1,-.3,.4,.8,-.7))
    counts<-rbind(c(2,1,0,1),c(0,2,3,1),c(1,2,1,0),c(0,1,1,4),c(3,1,1,0),c(1,2,1,1))
    w<-c(1,2,0,1,3,1);off<-c(.1,-.2,.3,0,-.1,.2)
    b<-c(-1,.1,1.2,.4,-.3);q<-3L;d<-length(b)
    logprob<-function(b,i) log(diff(c(0,plogis(b[1:3]-sum(x[i,]*b[4:5])-off[i]),1)))
    likelihood<-function(b) sum(vapply(seq_len(nrow(x)),function(i)sum(w[i]*counts[i,]*logprob(b,i)),numeric(1)))
    H<- -numDeriv::hessian(likelihood,b);U<-numDeriv::grad(likelihood,b)
    M<-matrix(0,d,d);rows<-matrix(0,nrow(x),d)
    for(i in seq_len(nrow(x))) {
        score<-numDeriv::jacobian(function(z)logprob(z,i),b)
        M<-M+crossprod(score,score*(w[i]*counts[i,]))
        rows[i,]<-colSums(score*(w[i]*counts[i,]))
    }
    block<-fastglm:::.ordinal_sandwich_block(x,b[1:3],b[4:5],off,integer(),counts,w)
    clustered<-fastglm:::.ordinal_sandwich_block(x,b[1:3],b[4:5],off,integer(),counts,w,independent=FALSE)
    expect_equal(block$information,H,tolerance=2e-7)
    expect_equal(unname(block$score),unname(U),tolerance=2e-7)
    expect_equal(block$meat,M,tolerance=2e-7)
    expect_equal(unname(clustered$rows),rows,tolerance=2e-7)
    expected<-fastglm:::ordinal_native_evaluate(x,integer(),counts,w,off,4L,b)$information
    expect_gt(max(abs(H-expected)),.1)
    normalized<-fastglm:::.ordinal_sandwich_block(x,b[1:3],b[4:5],off,integer(),counts,w,
                                               frequency=sum(counts*w))
    expect_equal(normalized$information,block$information/sum(counts*w),tolerance=1e-12)
    expect_equal(normalized$meat,block$meat/sum(counts*w),tolerance=1e-12)
    fit<-fastglm_ordinal(x,counts,weights=w,offset=off,tol=1e-10)
    at<-c(fit$alpha,fit$beta);B<-solve(-numDeriv::hessian(likelihood,at))
    for(i in seq_len(nrow(x)))
        rows[i,]<-numDeriv::grad(function(z)sum(w[i]*counts[i,]*logprob(z,i)),at)
    groups<-c("a","b","ignored","c","b","d")
    summed<-rowsum(rows[w>0,,drop=FALSE],groups[w>0],reorder=FALSE)
    expected<-B%*%crossprod(summed)%*%B
    expect_equal(c(fastglm_vcov(fit,cluster=groups,cadjust=FALSE,block.size=2L)),c(expected),tolerance=2e-7)
})

test_that("ordinal sandwiches preserve frequency expansion and cluster sums", {
    x<-cbind(x=c(-1,-.4,.2,.9,1.4,1.7))
    counts<-rbind(c(4,2,1),c(2,3,2),c(1,3,3),c(1,2,4),c(0,2,5),c(1,4,3))
    w<-c(2,1,0,3,1,2);off<-c(.1,-.1,0,.2,-.2,.1);groups<-c("a","b",NA,"a","c","d")
    ii<-rep(seq_len(nrow(x)),rowSums(counts*w))
    yy<-unlist(lapply(seq_len(nrow(x)),function(i)rep(1:3,counts[i,]*w[i])))
    for(br in c(FALSE,TRUE)) {
        a<-fastglm_ordinal(x,counts,weights=w,offset=off,firth=br,tol=1e-10)
        b<-fastglm_ordinal(x[ii,,drop=FALSE],ordered(yy,levels=1:3),offset=off[ii],firth=br,tol=1e-10)
        for(type in c("HC0","HC1")) {
            va<-fastglm_vcov(a,type,block.size=2L);vb<-fastglm_vcov(b,type,block.size=11L)
            expect_equal(c(va),c(vb),tolerance=2e-8)
            ca<-fastglm_vcov(a,type,cluster=groups,block.size=2L)
            cb<-fastglm_vcov(b,type,cluster=groups[ii],block.size=11L)
            expect_equal(c(ca),c(cb),tolerance=2e-8)
            expect_equal(attr(ca,"clusters"),4L)
            expect_equal(attr(ca,"frequency.total"),length(ii))
        }
        plain<-fastglm_vcov(a,"HC0");hc1<-fastglm_vcov(a,"HC1")
        expect_equal(c(hc1),c(plain)*a$frequency/(a$frequency-a$rank),tolerance=1e-12)
        clustered<-fastglm_vcov(a,"HC0",cluster=groups,cadjust=FALSE)
        corrected<-fastglm_vcov(a,"HC1",cluster=groups)
        expect_equal(c(corrected),c(clustered)*(4/3)*(a$frequency-1)/(a$frequency-a$rank),tolerance=1e-12)
        expect_equal(attr(corrected,"hc1.factor"),(a$frequency-1)/(a$frequency-a$rank))
        expect_equal(attr(corrected,"cluster.factor"),4/3)
        expect_identical(dimnames(plain),rep(list(names(coef(a))),2L))
        expect_equal(c(fastglm_vcov(a)),c(plain))
        expect_match(attr(plain,"bread"),"observed negative score Jacobian")
        if(br) expect_match(attr(plain,"covariance.scope"),"first-order ordinary-score")
    }
})

test_that("binary ordinal sandwiches reduce to ordinary logistic sandwiches", {
    skip_if_not_installed("sandwich")
    set.seed(48111);n<-75L
    x<-cbind(x=rnorm(n),z=runif(n,-1,1));off<-runif(n,-.2,.2)
    y<-rbinom(n,1,plogis(.2+drop(x%*%c(.5,-.3))+off));w<-rep(c(0,1,2),length.out=n)
    groups<-rep(1:9,length.out=n);ii<-rep(seq_len(n),w)
    fit<-fastglm_ordinal(x,ordered(y,levels=0:1),weights=w,offset=off,tol=1e-10)
    ref<-glm(y~x+z+offset(off),data=data.frame(y=y[ii],x[ii,,drop=FALSE],off=off[ii]),family=binomial())
    # glm retains the QR weights from the last IRLS step's starting estimate.
    # Restart at its own solution so sandwich's bread uses the final weights.
    ref<-update(ref,start=coef(ref),control=glm.control(epsilon=1e-12))
    T<-diag(c(-1,1,1))
    for(type in c("HC0","HC1")) {
        expected<-T%*%sandwich::vcovHC(ref,type=type)%*%T
        expect_equal(c(fastglm_vcov(fit,type)),c(expected),tolerance=2e-7)
        expected<-T%*%sandwich::vcovCL(ref,cluster=groups[ii],type=type,cadjust=TRUE)%*%T
        expect_equal(c(fastglm_vcov(fit,type,cluster=groups)),c(expected),tolerance=2e-7)
    }
    # At a BR estimate the convention uses ordinary scores and their Jacobian,
    # with no derivative of the global bias adjustment inserted into the bread.
    br<-fastglm_ordinal(x,ordered(y,levels=0:1),weights=w,offset=off,firth=TRUE,tol=1e-10)
    X<-cbind(1,x[ii,,drop=FALSE]);beta<-c(-br$alpha,br$beta)
    mu<-plogis(drop(X%*%beta)+off[ii]);H<-crossprod(X,X*(mu*(1-mu)))
    scores<-X*(y[ii]-mu);B<-solve(H)
    expected<-T%*%B%*%crossprod(scores)%*%B%*%T
    expect_equal(c(fastglm_vcov(br)),c(expected),tolerance=2e-10)
    expect_match(attr(fastglm_vcov(br),"covariance.scope"),"mean-bias-reduced estimate")
})

test_that("ordinal ML covariance agrees with the polr ordinary-score reference", {
    skip_if_not_installed("MASS");skip_if_not_installed("sandwich")
    set.seed(48112);x<-cbind(x=rnorm(250),z=rnorm(250))
    y<-ordered(findInterval(drop(x%*%c(.5,-.4))+rlogis(250),c(-.8,.6))+1,levels=1:3)
    fit<-fastglm_ordinal(x,y,tol=1e-10)
    ref<-MASS::polr(y~x+z,data=data.frame(y,x),Hess=TRUE,control=list(maxit=1000,reltol=1e-12))
    reference<-sandwich::sandwich(ref)
    order<-c(names(ref$zeta),names(coef(ref)))
    expect_equal(c(fastglm_vcov(fit)),c(reference[order,order]),tolerance=2e-6)
})

test_that("ordinal covariance respects affine coordinates and replication scaling", {
    set.seed(48113);x<-cbind(x=rnorm(130),z=runif(130,-1,1))
    y<-ordered(sample(1:4,130,TRUE),levels=1:4);off<-runif(130,-.2,.2)
    shift<-c(.8,-.2);scale<-c(.01,30)
    moved<-sweep(sweep(x,2,shift,"+"),2,scale,"*")
    for(br in c(FALSE,TRUE)) {
        a<-fastglm_ordinal(x,y,offset=off,firth=br,tol=1e-10)
        b<-fastglm_ordinal(moved,y,offset=off,firth=br,tol=1e-10)
        T<-diag(5);T[1:3,4:5]<-matrix(shift,3,2,byrow=TRUE);T[4:5,4:5]<-diag(1/scale)
        expect_equal(c(fastglm_vcov(b)),c(T%*%fastglm_vcov(a)%*%t(T)),tolerance=2e-8)
        expect_equal(c(fastglm_vcov(a,block.size=1)),c(fastglm_vcov(a,block.size=8192)),tolerance=2e-10)
    }
    a<-fastglm_ordinal(x,y,offset=off,tol=1e-10)
    b<-fastglm_ordinal(x,y,offset=off,weights=rep(17,nrow(x)),tol=1e-10)
    expect_equal(c(fastglm_vcov(b)),c(fastglm_vcov(a))/17,tolerance=2e-8)
    groups<-rep(1:13,length.out=nrow(x))
    expect_equal(c(fastglm_vcov(b,cluster=groups,cadjust=FALSE)),
                 c(fastglm_vcov(a,cluster=groups,cadjust=FALSE)),tolerance=2e-8)
})

test_that("ordinal cluster IDs align with formula omissions and ignore zero-frequency rows", {
    d<-data.frame(y=ordered(rep(1:3,4)),x=c(-1,NA,.3,.8,-.2,.5,1,-.5,.2,.4,.7,1e300),
                  w=c(rep(1,11),0),off=c(rep(.1,11),1e300))
    fit<-fastglm_ordinal(y~x+offset(off),data=d,weights=w,firth=TRUE,na.action=na.exclude)
    groups<-c(rep(1:3,3),1,2,NA)
    expect_equal(c(fastglm_vcov(fit,cluster=groups)),c(fastglm_vcov(fit,cluster=groups[-2])))
    clean<-fastglm_ordinal(y~x+offset(off),data=d[-c(2,12),],weights=w,firth=TRUE)
    expect_equal(c(fastglm_vcov(fit)),c(fastglm_vcov(clean)),tolerance=2e-8)
    expect_equal(c(fastglm_vcov(fit,cluster=groups)),
                 c(fastglm_vcov(clean,cluster=groups[-c(2,12)])),tolerance=2e-8)
    labels<-c(1,1+1e-15,2)[groups]
    expect_equal(c(fastglm_vcov(fit,cluster=labels)),c(fastglm_vcov(fit,cluster=groups)),tolerance=1e-12)
    expect_identical(attr(fastglm_vcov(fit,cluster=labels),"clusters"),3L)
    expect_error(fastglm_vcov(fit,cluster=1:10),"must match retained rows")
    expect_error(fastglm_vcov(fit,cluster=rep(NA,11)),"positive-frequency rows")
})

test_that("ordinal robust inference rejects unsupported boundaries and corrections", {
    fit<-fastglm_ordinal(matrix(c(-1,0,1,2),ncol=1),ordered(c(1,2,3,2),levels=1:3),firth=TRUE)
    expect_error(fastglm_vcov(fit,"HC2"),"HC0.*HC1")
    expect_error(fastglm_vcov(fit,cluster=rep(1,4)),"at least two")
    expect_error(fastglm_vcov(fit,cluster=matrix(1:4,4)),"vector of identifiers")
    expect_error(fastglm_vcov(fit,cadjust=1),"cadjust")
    expect_error(fastglm_vcov(fit,block.size=0),"block.size")
    fit$converged<-FALSE
    expect_error(fastglm_vcov(fit),"converged fit")
    for(counts in list(c(0,5,2),c(3,0,2))) {
        fit<-suppressWarnings(fastglm_ordinal(matrix(numeric(),1,0),matrix(counts,1),firth=TRUE))
        expect_error(fastglm_vcov(fit),"every declared category")
    }
    fit<-fastglm_ordinal(matrix(c(-1,1),ncol=1),ordered(1:2),firth=TRUE)
    expect_error(fastglm_vcov(fit,"HC1"),"total frequency greater")
    expect_error(fastglm:::.ordinal_observed_inverse(diag(c(1,0))),"not positive definite")
    extreme<-fastglm_ordinal(matrix(numeric(),1,0),matrix(rep(1e308,3),1),weights=1e-308,firth=TRUE)
    simple<-fastglm_ordinal(matrix(numeric(),1,0),matrix(rep(1,3),1),firth=TRUE)
    expect_equal(c(fastglm_vcov(extreme)),c(fastglm_vcov(simple)),tolerance=1e-10)
})

test_that("ordinal covariance conveniences preserve parameter order and supplied uncertainty", {
    set.seed(48114);x<-cbind(x=rnorm(150),z=runif(150,-1,1))
    y<-ordered(sample(1:3,150,TRUE),levels=1:3)
    fit<-fastglm_ordinal(x,y,firth=TRUE)
    v<-fastglm_vcov(fit)
    expect_equal(vcov(fit,type="HC0"),v)
    groups<-rep(1:15,each=10)
    expect_equal(vcov(fit,type="HC1",cluster=groups),fastglm_vcov(fit,"HC1",cluster=groups))
    location<-vcov(fit,component="location",type="HC0")
    expect_equal(c(location),c(v[3:4,3:4]))
    expect_identical(attr(location,"covariance.scope"),attr(v,"covariance.scope"))
    tab<-summary(fit,vcov.=v)
    expect_equal(unname(c(tab$thresholds[,2],tab$location[,2])),unname(sqrt(diag(v))))
    shuffled<-v[c(4,2,1,3),c(4,2,1,3)]
    expect_equal(summary(fit,vcov.=shuffled)$location,tab$location)
    expect_equal(summary(fit,vcov.=shuffled)$thresholds,tab$thresholds)
    if(requireNamespace("broom",quietly=TRUE)) {
        tidy<-broom::tidy(fit,vcov.=v,conf.int=TRUE)
        expect_equal(tidy$std.error,unname(sqrt(diag(v))))
        expect_equal(tidy$conf.high-tidy$estimate,qnorm(.975)*tidy$std.error)
    }
})
