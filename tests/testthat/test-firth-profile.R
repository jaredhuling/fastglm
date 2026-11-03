test_that("Firth native objective and adjusted score match independent calculations", {
    set.seed(14921);x<-cbind(1,rnorm(45),rbinom(45,1,.4));b<-c(-.4,.8,-.3)
    w<-rep(0:3,length.out=45);off<-runif(45,-.2,.2);y<-rbinom(45,1,.4)
    likelihood<-function(beta) {
        eta<-drop(x%*%beta)+off;mu<-plogis(eta)
        sum(w*dbinom(y,1,mu,log=TRUE))+.5*as.numeric(determinant(crossprod(x,x*(w*mu*(1-mu))),logarithm=TRUE)$modulus)
    }
    ev<-fastglm:::firth_profile_evaluate(x,y,w,off,b)
    gradient<-vapply(seq_along(b),function(j) {
        a<-z<-b;a[j]<-a[j]+1e-5;z[j]<-z[j]-1e-5;(likelihood(a)-likelihood(z))/2e-5
    },numeric(1))
    expect_equal(ev$value,likelihood(b),tolerance=1e-10)
    expect_equal(as.numeric(ev$gradient),gradient,tolerance=1e-7)
    keep<-w>0
    reduced<-fastglm:::firth_profile_evaluate(x[keep,,drop=FALSE],y[keep],w[keep],off[keep],b)
    expect_equal(ev$value,reduced$value,tolerance=1e-12)
    expect_equal(ev$gradient,reduced$gradient,tolerance=1e-12)
})

test_that("profile intervals and joint tests agree with logistf", {
    skip_if_not_installed("logistf")
    set.seed(14922);n<-120;d<-data.frame(x=rnorm(n),z=rbinom(n,1,.4))
    d$y<-rbinom(n,1,plogis(-.3+.6*d$x-.4*d$z))
    f<-fastglm(y~x+z,d,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    l<-logistf::logistf(y~x+z,d,control=logistf::logistf.control(maxit=300),
        plcontrol=logistf::logistpl.control(maxit=300))
    ci<-confint(f,parm=c("z","(Intercept)","x"),method="profile")
    expect_true(all(attr(ci,"diagnostics")$converged))
    expect_equal(unname(ci[,]),unname(cbind(l$ci.lower,l$ci.upper)[c(3,1,2),]),tolerance=2e-5)
    test<-fastglm_profile_test(f,c("z","x"),c(.2,-.1))
    reference<-logistf::logistftest(l,test=c(3,2),values=c(.2,-.1))
    expect_equal(unname(test$statistic),unname(2*(reference$loglik[1]-reference$loglik[2])),tolerance=1e-7)
    expect_equal(test$p.value,unname(reference$prob),tolerance=1e-7)
    expect_identical(names(test$null.value),c("z","x"))
    expect_equal(fastglm_profile_test(f,c(2,3),c(-.1,.2))$statistic,test$statistic,tolerance=1e-9)
    expect_equal(confint(f),confint(f,method="wald"))
})

test_that("profile inference remains finite under complete separation", {
    skip_if_not_installed("logistf")
    d<-data.frame(x=rep(0:1,each=10));d$y<-d$x
    f<-fastglm(y~x,d,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    l<-logistf::logistf(y~x,d,control=logistf::logistf.control(maxit=300),
        plcontrol=logistf::logistpl.control(maxit=300))
    ci<-confint(f,method="profile")
    expect_true(all(is.finite(ci)))
    expect_equal(unname(ci[,]),unname(cbind(l$ci.lower,l$ci.upper)),tolerance=2e-5)
    expect_lt(max(abs(attr(ci,"diagnostics")$lr.error)),2e-5)
    for(j in 1:2) for(side in 1:2) {
        endpoint<-fastglm_profile_test(f,j,ci[j,side])
        expect_equal(unname(endpoint$statistic),qchisq(.95,1),tolerance=2e-5)
        expect_equal(endpoint$p.value,.05,tolerance=2e-6)
    }
})

test_that("the published sex2 example agrees with logistf profile inference", {
    skip_if_not_installed("logistf")
    data("sex2",package="logistf",envir=environment())
    formula<-case~age+oc+vic+vicl+vis+dia
    f<-fastglm(formula,sex2,family=binomial(),firth=TRUE,tol=1e-10,maxit=1000)
    reference<-logistf::logistf(formula,sex2,plconf=c(2,7),
        control=logistf::logistf.control(maxit=300),plcontrol=logistf::logistpl.control(maxit=300))
    ci<-confint(f,c("dia","age"),method="profile")
    expect_true(all(attr(ci,"diagnostics")$converged))
    expect_equal(unname(ci[,]),unname(cbind(reference$ci.lower,reference$ci.upper)[c(7,2),]),tolerance=2e-5)
    tt<-fastglm_profile_test(f,c("oc","vic"))
    lr<-logistf::logistftest(reference,test=c(3,4))
    expect_equal(unname(tt$statistic),unname(2*(lr$loglik[1]-lr$loglik[2])),tolerance=1e-6)
    expect_lt(unname(fastglm_profile_test(f,c("oc","vic"),coef(f)[c("oc","vic")])$statistic),1e-8)
})

test_that("grouped responses retain frequency weights and offsets", {
    skip_if_not_installed("logistf")
    set.seed(14923);n<-35;d<-data.frame(x=rnorm(n),off=runif(n,-.3,.3))
    trials<-rep(5:9,length.out=n);w<-rep(0:3,length.out=n)
    d$s<-rbinom(n,trials,plogis(-.2+.5*d$x+d$off));d$f<-trials-d$s
    f<-fastglm(cbind(s,f)~x+offset(off),d,weights=w,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    dd<-d[rep(seq_len(n),2),];dd$y<-rep(0:1,each=n);ww<-c(w*d$f,w*d$s)
    keep<-ww>0
    l<-logistf::logistf(y~x+offset(off),dd[keep,],weights=ww[keep],
        control=logistf::logistf.control(maxit=300),plcontrol=logistf::logistpl.control(maxit=300))
    ci<-confint(f,method="profile")
    expect_equal(unname(ci[,]),unname(cbind(l$ci.lower,l$ci.upper)),tolerance=2e-5)
    test<-fastglm_profile_test(f,"x",-.2)
    reference<-logistf::logistftest(l,test=2,values=-.2)
    expect_equal(unname(test$statistic),unname(2*(reference$loglik[1]-reference$loglik[2])),tolerance=1e-7)
    f2<-fastglm(cbind(s,f)~x+offset(off),d[w>0,],weights=w[w>0],family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    expect_equal(confint(f2,method="profile")[,],ci[,],tolerance=1e-7)
})

test_that("profile constraints retain the full Jeffreys penalty", {
    set.seed(14924);n<-60;x<-cbind(intercept=1,x=rnorm(n),z=rnorm(n));y<-rbinom(n,1,plogis(x%*%c(-.4,.9,.3)))
    f<-fastglm(x,y,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    tt<-fastglm_profile_test(f,"x",.2)
    objective<-function(b) {
        mu<-plogis(drop(x%*%b));sum(dbinom(y,1,mu,log=TRUE))+
            .5*as.numeric(determinant(crossprod(x,x*mu*(1-mu)),logarithm=TRUE)$modulus)
    }
    nuisance<-optim(coef(f)[c(1,3)],function(g) -objective(c(g[1],.2,g[2])),method="BFGS",control=list(reltol=1e-12))
    expect_equal(tt$loglik[["restricted"]],-nuisance$value,tolerance=1e-8)
    expect_equal(unname(tt$statistic),2*(objective(coef(f))+nuisance$value),tolerance=1e-7)
    permutation<-c(3,1,2);fp<-fastglm(x[,permutation],y,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    expect_equal(fastglm_profile_test(fp,"x",.2)$statistic,tt$statistic,tolerance=1e-8)
    xs<-x;xs[,2]<-xs[,2]*1e6
    fs<-fastglm(xs,y,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    expect_equal(unname(confint(fs,"x",method="profile")[,])*1e6,
        unname(confint(f,"x",method="profile")[,]),tolerance=1e-6)
})

test_that("unsupported fits and failed profile computations are explicit", {
    d<-data.frame(x=rep(0:1,each=10));d$y<-d$x
    f<-fastglm(y~x,d,family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    ordinary<-f;ordinary$firth<-FALSE
    expect_error(confint(ordinary,method="profile"),"binomial-logit Firth")
    sparse<-f;sparse$x<-as(Matrix::Matrix(f$x,sparse=TRUE),"dgCMatrix")
    expect_error(confint(sparse,method="profile"),"retained dense")
    fractional<-f;fractional$prior.weights[1]<-.5
    expect_error(confint(fractional,method="profile"),"integer effective successes and failures")
    expect_error(fastglm_profile_test(f,c(2,2)),"distinct known")
    expect_error(fastglm_profile_test(f,"missing"),"distinct known")
    expect_warning(bad<-fastglm_profile_test(f,"x",0,control=list(maxit=1)),"profile test failed")
    expect_true(is.na(bad$p.value))
    expect_false(bad$diagnostics$converged)
    intercept<-fastglm(matrix(1,20,1),rep(0,20),family=binomial(),firth=TRUE,tol=1e-10,maxit=500)
    expect_warning(ci<-confint(intercept,level=.9999999,method="profile",control=list(max.expand=1)),"profile endpoints failed")
    expect_true(any(is.na(ci)))
    expect_true(any(grepl("bracketed",attr(ci,"diagnostics")$message)))
})
