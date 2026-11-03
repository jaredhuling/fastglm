local({
    fixture <- function(kind,dist="negbin",link="logit",estimated=dist=="negbin",n=35L) {
        x<-cbind(intercept=1,x=seq(-.7,.7,length.out=n))
        zz<-cbind(intercept=1,z=sin(seq_len(n))*.6)
        two<-kind %in% c("hurdle","zi","twopart")
        b<-c(intercept=if(kind=="nb" && link!="log") 1.6 else .6,x=.3)
        g<-c(intercept=if(link=="log") -1 else -.4,z=.2)
        if(link %in% c("identity","sqrt","inverse","1/mu^2"))
            g<-c(intercept=switch(link,identity=.6,sqrt=.7,inverse=1.7,`1/mu^2`=2.2),z=.1)
        f<-list(coefficients=if(two) list(count=b,zero=g) else b,dist=dist,link=link,
            theta=1.3,theta.estimated=estimated,family=if(kind=="nb") negbin(1.3,link=link))
        class(f)<-if(kind=="twopart") c("fastglm_twopart","fastglm_hurdle") else paste0("fastglm_",kind)
        if(kind=="nb") class(f)<-c(class(f),"fastglm")
        y<-rep(c(0,1,2,5,3),length.out=n)
        if(kind=="truncated") y<-pmax(y,1)
        if(kind=="twopart") y<-y/1.3
        z<-list(X=x,Z=if(two) zz,y=y,weights=rep(c(0,.4,1,2,3),length.out=n),
            offset=list(count=.08*cos(seq_len(n)),zero=.05*sin(seq_len(n))))
        list(object=f,block=z,kind=kind)
    }
    # Independent density evaluation: no package score/information helpers.
    # Gamma shape=1 differentiates the deliberately dispersion-unscaled
    # positive-mean equations used in the two-part sandwich, not the full
    # fitted Gamma likelihood with its estimated dispersion parameter.
    loglik <- function(parameters,a) {
        f<-a$object;z<-a$block;kind<-a$kind;y<-z$y
        mu<-if(kind=="nb") f$family$linkinv(drop(z$X%*%parameters[1:2])+z$offset$count) else
            exp(drop(z$X%*%parameters[1:2])+z$offset$count)
        theta<-if(f$theta.estimated) exp(tail(parameters,1)) else f$theta
        value<-switch(f$dist,negbin=dnbinom(y,mu=mu,size=theta,log=TRUE),
            poisson=dpois(y,mu,log=TRUE),Gamma=dgamma(y,shape=1,scale=mu,log=TRUE))
        lp0<-if(f$dist=="negbin") -theta*log1p(mu/theta) else -mu
        if(kind=="truncated") value<-value-log(-expm1(lp0))
        if(kind %in% c("hurdle","zi","twopart")) {
            pi<-make.link(f$link)$linkinv(drop(z$Z%*%parameters[3:4])+z$offset$zero)
            value<-switch(kind,
                zi=ifelse(y==0,log(pi+(1-pi)*exp(lp0)),log1p(-pi)+value),
                hurdle=ifelse(y==0,log1p(-pi),log(pi)+value-log(-expm1(lp0))),
                twopart=ifelse(y==0,log1p(-pi),log(pi)+value))
        }
        sum(z$weights*value)
    }

    test_that("analytic count information differentiates the weighted likelihood", {
        skip_if_not_installed("numDeriv")
        for(kind in c("nb","truncated","hurdle","zi","twopart")) {
            distributions<-if(kind=="nb") "negbin" else if(kind=="twopart") "Gamma" else c("poisson","negbin")
            for(dist in distributions) {
                links<-if(kind=="nb") c("log","sqrt","identity") else if(kind=="truncated") "log" else
                    c("logit","probit","cloglog","log")
                if(kind=="twopart") links<-c(links,"cauchit","identity","sqrt","inverse","1/mu^2")
                for(link in links) for(estimated in if(dist=="negbin") c(FALSE,TRUE) else FALSE) {
                    a<-fixture(kind,dist,link,estimated);parameters<-fastglm:::.count_parameters(a$object)
                    numerical<--numDeriv::hessian(loglik,parameters,a=a)
                    analytic<-fastglm:::.count_information_block(a$object,a$block,parameters)
                    label<-paste(kind,dist,link,estimated)
                    expect_equal(analytic,numerical,tolerance=2e-7,info=label)
                    expect_identical(analytic,t(analytic),info=label)
                }
            }
        }
    })

    test_that("information is additive over blocks and linear in prior weights", {
        a<-fixture("zi");f<-a$object;z<-a$block;b<-fastglm:::.count_parameters(f)
        information<-function(z) fastglm:::.count_information_block(f,z,b)
        part<-function(i) list(X=z$X[i,,drop=FALSE],Z=z$Z[i,,drop=FALSE],y=z$y[i],
            weights=z$weights[i],offset=lapply(z$offset,`[`,i))
        i<-seq_len(nrow(z$X))
        expect_equal(information(z),information(part(i%%2==0))+information(part(i%%2==1)),tolerance=1e-13)
        scaled<-z;scaled$weights<-3*z$weights
        expect_equal(information(scaled),3*information(z),tolerance=1e-13)
        zero<-z;zero$weights[]<-0;zero$offset$count[]<-1e300
        expect_identical(information(zero),matrix(0,length(b),length(b)))
        expect_identical(information(part(integer())),matrix(0,length(b),length(b)))
        invalid<-z;invalid$offset$count[]<-1000
        expect_error(fastglm:::.count_inverse_information(information(invalid)),
            "joint information is singular or not positive definite")
    })

    test_that("analytic curvature remains accurate for rare positive counts", {
        theta<-1.5
        for(mu in c(1e-6,1e-10,1e-14)) {
            f<-structure(list(coefficients=c(intercept=log(mu)),dist="poisson",theta=theta,
                theta.estimated=FALSE),class="fastglm_truncated")
            z<-list(X=matrix(1,1,1),Z=NULL,y=1,weights=1,offset=list(count=0,zero=0))
            ii<-fastglm:::.count_information_block(f,z,fastglm:::.count_parameters(f))
            expect_equal(ii[1,1]/(mu/2),1,tolerance=1e-5)
            f$dist<-"negbin";f$theta.estimated<-TRUE
            ii<-fastglm:::.count_information_block(f,z,fastglm:::.count_parameters(f))
            leading<-mu/2*matrix(c(1+1/theta,-1/theta,-1/theta,1/theta),2,2)
            expect_equal(ii/mu,leading/mu,tolerance=1e-5)
        }
        # At large theta the log-theta curvature is first order in 1/theta.
        theta<-1e7;mu<-2
        f<-structure(list(coefficients=c(intercept=log(mu)),family=negbin(theta),theta=theta,
            theta.estimated=TRUE),class=c("fastglm_nb","fastglm"))
        for(y in c(0,1,4)) {
            z$y<-y
            ii<-fastglm:::.count_information_block(f,z,fastglm:::.count_parameters(f))
            leading<-(y-(y-mu)^2)/(2*theta)
            expect_lt(abs(ii[2,2]-leading),20/theta^2)
        }
    })

    test_that("information respects the numerical limits of R links", {
        skip_if_not_installed("numDeriv")
        # At the limits, differentiate the retained score equations because
        # linkinv and mu.eta have different numerical floors in stats.
        for(kind in c("hurdle","zi","twopart")) for(link in c("logit","probit","cloglog","log")) {
            a<-fixture(kind,if(kind=="twopart") "Gamma" else "negbin",link)
            b<-fastglm:::.count_parameters(a$object)
            a$block$offset<-list(count=rep(0,35),zero=rep(0,35))
            for(extreme in switch(link,logit=c(-35,35),probit=c(-8.2,8.2),cloglog=c(-40,5),log=-40)) {
                b[3:4]<-c(extreme,0)
                numeric<--numDeriv::jacobian(function(bb) colSums(fastglm:::.count_score(a$object,a$block,bb)),b)
                numeric<-(numeric+t(numeric))/2
                analytic<-fastglm:::.count_information_block(a$object,a$block,b)
                expect_equal(analytic,numeric,tolerance=3e-6,info=paste(kind,link,extreme))
            }
        }
        for(kind in c("nb","truncated","hurdle","zi")) {
            a<-fixture(kind,link=if(kind %in% c("nb","truncated")) "log" else "logit")
            b<-fastglm:::.count_parameters(a$object);b[1:2]<-c(-40,0)
            a$block$offset$count[]<-0
            numeric<--numDeriv::jacobian(function(bb) colSums(fastglm:::.count_score(a$object,a$block,bb)),b)
            numeric<-(numeric+t(numeric))/2
            analytic<-fastglm:::.count_information_block(a$object,a$block,b)
            expect_equal(analytic,numeric,tolerance=3e-6,info=kind)
        }
        a<-fixture("twopart","Gamma","cauchit");b<-fastglm:::.count_parameters(a$object)
        for(extreme in c(-1e8,1e8,-2e15,2e15)) {
            b[3:4]<-c(extreme,0)
            numeric<--numDeriv::jacobian(function(bb) colSums(fastglm:::.count_score(a$object,a$block,bb)),b)
            expect_equal(fastglm:::.count_information_block(a$object,a$block,b),
                (numeric+t(numeric))/2,tolerance=3e-6)
        }
    })
})

test_that("two-part Cauchy links retain sandwich inference", {
    set.seed(9103);x<-runif(80,-1,1)
    y<-ifelse(rbinom(80,1,.6),rgamma(80,2,2),0)
    f<-fastglm_twopart(y~x,data=data.frame(x,y),link="cauchit",tol=1e-10)
    expect_true(f$converged)
    v<-fastglm_vcov(f)
    expect_true(all(is.finite(v)))
    expect_true(all(eigen(v,symmetric=TRUE,only.values=TRUE)$values>0))
})

test_that("joint information and robust covariance agree across count backends", {
    set.seed(8502);n<-360L
    x<-cbind(intercept=1,x=rbinom(n,1,.45));zz<-cbind(intercept=1,z=rbinom(n,1,.55))
    w<-rep(c(0,3,1,2,1),length.out=n);off<-list(count=runif(n,-.1,.1),zero=runif(n,-.1,.1))
    count<-rnbinom(n,mu=exp(1+.3*x[,2]),size=.8)
    continuous<-rbinom(n,1,.6)*rgamma(n,shape=2,scale=exp(.4+.2*x[,2])/2)
    groups<-rep(seq_len(36L),each=10L)
    for(kind in c("nb","hurdle","twopart")) {
        y<-if(kind=="twopart") continuous else count
        fit<-switch(kind,nb=fastglm_nb,hurdle=fastglm_hurdle,twopart=fastglm_twopart)
        extra<-switch(kind,nb=list(),hurdle=list(dist="negbin"),twopart=list())
        args<-if(kind=="nb") list(y=y,weights=w,offset=off$count,tol=1e-10,outer.tol=1e-9) else
            c(list(response=y,z=zz,weights=w,offset=off,tol=1e-10),extra)
        dense<-do.call(fit,c(list(x),args))
        sparse<-do.call(fit,c(list(as(Matrix::Matrix(x,sparse=TRUE),"dgCMatrix")),args))
        reads<-0L
        src<-fastglm_source(function(k) {
            reads<<-reads+1L;i<-seq.int((k-1L)*60L+1L,k*60L)
            list(X=x[i,,drop=FALSE],Z=zz[i,,drop=FALSE],y=y[i],weights=w[i],offset=lapply(off,`[`,i))
        },6L)
        streamed<-do.call(fit,c(list(src),extra,list(tol=1e-10)))
        information<-fastglm:::.count_information(dense,block.size=49L)$information
        for(f in list(sparse,streamed))
            expect_equal(unname(fastglm:::.count_information(f,block.size=43L)$information),unname(information),tolerance=3e-5)
        reads<-0L;v<-fastglm_vcov(streamed,type="HC0",full=TRUE)
        expect_identical(reads,6L)
        expect_equal(unname(v),unname(fastglm_vcov(dense,type="HC0",full=TRUE)),tolerance=3e-5)
        expect_equal(unname(fastglm_vcov(streamed,type="HC1",cluster=function(k) groups[((k-1)*60+1):(k*60)],full=TRUE)),
            unname(fastglm_vcov(dense,type="HC1",cluster=groups,full=TRUE)),tolerance=3e-5)
        if(kind=="nb") {
            reference<-fastglm:::.count_natural_covariance(solve(information),dense)
            expect_equal(vcov(dense,full=TRUE),reference,tolerance=1e-8)
            dense$theta.boundary<-TRUE
            expect_error(fastglm_vcov(dense,full=TRUE),"theta boundary")
        }
        if(kind=="twopart") {
            dense$positive$firth<-TRUE
            expect_error(fastglm_vcov(dense),"adjusted scores")
        }
    }
})
