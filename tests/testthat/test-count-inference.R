test_that("joint count scores differentiate the observation likelihood", {
    set.seed(7401)
    n<-280;d<-data.frame(x=rnorm(n));d$y<-rnbinom(n,mu=exp(.4+.3*d$x),size=1.4)
    d$y[runif(n)<.25]<-0
    objects<-list(
        fastglm_nb(model.matrix(~x,d),d$y,tol=1e-10,outer.tol=1e-9),
        fastglm_hurdle(y~x,d,dist="negbin",outer.tol=1e-9,outer.maxit=150),
        fastglm_truncated(y~x,data=subset(d,y>0),dist="negbin",outer.tol=1e-9),
        fastglm_zi(y~x,d,dist="negbin",em.maxit=500,em.tol=1e-10))
    for(f in objects) {
        pars<-fastglm:::.count_parameters(f)
        blocks<-list();fastglm:::.count_blocks(f,function(z,i,k) blocks[[length(blocks)+1L]]<<-z)
        z<-blocks[[1L]];pc<-if(is.list(f$coefficients)) length(f$coefficients$count) else length(coef(f))
        ll<-function(b) {
            mu<-exp(drop(z$X%*%b[seq_len(pc)])+z$offset$count);theta<-exp(tail(b,1))
            val<-dnbinom(z$y,mu=mu,size=theta,log=TRUE)
            p0<-dnbinom(0,mu=mu,size=theta)
            if(inherits(f,"fastglm_truncated")) val<-val-log1p(-p0)
            if(inherits(f,c("fastglm_hurdle","fastglm_zi"))) {
                pi<-plogis(drop(z$Z%*%b[pc+seq_len(ncol(z$Z))])+z$offset$zero)
                if(inherits(f,"fastglm_zi")) val<-ifelse(z$y==0,log(pi+(1-pi)*p0),log1p(-pi)+val)
                else val<-ifelse(z$y==0,log1p(-pi),log(pi)+val-log1p(-p0))
            }
            z$weights*val
        }
        numeric.score<-vapply(seq_along(pars),function(j) {
            a<-b<-pars;h<-1e-5;a[j]<-a[j]+h;b[j]<-b[j]-h;(ll(a)-ll(b))/(2*h)
        },numeric(length(z$y)))
        expect_equal(unname(fastglm:::.count_score(f,z,pars)),unname(numeric.score),tolerance=2e-7)
        v<-fastglm_vcov(f,type="HC0",full=TRUE)
        expect_equal(dim(v),c(length(pars),length(pars)))
        expect_true(all(eigen(v,symmetric=TRUE,only.values=TRUE)$values>0))
    }
})

test_that("two-part cluster covariance includes cross-component scores", {
    set.seed(7402);n<-300;x<-cbind(1,rnorm(n));groups<-rep(1:30,each=10)
    y<-rbinom(n,1,plogis(.3+.2*x[,2]))*rgamma(n,2,scale=exp(.1+.3*x[,2])/2)
    w<-rep(c(1,2,3),length.out=n);w[groups==1]<-0
    f<-fastglm_twopart(x,response=y,weights=w,method=2,tol=1e-11)
    mu<-exp(drop(x%*%coef(f,"count")));pi<-plogis(drop(x%*%coef(f,"zero")))
    s<-cbind(x*(w*(y>0)*(y/mu-1)),x*(w*((y>0)-pi)))
    h<-matrix(0,4,4);h[1:2,1:2]<-crossprod(x,x*(w*(y>0)*y/mu))
    h[3:4,3:4]<-crossprod(x,x*(w*pi*(1-pi)))
    b<-solve(h);ss<-rowsum(s[w>0,,drop=FALSE],groups[w>0]);nn<-sum(w>0);g<-nrow(ss)
    reference<-b%*%crossprod(ss)%*%b*g/(g-1)*(nn-1)/(nn-4)
    v<-fastglm_vcov(f,type="HC1",cluster=groups)
    expect_equal(unname(v),reference,tolerance=1e-7)
    expect_true(any(abs(v[1:2,3:4])>1e-6))
    expect_error(fastglm_vcov(f,type="HC3"),"HC0 or HC1")
})

test_that("count averages use joint uncertainty and component contrasts", {
    set.seed(7403);n<-240;x<-cbind(1,rnorm(n));colnames(x)<-c("intercept","x")
    y<-rbinom(n,1,.6)*rgamma(n,2,scale=exp(.2+.4*x[,2])/2)
    f<-fastglm_twopart(x,response=y,tol=1e-11)
    v<-fastglm_vcov(f,type="HC0",full=TRUE)
    pi<-plogis(drop(x%*%coef(f,"zero")));mu<-exp(drop(x%*%coef(f,"count")))
    g<-c(colMeans(x*pi*mu),colMeans(x*mu*pi*(1-pi)))
    a<-fastglm_average(f,vcov.=v)
    expect_equal(a$estimate,mean(pi*mu),tolerance=1e-10)
    expect_equal(a$std.error,sqrt(drop(g%*%v%*%g)),tolerance=1e-7)
    alter<-function(design) {design$count[,2]<-design$count[,2]+1;design}
    ac<-fastglm_average(f,contrast=alter,vcov.=v)
    expect_equal(ac$estimate,mean(pi*mu)*(exp(coef(f,"count")[2])-1),tolerance=1e-9,ignore_attr=TRUE)
    cb<-function(k) {i<-((k-1)*60+1):(k*60);list(X=x[i,,drop=FALSE],y=y[i])}
    stream<-fastglm_twopart(fastglm_source(cb,4),tol=1e-11)
    expect_equal(fastglm_average(stream),fastglm_average(f),tolerance=1e-7)
    expect_equal(fastglm_vcov(stream),fastglm_vcov(f),tolerance=1e-7)
    expect_equal(predict(stream,se.fit=TRUE)$se.fit,predict(f,se.fit=TRUE)$se.fit,tolerance=1e-7)
    values<-list();predict(stream,sink=function(value,k) values[[k]]<<-value)
    expect_equal(unlist(values,use.names=FALSE),predict(f),tolerance=1e-8)
    expect_equal(residuals(stream),residuals(f),tolerance=1e-8)
    expect_equal(nobs(stream),n)
})

test_that("bootstrap refits reproduce explicit row and cluster samples", {
    set.seed(7404);n<-100;x<-cbind(1,rnorm(n));y<-rnorm(n);w<-runif(n,.5,2);off<-runif(n,-.2,.2)
    f<-fastglm(x,y,weights=w,offset=off)
    old<-.Random.seed
    b<-fastglm_bootstrap(f,R=7,seed=32)
    expect_identical(.Random.seed,old)
    b2<-fastglm_bootstrap(f,R=7,seed=32)
    expect_identical(b$replicates,b2$replicates)
    set.seed(b$seeds[1]);i<-sample.int(n,n,replace=TRUE)
    reference<-glm.fit(x[i,,drop=FALSE],y[i],weights=w[i],offset=off[i])
    expect_equal(unname(b$replicates[1,]),unname(coef(reference)),tolerance=1e-9)
    groups<-rep(1:20,each=5)
    c<-fastglm_bootstrap(f,R=7,seed=33,type="cluster",cluster=groups)
    units<-split(seq_len(n),as.character(groups));set.seed(c$seeds[1])
    i<-unlist(units[sample.int(length(units),length(units),replace=TRUE)],use.names=FALSE)
    reference<-glm.fit(x[i,,drop=FALSE],y[i],weights=w[i],offset=off[i])
    expect_equal(unname(c$replicates[1,]),unname(coef(reference)),tolerance=1e-9)
    expect_warning(bad<-fastglm_bootstrap(f,R=3,seed=34,control=list(start=Inf)),"refits failed")
    expect_true(all(!bad$diagnostics$converged))
    expect_true(all(is.na(bad$conf.int)))
    expect_true(all(nzchar(bad$diagnostics$error)))
})

test_that("bootstrap supports count models and repeatable streaming sources", {
    set.seed(7405);n<-240;x<-cbind(1,rnorm(n));colnames(x)<-c("intercept","x")
    y<-rnbinom(n,mu=exp(.5+.3*x[,2]),size=1.5)
    d<-data.frame(y=y,x=x[,2])
    fits<-list(fastglm_nb(x,y),fastglm_hurdle(y~x,d,dist="poisson"),
        fastglm_truncated(x[y>0,,drop=FALSE],y[y>0],dist="poisson"))
    for(f in fits) {
        b<-fastglm_bootstrap(f,R=4,seed=35,type="parametric")
        expect_equal(b$successful,4L)
        expect_equal(ncol(b$replicates),length(b$estimate))
    }
    cb<-function(k) {i<-((k-1)*60+1):(k*60);list(X=x[i,,drop=FALSE],y=y[i])}
    f<-fastglm_nb(fastglm_source(cb,4))
    a<-fastglm_bootstrap(f,R=4,seed=36,type="pairs")
    b<-fastglm_bootstrap(f,R=4,seed=36,type="pairs")
    expect_equal(a$successful,4L)
    expect_identical(a$replicates,b$replicates)
})

test_that("cluster bootstrap scans skipped blocks in sequential sources", {
    set.seed(7410);n<-120;x<-cbind(intercept=1,x=rnorm(n));y<-rnorm(n)
    position<-0L
    cb<-function(k) {
        if(k!=position+1L) stop("reader skipped a block")
        position<<-k;i<-seq.int((k-1L)*20L+1L,k*20L)
        list(X=x[i,,drop=FALSE],y=y[i])
    }
    reset<-function() {
        if(!position %in% c(0L,6L)) stop("reader reset before the final block")
        position<<-0L
    }
    f<-fastglm_streaming(fastglm_source(cb,6L,reset=reset),family=gaussian())
    b<-fastglm_bootstrap(f,R=3L,type="cluster",cluster=function(k) rep(k,20L),seed=11)
    expect_equal(b$successful,3L)
    units<-split(seq_len(n),rep(1:6,each=20))
    for(j in 1:3) {
        set.seed(b$seeds[j]);i<-unlist(units[sample.int(6L,6L,replace=TRUE)],use.names=FALSE)
        expect_equal(unname(b$replicates[j,]),unname(coef(glm.fit(x[i,,drop=FALSE],y[i]))),tolerance=1e-9)
    }
})

test_that("information failures retain the source reader error", {
    expect_error(fastglm:::.count_inverse_information(stop("reader failed")),"reader failed")
    expect_error(fastglm:::.count_inverse_information(matrix(0,2,2)),"joint information is singular")
})

test_that("parametric binomial bootstrap retains proportion trial counts", {
    set.seed(7411);n<-100L;x<-matrix(1,n,1L);trials<-rep(10,n)
    y<-rbinom(n,trials,.3)/trials
    f<-fastglm(x,y,weights=trials,family=binomial())
    b<-fastglm_bootstrap(f,R=3L,type="parametric",seed=12)
    expect_equal(b$successful,3L)
    for(j in 1:3) {
        set.seed(b$seeds[j]);yy<-rbinom(n,trials,plogis(coef(f)))/trials
        reference<-glm.fit(x,yy,weights=trials,family=binomial())
        expect_equal(unname(b$replicates[j,]),unname(coef(reference)),tolerance=1e-8)
    }
    z<-fastglm:::.bootstrap_input(f,FALSE)$data;z$weights[1L]<-.5
    expect_error(fastglm:::.bootstrap_response(f,z),"integer trial counts")
})

test_that("two-component bootstrap preserves sparse designs", {
    set.seed(7412);n<-160L;d<-data.frame(a=rbinom(n,1,.4),b=rbinom(n,1,.6))
    x<-as(Matrix::Matrix(model.matrix(~a,d),sparse=TRUE),"dgCMatrix")
    z<-as(Matrix::Matrix(model.matrix(~b,d),sparse=TRUE),"dgCMatrix")
    w<-rep(1:3,length.out=n);off<-cbind(runif(n,-.1,.1),runif(n,-.1,.1))
    responses<-list(rpois(n,exp(.4+.3*d$a)),rbinom(n,1,.6)*rgamma(n,2,scale=exp(.4+.3*d$a)/2))
    for(j in seq_along(responses)) {
        fn<-if(j==1L) fastglm_hurdle else fastglm_twopart
        f<-fn(x,response=responses[[j]],z=z,weights=w,offset=off,tol=1e-10)
        statistic<-function(fit) {
            if(!inherits(fit$x$count,"dgCMatrix") || !inherits(fit$x$zero,"dgCMatrix"))
                stop("refit lost its sparse component designs")
            coef(fit)
        }
        b<-fastglm_bootstrap(f,R=3L,seed=13,statistic=statistic)
        expect_equal(b$successful,3L)
        set.seed(b$seeds[1L]);i<-sample.int(n,n,replace=TRUE)
        dd<-d[i,,drop=FALSE];dd$y<-responses[[j]][i]
        reference<-fn(y~a|b,data=dd,weights=w[i],offset=off[i,,drop=FALSE],tol=1e-10)
        expect_equal(unname(b$replicates[1L,]),unname(coef(reference)),tolerance=1e-7)
    }
})

test_that("streaming component predictions retain external source offsets", {
    set.seed(7406);n<-240;d<-data.frame(x=rnorm(n),e=runif(n,.8,1.2),o=runif(n,-.2,.2))
    d$y<-rbinom(n,1,plogis(.4+d$o))*rgamma(n,2,scale=d$e*exp(.2+.3*d$x)/2)
    cb<-function(k) d[((k-1)*60+1):(k*60),]
    src<-fastglm_source(cb,4,formula=y~x|x,prototype=d,offset=c("o","o"))
    f<-fastglm_twopart(src)
    expect_equal(predict(f,newdata=d),predict(f),tolerance=1e-9,ignore_attr=TRUE)
    src<-fastglm_source(cb,4,formula=y~x+offset(log(e)),prototype=d,offset="o")
    f<-fastglm_twopart(src)
    expect_equal(predict(f,newdata=d),predict(f),tolerance=1e-9,ignore_attr=TRUE)
})

test_that("two-part marginal cluster uncertainty agrees with cluster sample means", {
    set.seed(7407);g<-40L;m<-8L;groups<-rep(seq_len(g),each=m)
    probability<-rbeta(g,6,4)[groups]
    y<-rbinom(g*m,1,probability)*rgamma(g*m,2,scale=(1+2*probability)/2)
    f<-fastglm_twopart(matrix(1,g*m,1),response=y,tol=1e-11)
    v<-fastglm_vcov(f,type="HC0",cluster=groups,full=TRUE)
    a<-fastglm_average(f,vcov.=v)
    reference<-sd(tapply(y,groups,mean))/sqrt(g)
    expect_equal(a$estimate,mean(y),tolerance=1e-9)
    expect_equal(a$std.error,reference,tolerance=1e-7)
})

test_that("joint count inference respects numerical theta boundaries", {
    set.seed(7408);x<-cbind(1,rnorm(180));y<-rnbinom(180,mu=exp(.4+.3*x[,2]),size=2)
    f<-fastglm_nb(x,y)
    f$theta.boundary<-TRUE
    expect_error(fastglm_vcov(f,full=TRUE),"theta boundary")
    expect_error(fastglm_bootstrap(f,R=3,seed=3),"theta boundary")
    v<-vcov(f,full=TRUE);v[nrow(v),]<-v[,ncol(v)]<-NA_real_;f$vcov.full<-v
    expect_true(is.finite(fastglm_average(f)$std.error))
})

test_that("count scores preserve small-rate and Poisson-boundary limits", {
    mu<-1e-6;theta<-1.5
    f<-structure(list(coefficients=c(intercept=log(mu)),dist="poisson",theta=NA_real_,theta.estimated=FALSE),class="fastglm_truncated")
    z<-list(X=matrix(1,2,1),Z=NULL,y=c(1,2),weights=c(1,0),offset=list(count=c(0,0),zero=c(0,0)))
    score<-fastglm:::.count_score(f,z,coef(f))
    expect_equal(unname(score[1,1]/(-mu/2)),1,tolerance=1e-5)
    expect_identical(unname(score[2,]),0)
    f$dist<-"negbin";f$theta<-theta;f$theta.estimated<-TRUE
    score<-fastglm:::.count_score(f,z,fastglm:::.count_parameters(f))
    expect_equal(unname(score[1,1]/(-mu*(1+1/theta)/2)),1,tolerance=1e-5)
    expect_equal(unname(score[1,2]/(mu/(2*theta))),1,tolerance=1e-5)
    theta<-1e7;mu<-2
    f<-structure(list(coefficients=c(intercept=log(mu)),family=negbin(theta),theta=theta,theta.estimated=TRUE),class=c("fastglm_nb","fastglm"))
    z$y<-c(0,4);z$weights<-c(1,1)
    score<-fastglm:::.count_score(f,z,fastglm:::.count_parameters(f))
    leading<-(z$y-(z$y-mu)^2)/(2*theta)
    expect_lt(max(abs(unname(score[,2])-leading)),20/theta^2)
})
