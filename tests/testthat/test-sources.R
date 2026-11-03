test_that("formula sources preserve both component designs and transformations", {
    set.seed(8021)
    d <- data.frame(y=rpois(100,2),x=rnorm(100),z=rnorm(100),
        g=factor(rep(letters[1:4],25)),e=runif(100,1,2),w=runif(100))
    reader <- function(k) d[((k-1)*25+1):(k*25),]
    src <- fastglm_source(reader,4,
        formula=y~splines::ns(x,df=3)+g+offset(log(e))|z,
        prototype=d,weights="w")
    blocks <- lapply(1:4,src$callback)
    ref <- model.matrix(~splines::ns(x,df=3)+g+offset(log(e)),d)
    expect_equal(unname(do.call(rbind,lapply(blocks,`[[`,"X"))),
                 unname(ref),ignore_attr=TRUE)
    expect_equal(do.call(rbind,lapply(blocks,`[[`,"Z")),model.matrix(~z,d),ignore_attr=TRUE)
    expect_equal(blocks[[1]]$offset$count,log(d$e[1:25]))
    expect_equal(blocks[[1]]$offset$zero,rep(0,25))
    expect_equal(blocks[[1]]$weights,d$w[1:25])
})

test_that("Parquet sources preserve factor levels, offsets, and fitted models", {
    skip_if_not_installed("arrow")
    set.seed(8022)
    d <- data.frame(x=rnorm(120),g=factor(rep(c("a","b","c"),each=40)),
                    exposure=runif(120,.5,2),w=rep(c(0,1,2),40))
    d$y <- rpois(120,d$exposure*exp(.2+.3*d$x))
    directory <- tempfile("parquet-");dir.create(directory)
    on.exit(unlink(directory,recursive=TRUE))
    paths <- file.path(directory,paste0(1:3,".parquet"))
    for(k in 1:3) arrow::write_parquet(d[((k-1)*40+1):(k*40),],paths[k])
    src <- fastglm_source_arrow(paths,formula=y~x+g+offset(log(exposure)),
                                prototype=d,weights="w")
    f <- fastglm_streaming(src,family=poisson(),tol=1e-10)
    ref <- fastglm(y~x+g+offset(log(exposure)),d,poisson(),weights=w,
                   method="llt",tol=1e-10)
    expect_equal(unname(coef(f)),unname(coef(ref)),tolerance=1e-8)
    expect_equal(predict(f,type="response"),as.numeric(predict(ref,type="response")),tolerance=1e-8)
    expect_equal(f$aic,ref$aic,tolerance=1e-8)
})

test_that("repeated reads do not duplicate family initialization warnings", {
    set.seed(8023);x<-cbind(1,rnorm(100));y<-rbinom(100,1,.5)
    reader<-function(k) {i<-((k-1)*25+1):(k*25);list(X=x[i,],y=y[i],weights=rep(.5,25))}
    messages<-character()
    f<-withCallingHandlers(fastglm_streaming(reader,4,binomial()),warning=function(w) {
        messages<<-c(messages,conditionMessage(w));invokeRestart("muffleWarning")
    })
    expect_true(f$converged)
    expect_equal(sum(grepl("non-integer #successes",messages,fixed=TRUE)),1L)
})

test_that("new-data prediction retains source offset columns", {
    set.seed(8024);d<-data.frame(x=rnorm(120),off=runif(120,-.3,.3))
    d$y<-rpois(120,exp(.2+.3*d$x+d$off))
    source<-fastglm_source(function(k)d[((k-1)*40+1):(k*40),],3,
                           formula=y~x,prototype=d,offset="off")
    f<-fastglm_streaming(source,family=poisson())
    expect_equal(unname(predict(f,newdata=d,type="response")),predict(f,type="response"))
    nd<-d[1:4,];nd$off<-nd$off+.5
    expected<-exp(drop(model.matrix(~x,nd)%*%coef(f))+nd$off)
    expect_equal(predict(f,newdata=nd,type="response"),expected)
    expect_error(predict(f,newdata=nd[,c("y","x")]),"offset column")
})

test_that("fixed-theta likelihoods remain stable near the Poisson limit", {
    x<-cbind(1,seq_len(120)/120);y<-rep(0:2,40)
    families<-list(negbin(1e8))
    if(requireNamespace("MASS",quietly=TRUE)) families<-c(families,list(MASS::negative.binomial(1e8)))
    for(family in families) {
        f<-fastglm(x,y,family)
        expected<- -2*sum(dnbinom(y,mu=f$fitted.values,size=1e8,log=TRUE))+2*f$rank
        expect_equal(f$aic,expected,tolerance=1e-12)
        reader<-function(k) {i<-((k-1)*40+1):(k*40);list(X=x[i,],y=y[i])}
        s<-fastglm_streaming(reader,3,family)
        expect_equal(s$aic,expected,tolerance=1e-9)
    }
    family<-negbin(1e8);original<-family$aic
    family$aic<-function(y,n,mu,wt,dev) original(y,n,mu,wt,dev)+7
    f<-fastglm(x,y,family)
    expect_equal(f$aic,-2*sum(dnbinom(y,mu=f$fitted.values,size=1e8,log=TRUE))+2*f$rank+7,tolerance=1e-12)
})
