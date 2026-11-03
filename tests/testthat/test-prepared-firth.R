test_that("prepared bias-reduced fits share setup and match independent fits", {
    set.seed(921);n<-180;x<-cbind(1,matrix(rnorm(n*2),n,2))
    weights<-rep(c(0,.5,1,2,3),length.out=n);offset<-rnorm(n,0,.03)
    families<-list(gaussian(),Gamma("log"),poisson("sqrt"))
    responses<-list(matrix(rnorm(n*3),n),matrix(rgamma(n*3,2),n),matrix(rpois(n*3,6),n))
    for(k in seq_along(families)) for(method in list("auto",2L)) {
        design<-fastglm_prepare(x,weights=weights,offset=offset,firth=TRUE)
        fits<-fastglm_batch(design,responses[[k]],families[[k]],method=method,
            warm.start=FALSE,tol=1e-10)
        expect_equal(design$cache$builds,1L)
        for(j in 1:3) {
            ref<-fastglm(x,responses[[k]][,j],families[[k]],weights=weights,
                offset=offset,method=method,firth=TRUE,tol=1e-10)
            expect_s3_class(fits[[j]],"fastglm_firth")
            expect_true(fits[[j]]$converged)
            expect_equal(coef(fits[[j]]),coef(ref),tolerance=1e-7)
            expect_equal(vcov(fits[[j]]),vcov(ref),tolerance=1e-7)
            expect_equal(fits[[j]]$dispersion,ref$dispersion,tolerance=1e-7)
            expect_equal(as.numeric(logLik(fits[[j]])),as.numeric(logLik(ref)),tolerance=1e-7)
        }
    }
})

test_that("prepared factors invalidate after changing predictors or weights", {
    set.seed(922);x<-cbind(1,rnorm(150));y<-matrix(rgamma(300,2),150)
    d<-fastglm_prepare(x,firth=TRUE)
    fastglm_batch(d,y,Gamma("log"),warm.start=FALSE)
    d$x[1,2]<-d$x[1,2]+1
    changed<-fastglm_batch(d,y,Gamma("log"),warm.start=FALSE)
    expect_equal(d$cache$builds,2L)
    ref<-fastglm(d$x,y[,1],Gamma("log"),firth=TRUE,method="auto")
    expect_equal(coef(changed[[1]]),coef(ref),tolerance=1e-7)
    d$weights[1]<-2
    changed<-fastglm_batch(d,y,Gamma("log"),warm.start=FALSE)
    expect_equal(d$cache$builds,3L)
    ref<-fastglm(d$x,y[,1],Gamma("log"),weights=d$weights,firth=TRUE,method="auto")
    expect_equal(vcov(changed[[1]]),vcov(ref),tolerance=1e-7)
    d$offset[2]<-.2
    changed<-fastglm_batch(d,y,Gamma("log"),warm.start=FALSE)
    expect_equal(d$cache$builds,3L)
    ref<-fastglm(d$x,y[,1],Gamma("log"),weights=d$weights,offset=d$offset,firth=TRUE,method="auto")
    expect_equal(coef(changed[[1]]),coef(ref),tolerance=1e-7)
})

test_that("integer predictors are converted once in the shared setup", {
    set.seed(927);x<-cbind(1L,rep(0:1,50));y<-matrix(rgamma(200,2),100)
    d<-fastglm_prepare(x,firth=TRUE)
    fits<-fastglm_batch(d,y,Gamma("log"),method=2L,warm.start=FALSE)
    expect_type(x,"integer")
    expect_type(d$x,"double")
    expect_type(d$cache$factor$x,"double")
    expect_equal(d$cache$builds,1L)
    ref<-fastglm(d$x,y[,1],Gamma("log"),firth=TRUE,method=2L)
    expect_equal(coef(fits[[1]]),coef(ref),tolerance=1e-8)
})

test_that("prepared fits respect response-specific weights and offsets", {
    set.seed(923);n<-140;x<-cbind(1,rnorm(n));y<-matrix(rgamma(n*3,2),n)
    w<-cbind(rep(1,n),rep(c(0,1,2),length.out=n),runif(n,.5,2))
    off<-matrix(rnorm(n*3,0,.1),n)
    d<-fastglm_prepare(x,firth=TRUE)
    fit<-fastglm_batch(d,y,Gamma("log"),weights=w,offset=off,tol=1e-10)
    expect_equal(d$cache$builds,3L)
    for(j in 1:3) {
        ref<-fastglm(x,y[,j],Gamma("log"),weights=w[,j],offset=off[,j],firth=TRUE,method="auto",tol=1e-10)
        expect_equal(coef(fit[[j]]),coef(ref),tolerance=1e-7)
        expect_equal(fit[[j]]$dispersion,ref$dispersion,tolerance=1e-7)
    }
})

test_that("batch warm starts reset when new offsets leave the link domain", {
    set.seed(928);x<-matrix(1,100,1);y<-matrix(rpois(200,6),100)
    off<-cbind(rep(0,100),rep(-10,100))
    for(method in c("auto","qr")) {
        warm<-fastglm_batch(x,y,poisson("sqrt"),offset=off,firth=TRUE,method=method,warm.start=TRUE)
        cold<-fastglm_batch(x,y,poisson("sqrt"),offset=off,firth=TRUE,method=method,warm.start=FALSE)
        expect_equal(coef(warm[[2]]),coef(cold[[2]]),tolerance=1e-8)
        expect_true(warm[[2]]$converged)
    }
})

test_that("prepared inference levels and serialized caches retain their meaning", {
    set.seed(924);x<-cbind(1,rnorm(100));y<-matrix(rgamma(200,2),100)
    d<-fastglm_prepare(x,firth=TRUE)
    none<-fastglm_batch(d,y,Gamma("log"),inference="none",retain.data=FALSE)
    expect_length(d$cache$factor$covariance,0L)
    expect_null(none[[1]]$x);expect_null(none[[1]]$y);expect_null(none[[1]]$se)
    diagonal<-fastglm_batch(d,y,Gamma("log"),inference="diagonal")
    full<-fastglm_batch(d,y,Gamma("log"))
    expect_null(diagonal[[1]]$cov.unscaled)
    expect_equal(diagonal[[1]]$se,full[[1]]$se,tolerance=1e-8)
    tmp<-tempfile(fileext=".rds");on.exit(unlink(tmp))
    saveRDS(d,tmp);restored<-readRDS(tmp)
    after<-fastglm_batch(restored,y,Gamma("log"))
    expect_equal(coef(after[[1]]),coef(full[[1]]),tolerance=1e-8)
    expect_equal(restored$cache$builds,1L)
})

test_that("batch Firth settings and ordinary solver fallbacks remain explicit", {
    set.seed(925);x<-cbind(1,rnorm(100));y<-matrix(rbinom(200,1,.4),100)
    d<-fastglm_prepare(x,firth=TRUE)
    f<-fastglm_batch(d,y,binomial(),method="qr",warm.start=FALSE)
    ref<-fastglm(x,y[,1],binomial(),method="qr",firth=TRUE)
    expect_equal(coef(f[[1]]),coef(ref),tolerance=1e-8)
    expect_true(f[[1]]$firth)
    ordinary<-fastglm_batch(d,y,binomial(),firth=FALSE)
    expect_false(isTRUE(ordinary[[1]]$firth))
    custom<-gaussian();custom$variance<-function(mu)rep(2,length(mu))
    expect_error(fastglm_batch(d,y,custom),"unmodified")
    custom<-gaussian();custom$aic<-function(...)123
    a<-fastglm_batch(d,matrix(rnorm(200),100),custom)
    expect_equal(a[[1]]$aic,123+2*ncol(x))
    fixed<-gaussian();fixed$dispersion<-2
    expect_error(fastglm_batch(d,y,fixed),"fixed dispersion")
    expect_error(fastglm_batch(d,y,Gamma("log")),"positive")
})

test_that("prepared Gamma fits agree across sequential and independent processes", {
    skip_on_cran()
    set.seed(926);x<-cbind(1,rnorm(120));y<-matrix(rgamma(360,2),120)
    d<-fastglm_prepare(x,firth=TRUE)
    a<-fastglm_batch(d,y,Gamma("log"),warm.start=FALSE,tol=1e-10)
    b<-fastglm_batch(d,y,Gamma("log"),warm.start=FALSE,workers=2,tol=1e-10)
    expect_equal(lapply(a,coef),lapply(b,coef),tolerance=1e-8)
    cl<-parallel::makePSOCKcluster(2L);on.exit(parallel::stopCluster(cl))
    init<-function(paths){.libPaths(paths);loadNamespace("fastglm");NULL};environment(init)<-baseenv()
    parallel::clusterCall(cl,init,.libPaths())
    run<-function(j,design,responses)fastglm::fastglm_batch(design,responses[,j,drop=FALSE],
        Gamma("log"),warm.start=FALSE,tol=1e-10)[[1]]
    b<-parallel::parLapply(cl,1:3,run,design=d,responses=y)
    expect_equal(lapply(a,coef),lapply(b,coef),tolerance=1e-8)
})
