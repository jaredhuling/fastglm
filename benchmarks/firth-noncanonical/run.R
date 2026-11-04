# Frozen confirmation protocol: ordinary fits plus independent numerical stress cases.
# Run each library in its own process, with no other fitting jobs running:
# Rscript benchmarks/firth-noncanonical/run.R LIBRARY LABEL OUTPUT_DIRECTORY
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L,dir.exists(args[1L]))
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
destination <- args[3L];dir.create(destination,recursive=TRUE,showWarnings=FALSE)
output <- file.path(destination,paste0(args[2L],".csv"))
if(file.exists(output)) stop("Results already exist")
capture.output(sessionInfo(),file=file.path(destination,paste0(args[2L],"-session.txt")))
capture.output(tools::md5sum(list.files(system.file("libs",package="fastglm"),full.names=TRUE,recursive=TRUE)),
    file=file.path(destination,paste0(args[2L],"-library-md5.txt")))
set.seed(111901)
n <- 5000L
x <- cbind("(Intercept)"=1,matrix(rnorm(n*5),n,5,dimnames=list(NULL,paste0("x",1:5))))
eta <- .2+.3*x[,2]-.2*x[,3]+.1*x[,4]
cases <- list()
for(family in list(binomial(),binomial("probit"),binomial("cloglog"),poisson(),Gamma("log"),gaussian())) {
    y <- switch(family$family,binomial=rbinom(n,1,family$linkinv(eta)),
        poisson=rpois(n,exp(eta)),Gamma=rgamma(n,shape=.8,scale=exp(eta)/.8),gaussian=eta+rnorm(n))
    name <- paste("ordinary",family$family,family$link,sep="-")
    cases[[name]] <- list(x=x,y=y,family=family,weights=rep(1,n),offset=rep(0,n),stress=FALSE)
}
for(link in c("probit","cloglog")) {
    n <- 240L;x <- cbind("(Intercept)"=1,x=seq(-2,2,length.out=n),z=rnorm(n,sd=.25))
    w <- rep(c(0,1,2,1),length.out=n);off <- cos(seq_len(n))/8
    outcomes <- list(separation=as.numeric(x[,2]>0),all_zero=numeric(n),all_one=rep(1,n))
    m <- rep(c(4,10,20),length.out=n)
    k <- rbinom(n,m,binomial(link)$linkinv(-1+1.2*x[,2]-.2*x[,3]+off))
    outcomes$grouped <- cbind(k,m-k)
    for(name in names(outcomes)) cases[[paste(name,link,sep="-")]] <-
        list(x=x,y=outcomes[[name]],family=binomial(link),weights=w,offset=off,stress=TRUE)
}
records <- list()
for(name in names(cases)) {
    d <- cases[[name]]
    reference <- tryCatch(suppressWarnings(brglm2::brglmFit(d$x,d$y,family=d$family,
        weights=d$weights,offset=d$offset,start=numeric(ncol(d$x)),
        control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10,maxit=500))),error=function(e) NULL)
    backends <- if(d$stress) c("dense","sparse","mapped","stream_qr","stream_llt") else "dense"
    for(backend in backends) {
        design <- switch(backend,sparse=as(d$x,"dgCMatrix"),mapped=bigmemory::as.big.matrix(d$x),d$x)
        read <- function(k) {i<-((k-1)*40+1):(k*40);list(X=d$x[i,],
            y=if(is.matrix(d$y)) d$y[i,,drop=FALSE] else d$y[i],weights=d$weights[i],offset=d$offset[i])}
        run <- function() if(startsWith(backend,"stream"))
            fastglm_streaming(read,nrow(d$x)/40,family=d$family,
                method=if(backend=="stream_qr") "qr" else "llt",firth=TRUE,tol=1e-9,maxit=300) else
            fastglm(design,d$y,d$family,weights=d$weights,offset=d$offset,method="llt",firth=TRUE,tol=1e-9,maxit=300)
        error <- "";warnings <- character()
        fit <- tryCatch(withCallingHandlers(run(),warning=function(w) {
            warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")
        }),error=function(e) {error <<- conditionMessage(e);NULL})
        good <- !is.null(fit) && isTRUE(fit$converged) && all(is.finite(coef(fit)))
        seconds <- if(good) median(replicate(if(d$stress) 3 else 9,system.time(suppressWarnings(run()))[["elapsed"]])) else NA_real_
        reference.ok <- !is.null(reference) && isTRUE(reference$converged) && all(is.finite(coef(reference)))
        score <- if(!is.null(fit$standardized.adjusted.score)) max(abs(fit$standardized.adjusted.score)) else NA_real_
        records[[length(records)+1L]] <- data.frame(case=name,backend,converged=good,
            iterations=if(is.null(fit)) NA_integer_ else fit$iter,
            reductions=if(is.null(fit$step.reductions)) NA_integer_ else fit$step.reductions,
            seconds,adjusted_score=score,reference_converged=reference.ok,
            coefficient_error=if(good && reference.ok) max(abs(unname(coef(fit))-unname(coef(reference)))) else NA_real_,
            error,warnings=paste(unique(warnings),collapse="; "))
        write.csv(do.call(rbind,records),output,row.names=FALSE)
        cat(name,backend,good,"\n");flush.console()
    }
}
