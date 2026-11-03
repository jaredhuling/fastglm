# Rscript benchmarks/benchmark.R <output-directory> [library-directory]
# Fits are timed only after verifying convergence and agreement of fitted means.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)<1L) stop("Supply an output directory")
if(length(args)>1L) .libPaths(c(args[2L],.libPaths()))
library(fastglm)
if(!requireNamespace("microbenchmark",quietly=TRUE)) stop("microbenchmark is required")
dir.create(args[1L],recursive=TRUE,showWarnings=FALSE)
set.seed(7001)
results <- list()
measure <- function(label,fun,reference,n,p) {
    value <- fun()
    stopifnot(isTRUE(value$converged))
    error <- max(abs(value$fitted.values-reference$fitted.values))
    if(!is.finite(error) || error>1e-4*(1+max(abs(reference$fitted.values)))) stop(label,": accuracy check failed")
    timing <- microbenchmark::microbenchmark(fun(),times=7L)
    results[[length(results)+1L]] <<- data.frame(case=label,n=n,p=p,
        median_ms=median(timing$time)/1e6,min_ms=min(timing$time)/1e6,
        max_fitted_error=error,iterations=value$iter)
}
for(n in c(10000L,100000L)) {
    p <- 30L; x <- cbind(1,matrix(rnorm(n*(p-1)),n,p-1)); eta<-drop(x%*%c(.2,rep(.07,p-1)))
    for(name in c("binomial","poisson","Gamma")) {
        family<-switch(name,binomial=binomial(),poisson=poisson(),Gamma=Gamma("log"))
        y<-switch(name,binomial=rbinom(n,1,plogis(eta)),poisson=rpois(n,exp(eta)),Gamma=rgamma(n,shape=2,scale=exp(eta)/2))
        reference<-glm.fit(x,y,family=family,control=list(epsilon=1e-10))
        measure(paste(name,"llt",sep="_"),function() fastglm(x,y,family,method=2,tol=1e-8),reference,n,p)
        if("inference" %in% names(formals(fastglm:::fastglm.default)))
            measure(paste(name,"coefficients",sep="_"),function() fastglm(x,y,family,method=2,tol=1e-8,inference="none"),reference,n,p)
    }
}
set.seed(7002)
n<-20000L;p<-200L
x<-cbind(1,Matrix::rsparsematrix(n,p-1,density=.005));x<-as(x,"dgCMatrix")
y<-rbinom(n,1,plogis(as.numeric(x%*%rep(.1,p))))
reference<-glm.fit(as.matrix(x),y,family=binomial(),control=list(epsilon=1e-10))
measure("sparse_binomial",function() fastglm(x,y,binomial(),method=2),reference,n,p)
write.csv(do.call(rbind,results),file.path(args[1L],"timings.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(args[1L],"session-info.txt"))
