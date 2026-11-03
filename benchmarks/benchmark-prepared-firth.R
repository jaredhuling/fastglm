# Compare repeated mean-bias-reduced fits with shared prepared setup.
# Optional arguments: rows, predictors, responses, repetitions, output RDS.
args<-commandArgs(TRUE)
size<-c(50000L,30L,8L,3L)
if(length(args)) size[seq_len(min(4L,length(args)))]<-as.integer(args[seq_len(min(4L,length(args)))])
stopifnot(all(is.finite(size)),all(size>0),size[2]>1)
library(fastglm)
has.prepared<-"firth" %in% names(formals(fastglm_batch))
n<-size[1];p<-size[2];m<-size[3];repetitions<-size[4]
set.seed(20261102)
x<-cbind(1,matrix(rnorm(n*(p-1)),n,p-1))
eta<-drop(x%*%c(.4,rep(.1/sqrt(p-1),p-1)))
families<-list(gaussian=gaussian(),gamma_log=Gamma("log"),poisson_sqrt=poisson("sqrt"))
responses<-list(gaussian=matrix(rnorm(n*m),n)+eta,
    gamma_log=matrix(rgamma(n*m,shape=2),n)*exp(eta)/2,
    poisson_sqrt=matrix(rpois(n*m,(2+eta)^2),n))
out<-list()
for(name in names(families)) {
    family<-families[[name]];y<-responses[[name]]
    prepared<-if(has.prepared)fastglm_prepare(x,firth=TRUE) else NULL
    independent<-function()lapply(seq_len(m),function(j)fastglm(x,y[,j],family,
        firth=TRUE,method=2L,tol=1e-8,retain.data=FALSE))
    cold<-function()fastglm_batch(fastglm_prepare(x,firth=TRUE),y,family,
        method=2L,tol=1e-8,warm.start=FALSE,retain.data=FALSE)
    warm<-function()fastglm_batch(prepared,y,family,
        method=2L,tol=1e-8,warm.start=FALSE,retain.data=FALSE)
    reference<-independent();cached<-if(has.prepared)warm() else reference
    coef.error<-max(vapply(seq_len(m),function(j)max(abs(coef(reference[[j]])-coef(cached[[j]]))),numeric(1)))
    se.error<-max(vapply(seq_len(m),function(j)max(abs(reference[[j]]$se-cached[[j]]$se)),numeric(1)))
    stopifnot(coef.error<1e-6,se.error<1e-6,all(vapply(cached,`[[`,logical(1),"converged")))
    for(mode in if(has.prepared)c("independent","prepared_cold","prepared_warm") else "independent") {
        fn<-switch(mode,independent=independent,prepared_cold=cold,prepared_warm=warm)
        elapsed<-numeric(repetitions)
        for(j in seq_len(repetitions)) {gc();elapsed[j]<-system.time(fn())[["elapsed"]]}
        out[[length(out)+1L]]<-data.frame(family=name,mode=mode,n=n,p=p,responses=m,
            repetitions=repetitions,median_seconds=median(elapsed),minimum_seconds=min(elapsed),
            maximum_seconds=max(elapsed),coefficient_error=coef.error,se_error=se.error)
    }
}
result<-list(results=do.call(rbind,out),session=sessionInfo(),call=commandArgs())
print(result$results,row.names=FALSE)
if(length(args)>=5L)saveRDS(result,args[5])
