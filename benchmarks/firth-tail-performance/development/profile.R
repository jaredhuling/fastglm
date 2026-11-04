# Development profiling: separate fitting and post-fit bookkeeping.
# Rscript benchmarks/firth-tail-performance/development/profile.R LIBRARY OUTPUT_DIRECTORY
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2)
.libPaths(c(normalizePath(args[1]),.libPaths()));library(fastglm)
out<-args[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(!file.exists(file.path(out,'components.csv')))
capture.output(sessionInfo(),file=file.path(out,'session.txt'))
set.seed(113101);records<-list()
for(n in c(12000L,100000L)) {
 x<-cbind(1,matrix(rnorm(n*5),n,5));eta<-.2+.3*x[,2]-.2*x[,3]
 for(link in c('probit','cloglog')) {
  family<-binomial(link);y<-rbinom(n,1,family$linkinv(eta));w<-rep(1,n);off<-numeric(n)
  raw<-fastglm:::fastglmPure(x,y,family,w,off,method='llt',tol=1e-9,firth=TRUE)
  fit<-fastglm(x,y,family,method='llt',tol=1e-9,firth=TRUE)
  complete<-function() fastglm(x,y,family,method='llt',tol=1e-9,firth=TRUE)
  fitting<-function() fastglm:::fastglmPure(x,y,family,w,off,method='llt',tol=1e-9,firth=TRUE)
  finish<-function() fastglm:::.finish_glm(raw,off)
  derivative<-function() fastglm:::.glm_mu_eta(fit,fit$linear.predictors)
  for(operation in c('complete','fitting','finish','derivative')) {
   gc();t<-microbenchmark::microbenchmark(get(operation)(),times=21L)
   records[[length(records)+1L]]<-data.frame(n,link,operation,repetition=seq_len(nrow(t)),nanoseconds=t$time)
  }
  prefix<-file.path(out,paste(n,link,sep='-'))
  Rprof(paste0(prefix,'.Rprof'),interval=.001,memory.profiling=TRUE)
  for(i in 1:15) complete()
  Rprof(NULL);s<-summaryRprof(paste0(prefix,'.Rprof'),memory='both')
  write.csv(s$by.total,paste0(prefix,'-profile.csv'))
  write.csv(do.call(rbind,records),file.path(out,'components.csv'),row.names=FALSE)
 }
}
print(aggregate(nanoseconds~n+link+operation,do.call(rbind,records),median))
