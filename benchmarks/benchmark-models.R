# Rscript benchmarks/benchmark-models.R <output-directory> [library-directory]
# Compares ordinary dense fits across design sizes and Gamma bias reduction
# under equal, repeated, and distinct prior weights. Run against each library
# separately on an otherwise idle machine; compare the saved estimates and
# covariance matrices before interpreting timing differences.
args <- commandArgs(TRUE)
if (!length(args)) stop('Supply an output directory')
if (length(args)>1L) .libPaths(c(args[2L],.libPaths()))
dir.create(args[1L],recursive=TRUE,showWarnings=FALSE)
library(fastglm)
if (!requireNamespace('microbenchmark',quietly=TRUE)) stop('microbenchmark is required')
set.seed(7051);results<-list();fits<-list()
measure<-function(label,fun,n,p,reference=NULL) {
 f<-fun();stopifnot(f$converged)
 if(!is.null(reference)) stopifnot(max(abs(f$fitted.values-reference$fitted.values))<1e-4*(1+max(abs(reference$fitted.values))))
 t<-microbenchmark::microbenchmark(fun(),times=5L)
 results[[label]]<<-data.frame(case=label,n=n,p=p,median_ms=median(t$time)/1e6,min_ms=min(t$time)/1e6,iterations=f$iter,dispersion=f$dispersion)
 fits[[label]]<<-list(coefficients=coef(f),covariance=vcov(f),fitted=f$fitted.values,deviance=f$deviance,dispersion=f$dispersion)
}
for(shape in list(c(10000L,6L),c(100000L,30L),c(20000L,100L))) {
 n<-shape[1];p<-shape[2];x<-cbind(1,matrix(rnorm(n*(p-1)),n,p-1));eta<-drop(x%*%c(.2,rep(.5/sqrt(p),p-1)))
 for(name in c('binomial','poisson','Gamma')) {
  fam<-switch(name,binomial=binomial(),poisson=poisson(),Gamma=Gamma('log'))
  y<-switch(name,binomial=rbinom(n,1,plogis(eta)),poisson=rpois(n,exp(eta)),Gamma=rgamma(n,shape=2,scale=exp(eta)/2))
  reference<-glm.fit(x,y,family=fam,control=list(epsilon=1e-11))
  measure(paste(name,n,p,sep='_'),function()fastglm(x,y,fam,method=2,tol=1e-9),n,p,reference)
 }
}
set.seed(7052);n<-2000L;p<-6L;x<-cbind(1,matrix(rnorm(n*(p-1)),n,p-1));mu<-exp(drop(x%*%c(.2,rep(.1,p-1))))
prior<-list(unit=rep(1,n),repeated=rep(c(0,.25,1,2),length.out=n),distinct=runif(n,.25,2))
for(shape in c(.3,5)) {
 y<-rgamma(n,shape=shape,scale=mu/shape)
 for(nm in names(prior)) {
  w<-prior[[nm]];start<-c(log(weighted.mean(y,w)),rep(0,p-1))
  measure(paste('firth_Gamma',shape,nm,sep='_'),function()fastglm(x,y,Gamma('log'),weights=w,start=start,firth=TRUE,tol=1e-9),n,p)
 }
}
write.csv(do.call(rbind,results),file.path(args[1L],'timings.csv'),row.names=FALSE)
saveRDS(fits,file.path(args[1L],'estimates.rds'))
writeLines(capture.output(sessionInfo()),file.path(args[1L],'session-info.txt'))
