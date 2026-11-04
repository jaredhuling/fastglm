# Rscript benchmarks/firth-tail-performance/run.R LIBRARY LABEL OUTPUT_DIRECTORY
# Run libraries in separate processes, without competing fitting/compilation jobs.
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==3L)
.libPaths(c(normalizePath(args[1]),.libPaths()));library(fastglm)
out <- file.path(args[3],args[2]);dir.create(out,recursive=TRUE,showWarnings=FALSE)
stopifnot(!file.exists(file.path(out,'timing.csv')))
capture.output(sessionInfo(),file=file.path(out,'session.txt'))
capture.output(tools::md5sum(list.files(system.file('libs',package='fastglm'),full.names=TRUE)),
    file=file.path(out,'library-md5.txt'))
set.seed(113201);records <- list();solutions <- list()
designs <- data.frame(name=c('small','ordinary','wide','large','grouped'),
                     n=c(600L,12000L,12000L,120000L,12000L),p=c(3L,6L,20L,6L,6L))
for(j in seq_len(nrow(designs))) {
 n<-designs$n[j];p<-designs$p[j];name<-designs$name[j]
 x<-cbind('(Intercept)'=1,matrix(rnorm(n*(p-1L)),n,p-1L,dimnames=list(NULL,paste0('x',seq_len(p-1L)))))
 offset<-if(name=='grouped') sin(seq_len(n))/10 else numeric(n)
 eta<-.2+.35*x[,2]-.15*x[,3]+offset
 for(link in if(name=='ordinary') c('logit','probit','cloglog') else c('probit','cloglog')) {
  family<-binomial(link)
  if(name=='grouped') {
   m<-rep(c(4,9,20),length.out=n);y1<-rbinom(n,m,family$linkinv(eta));y<-cbind(y1,m-y1)
   weights<-rep(c(0,1,2,1),length.out=n)
  } else {y<-rbinom(n,1,family$linkinv(eta));weights<-rep(1,n)}
  for(method in c('qr','llt')) {
   fitfun<-function() fastglm(x,y,family,weights=weights,offset=offset,method=method,firth=TRUE,tol=1e-9)
   fit<-fitfun();stopifnot(fit$converged)
   key<-paste(name,link,method,sep='-')
   solutions[[key]]<-list(coefficients=coef(fit),covariance=vcov(fit),loglik=as.numeric(logLik(fit)),
       deviance=deviance(fit),weights=fit$weights,residuals=fit$residuals,adjusted_score=fit$adjusted.score,
       predictions=predict(fit,type='response',se.fit=TRUE),iterations=fit$iter)
   derivative<-function() fastglm:::.glm_mu_eta(fit,fit$linear.predictors)
   for(operation in c('fit','derivative')) {
    operationfun<-if(operation=='fit') fitfun else derivative
    gc();timing<-microbenchmark::microbenchmark(operationfun(),times=11L)
    records[[length(records)+1L]]<-data.frame(case=name,n,p,link,method,operation,
      repetition=seq_len(nrow(timing)),nanoseconds=timing$time,iterations=fit$iter,converged=fit$converged)
   }
   write.csv(do.call(rbind,records),file.path(out,'timing.csv'),row.names=FALSE)
   saveRDS(solutions,file.path(out,'solutions.rds'))
  }
 }
 cat(name,'complete\n');flush.console()
}
