# Run each library separately, without competing benchmarks or compilation.
# Rscript benchmarks/ordinal-kernel-performance/run.R LIBRARY LABEL
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
.libPaths(c(normalizePath(args[1]),.libPaths()));library(fastglm)
out <- file.path("benchmarks/ordinal-kernel-performance/results",args[2]);stopifnot(!file.exists(paste0(out,".csv")))
capture.output(sessionInfo(),file=paste0(out,"-session.txt"))
capture.output(tools::md5sum(list.files(system.file("libs",package="fastglm"),full.names=TRUE)),file=paste0(out,"-library-md5.txt"))
set.seed(112301);records<-list();solutions<-list()
for(n in c(10000L,100000L,1000000L)) {
 p <- if(n==100000L) 6L else 3L;K <- if(n==10000L) 8L else if(n==100000L) 5L else 4L
 x<-matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
 y<-ordered(cut(.4*x[,1]-.2*x[,2]+rlogis(n),c(-Inf,seq(-1.2,1.2,length.out=K-1),Inf)),labels=seq_len(K))
 for(firth in c(FALSE,TRUE)) {
  prepared<-fastglm_ordinal_prepare(x,firth=firth)
  for(operation in c("ordinary","prepared")) {
   run<-function() fastglm_ordinal(if(operation=="prepared") prepared else x,y,firth=firth)
   fit<-run();stopifnot(fit$converged)
   key<-paste(n,firth,operation,sep="-")
   solutions[[key]]<-list(coefficients=coef(fit),covariance=vcov(fit),loglik=as.numeric(logLik(fit)),
     predictions=predict(fit,x[1:20,,drop=FALSE]),score=fit$score,iterations=fit$iterations)
   for(rep in 1:3) {
    gc();elapsed<-system.time(fit<-run())[["elapsed"]]
    records[[length(records)+1L]]<-data.frame(n,p,K,estimation=if(firth) "meanBR" else "ML",operation,
      repetition=rep,seconds=elapsed,iterations=fit$iterations,converged=fit$converged)
   }
   write.csv(do.call(rbind,records),paste0(out,".csv"),row.names=FALSE)
   saveRDS(solutions,paste0(out,"-solutions.rds"))
  }
  cat(n,firth,"finished\n");flush.console()
 }
}
