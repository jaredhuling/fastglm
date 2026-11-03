args<-commandArgs(TRUE)
directory<-if(length(args))args[1L] else 'benchmarks/ordinal-uncertainty-validation/development-v1'
files<-list.files(file.path(directory,'records'),full.names=TRUE,pattern='\\.rds$')
maximum.literal<-maximum.exact<-0;bad<-list();n<-0L
for(file in files) {
 z<-readRDS(file)
 for(f in z$fits) if(!is.null(f$bootstrap)&&f$bootstrap$inference.available) {
  b<-f$bootstrap;n<-n+1L
  a<-t(apply(b$replicates,2,quantile,probs=c(.025,.975),names=FALSE))
  exact<-t(apply(b$replicates,2,quantile,probs=c((1-b$level)/2,(1+b$level)/2),names=FALSE))
  dl<-max(abs(a-b$conf.int));de<-max(abs(exact-b$conf.int))
  maximum.literal<-max(maximum.literal,dl);maximum.exact<-max(maximum.exact,de)
  if(!isTRUE(all.equal(unname(a),unname(b$conf.int),tolerance=1e-14)))
   bad[[length(bad)+1L]]<-list(file=file,estimator=f$estimator,level=b$level,literal=dl,exact=de,
       comparison=all.equal(unname(a),unname(b$conf.int),tolerance=1e-14))
 }
}
print(list(available.bootstrap.objects=n,maximum.literal.difference=maximum.literal,
 maximum.exact.expression.difference=maximum.exact,failed.strict.literal.comparisons=length(bad),first=bad[1]))
saveRDS(list(available=n,literal.maximum=maximum.literal,exact.maximum=maximum.exact,bad=bad),
 file.path(directory,'endpoint-recomputation.rds'))
