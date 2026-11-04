args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || dir.exists(args[1L])) stop("Supply one new output directory")
dir.create(args[1L],recursive=TRUE);output<-normalizePath(args[1L])
pkgload::load_all(".",quiet=TRUE,compile=FALSE)
stopifnot(requireNamespace("numDeriv",quietly=TRUE))
original<-system2("git",c("show","17c4a8a:R/ordinal-inference.R"),stdout=TRUE)
stopifnot(is.null(attr(original,"status")))
writeLines(original,file.path(output,"baseline-ordinal-inference.R"))
reference<-new.env(parent=asNamespace("fastglm"))
sys.source(file.path(output,"baseline-ordinal-inference.R"),envir=reference)
n<-15L;x<-cbind(x=seq(-1,1,length.out=n),z=cos(seq_len(n)))
counts<-cbind(low=3+(seq_len(n)%%3),middle=5+(seq_len(n)%%2),high=4+(seq_len(n)%%4))
w<-c(rep(0,3),rep(c(1,2),6));off<-sin(seq_len(n))/10
groups<-c(1,1+1e-15,2,3,4)[rep(1:5,3)];groups[w==0]<-NA
rows<-split(seq_len(n),ceiling(seq_len(n)/3))
source<-fastglm_source(function(k) {
    at<-rows[[k]];list(X=x[at,,drop=FALSE],y=counts[at,,drop=FALSE],weights=w[at],offset=off[at])
},length(rows))
cluster<-function(k) groups[rows[[k]]]
checks<-list()
record<-function(id,error,tolerance) {
    checks[[length(checks)+1L]]<<-data.frame(check=id,error=error,tolerance=tolerance,passed=is.finite(error)&&error<=tolerance)
}
for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) for(partial in c(FALSE,TRUE)) {
    id<-paste(link,if(br) "meanBR" else "ML",if(partial) "partial" else "common",sep="/")
    selected<-if(partial) "z" else NULL
    fit<-fastglm_ordinal(x,counts,weights=w,offset=off,nonparallel=selected,link=link,firth=br,tol=1e-10)
    stream<-fastglm_ordinal_streaming(source,nonparallel=selected,link=link,firth=br,tol=1e-10)
    stopifnot(fit$converged,stream$converged)
    for(type in c("HC0","HC1")) for(clustered in c(FALSE,TRUE)) {
        label<-paste(id,type,if(clustered) "clustered" else "independent",sep="/")
        previous<-reference$.ordinal_robust_vcov(fit,type,cluster=if(clustered) groups,block.size=2L)
        current<-fastglm_vcov(fit,type,cluster=if(clustered) groups,block.size=2L)
        record(paste0(label,"/baseline-exact"),as.numeric(!identical(previous,current)),0)
        streamed<-fastglm_vcov(stream,type,cluster=if(clustered) cluster,block.size=2L)
        record(paste0(label,"/source"),max(abs(streamed-current))/max(1,max(abs(current))),2e-8)
    }
    q<-2L;common<-if(partial) 1L else 1:2;varying<-if(partial) 2L else integer()
    pc<-length(common);pn<-length(varying);b<-coef(fit);d<-length(b)
    logprob<-function(theta) {
        eta<-matrix(theta[seq_len(q)],n,q,byrow=TRUE)-drop(x[,common,drop=FALSE]%*%theta[q+seq_len(pc)])-off
        if(pn) eta<-eta-x[,varying,drop=FALSE]%*%matrix(theta[q+pc+seq_len(q*pn)],pn,q)
        G<-switch(link,logit=plogis,probit=pnorm,cloglog=function(z)-expm1(-exp(z)))
        F<-G(eta);log(cbind(F,1)-cbind(0,F))
    }
    H<- -numDeriv::hessian(function(theta) sum(counts*w*logprob(theta)),b)
    M<-matrix(0,d,d);scores<-matrix(0,n,d)
    for(i in seq_len(n)) {
        S<-numDeriv::jacobian(function(theta) logprob(theta)[i,],b)
        M<-M+crossprod(S,S*(counts[i,]*w[i]))
        scores[i,]<-colSums(S*(counts[i,]*w[i]))
    }
    B<-solve(H);independent<-B%*%M%*%B
    summed<-rowsum(scores[w>0,,drop=FALSE],match(groups[w>0],unique(groups[w>0])),reorder=FALSE)
    clustered<-B%*%crossprod(summed)%*%B
    record(paste0(id,"/independent-numerical-reference"),
        max(abs(fastglm_vcov(stream)-independent))/max(1,max(abs(independent))),3e-7)
    record(paste0(id,"/clustered-numerical-reference"),
        max(abs(fastglm_vcov(stream,cluster=cluster,cadjust=FALSE)-clustered))/max(1,max(abs(clustered))),3e-7)
}
checks<-do.call(rbind,checks)
write.csv(checks,file.path(output,"checks.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(output,"session.txt"))
paths<-c("R/ordinal-inference.R","R/ordinal-stream-inference.R","R/ordinal-partial-robust.R",
    "R/ordinal-partial.R","R/ordinal.R","R/ordinal-links.R","R/stream-methods.R",
    "benchmarks/ordinal-source-covariance/validate.R")
paths<-paths[file.exists(paths)]
write.csv(data.frame(file=paths,md5=unname(tools::md5sum(paths))),file.path(output,"source-hashes.csv"),row.names=FALSE)
cat(sum(checks$passed),"/",nrow(checks),"checks passed; maximum error",max(checks$error),"\n")
if(!all(checks$passed)) stop("Independent source covariance validation failed")
