# Run from the repository root: Rscript benchmarks/binomial-tail-accuracy/run.R LIBRARY LABEL
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
.libPaths(c(normalizePath(args[1]),.libPaths()));library(fastglm)
source("benchmarks/binomial-tail-accuracy/reference.R")
out <- file.path("benchmarks/binomial-tail-accuracy/results",args[2]);stopifnot(!file.exists(paste0(out,".csv")))
capture.output(sessionInfo(),file=paste0(out,"-session.txt"))
capture.output(tools::md5sum(list.files(system.file("libs",package="fastglm"),full.names=TRUE)),file=paste0(out,"-library-md5.txt"))
# Preserve the previous panel's exact RNG sequence; its results remain unchanged.
protocol <- readLines("benchmarks/firth-noncanonical/run.R")
eval(parse(text=protocol[grep("^set.seed",protocol):(grep("^records <-",protocol)-1)]))
cases <- cases[vapply(cases,function(d) d$family$family=="binomial" && d$family$link!="logit",logical(1))]
# New numerical confirmation inputs; no inference calibration or tuning.
set.seed(112201)
for(n in c(120L,300L)) for(link in c("probit","cloglog")) {
    x <- cbind("(Intercept)"=1,x=seq(-3,3,length.out=n),z=rnorm(n,sd=.4))
    w <- rep(c(1,2,0,3,1),length.out=n);off <- .2*sin(seq_len(n))
    m <- rep(c(3,10,30),length.out=n);k <- rbinom(n,m,binomial(link)$linkinv(-1+.8*x[,2]+off))
    outcomes <- list(separated=as.numeric(x[,2]+.1*x[,3]>.2),grouped=cbind(k,m-k),zero=numeric(n),one=rep(1,n))
    for(name in names(outcomes)) cases[[paste("fresh",n,link,name,sep="-")]] <-
        list(x=x,y=outcomes[[name]],family=binomial(link),weights=w,offset=off,stress=TRUE)
}
records <- list()
for(name in names(cases)) {
 d <- cases[[name]];x<-d$x;y<-d$y;w<-d$weights;off<-d$offset
 for(backend in if(d$stress) c("dense_qr","dense_llt","sparse","mapped","stream_qr","stream_llt") else "dense_llt") {
  design <- switch(backend,sparse=as(x,"dgCMatrix"),mapped=bigmemory::as.big.matrix(x),x)
  read <- function(k) {i<-((k-1)*30+1):min(k*30,nrow(x));list(X=x[i,],y=if(is.matrix(y)) y[i,] else y[i],weights=w[i],offset=off[i])}
  warn<-character();error<-""
  fit <- tryCatch(withCallingHandlers(if(startsWith(backend,"stream")) fastglm_streaming(read,ceiling(nrow(x)/30),
      family=d$family,firth=TRUE,method=if(backend=="stream_qr") "qr" else "llt",tol=1e-9,maxit=500) else
      fastglm(design,y,d$family,weights=w,offset=off,firth=TRUE,
          method=if(backend=="dense_qr") "qr" else "llt",tol=1e-9,maxit=500),
      warning=function(e) {warn<<-c(warn,conditionMessage(e));invokeRestart("muffleWarning")}),error=function(e) {error<<-conditionMessage(e);NULL})
  trials<-if(is.matrix(y)) rowSums(y) else rep(1,length(y));response<-if(is.matrix(y)) y[,1]/trials else y
  ref <- if(!is.null(fit)) tryCatch(tail_reference(x,response,w*trials,off,coef(fit),d$family$link),error=function(e) NULL)
  records[[length(records)+1]] <- data.frame(case=name,backend,converged=isTRUE(fit$converged),
    iterations=if(is.null(fit)) NA else fit$iter,
    exact_score=if(is.null(ref)) NA else max(abs(ref$score/sqrt(diag(ref$information)))),
    covariance_relative_error=if(is.null(ref)) NA else max(abs(unname(vcov(fit))-ref$covariance))/(1+max(abs(ref$covariance))),
    reported_score_error=if(is.null(ref)) NA else max(abs(fit$adjusted.score-ref$score)),
    error,warnings=paste(unique(warn),collapse="; "))
  write.csv(do.call(rbind,records),paste0(out,".csv"),row.names=FALSE)
 }
 cat(name,"finished\n");flush.console()
}
