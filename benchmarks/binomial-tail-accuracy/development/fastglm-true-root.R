args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
library(fastglm,lib.loc=args[1L])
code<-readLines("benchmarks/firth-noncanonical/run.R")
eval(parse(text=code[grep("^set.seed",code):(grep("^records <-",code)-1)]))
d<-cases[["separation-cloglog"]]
x<-d$x;y<-d$y;m<-d$weights;off<-d$offset
state<-function(b,clipped=FALSE) {
 eta<-drop(x%*%b)+off;t<-exp(eta);mu<--expm1(-t)
 if(clipped) {mu<-d$family$linkinv(eta);dm<-d$family$mu.eta(eta);w<-m*dm^2/(mu*(1-mu));s<-m*(y-mu)*dm/(mu*(1-mu))}
 else {w<-m*exp(2*eta-t-log(mu));s<-m*(y*exp(eta-t-log(mu))-(1-y)*t)}
 q<-qr(x*sqrt(w));I<-crossprod(x,x*w);h<-w*rowSums((x%*%solve(I))*x)
 U<-drop(crossprod(x,s+.5*h*(1-t)))
 list(score=U,information=I,step=drop(solve(I,U)),remaining=max(abs(x%*%solve(I,U))))
}

cat("True-score root search using row quadratic forms\n")
for(slow in c(1,.2)) {
 b0<-numeric(3)
 for(iter in 1:5000) {
  s<-state(b0,FALSE)
  if(s$remaining<1e-8) break
  v<-s$step*slow;v<-v/max(1,max(abs(x%*%v)))
  b0<-b0+v
 }
 print(list(slow=slow,b=b0,iterations=iter,score=state(b0,FALSE)$score,remaining=state(b0,FALSE)$remaining))
}
for(start in list(c(-.7,14,1.5),c(-1,25,2),c(-1,40,3))) {
 metric<-solve(state(start)$information)
 objective<-function(b) tryCatch({s<-state(b)$score;drop(crossprod(s,metric%*%s))},error=function(e) 1e100)
 fit<-nlminb(start,objective,control=list(iter.max=1000,eval.max=3000,rel.tol=1e-12,x.tol=1e-12))
 print(list(start=start,b=fit$par,objective=fit$objective,remaining=state(fit$par)$remaining,convergence=fit$convergence))
}
