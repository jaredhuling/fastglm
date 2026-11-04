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
 q<-qr(x*sqrt(w));h<-rowSums(qr.Q(q)^2);I<-crossprod(x,x*w)
 U<-drop(crossprod(x,s+.5*h*(1-t)))
 list(score=U,information=I,step=drop(solve(I,U)),remaining=max(abs(x%*%solve(I,U))))
}
b<-suppressWarnings(brglm2::brglmFit(x,y,weights=m,offset=off,family=d$family,start=numeric(3),control=brglm2::brglmControl(type="AS_mean",slowit=.1,epsilon=1e-11,maxit=1000)))$coefficients
cat("clipped root\n");print(b)
for(clipped in c(TRUE,FALSE)) {
 s<-state(b,clipped);J<-numDeriv::jacobian(function(a) state(a,clipped)$score,b)
 print(list(clipped=clipped,score=s$score,remaining=s$remaining,scoring_eigen=eigen(diag(3)+solve(s$information,J))$values))
}
for(clipped in c(FALSE,TRUE)) {
 b0<-numeric(3)
 for(iter in 1:500) {
  s<-state(b0,clipped)
  if(s$remaining<1e-9) break
  direction<-s$step;direction<-direction/max(1,max(abs(x%*%direction)))
  merit<-drop(crossprod(s$score,solve(s$information,s$score)))
  for(j in 0:30) {
   b1<-b0+direction*2^-j;a<-state(b1,clipped)
   next.merit<-drop(crossprod(a$score,solve(s$information,a$score)))
   if(next.merit<=merit*(1-1e-4*2^-j)) break
  }
  if(j==30) {cat("no descent at",iter,"\n");break}
  b0<-b1
 }
 print(list(clipped=clipped,b=b0,iterations=iter,remaining=state(b0,clipped)$remaining))
 J<-numDeriv::jacobian(function(a) state(a,clipped)$score,b0)
 print(eigen(diag(3)+solve(state(b0,clipped)$information,J))$values)
}
cat("Newton steps for true equations\n")
b0<-c(-.7,18,2)
for(iter in 1:100) {
 s<-state(b0,FALSE);J<-numDeriv::jacobian(function(a) state(a,FALSE)$score,b0)
 if(s$remaining<1e-8) break
 direction<-drop(solve(J,-s$score));direction<-direction/max(1,max(abs(x%*%direction))/5)
 merit<-drop(crossprod(s$score,solve(s$information,s$score)))
 for(j in 0:30) {
  b1<-b0+direction*2^-j;a<-state(b1,FALSE)
  next.merit<-drop(crossprod(a$score,solve(s$information,a$score)))
  if(next.merit<=merit*(1-1e-4*2^-j)) break
 }
 if(j==30) {cat("failed",iter,"\n");break}
 b0<-b1
}
print(list(b=b0,iterations=iter,score=state(b0,FALSE)$score,remaining=state(b0,FALSE)$remaining))
for(b in list(c(-.6380102,14.2126086,1.4891041),b0)) {
 ordinal<-fastglm:::ordinal_native_evaluate(x[,-1],integer(),cbind(event=y,none=1-y),m,-off,2,b*c(1,-1,-1),TRUE,link=2L)
 print(cbind(R=state(b,FALSE)$score,ordinal=drop(ordinal$score)*c(1,-1,-1)))
}
