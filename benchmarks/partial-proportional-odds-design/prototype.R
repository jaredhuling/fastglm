# Mathematical prototype for a future partial proportional-odds implementation.
# This script is not a package fitting interface or a performance benchmark.
# Run from repository root: Rscript benchmarks/partial-proportional-odds-design/prototype.R LIBRARY OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)%in%1:2)
.libPaths(c(normalizePath(args[1]),.libPaths()));library(fastglm)
stopifnot(requireNamespace('numDeriv',quietly=TRUE),requireNamespace('ordinal',quietly=TRUE))
out <- if(length(args)==2L) args[2L] else 'benchmarks/partial-proportional-odds-design/results'
if(file.exists(file.path(out,'checks.csv'))) stop('Prototype results already exist')
dir.create(out,recursive=TRUE,showWarnings=FALSE)

# x: common slopes; z: slopes allowed to differ at each cumulative threshold.
# Coordinates are alpha, beta_common, then one gamma block per threshold.
ppo_state <- function(theta,x,z,counts,weights=rep(1,nrow(x)),offset=numeric(nrow(x)),
                      link='logit',adjust=TRUE) {
 q <- ncol(counts)-1L;pc<-ncol(x);pn<-ncol(z);d<-q+pc+q*pn
 stopifnot(length(theta)==d)
 G <- switch(link,logit=plogis,probit=pnorm,cloglog=function(a) -expm1(-exp(a)))
 density <- switch(link,logit=function(a) dlogis(a),probit=dnorm,cloglog=function(a) exp(a-exp(a)))
 curvature <- switch(link,logit=function(a) dlogis(a)*(1-2*plogis(a)),
                     probit=function(a) -a*dnorm(a),cloglog=function(a) exp(a-exp(a))*(1-exp(a)))
 I <- matrix(0,d,d);U<-numeric(d);ll<-0;rows<-vector('list',nrow(x));minimum<-Inf
 for(i in seq_len(nrow(x))) {
  Z<-matrix(0,q,d);Z[,seq_len(q)]<-diag(q)
  if(pc) Z[,q+seq_len(pc)] <- matrix(-x[i,],q,pc,byrow=TRUE)
  if(pn) for(j in seq_len(q)) Z[j,q+pc+(j-1L)*pn+seq_len(pn)] <- -z[i,]
  eta<-drop(Z%*%theta)-offset[i]
  if(q>1L) minimum<-min(minimum,diff(eta))
  P<-diff(c(0,G(eta),1));if(any(P<=0)) stop('outside the positive-probability domain')
  g<-density(eta);gp<-curvature(eta);D<-Z*g
  J<-rbind(D,rep(0,d))-rbind(rep(0,d),D)
  effective<-weights[i]*counts[i,];m<-sum(effective)
  ll<-ll+sum(effective*log(P));U<-U+drop(crossprod(J,effective/P))
  I<-I+m*crossprod(J,J/P)
  rows[[i]]<-list(Z=Z,J=J,P=P,gprime=gp,m=m)
 }
 V<-solve(I);A<-numeric(d)
 if(adjust) for(r in rows) {
  v<-rowSums((r$Z%*%V)*r$Z)
  correction<-.5*r$m*r$gprime*v
  A<-A+drop(crossprod(r$J,diff(c(0,correction,0))/r$P))
 }
 list(loglik=ll,score=U,information=I,adjustment=A,adjusted.score=U+A,
      covariance=V,minimum.gap=minimum,rows=rows)
}

checks<-list()
record<-function(name,error,tolerance) {
 checks[[length(checks)+1L]]<<-data.frame(check=name,error,tolerance,passed=is.finite(error)&&error<tolerance)
}
x<-matrix(c(-.7,-.3,.1,.4,.8,1.1),6,1);z<-matrix(c(0,1,0,1,0,1),6,1)
counts<-matrix(c(4,2,3,5,1,2, 3,6,2,1,4,2, 2,1,4,3,2,5),6,3)
w<-c(1,2,0,1,3,1);off<-c(.1,-.2,0,.1,-.1,.2)
b<-c(-.7,.9,.3,.2,-.1)
for(link in c('logit','probit','cloglog')) {
 s<-ppo_state(b,x,z,counts,w,off,link)
 gradient<-numDeriv::grad(function(t) ppo_state(t,x,z,counts,w,off,link,FALSE)$loglik,b)
 # Expected score curvature holds each row's category probabilities fixed.
 expected<-do.call(rbind,lapply(seq_len(nrow(x)),function(i) sum(counts[i,])*s$rows[[i]]$P))
 info<--numDeriv::hessian(function(t) ppo_state(t,x,z,expected,w,off,link,FALSE)$loglik,b)
 record(paste(link,'score'),max(abs(s$score-gradient)),1e-7)
 record(paste(link,'information'),max(abs(s$information-info)),2e-6)
 # Restricted common-slope design must reproduce the current production kernel.
 Z0<-matrix(numeric(),nrow(x),0);po<-ppo_state(c(-.7,.9,.3),x,Z0,counts,w,off,link)
 native<-fastglm:::ordinal_native_evaluate(x,integer(),counts,w,off,3L,c(-.7,.9,.3),TRUE,link=match(link,c('logit','probit','cloglog'))-1L)
 record(paste(link,'PO reduction score'),max(abs(po$adjusted.score-native$score)),1e-10)
 record(paste(link,'PO reduction information'),max(abs(po$information-native$information)),1e-10)
}
# A bounded interval certifies validity, while unbounded extrapolation can fail.
alpha<-c(-1,1);gamma<-c(0,.5)
gaps<-function(z) diff(alpha)-z*diff(gamma)
record('valid interval endpoints',as.numeric(any(gaps(c(-1,1))<=0)),.5)
record('invalid extrapolation detected',as.numeric(gaps(5)>0),.5)

# Published external example: Kosmidis (2014), Table 6.
counts<-rbind(c(4,9,5,0,0),c(1,7,8,2,0),c(0,5,8,3,2),c(0,1,5,7,5))
x<-matrix(c(0,1,0,1),4,1);z<-matrix(c(0,0,1,1),4,1)
alpha<-qlogis(cumsum(colSums(counts))[1:4]/sum(counts));b<-c(alpha,0,rep(0,4))
trace<-list()
for(iter in 1:2000) {
 s<-ppo_state(b,x,z,counts);step<-drop(solve(s$information,s$adjusted.score))
 predictor<-max(vapply(s$rows,function(r) max(abs(r$Z%*%step)),numeric(1)))
 scaled<-max(abs(s$adjusted.score/sqrt(diag(s$information))))
 trace[[iter]]<-data.frame(iter,loglik=s$loglik,scaled.score=scaled,predictor.step=predictor,minimum.gap=s$minimum.gap)
 if(predictor<1e-9 && scaled<1e-9) break
 step<-step/max(1,predictor)
 fraction<-1
 for(r in s$rows) {
  gap<-diff(drop(r$Z%*%b));change<-diff(drop(r$Z%*%step));decreasing<-change<0
  if(any(decreasing)) fraction<-min(fraction,.9*min(-gap[decreasing]/change[decreasing]))
 }
 b<-b+fraction*step
}
reference<-c(-1.19,1.06,3.50,5.20,1.40,2.62,2.05,2.65,2.96)
record('wine published coefficients (two-decimal table)',max(abs(b-reference)),.006)
record('wine adjusted-score root',scaled,1e-8)
record('wine remaining predictor change',predictor,1e-8)
write.csv(do.call(rbind,checks),file.path(out,'checks.csv'),row.names=FALSE)
write.csv(do.call(rbind,trace),file.path(out,'wine-iteration.csv'),row.names=FALSE)
write.csv(data.frame(parameter=c(paste0('alpha',1:4),'contact',paste0('temperature',1:4)),
 estimate=b,standard.error=sqrt(diag(s$covariance)),published=reference),file.path(out,'wine-fit.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'session.txt'))
print(do.call(rbind,checks));stopifnot(all(vapply(checks,function(x) x$passed,logical(1))))
