# Rscript benchmarks/firth-tail-performance/summarize.R RESULTS_DIRECTORY BASELINE_LABEL CANDIDATE_LABEL
args<-commandArgs(trailingOnly=TRUE);stopifnot(length(args)==3L)
path<-function(label,name) file.path(args[1],label,name)
b<-read.csv(path(args[2],'timing.csv'));a<-read.csv(path(args[3],'timing.csv'))
medians<-function(d) aggregate(nanoseconds~case+n+p+link+method+operation,d,median)
summary<-merge(medians(b),medians(a),by=c('case','n','p','link','method','operation'),suffixes=c('_before','_after'))
summary$speedup<-summary$nanoseconds_before/summary$nanoseconds_after
write.csv(summary,file.path(args[1],'comparison.csv'),row.names=FALSE);print(summary)
b<-readRDS(path(args[2],'solutions.rds'));a<-readRDS(path(args[3],'solutions.rds'))
stopifnot(identical(names(b),names(a)))
agreement<-do.call(rbind,lapply(names(b),function(k) {
 before<-b[[k]];after<-a[[k]]
 data.frame(case=k,coefficient_error=max(abs(before$coefficients-after$coefficients)),
  covariance_error=max(abs(before$covariance-after$covariance)),
  loglik_relative_error=abs(before$loglik-after$loglik)/(1+abs(before$loglik)),
  deviance_relative_error=abs(before$deviance-after$deviance)/(1+abs(before$deviance)),
  weight_error=max(abs(before$weights-after$weights)),
  residual_error=max(abs(before$residuals-after$residuals)),
  adjusted_score_error=max(abs(before$adjusted_score-after$adjusted_score)),
  probability_error=max(abs(before$predictions$fit-after$predictions$fit)),
  se_error=max(abs(before$predictions$se.fit-after$predictions$se.fit)),
  iterations_identical=before$iterations==after$iterations)
}))
write.csv(agreement,file.path(args[1],'agreement.csv'),row.names=FALSE);print(agreement)
stopifnot(all(agreement$iterations_identical),all(agreement[,2:10]<1e-9))
