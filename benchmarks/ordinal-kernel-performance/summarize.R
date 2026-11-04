prefix <- 'benchmarks/ordinal-kernel-performance/results'
b <- read.csv(file.path(prefix,'baseline.csv'));a <- read.csv(file.path(prefix,'candidate.csv'))
medians <- function(d) aggregate(seconds~n+p+K+estimation+operation,d,median)
d <- merge(medians(b),medians(a),by=c('n','p','K','estimation','operation'),suffixes=c('_before','_after'))
d$speedup <- d$seconds_before/d$seconds_after
write.csv(d,file.path(prefix,'comparison.csv'),row.names=FALSE);print(d)
b <- readRDS(file.path(prefix,'baseline-solutions.rds'));a <- readRDS(file.path(prefix,'candidate-solutions.rds'))
agreement <- do.call(rbind,lapply(names(b),function(k) data.frame(case=k,
 coefficient_error=max(abs(b[[k]]$coefficients-a[[k]]$coefficients)),
 covariance_error=max(abs(b[[k]]$covariance-a[[k]]$covariance)),
 loglik_error=abs(b[[k]]$loglik-a[[k]]$loglik),
 prediction_error=max(abs(b[[k]]$predictions-a[[k]]$predictions)),
 iterations_same=b[[k]]$iterations==a[[k]]$iterations)))
write.csv(agreement,file.path(prefix,'agreement.csv'),row.names=FALSE);print(agreement)
