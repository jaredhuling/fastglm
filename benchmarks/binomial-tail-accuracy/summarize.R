# Rscript benchmarks/binomial-tail-accuracy/summarize.R FINAL_LABEL
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1L)
root <- 'benchmarks/binomial-tail-accuracy/results';label <- args[1]
b <- read.csv(file.path(root,'baseline.csv'));a <- read.csv(file.path(root,paste0(label,'.csv')))
summary <- do.call(rbind,lapply(list(baseline=b,candidate=a),function(d) data.frame(fits=nrow(d),
 converged=sum(d$converged),exact_score_below_1e_6=sum(d$converged & d$exact_score<1e-6,na.rm=TRUE),
 maximum_exact_score=max(d$exact_score,na.rm=TRUE),
 maximum_covariance_relative_error=max(d$covariance_relative_error,na.rm=TRUE),
 maximum_reported_score_error=max(d$reported_score_error,na.rm=TRUE))))
write.csv(summary,file.path(root,'accuracy-summary.csv'));print(summary)
b <- read.csv(file.path(root,'baseline-ordinary-timing.csv'))
a <- read.csv(file.path(root,paste0(label,'-ordinary-timing.csv')))
medians <- function(d) aggregate(nanoseconds~family+link,d,median)
d <- merge(medians(b),medians(a),by=c('family','link'),suffixes=c('_before','_after'))
d$ratio_after_before <- d$nanoseconds_after/d$nanoseconds_before
write.csv(d,file.path(root,'timing-comparison.csv'),row.names=FALSE);print(d)
