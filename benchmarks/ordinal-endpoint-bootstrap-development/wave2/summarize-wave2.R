# Audit and summarize the completed wave-2 paired simulation.
args <- commandArgs(trailingOnly=TRUE)
outdir <- if(length(args)) args[1L] else
  "benchmarks/ordinal-endpoint-bootstrap-development/wave2"
raw <- utils::read.csv(file.path(outdir,"wave2-raw.csv"),stringsAsFactors=FALSE,
                       check.names=FALSE)
parameters <- c("alpha1","alpha2","beta1","beta2")
category.retained <- function(x) is.na(x) | !nzchar(x)

summarize_parameter <- function(d,parameter) {
  estimate <- d[[paste0("estimate.",parameter)]]
  truth <- d[[paste0("true.",parameter)]]
  eligible.original <- d$fit.converged & category.retained(d$original.empty.interior) &
    is.finite(estimate) & is.finite(truth)
  point <- estimate[eligible.original]
  delta <- point-truth[eligible.original]
  available <- d$candidate.available & is.finite(d[[paste0("lower.",parameter)]]) &
    is.finite(d[[paste0("upper.",parameter)]]) & is.finite(truth)
  lower <- d[[paste0("lower.",parameter)]][available]
  upper <- d[[paste0("upper.",parameter)]][available]
  target <- truth[available]
  hit <- lower<=target & upper>=target
  list(point.n=length(point),bias=if(length(delta)) mean(delta) else NA_real_,
       empirical.sd=if(length(point)>1L) stats::sd(point) else NA_real_,
       available.n=sum(available),coverage=if(length(hit)) mean(hit) else NA_real_,
       coverage.mcse=if(length(hit)) sqrt(mean(hit)*(1-mean(hit))/length(hit)) else NA_real_,
       lower.miss=if(length(hit)) mean(upper<target) else NA_real_,
       upper.miss=if(length(hit)) mean(lower>target) else NA_real_,
       median.width=if(length(hit)) stats::median(upper-lower) else NA_real_)
}

keys <- unique(raw[c("scenario","n")])
summary <- do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
  d <- raw[raw$scenario==keys$scenario[i] & raw$n==keys$n[i],,drop=FALSE]
  row <- data.frame(scenario=keys$scenario[i],n=keys$n[i],datasets=nrow(d),
    predictors=unique(d$predictors),fit.converged=mean(d$fit.converged),
    original.interior.retained=mean(d$fit.converged & category.retained(d$original.empty.interior)),
    current.availability=mean(d$current.available),
    endpoint.allowance.availability=mean(d$candidate.available),
    endpoint.empty.refit.rate=mean(d$refit.endpoint.empty.rate,na.rm=TRUE),
    interior.empty.refit.rate=mean(d$refit.interior.empty.rate,na.rm=TRUE),
    refit.failure.rate=mean(d$refit.failure.rate,na.rm=TRUE),
    statistic.failure.rate=mean(d$refit.statistic.failure.rate,na.rm=TRUE))
  for(parameter in parameters) for (metric in names(summarize_parameter(d,parameter))) {
    val <- summarize_parameter(d,parameter)[[metric]]
    row[[paste(parameter,metric,sep=".")]] <- val
  }
  row
}))
utils::write.csv(summary,file.path(outdir,"wave2-summary.csv"),row.names=FALSE)
writeLines(capture.output(print(summary,row.names=FALSE,digits=4)),
           file.path(outdir,"wave2-summary.txt"))
print(summary,row.names=FALSE,digits=4)

expected <- do.call(rbind,lapply(names(list(
  rare_lower_endpoint=c(60,120,240),rare_upper_endpoint=c(60,120,240),
  moderate_control=c(60,120,240),rare_lower_two_predictors=c(90,180,360))),
  function(scenario) {
    ns <- switch(scenario,rare_lower_endpoint=c(60,120,240),
      rare_upper_endpoint=c(60,120,240),moderate_control=c(60,120,240),
      rare_lower_two_predictors=c(90,180,360))
    data.frame(scenario=scenario,n=ns,datasets=500L)
  }))
observed <- aggregate(replicate~scenario+n,raw,length)
names(observed)[3L] <- "datasets"
audit <- merge(expected,observed,by=c("scenario","n"),all=TRUE,
               suffixes=c(".expected",".observed"))
utils::write.csv(audit,file.path(outdir,"wave2-cell-audit.csv"),row.names=FALSE)
if(nrow(audit)!=12L || any(audit$datasets.expected!=500L) ||
   any(audit$datasets.observed!=500L))
  stop("wave-2 output does not match the frozen 12-cell by 500-dataset design")
fit.errors <- !is.na(raw$fit.error) & nzchar(raw$fit.error)
if(any(!raw$fit.converged) || any(fit.errors) ||
   any(raw$refit.failure.rate>0,na.rm=TRUE) ||
   any(raw$refit.statistic.failure.rate>0,na.rm=TRUE))
  stop("fit failures require review; raw rows were retained")
