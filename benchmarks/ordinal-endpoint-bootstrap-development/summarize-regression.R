# Audit and summarize the completed paired regression simulation.
args <- commandArgs(trailingOnly=TRUE)
outdir <- if(length(args)) args[1L] else
  "benchmarks/ordinal-endpoint-bootstrap-development"
raw.path <- file.path(outdir,"regression-raw.csv")
raw <- utils::read.csv(raw.path,stringsAsFactors=FALSE,check.names=FALSE)
parameters <- c("alpha1","alpha2","beta")

summarize_parameter <- function(d, parameter) {
  estimate <- d[[paste0("estimate.",parameter)]]
  truth <- d[[paste0("true.",parameter)]]
  eligible <- d$fit.converged & !nzchar(d$original.empty.interior)
  point <- estimate[eligible & is.finite(estimate)]
  delta <- point-truth[eligible & is.finite(estimate)]
  available <- d$candidate.available &
    is.finite(d[[paste0("lower.",parameter)]]) &
    is.finite(d[[paste0("upper.",parameter)]])
  lower <- d[[paste0("lower.",parameter)]][available]
  upper <- d[[paste0("upper.",parameter)]][available]
  target <- truth[available]
  hit <- lower<=target & upper>=target
  list(point.n=length(point),bias=if(length(delta)) mean(delta) else NA_real_,
       empirical.sd=if(length(point)>1L) stats::sd(point) else NA_real_,
       candidate.available.n=sum(available),
       coverage=if(length(hit)) mean(hit) else NA_real_,
       coverage.mcse=if(length(hit)) sqrt(mean(hit)*(1-mean(hit))/length(hit)) else NA_real_,
       lower.miss=if(length(hit)) mean(upper<target) else NA_real_,
       upper.miss=if(length(hit)) mean(lower>target) else NA_real_,
       median.width=if(length(hit)) stats::median(upper-lower) else NA_real_)
}

keys <- unique(raw[c("scenario","n")])
summary <- do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
  d <- raw[raw$scenario==keys$scenario[i] & raw$n==keys$n[i],,drop=FALSE]
  row <- data.frame(scenario=keys$scenario[i],n=keys$n[i],
    datasets=nrow(d),fit.converged=mean(d$fit.converged),
    original.interior.retained=mean(d$fit.converged & !nzchar(d$original.empty.interior)),
    current.availability=mean(d$current.available),
    endpoint.allowance.availability=mean(d$candidate.available),
    endpoint.empty.refit.rate=mean(d$refit.endpoint.empty.rate,na.rm=TRUE),
    interior.empty.refit.rate=mean(d$refit.interior.empty.rate,na.rm=TRUE),
    refit.failure.rate=mean(d$refit.failure.rate,na.rm=TRUE),
    statistic.failure.rate=mean(d$refit.statistic.failure.rate,na.rm=TRUE))
  for (parameter in parameters) {
    values <- summarize_parameter(d,parameter)
    for (metric in names(values)) row[[paste(parameter,metric,sep=".")]] <- values[[metric]]
  }
  row
}))
utils::write.csv(summary,file.path(outdir,"regression-summary.csv"),row.names=FALSE)
writeLines(capture.output(print(summary,row.names=FALSE,digits=4)),
  file.path(outdir,"regression-summary.txt"))
print(summary,row.names=FALSE,digits=4)

# Basic completion audit: the frozen protocol expects all 9 cells by 250 rows.
expected <- expand.grid(scenario=unique(raw$scenario),n=unique(raw$n),
                        stringsAsFactors=FALSE)
expected$rows <- vapply(seq_len(nrow(expected)),function(i)
  sum(raw$scenario==expected$scenario[i] & raw$n==expected$n[i]),integer(1))
utils::write.csv(expected,file.path(outdir,"regression-cell-audit.csv"),row.names=FALSE)
if(nrow(expected)!=9L || any(expected$rows!=250L))
  stop("completed output does not match the frozen 9-cell by 250-dataset design")
fit.errors <- !is.na(raw$fit.error) & nzchar(raw$fit.error)
if(any(!raw$fit.converged) || any(fit.errors) ||
   any(raw$refit.failure.rate>0,na.rm=TRUE) ||
   any(raw$refit.statistic.failure.rate>0,na.rm=TRUE))
  stop("one or more fit failures need review; raw rows were retained")
