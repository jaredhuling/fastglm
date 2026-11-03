# Paired parametric-bootstrap development simulation for ordinal regression.
#
# Run after installing fastglm 0.1.3 into a library on R_LIBS. Environment
# overrides are provided for timing pilots; the planned study uses 250 outer
# datasets per cell and 199 parametric refits per dataset.

if (utils::packageVersion("fastglm") != "0.1.3")
  stop("this study requires fastglm 0.1.3 on the active library path")

outer.reps <- as.integer(Sys.getenv("FASTGLM_OUTER", "250"))
B <- as.integer(Sys.getenv("FASTGLM_B", "199"))
outdir <- Sys.getenv("FASTGLM_OUTDIR", "benchmarks/ordinal-endpoint-bootstrap-development")
if (is.na(outer.reps) || outer.reps < 1L || is.na(B) || B < 2L)
  stop("FASTGLM_OUTER must be positive and FASTGLM_B must be at least two")
dir.create(outdir, recursive=TRUE, showWarnings=FALSE)

scenarios <- list(
  rare_lower_endpoint=list(alpha=c(-4,1.2), beta=.5),
  sparse_interior=list(alpha=c(-2.2,-2.05), beta=.3),
  moderate_control=list(alpha=c(-1,1), beta=.5)
)
sample.sizes <- c(60L,120L,240L)
category.levels <- c("low","middle","high")
master.seed <- 20261104L

simulate_response <- function(x, alpha, beta, seed) {
  set.seed(seed)
  u <- stats::runif(length(x))
  lower <- stats::plogis(alpha[1]-beta*x)
  upper <- stats::plogis(alpha[2]-beta*x)
  code <- ifelse(u <= lower, 1L, ifelse(u <= upper, 2L, 3L))
  ordered(code, levels=seq_along(category.levels), labels=category.levels)
}

run_dataset <- function(scenario.name, scenario.index, n, size.index, replicate) {
  scenario <- scenarios[[scenario.name]]
  # A fixed, balanced design isolates response sparsity from random design
  # imbalance and is held fixed in each parametric bootstrap.
  x <- rep(c(-1,0,1), length.out=n)
  data.seed <- master.seed + scenario.index*1000000L + size.index*10000L + replicate
  bootstrap.seed <- data.seed + 500000L
  y <- simulate_response(x,scenario$alpha,scenario$beta,data.seed)
  fit.warning <- character()
  fit <- tryCatch(withCallingHandlers(
    fastglm::fastglm_ordinal(y~x,data=data.frame(y=y,x=x),firth=TRUE),
    warning=function(w) {fit.warning <<- c(fit.warning,conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) e)
  base <- list(scenario=scenario.name,n=n,replicate=replicate,
    data.seed=data.seed,bootstrap.seed=bootstrap.seed,
    true.alpha1=scenario$alpha[1],true.alpha2=scenario$alpha[2],true.beta=scenario$beta,
    fit.converged=FALSE,fit.error="",fit.warning=paste(unique(fit.warning),collapse="; "),
    original.empty.end=NA_character_,original.empty.interior=NA_character_,
    current.available=FALSE,candidate.available=FALSE,
    refit.endpoint.empty.rate=NA_real_,refit.interior.empty.rate=NA_real_,
    refit.failure.rate=NA_real_,refit.statistic.failure.rate=NA_real_,
    estimate.alpha1=NA_real_,estimate.alpha2=NA_real_,estimate.beta=NA_real_)
  if (inherits(fit,"error")) {
    base$fit.error <- conditionMessage(fit)
    return(as.data.frame(base,check.names=FALSE))
  }
  base$fit.converged <- isTRUE(fit$converged)
  base$original.empty.end <- paste(fit$levels[fit$category$empty.end.categories],collapse="|")
  base$original.empty.interior <- paste(fit$levels[fit$category$collapsed.categories],collapse="|")
  estimate <- stats::coef(fit)
  alpha.names <- paste0("threshold:",fit$levels[-length(fit$levels)],"|",fit$levels[-1L])
  beta.names <- names(fit$beta)
  base$estimate.alpha1 <- unname(estimate[alpha.names[1L]])
  base$estimate.alpha2 <- unname(estimate[alpha.names[2L]])
  base$estimate.beta <- unname(estimate[beta.names[1L]])
  for (nm in c("lower.alpha1","upper.alpha1","lower.alpha2","upper.alpha2",
               "lower.beta","upper.beta")) base[[nm]] <- NA_real_
  if (!base$fit.converged) {
    base$fit.error <- paste0("original meanBR fit did not converge: ",fit$convergence.reason)
    return(as.data.frame(base,check.names=FALSE))
  }
  # Once an interior category is empty, the original fit has a reduced model
  # and the full threshold-vector bootstrap target is not defined here.
  if (isTRUE(fit$category$boundary))
    return(as.data.frame(base,check.names=FALSE))

  b.warning <- character()
  boot <- tryCatch(withCallingHandlers(
    fastglm::fastglm_bootstrap(fit,R=B,type="parametric",seed=bootstrap.seed),
    warning=function(w) {b.warning <<- c(b.warning,conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) e)
  if (inherits(boot,"error")) {
    base$fit.error <- paste0("bootstrap error: ",conditionMessage(boot))
    return(as.data.frame(base,check.names=FALSE))
  }
  d <- boot$diagnostics
  base$current.available <- isTRUE(boot$inference.available)
  base$refit.endpoint.empty.rate <- mean(lengths(d$empty.end.categories)>0L)
  base$refit.interior.empty.rate <- mean(d$boundary %in% TRUE,na.rm=TRUE)
  base$refit.failure.rate <- mean(!d$converged)
  base$refit.statistic.failure.rate <- mean(!d$statistic.ok)
  candidate <- !isTRUE(fit$category$boundary) &&
    all(d$converged) && all(d$statistic.ok) &&
    all(!is.na(d$boundary) & !d$boundary) &&
    all(is.finite(boot$replicates))
  base$candidate.available <- candidate
  if (candidate) {
    ci <- t(apply(boot$replicates,2L,stats::quantile,
                  probs=c(.025,.975),names=FALSE,type=7))
    colnames(ci) <- c("lower","upper")
    for (nm in c("alpha1","alpha2","beta")) {
      key <- switch(nm,alpha1=alpha.names[1L],alpha2=alpha.names[2L],beta=beta.names[1L])
      base[[paste0("lower.",nm)]] <- ci[key,"lower"]
      base[[paste0("upper.",nm)]] <- ci[key,"upper"]
    }
  }
  base$fit.warning <- paste(unique(c(base$fit.warning,b.warning)),collapse="; ")
  as.data.frame(base,check.names=FALSE)
}

rows <- vector("list",length(scenarios)*length(sample.sizes)*outer.reps)
at <- 0L
for (s in seq_along(scenarios)) for (j in seq_along(sample.sizes)) {
  scenario.name <- names(scenarios)[s]
  n <- sample.sizes[j]
  message("Running ",scenario.name,", n=",n,", datasets=",outer.reps,
          ", refits per eligible fit=",B)
  for (r in seq_len(outer.reps)) {
    at <- at+1L
    rows[[at]] <- run_dataset(scenario.name,s,n,j,r)
  }
  partial <- do.call(rbind,rows[seq_len(at)])
  utils::write.csv(partial,file.path(outdir,"regression-raw.csv"),row.names=FALSE)
}
raw <- do.call(rbind,rows)

summarize_cell <- function(d) {
  available <- d$candidate.available
  eligible.original <- !nzchar(d$original.empty.interior) & d$fit.converged
  cover <- function(parm) {
    lo <- d[[paste0("lower.",parm)]]; hi <- d[[paste0("upper.",parm)]]
    truth <- d[[paste0("true.",parm)]]
    ok <- available & is.finite(lo) & is.finite(hi)
    hit <- ok & lo<=truth & hi>=truth
    c(n.available=sum(ok),coverage=if(any(ok)) mean(hit[ok]) else NA_real_,
      mcse=if(any(ok)) sqrt(mean(hit[ok])*(1-mean(hit[ok]))/sum(ok)) else NA_real_)
  }
  c(fit.converged=mean(d$fit.converged),
    original.interior.retained=mean(eligible.original),
    current.availability=mean(d$current.available),
    endpoint.allowance.availability=mean(available),
    endpoint.empty.refit.rate=mean(d$refit.endpoint.empty.rate,na.rm=TRUE),
    interior.empty.refit.rate=mean(d$refit.interior.empty.rate,na.rm=TRUE),
    refit.failure.rate=mean(d$refit.failure.rate,na.rm=TRUE),
    c(alpha1=cover("alpha1"),alpha2=cover("alpha2"),beta=cover("beta")))
}
keys <- unique(raw[c("scenario","n")])
summary <- do.call(rbind,lapply(seq_len(nrow(keys)),function(i) {
  d <- raw[raw$scenario==keys$scenario[i] & raw$n==keys$n[i],,drop=FALSE]
  data.frame(scenario=keys$scenario[i],n=as.integer(keys$n[i]),
    as.list(summarize_cell(d)),check.names=FALSE)
}))
utils::write.csv(raw,file.path(outdir,"regression-raw.csv"),row.names=FALSE)
utils::write.csv(summary,file.path(outdir,"regression-summary.csv"),row.names=FALSE)
writeLines(capture.output(print(summary,row.names=FALSE,digits=4)),
           file.path(outdir,"regression-summary.txt"))
print(summary,row.names=FALSE,digits=4)
