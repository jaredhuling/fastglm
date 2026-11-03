# Higher-precision follow-up for endpoint-empty meanBR bootstrap inference.
# Run with fastglm 0.1.3 on R_LIBS. The frozen run uses 500 outer datasets
# and 499 parametric refits per eligible original fit.

if (utils::packageVersion("fastglm") != "0.1.3")
  stop("this study requires fastglm 0.1.3 on the active library path")
outer.reps <- as.integer(Sys.getenv("FASTGLM_OUTER", "500"))
B <- as.integer(Sys.getenv("FASTGLM_B", "499"))
outdir <- Sys.getenv("FASTGLM_OUTDIR", "benchmarks/ordinal-endpoint-bootstrap-development/wave2")
if (is.na(outer.reps) || outer.reps < 1L || is.na(B) || B < 2L)
  stop("FASTGLM_OUTER must be positive and FASTGLM_B must be at least two")
dir.create(outdir,recursive=TRUE,showWarnings=FALSE)

scenarios <- list(
  rare_lower_endpoint=list(alpha=c(-4,1.2),beta=c(.5),p=1L,ns=c(60L,120L,240L)),
  rare_upper_endpoint=list(alpha=c(-1.2,4),beta=c(-.5),p=1L,ns=c(60L,120L,240L)),
  moderate_control=list(alpha=c(-1,1),beta=c(.5),p=1L,ns=c(60L,120L,240L)),
  rare_lower_two_predictors=list(alpha=c(-4,1.2),beta=c(.35,.35),p=2L,
                                 ns=c(90L,180L,360L))
)
levels <- c("low","middle","high")
master.seed <- as.integer(Sys.getenv("FASTGLM_SEED","20261105"))

make_design <- function(n,p) {
  if (p==1L) return(data.frame(x1=rep(c(-1,0,1),length.out=n)))
  if (n%%9L) stop("two-predictor sample sizes must be divisible by nine")
  grid <- expand.grid(x1=c(-1,0,1),x2=c(-1,0,1))
  grid[rep(seq_len(nrow(grid)),each=n/9L),,drop=FALSE]
}

simulate_response <- function(x,scenario,seed) {
  set.seed(seed)
  location <- as.vector(as.matrix(x) %*% scenario$beta)
  u <- stats::runif(nrow(x))
  f1 <- stats::plogis(scenario$alpha[1]-location)
  f2 <- stats::plogis(scenario$alpha[2]-location)
  code <- ifelse(u<=f1,1L,ifelse(u<=f2,2L,3L))
  ordered(code,levels=seq_along(levels),labels=levels)
}

run_dataset <- function(scenario.name,scenario.index,n,size.index,replicate) {
  scenario <- scenarios[[scenario.name]]
  x <- make_design(n,scenario$p)
  seed <- master.seed+scenario.index*1000000L+size.index*10000L+replicate
  bootstrap.seed <- seed+500000L
  y <- simulate_response(x,scenario,seed)
  dat <- cbind(data.frame(y=y),x)
  form <- if(scenario$p==1L) y~x1 else y~x1+x2
  fit.warning <- character()
  fit <- tryCatch(withCallingHandlers(
    fastglm::fastglm_ordinal(form,data=dat,firth=TRUE),
    warning=function(w) {fit.warning<<-c(fit.warning,conditionMessage(w));invokeRestart("muffleWarning")}),
    error=function(e)e)
  base <- list(scenario=scenario.name,n=n,replicate=replicate,data.seed=seed,
    bootstrap.seed=bootstrap.seed,predictors=scenario$p,
    true.alpha1=scenario$alpha[1],true.alpha2=scenario$alpha[2],
    true.beta1=scenario$beta[1],true.beta2=if(scenario$p==2L) scenario$beta[2] else NA_real_,
    fit.converged=FALSE,fit.error="",fit.warning="",
    original.empty.end=NA_character_,original.empty.interior=NA_character_,
    current.available=FALSE,candidate.available=FALSE,
    refit.endpoint.empty.rate=NA_real_,refit.interior.empty.rate=NA_real_,
    refit.failure.rate=NA_real_,refit.statistic.failure.rate=NA_real_,
    estimate.alpha1=NA_real_,estimate.alpha2=NA_real_,estimate.beta1=NA_real_,
    estimate.beta2=NA_real_,lower.alpha1=NA_real_,upper.alpha1=NA_real_,
    lower.alpha2=NA_real_,upper.alpha2=NA_real_,lower.beta1=NA_real_,
    upper.beta1=NA_real_,lower.beta2=NA_real_,upper.beta2=NA_real_)
  if (inherits(fit,"error")) {
    base$fit.error <- conditionMessage(fit)
    base$fit.warning <- paste(unique(fit.warning),collapse="; ")
    return(as.data.frame(base,check.names=FALSE))
  }
  base$fit.converged <- isTRUE(fit$converged)
  base$fit.warning <- paste(unique(fit.warning),collapse="; ")
  base$original.empty.end <- paste(fit$levels[fit$category$empty.end.categories],collapse="|")
  base$original.empty.interior <- paste(fit$levels[fit$category$collapsed.categories],collapse="|")
  estimates <- stats::coef(fit)
  alpha.names <- paste0("threshold:",fit$levels[-length(fit$levels)],"|",fit$levels[-1L])
  base$estimate.alpha1 <- unname(estimates[alpha.names[1L]])
  base$estimate.alpha2 <- unname(estimates[alpha.names[2L]])
  base$estimate.beta1 <- unname(estimates["x1"])
  if (scenario$p==2L) base$estimate.beta2 <- unname(estimates["x2"])
  if (!base$fit.converged) {
    base$fit.error <- paste0("original meanBR fit did not converge: ",fit$convergence.reason)
    return(as.data.frame(base,check.names=FALSE))
  }
  if (isTRUE(fit$category$boundary))
    return(as.data.frame(base,check.names=FALSE))

  b.warning <- character()
  boot <- tryCatch(withCallingHandlers(
    fastglm::fastglm_bootstrap(fit,R=B,type="parametric",seed=bootstrap.seed),
    warning=function(w) {b.warning<<-c(b.warning,conditionMessage(w));invokeRestart("muffleWarning")}),
    error=function(e)e)
  if (inherits(boot,"error")) {
    base$fit.error <- paste0("bootstrap error: ",conditionMessage(boot))
    base$fit.warning <- paste(unique(c(base$fit.warning,b.warning)),collapse="; ")
    return(as.data.frame(base,check.names=FALSE))
  }
  d <- boot$diagnostics
  base$current.available <- isTRUE(boot$inference.available)
  base$refit.endpoint.empty.rate <- mean(lengths(d$empty.end.categories)>0L)
  base$refit.interior.empty.rate <- mean(d$boundary%in%TRUE,na.rm=TRUE)
  base$refit.failure.rate <- mean(!d$converged)
  base$refit.statistic.failure.rate <- mean(!d$statistic.ok)
  base$candidate.available <- all(d$converged) && all(d$statistic.ok) &&
    all(!is.na(d$boundary)&!d$boundary) && all(is.finite(boot$replicates))
  if (base$candidate.available) {
    ci <- t(apply(boot$replicates,2L,stats::quantile,
                  probs=c(.025,.975),names=FALSE,type=7))
    colnames(ci) <- c("lower","upper")
    stat.names <- c(alpha1=alpha.names[1L],alpha2=alpha.names[2L],
                    beta1="x1",beta2="x2")
    active <- if(scenario$p==1L) c("alpha1","alpha2","beta1") else names(stat.names)
    for (parameter in active) {
      key <- stat.names[[parameter]]
      base[[paste0("lower.",parameter)]] <- ci[key,"lower"]
      base[[paste0("upper.",parameter)]] <- ci[key,"upper"]
    }
  }
  base$fit.warning <- paste(unique(c(base$fit.warning,b.warning)),collapse="; ")
  as.data.frame(base,check.names=FALSE)
}

rows <- vector("list",length(unlist(lapply(scenarios,`[[`,"ns")))*outer.reps)
at <- 0L
for (s in seq_along(scenarios)) for (j in seq_along(scenarios[[s]]$ns)) {
  scenario.name <- names(scenarios)[s]
  n <- scenarios[[s]]$ns[j]
  message("Running ",scenario.name,", n=",n,", datasets=",outer.reps,
          ", refits per eligible fit=",B)
  for (r in seq_len(outer.reps)) {
    at <- at+1L
    rows[[at]] <- run_dataset(scenario.name,s,n,j,r)
  }
  utils::write.csv(do.call(rbind,rows[seq_len(at)]),
                   file.path(outdir,"wave2-raw.csv"),row.names=FALSE)
}
raw <- do.call(rbind,rows)
utils::write.csv(raw,file.path(outdir,"wave2-raw.csv"),row.names=FALSE)
