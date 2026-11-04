# Higher-resolution timing check for ordinary inputs; separate from stress tests.
# Rscript benchmarks/firth-noncanonical/time-ordinary.R LIBRARY LABEL OUTPUT_DIRECTORY
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L,dir.exists(args[1L]))
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
output <- file.path(args[3L],paste0(args[2L],"-ordinary-timing.csv"))
if(file.exists(output)) stop("Results already exist")
set.seed(112001)
n <- 12000L
x <- cbind("(Intercept)"=1,matrix(rnorm(n*5),n,5))
eta <- -.3+.2*x[,2]-.1*x[,3]+.2*x[,4]
records <- list()
for(family in list(binomial(),binomial("probit"),binomial("cloglog"),poisson(),Gamma("log"),gaussian())) {
    y <- switch(family$family,binomial=rbinom(n,1,family$linkinv(eta)),
        poisson=rpois(n,exp(eta)),Gamma=rgamma(n,shape=1.2,scale=exp(eta)/1.2),gaussian=eta+rnorm(n))
    run <- function() fastglm(x,y,family,firth=TRUE,tol=1e-9)
    fit <- run();stopifnot(fit$converged)
    gc()
    timing <- microbenchmark::microbenchmark(run(),times=41L)
    records[[length(records)+1L]] <- data.frame(family=family$family,link=family$link,
        repetition=seq_len(nrow(timing)),nanoseconds=timing$time,iterations=fit$iter,
        reductions=if(is.null(fit$step.reductions)) NA_integer_ else fit$step.reductions)
    write.csv(do.call(rbind,records),output,row.names=FALSE)
}
