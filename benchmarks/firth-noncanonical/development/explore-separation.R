# Post-confirmation diagnosis; this case is now development evidence.
# Rscript benchmarks/firth-noncanonical/development/explore-separation.R CANDIDATE_LIBRARY
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1L)
library(fastglm,lib.loc=args[1L])
code <- readLines("benchmarks/firth-noncanonical/run.R")
eval(parse(text=code[grep("^set.seed",code):(grep("^records <-",code)-1)]))
d <- cases[["separation-cloglog"]]
counts <- cbind(event=d$y,none=1-d$y)
ordinal <- suppressWarnings(fastglm_ordinal(d$x[,-1],counts,weights=d$weights,
    offset=-d$offset,link="cloglog",firth=TRUE,maxit=500))
print(fastglm_diagnostics(ordinal))
for(slowit in c(.5,.1)) {
    reference <- suppressWarnings(brglm2::brglmFit(d$x,d$y,family=d$family,
        weights=d$weights,offset=d$offset,start=numeric(3),
        control=brglm2::brglmControl(type="AS_mean",epsilon=1e-9,maxit=1000,slowit=slowit)))
    print(list(slowit=slowit,converged=reference$converged,iterations=reference$iter,coefficients=coef(reference)))
    if(reference$converged) {
        check <- suppressWarnings(fastglm(d$x,d$y,d$family,weights=d$weights,offset=d$offset,
            firth=TRUE,start=coef(reference),tol=1e-9,maxit=1))
        print(list(converged=check$converged,coefficient.change=max(abs(coef(check)-coef(reference))),
            standardized.adjusted.score=max(abs(check$standardized.adjusted.score))))
    }
}
