# Rscript benchmarks/binomial-tail-accuracy/diagnose.R LIBRARY
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==1L)
.libPaths(c(normalizePath(args[1]),.libPaths()));library(fastglm)
source('benchmarks/binomial-tail-accuracy/reference.R')
code <- readLines('benchmarks/firth-noncanonical/run.R')
eval(parse(text=code[grep('^set.seed',code):(grep('^records <-',code)-1)]))
d <- cases[['separation-cloglog']]
reference <- suppressWarnings(brglm2::brglmFit(d$x,d$y,weights=d$weights,offset=d$offset,
 family=d$family,start=numeric(3),control=brglm2::brglmControl(type='AS_mean',slowit=.1,epsilon=1e-11,maxit=1000)))
fit <- suppressWarnings(fastglm(d$x,d$y,d$family,weights=d$weights,offset=d$offset,
 firth=TRUE,tol=1e-10,maxit=500))
for(name in c('reference','fit')) {
 object <- get(name);exact <- tail_reference(d$x,d$y,d$weights,d$offset,coef(object),'cloglog')
 cat('\n',name,'\n');print(list(coefficients=coef(object),converged=object$converged,
 exact_score=exact$score,standardized_score=max(abs(exact$score/sqrt(diag(exact$information)))),
 remaining_predictor_change=max(abs(d$x%*%solve(exact$information,exact$score)))))
}
# This independently implemented ordinal evaluator uses the same binary model.
ordinal <- fastglm_ordinal(-d$x[,-1],cbind(low=d$y,high=1-d$y),weights=d$weights,
 offset=-d$offset,link='cloglog',firth=TRUE,tol=1e-10,maxit=500)
print(list(binary_ordinal_converged=ordinal$converged,
 coefficient_difference=max(abs(unname(coef(ordinal))-unname(coef(fit))))))
