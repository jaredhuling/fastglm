# Run this script separately for each installed library. Loading two native
# versions into one R session would confound both provenance and measurement.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L || file.exists(args[2L]))
    stop("Supply an installed library and a new output RDS filename")
library(fastglm,lib.loc=args[1L])
set.seed(154200)
n <- 1200L;K <- 4L;q <- K-1L
x <- matrix(runif(n*4,-1,1),n,4,dimnames=list(NULL,paste0("x",1:4)))
alpha <- c(-1,0,1);beta <- c(.15,-.2,.1,-.05)
coefficients <- list(common=c(alpha,beta),partial=c(alpha,beta[1:2],
    .08,-.06,.1,-.05,.12,-.04))
native.common <- getFromNamespace("ordinal_native_evaluate","fastglm")
native.partial <- getFromNamespace("ordinal_partial_native_evaluate","fastglm")
links <- c("logit","probit","cloglog")
timed <- function(fun) {
    fun();pilot <- unname(system.time(fun())["elapsed"])
    batch <- max(1L,min(20L,ceiling(.03/max(pilot,.001))))
    elapsed <- replicate(5L,unname(system.time(for(i in seq_len(batch)) fun())["elapsed"])/batch)
    list(seconds=median(elapsed),elapsed=elapsed,batch=batch)
}
grid <- expand.grid(link=links,firth=c(FALSE,TRUE),model=c("common","partial"),stringsAsFactors=FALSE)
records <- vector("list",nrow(grid))
for(i in seq_len(nrow(grid))) {
    cell <- grid[i,];eta <- matrix(alpha,n,q,byrow=TRUE)-drop(x%*%beta)
    G <- switch(cell$link,logit=plogis,probit=pnorm,cloglog=function(t)-expm1(-exp(t)))
    F <- G(eta);counts <- 1+round(100*(cbind(F,1)-cbind(0,F)))
    colnames(counts) <- paste0("category",1:K)
    evaluate <- if(cell$model=="common") function() native.common(x,integer(),counts,
        rep(1,n),numeric(n),K,coefficients$common,cell$firth,link=match(cell$link,links)-1L) else
        function() native.partial(x[,1:2,drop=FALSE],x[,3:4,drop=FALSE],integer(),counts,
            rep(1,n),numeric(n),K,coefficients$partial,cell$firth,link=match(cell$link,links)-1L)
    fit <- function() fastglm_ordinal(x,counts,link=cell$link,firth=cell$firth,
        nonparallel=if(cell$model=="partial") 3:4 else NULL,tol=1e-8)
    value <- fit();stopifnot(value$converged)
    records[[i]] <- c(as.list(cell),list(evaluate=timed(evaluate),fit=timed(fit),
        coefficients=coef(value),covariance=vcov(value),evaluation=evaluate(),
        iterations=value$iterations,evaluations=value$evaluations))
}
package <- find.package("fastglm")
saveRDS(list(library=normalizePath(args[1L]),package=package,
    binary.hash=tools::md5sum(list.files(file.path(package,"libs"),recursive=TRUE,full.names=TRUE)),
    version=as.character(packageVersion("fastglm")),session=sessionInfo(),records=records),args[2L])
