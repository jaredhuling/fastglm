# Run baseline and candidate in separate R processes using the same cases.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L || dir.exists(args[2L]))
    stop("Supply an installed library and a new output directory")
library(fastglm,lib.loc=args[1L])
dir.create(args[2L],recursive=TRUE)
output <- normalizePath(args[2L])
config <- data.frame(n=c(20000L,100000L,1000000L,10000L),
    K=c(4L,5L,4L,10L),r=c(2L,4L,2L,12L),seed=154601L+1:4)
saveRDS(list(config=config,repetitions=3L,source.block=10000L),
    file.path(output,"configuration.rds"))
package <- find.package("fastglm")
saveRDS(list(time=Sys.time(),session=sessionInfo(),library=package,
    binary.hash=tools::md5sum(list.files(file.path(package,"libs"),recursive=TRUE,full.names=TRUE)),
    headers=tools::md5sum(file.path(package,"include",c("ordinal_partial.h","ordinal.h"))),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))),
    file.path(output,"provenance.rds"))
records <- list()
for(i in seq_len(nrow(config))) {
    cell <- config[i,];set.seed(cell$seed)
    x <- matrix(runif(cell$n*(2L+cell$r),-1,1),cell$n)
    colnames(x) <- paste0("x",seq_len(ncol(x)))
    q <- cell$K-1L;alpha <- seq(-1.7,1.4,length.out=q)
    beta <- c(.17,-.11)
    gamma <- outer(rep(.08/cell$r,cell$r),rep(1,q))+
        outer(rep(.021/cell$r,cell$r),cos(seq_len(q)))
    offset <- sin(seq_len(cell$n))/20
    eta <- matrix(alpha,cell$n,q,byrow=TRUE)-drop(x[,1:2,drop=FALSE]%*%beta)-
        x[,-(1:2),drop=FALSE]%*%gamma-offset
    weights <- rep(c(0,1,2,1,1),length.out=cell$n)
    links <- if(cell$n==1000000L) "logit" else c("logit","probit","cloglog")
    for(link in links) {
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(a)-expm1(-exp(a)))
        F <- inverse(eta);counts <- 1+round(1000*(cbind(F,1)-cbind(0,F)))
        colnames(counts) <- paste0("category",seq_len(cell$K))
        chunks <- ceiling(cell$n/10000L)
        source <- fastglm_source(function(k) {
            at <- seq.int((k-1L)*10000L+1L,min(k*10000L,cell$n))
            list(X=x[at,,drop=FALSE],y=counts[at,,drop=FALSE],
                weights=weights[at],offset=offset[at])
        },chunks)
        for(br in c(FALSE,TRUE)) for(representation in c("dense","source")) {
            message("Compare n=",cell$n," d=",q+2+q*cell$r," ",link,
                " meanBR=",br," ",representation)
            fun <- if(representation=="dense") function() fastglm_ordinal(x,counts,
                weights=weights,offset=offset,nonparallel=seq.int(3L,ncol(x)),link=link,
                firth=br,tol=1e-8) else function() fastglm_ordinal_streaming(source,
                nonparallel=seq.int(3L,ncol(x)),link=link,firth=br,tol=1e-8)
            records[[length(records)+1L]] <- tryCatch({
                fit <- fun()
                result <- list(n=cell$n,K=cell$K,r=cell$r,d=length(coef(fit)),link=link,
                    firth=br,representation=representation,status=fit$convergence.reason,
                    converged=fit$converged,coefficients=coef(fit),covariance=fit$cov.unscaled,
                    history=fit$history,iterations=fit$iterations,evaluations=fit$evaluations,
                    fallbacks=fit$fallbacks,source.passes=fit$source.passes,
                    scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,
                    minimum.gap=fit$minimum.gap,loglik=fit$loglik,
                    equations=fit$working[c("score","ordinary.score","adjustment","information","step")])
                if(fit$converged) {
                    elapsed <- replicate(3L,unname(system.time(fun())["elapsed"]))
                    result$elapsed <- elapsed;result$median <- median(elapsed)
                }
                result
            },error=function(e) list(n=cell$n,K=cell$K,r=cell$r,link=link,firth=br,
                representation=representation,status=conditionMessage(e),converged=FALSE))
            saveRDS(records,file.path(output,"records.rds"))
        }
    }
}
