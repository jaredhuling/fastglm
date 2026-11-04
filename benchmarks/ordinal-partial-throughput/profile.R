args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L || dir.exists(args[2L]))
    stop("Supply an installed baseline library and a new output directory")
library(fastglm,lib.loc=args[1L])
dir.create(args[2L],recursive=TRUE)
output <- normalizePath(args[2L])
header.root <- system.file("include",package="fastglm")
previous <- Sys.getenv("PKG_CPPFLAGS")
Sys.setenv(PKG_CPPFLAGS=paste(previous,paste0('-I"',header.root,'"')))
Rcpp::sourceCpp("benchmarks/ordinal-partial-throughput/profile.cpp",
    cacheDir=file.path(tempdir(),"ordinal-operation-profile"),rebuild=TRUE)
Sys.setenv(PKG_CPPFLAGS=previous)
config <- data.frame(n=c(20000L,200000L,1000000L,10000L,2000L),
    K=c(5L,5L,5L,10L,20L),r=c(4L,4L,4L,12L,24L))
saveRDS(list(config=config,repetitions=5L,seed=154501L),file.path(output,"configuration.rds"))
saveRDS(list(session=sessionInfo(),library=find.package("fastglm"),
    hashes=tools::md5sum(c(file.path(header.root,"ordinal_partial.h"),file.path(header.root,"ordinal.h"),
        "benchmarks/ordinal-partial-throughput/profile.cpp")),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))),
    file.path(output,"provenance.rds"))
native <- getFromNamespace("ordinal_partial_native_evaluate","fastglm")
records <- list()
for(i in seq_len(nrow(config))) {
    cell <- config[i,];set.seed(154501L+i)
    x <- matrix(runif(cell$n*2,-1,1),cell$n,2)
    z <- matrix(runif(cell$n*cell$r,-1,1),cell$n,cell$r)
    q <- cell$K-1L;alpha <- seq(-1.5,1.5,length.out=q)
    beta <- c(.2,-.15)
    gamma <- outer(rep(.1/cell$r,cell$r),rep(1,q))+
        outer(rep(.025/cell$r,cell$r),sin(seq_len(q)))
    theta <- c(alpha,beta,gamma)
    eta <- matrix(alpha,cell$n,q,byrow=TRUE)-drop(x%*%beta)-z%*%gamma
    links <- if(cell$n==1000000L) "logit" else c("logit","probit","cloglog")
    for(link in links) {
        inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(a)-expm1(-exp(a)))
        F <- inverse(eta);counts <- 1+round(1000*(cbind(F,1)-cbind(0,F)))
        weights <- rep(1,cell$n);offset <- numeric(cell$n)
        code <- match(link,c("logit","probit","cloglog"))-1L
        for(br in c(FALSE,TRUE)) {
            message("Profile n=",cell$n," K=",cell$K," r=",cell$r," ",link," meanBR=",br)
            ref <- native(x,z,integer(),counts,weights,offset,cell$K,theta,br,link=code)
            measured <- profile_partial_operations(x,z,counts,weights,offset,cell$K,theta,br,code)
            keys <- c("loglik","score","ordinary.score","adjustment","information",
                "cov.unscaled","step","scaled.score","predictor.step","minimum.gap")
            differences <- vapply(keys,function(key) max(abs(ref[[key]]-measured$evaluation[[key]])),numeric(1))
            normalized <- vapply(keys,function(key) max(abs(ref[[key]]-measured$evaluation[[key]])/(1+abs(ref[[key]]))),numeric(1))
            stopifnot(all(normalized<=5e-13))
            records[[length(records)+1L]] <- list(n=cell$n,K=cell$K,r=cell$r,d=length(theta),
                link=link,firth=br,seconds=measured$seconds,differences=differences,normalized=normalized)
            saveRDS(records,file.path(output,"records.rds"))
        }
    }
}
table <- do.call(rbind,lapply(records,function(z) {
    value <- apply(z$seconds,2,median)
    data.frame(n=z$n,K=z$K,r=z$r,d=z$d,link=z$link,firth=z$firth,
        as.list(value),check.names=FALSE)
}))
write.csv(table,file.path(output,"components.csv"),row.names=FALSE)
