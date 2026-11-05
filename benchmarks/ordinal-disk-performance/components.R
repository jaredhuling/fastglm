args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=3L || dir.exists(args[3L])) stop("Supply library, precompiled cache, and new output directory")
library(fastglm,lib.loc=args[1L]);cache<-normalizePath(args[2L])
compiled<-readRDS(file.path(cache,"loader.rds"))
stopifnot(identical(compiled$library,find.package("fastglm")),
    identical(unname(compiled$headers),unname(tools::md5sum(file.path(system.file("include",package="fastglm"),
        c("ordinal_partial.h","ordinal.h"))))),
    identical(unname(compiled$source),unname(tools::md5sum("benchmarks/ordinal-disk-performance/components.cpp"))))
source(file.path(cache,compiled$loader))
source("benchmarks/ordinal-disk-performance/config.R")
config<-disk_configuration();dir.create(args[3L],recursive=TRUE);output<-normalizePath(args[3L])
saveRDS(list(config=config,compiled=compiled,session=sessionInfo(),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))),
    file.path(output,"configuration.rds"))
native<-getFromNamespace("ordinal_partial_native_evaluate","fastglm")
records<-list()
for(i in seq_len(nrow(config$components))) for(link in config$links) for(br in c(FALSE,TRUE)) {
    cell<-config$components[i,]
    z<-disk_fixture(cell$n,cell$K,cell$r,link,config$seed+1000000L+i)
    x<-z$X[,1:2,drop=FALSE];varying<-z$X[,-(1:2),drop=FALSE]
    code<-match(link,config$links)-1L
    message("Components ",cell$id," ",link," meanBR=",br)
    reference<-native(x,varying,integer(),z$y,z$weights,z$offset,cell$K,z$theta,br,link=code)
    value<-profile_partial_operations(x,varying,z$y,z$weights,z$offset,cell$K,z$theta,br,code,5L)
    fields<-c("loglik","information","score","ordinary.score","adjustment","cov.unscaled",
        "step","scaled.score","predictor.step","minimum.gap")
    error<-vapply(fields,function(nm) max(abs(reference[[nm]]-value$evaluation[[nm]])/(1+abs(reference[[nm]]))),numeric(1))
    records[[length(records)+1L]]<-list(cell=cell,link=link,firth=br,d=length(z$theta),
        seconds=value$seconds,evaluation=value$evaluation,error=error,passed=all(error<=5e-13))
    saveRDS(records,file.path(output,"records.rds"))
}
stopifnot(all(vapply(records,`[[`,logical(1),"passed")))
