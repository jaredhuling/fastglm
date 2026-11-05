args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=2L || dir.exists(args[2L])) stop("Supply an installed library and a new compiler-cache directory")
library(fastglm,lib.loc=args[1L]);dir.create(args[2L],recursive=TRUE);cache<-normalizePath(args[2L])
header.root<-system.file("include",package="fastglm")
previous<-Sys.getenv("PKG_CPPFLAGS")
Sys.setenv(PKG_CPPFLAGS=paste(previous,paste0('-I"',header.root,'"')))
Rcpp::sourceCpp("benchmarks/ordinal-disk-performance/components.cpp",cacheDir=cache,rebuild=TRUE)
Sys.setenv(PKG_CPPFLAGS=previous)
loader<-list.files(cache,pattern="\\.R$",recursive=TRUE,full.names=TRUE)
stopifnot(length(loader)==1L)
saveRDS(list(loader=substring(loader,nchar(cache)+2L),library=find.package("fastglm"),
    headers=tools::md5sum(file.path(header.root,c("ordinal_partial.h","ordinal.h"))),
    source=tools::md5sum("benchmarks/ordinal-disk-performance/components.cpp")),file.path(cache,"loader.rds"))
cat("Precompiled component loader:",loader,"\n")
