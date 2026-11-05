args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=4L || dir.exists(args[4L]))
    stop("Supply baseline library, candidate library, existing inputs, and new results directory")
source("benchmarks/ordinal-disk-performance/config.R")
config<-disk_configuration();inputs<-normalizePath(args[3L])
package.provenance<-lapply(args[1:2],function(library) {
    package<-file.path(normalizePath(library),"fastglm")
    binary<-list.files(file.path(package,"libs"),pattern="[.](so|dll)$",full.names=TRUE,recursive=TRUE)
    stopifnot(length(binary)>0L)
    list(binary=tools::md5sum(binary),headers=tools::md5sum(file.path(package,"include",
        c("ordinal.h","ordinal_partial.h"))))
})
if(identical(unname(package.provenance[[1L]]$binary),unname(package.provenance[[2L]]$binary)))
    stop("Baseline and candidate binaries are identical; rebuild the native candidate before timing")
if(!identical(unname(package.provenance[[2L]]$headers),
    unname(tools::md5sum(file.path("inst/include",c("ordinal.h","ordinal_partial.h"))))))
    stop("Installed candidate headers differ from the source under study")
manifest<-readRDS(file.path(inputs,"inputs.rds"))
stopifnot(identical(config,manifest$configuration))
for(dataset in manifest$datasets)
    stopifnot(identical(unname(tools::md5sum(file.path(inputs,dataset$files))),dataset$md5))
dir.create(args[4L],recursive=TRUE);output<-normalizePath(args[4L]);grid<-disk_grid(config)
saveRDS(list(configuration=config,grid=grid,inputs=manifest,
    libraries=normalizePath(args[1:2]),package.provenance=package.provenance,
    session=sessionInfo()),file.path(output,"configuration.rds"))
Sys.setenv(OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,VECLIB_MAXIMUM_THREADS=1)
mac<-Sys.info()[["sysname"]]=="Darwin"
if(!mac && Sys.info()[["sysname"]]!="Linux") stop("Process resource collection supports macOS or Linux")
rscript<-file.path(R.home("bin"),"Rscript")
status<-list()
for(i in seq_len(nrow(grid))) for(repetition in seq_len(config$repetitions)) {
    cell<-grid[i,,drop=FALSE];cell$repetition<-repetition
    cell.file<-file.path(output,sprintf("cell-%03d-rep-%02d.rds",i,repetition));saveRDS(cell,cell.file)
    order<-if((i+repetition)%%2L) c("baseline","current") else c("current","baseline")
    for(version in order) {
        stem<-sprintf("cell-%03d-rep-%02d-%s",i,repetition,version)
        result<-file.path(output,paste0(stem,".rds"))
        message(stem," ",cell$case," ",cell$link," ",cell$model," ",cell$representation," meanBR=",cell$firth)
        library<-normalizePath(args[if(version=="baseline") 1L else 2L])
        command<-c(if(mac) "-l" else "-v",shQuote(rscript),
            shQuote("benchmarks/ordinal-disk-performance/worker.R"),shQuote(library),shQuote(inputs),
            shQuote(cell.file),shQuote(result),version)
        started<-proc.time()[["elapsed"]]
        exit<-system2("/usr/bin/time",command,stdout=file.path(output,paste0(stem,".log")),
            stderr=file.path(output,paste0(stem,".resources")))
        status[[length(status)+1L]]<-data.frame(cell=i,repetition=repetition,version=version,
            exit=exit,process.elapsed=proc.time()[["elapsed"]]-started,result=basename(result))
        write.csv(do.call(rbind,status),file.path(output,"execution.csv"),row.names=FALSE)
        if(exit!=0L || !file.exists(result)) stop("Worker failed to produce a result: ",stem)
    }
}
