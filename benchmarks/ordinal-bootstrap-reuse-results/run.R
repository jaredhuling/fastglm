# Rscript run.R baseline|candidate library output-directory
# Run baseline first, then candidate with the same output directory.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3L || !args[1L]%in%c("baseline","candidate"))
    stop("usage: Rscript run.R baseline|candidate library output-directory")
mode <- args[1L];lib <- normalizePath(args[2L]);output <- args[3L]
script <- sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1L])
directory <- dirname(normalizePath(script))
fixture <- file.path(directory,"..","ordinal-prepared-results","fixtures.rds")
dir.create(output,recursive=TRUE,showWarnings=FALSE)
result.path <- file.path(output,paste0(mode,".rds"))
if(file.exists(result.path)) stop("output exists; use a fresh output directory")
.libPaths(c(lib,.libPaths()));library(fastglm)
RNGkind("Mersenne-Twister","Inversion","Rejection")
data <- readRDS(fixture)
reference <- if(mode=="candidate") readRDS(file.path(output,"baseline.rds"))
measure <- function(fn,key) {
    elapsed <- numeric(3L)
    for(j in seq_along(elapsed)) {
        gc()
        elapsed[j] <- system.time(value <- fn())[["elapsed"]]
    }
    profile <- paste0(mode,"-",key,"-Rprofmem.out")
    Rprofmem(file.path(output,profile))
    value <- tryCatch(fn(),finally=Rprofmem(NULL))
    bytes <- suppressWarnings(as.numeric(sub(" .*","",readLines(file.path(output,profile),warn=FALSE))))
    list(elapsed=elapsed,R.allocation.bytes=sum(bytes,na.rm=TRUE),profile=profile)
}
checks <- list();measurements <- list();rows <- list()
for(d in data) for(br in c(FALSE,TRUE)) {
    key <- paste(d$n,d$p,d$K,if(br) "BR" else "ML",sep="_")
    fit <- fastglm_ordinal(d$x,d$y[[1L]],weights=d$weights,offset=d$offset,firth=br)
    stopifnot(fit$converged)
    operation <- function()fastglm_bootstrap(fit,R=19L,type="parametric",seed=8402)
    check <- operation()
    stopifnot(check$successful==19L,isTRUE(check$inference.available))
    check$call <- NULL
    if(mode=="candidate") stopifnot(identical(check,reference$checks[[key]]))
    checks[[key]] <- check
    measurements[[key]] <- measure(operation,key)
    m <- measurements[[key]]
    rows[[key]] <- data.frame(case=key,n=d$n,p=d$p,K=d$K,firth=br,refits=19L,
        median.seconds=median(m$elapsed),min.seconds=min(m$elapsed),max.seconds=max(m$elapsed),
        R.allocation.bytes=m$R.allocation.bytes,successful=check$successful,
        exact.baseline.agreement=if(mode=="candidate") TRUE else NA)
    metadata <- list(library=find.package("fastglm"),session=sessionInfo(),rng.kind=RNGkind(),
        dll.md5=tools::md5sum(getLoadedDLLs()[["fastglm"]][["path"]]),
        namespace.md5=tools::md5sum(file.path(find.package("fastglm"),"R",c("fastglm.rdb","fastglm.rdx"))),
        fixture.md5=tools::md5sum(fixture),runner.md5=tools::md5sum(script))
    saveRDS(list(checks=checks,measurements=measurements,table=do.call(rbind,rows),metadata=metadata),
        result.path,compress="xz")
    write.csv(do.call(rbind,rows),file.path(output,paste0(mode,".csv")),row.names=FALSE)
    cat(key,"complete\n");flush.console()
}
