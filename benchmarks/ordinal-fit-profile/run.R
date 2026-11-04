# Development profiling of the committed implementation, without modifying fits.
# Run from the repository root: Rscript benchmarks/ordinal-fit-profile/run.R LIBRARY
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L,dir.exists(args[1L]))
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
destination <- "benchmarks/ordinal-fit-profile/results"
dir.create(destination,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(destination,"timing.csv"))) stop("Profile records already exist")
capture.output(sessionInfo(),file=file.path(destination,"session.txt"))
writeLines("3f4a407",file.path(destination,"baseline.txt"))
set.seed(111601)
records <- list()
for(n in c(100000L,1000000L)) {
    p <- if(n==100000L) 6L else 3L; K <- if(p==6L) 5L else 4L
    x <- matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    y <- ordered(cut(.4*x[,1]-.2*x[,2]+rlogis(n),c(-Inf,seq(-1.2,1.2,length.out=K-1L),Inf)),labels=seq_len(K))
    for(firth in c(FALSE,TRUE)) {
        estimation <- if(firth) "meanBR" else "ML"
        prefix <- file.path(destination,paste(n,estimation,sep="-"))
        gc();Rprof(paste0(prefix,".Rprof"),interval=.002,memory.profiling=TRUE)
        elapsed <- system.time(fit <- fastglm_ordinal(x,y,firth=firth))[["elapsed"]]
        Rprof(NULL)
        s <- summaryRprof(paste0(prefix,".Rprof"),memory="both")
        write.csv(s$by.total,paste0(prefix,"-profile.csv"))
        records[[length(records)+1L]] <- data.frame(n,p,K,estimation,operation="fit",seconds=elapsed,
            converged=fit$converged,iterations=fit$iterations,evaluations=fit$evaluations,fallbacks=fit$fallbacks)
        prepared <- fastglm_ordinal_prepare(x,firth=firth)
        invisible(fastglm_ordinal(prepared,y))
        elapsed <- median(replicate(3,system.time(fastglm_ordinal(prepared,y))[["elapsed"]]))
        records[[length(records)+1L]] <- transform(records[[length(records)]],operation="prepared_fit",seconds=elapsed)
        write.csv(do.call(rbind,records),file.path(destination,"timing.csv"),row.names=FALSE)
        cat(n,estimation,"finished\n");flush.console()
    }
}
