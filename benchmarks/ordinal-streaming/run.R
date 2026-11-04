# Bounded-memory development comparison. Run each case in a fresh R process:
# /usr/bin/time -l Rscript benchmarks/ordinal-streaming/run.R LIBRARY N METHOD OUTPUT
# N: positive row count; METHOD: dense or source. Capture time -l separately.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L,dir.exists(args[1L]))
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
n <- as.integer(args[2L]);method <- match.arg(args[3L],c("dense","source"));out <- args[4L]
stopifnot(n>0L,!file.exists(paste0(out,".csv")))
dir.create(dirname(out),recursive=TRUE,showWarnings=FALSE)
capture.output(sessionInfo(),file=paste0(out,"-session.txt"))
capture.output(tools::md5sum(c(list.files(system.file("libs",package="fastglm"),full.names=TRUE),
    system.file("R","fastglm.rdb",package="fastglm"),system.file("R","fastglm.rdx",package="fastglm"))),
    file=paste0(out,"-library-md5.txt"))
root <- tempfile("ordinal-source-data-");dir.create(root)
# The temporary data are generated and stored one block at a time. The source
# function captures only filenames; it does not close over the complete design.
make_data <- function(n,root,size=2000L) {
    set.seed(114101)
    files <- character(ceiling(n/size))
    for(k in seq_along(files)) {
        i <- ((k-1L)*size+1L):min(k*size,n)
        x <- matrix(rnorm(length(i)*6L),length(i),6L,dimnames=list(NULL,paste0("x",1:6)))
        off <- sin(i)/8;w <- rep(c(1,2,0,3),length.out=length(i))
        y <- ordered(cut(.4*x[,1]-.2*x[,2]+off+rlogis(length(i)),c(-Inf,-1,-.2,.6,Inf)),
            labels=c("low","moderate","high","very_high"))
        files[k] <- file.path(root,sprintf("part-%05d.rds",k))
        saveRDS(list(X=x,y=y,weights=w,offset=off),files[k],compress=FALSE)
    }
    files
}
files <- make_data(n,root)
make_source <- function(paths) {
    force(paths)
    fastglm_source(function(k) readRDS(paths[k]),length(paths))
}
source <- make_source(files)
if(method=="dense") {
    blocks <- lapply(files,readRDS)
    x <- do.call(rbind,lapply(blocks,`[[`,"X"));y <- do.call(c,lapply(blocks,`[[`,"y"))
    w <- unlist(lapply(blocks,`[[`,"weights"));off <- unlist(lapply(blocks,`[[`,"offset"))
    rm(blocks)
}
records <- list();solutions <- list()
for(firth in c(FALSE,TRUE)) {
    heap.before <- gc(reset=TRUE)
    elapsed <- system.time(fit <- if(method=="source") fastglm_ordinal_streaming(source,firth=firth,tol=1e-9) else
        fastglm_ordinal(x,y,weights=w,offset=off,firth=firth,tol=1e-9))[["elapsed"]]
    stopifnot(fit$converged)
    name <- if(firth) "meanBR" else "ML"
    heap.after <- gc()
    maximum.column <- match("max used",colnames(heap.after))
    write.csv(data.frame(estimation=name,kind=rownames(heap.before),
        used_before_mb=heap.before[,2],used_after_mb=heap.after[,2],
        maximum_used_cells=heap.after[,maximum.column],maximum_used_mb=heap.after[,maximum.column+1L]),
        paste0(out,"-",name,"-heap.csv"),row.names=FALSE)
    probe <- readRDS(files[1L]);probe <- lapply(probe,function(z) if(is.matrix(z)) z[1:10,,drop=FALSE] else z[1:10])
    solutions[[name]] <- list(coefficients=coef(fit),covariance=vcov(fit),score=fit$effective$score,
        loglik=as.numeric(logLik(fit)),deviance=deviance(fit),
        prediction=predict(fit,probe$X,offset=probe$offset,se.fit=TRUE))
    records[[name]] <- data.frame(n,method,estimation=name,seconds=elapsed,converged=fit$converged,
        iterations=fit$iterations,evaluations=fit$evaluations,scaled_score=fit$scaled.score,
        predictor_step=fit$predictor.step,source_passes=if(method=="source") fit$source.passes else NA_integer_,
        source_reads=if(method=="source") fit$source.reads else NA_integer_,
        maximum_block_rows=if(method=="source") fit$maximum.block.rows else n,
        retained_design_bytes=as.numeric(object.size(fit$x)),reported_object_bytes=as.numeric(object.size(fit)))
    write.csv(do.call(rbind,records),paste0(out,".csv"),row.names=FALSE)
    saveRDS(solutions,paste0(out,"-solutions.rds"))
    cat(n,method,name,elapsed,"seconds\n");flush.console()
}
unlink(root,recursive=TRUE)
