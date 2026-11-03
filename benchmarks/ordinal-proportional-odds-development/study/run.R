# Rscript benchmarks/ordinal-proportional-odds-development/study/run.R pilot|run|summarize|audit
args <- commandArgs(trailingOnly=TRUE)
action <- if(length(args)) args[1] else "audit"
root <- "benchmarks/ordinal-proportional-odds-development/study"
stopifnot(dir.exists(root))
study.library <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(study.library)) .libPaths(c(study.library,.libPaths()))
RNGkind("Mersenne-Twister","Inversion","Rejection")
source(file.path(root,"config.R"))
source(file.path(root,"methods.R"))
source(file.path(root,"..","prototype.R"))
source(file.path(root,"summarize.R"))
frozen <- file.path(root,c("PROTOCOL.md","config.R","methods.R","run.R","summarize.R","../prototype.R"))
identity <- list(files=setNames(unname(tools::md5sum(frozen)),basename(frozen)),
    packages=vapply(c("fastglm","MASS","car"),function(x) as.character(packageVersion(x)),character(1)),
    candidate.revision="b5e08af",R=R.version.string,RNG=RNGkind())
manifest <- file.path(root,"manifest.rds")
if(action%in%c("pilot","run")) {
    if(file.exists(manifest)) stopifnot(identical(readRDS(manifest),identity))
    else {
        saveRDS(identity,manifest,version=2)
        dput(identity,file=file.path(root,"manifest.txt"))
        capture.output(sessionInfo(),file=file.path(root,"session.txt"))
        dput(getS3method("poTest","polr",envir=asNamespace("car")),
             file=file.path(root,"car-source.txt"))
    }
}
if(action=="pilot") {
    pilot <- lapply(seq_len(nrow(po_cells)),function(i)
        po_run_cell(po_cells[i,],9001:9003))
    saveRDS(pilot,file.path(root,"pilot.rds"),version=2)
    timings <- do.call(rbind,lapply(seq_along(pilot),function(i) {
        z <- pilot[[i]]
        data.frame(po_cells[i,],seconds.per.dataset=sum(z$elapsed)/3)
    }))
    write.csv(timings,file.path(root,"pilot-timings.csv"),row.names=FALSE)
    print(timings)
    cat("Estimated total compute seconds:",sum(timings$seconds.per.dataset)*po_repetitions,"\n")
} else if(action=="run") {
    dir.create(file.path(root,"raw"),showWarnings=FALSE)
    tasks <- seq_len(nrow(po_cells))
    completed <- function(i) file.exists(file.path(root,"raw",sprintf("cell-%02d.rds",i)))
    tasks <- tasks[!vapply(tasks,completed,logical(1))]
    workers <- as.integer(Sys.getenv("FASTGLM_STUDY_WORKERS","3"))
    stopifnot(is.finite(workers),workers>=1L)
    job <- function(i) {
        path <- file.path(root,"raw",sprintf("cell-%02d.rds",i))
        start <- proc.time()[3]
        z <- po_run_cell(po_cells[i,],seq_len(po_repetitions),path)
        cat(sprintf("Completed cell %02d: %s, n=%d, %.1f seconds\n",i,po_cells$scenario[i],po_cells$n[i],proc.time()[3]-start))
        nrow(z)
    }
    if(.Platform$OS.type=="unix" && workers>1L)
        answer <- parallel::mclapply(tasks,job,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
    else answer <- lapply(tasks,job)
    stopifnot(all(vapply(answer,function(x) is.numeric(x) && x==po_repetitions*length(po_methods),logical(1))))
    stopifnot(identical(readRDS(manifest),identity))
    po_summarize(root,identity,audit.only=FALSE)
} else if(action%in%c("summarize","audit")) {
    stopifnot(identical(readRDS(manifest),identity))
    po_summarize(root,identity,audit.only=action=="audit")
} else stop("unknown action")
