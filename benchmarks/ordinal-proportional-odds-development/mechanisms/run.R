# Rscript benchmarks/ordinal-proportional-odds-development/mechanisms/run.R pilot|run|audit|summarize
args <- commandArgs(trailingOnly=TRUE); action <- if(length(args)) args[1] else "audit"
root <- "benchmarks/ordinal-proportional-odds-development/mechanisms"
study.library <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(study.library)) .libPaths(c(study.library,.libPaths()))
RNGkind("Mersenne-Twister","Inversion","Rejection")
for(f in c("algebra.R","config.R","banks.R","summarize.R")) source(file.path(root,f))
files <- c("PROTOCOL.md","THEORY.md","algebra.R","config.R","banks.R","run.R","summarize.R","validate.R")
installed <- system.file(package="fastglm")
installed.files <- list.files(installed,recursive=TRUE,full.names=TRUE)
identity <- list(files=setNames(unname(tools::md5sum(file.path(root,files))),files),
    installed=setNames(unname(tools::md5sum(installed.files)),substring(installed.files,nchar(installed)+2L)),
    packages=vapply(c("fastglm","numDeriv"),function(p) as.character(packageVersion(p)),character(1)),
    candidate.revision="b5e08af",R=R.version.string,RNG=RNGkind())
manifest <- file.path(root,"manifest.rds")
if(action%in%c("pilot","run")) {
    if(file.exists(manifest)) stopifnot(identical(readRDS(manifest),identity))
    else {
        saveRDS(identity,manifest,version=2); dput(identity,file=file.path(root,"manifest.txt"))
        capture.output(sessionInfo(),file=file.path(root,"session.txt"))
        for(i in seq_len(nrow(mechanism.cells))) saveRDS(po_fixed_design(mechanism.cells[i,]),
            file.path(root,sprintf("design-%02d.rds",i)),version=2)
    }
}
if(action=="pilot") {
    z <- lapply(seq_len(nrow(mechanism.cells)),function(i)
        po_run_bank(mechanism.cells[i,],readRDS(file.path(root,sprintf("design-%02d.rds",i))),1L,9001:9002))
    saveRDS(z,file.path(root,"pilot.rds"),version=2)
    timings <- do.call(rbind,lapply(seq_along(z),function(i) data.frame(mechanism.cells[i,],
        seconds.per.dataset=mean(z[[i]]$meta$elapsed),binary.failed=sum(!z[[i]]$meta$fit.ok))))
    write.csv(timings,file.path(root,"pilot-timings.csv"),row.names=FALSE); print(timings)
    cat("Estimated compute seconds:",sum(timings$seconds.per.dataset)*sum(mechanism.banks),"\n")
} else if(action=="run") {
    dir.create(file.path(root,"raw"),showWarnings=FALSE)
    job <- function(i) {
        design <- readRDS(file.path(root,sprintf("design-%02d.rds",i)))
        for(b in seq_along(mechanism.banks)) {
            path <- file.path(root,"raw",sprintf("cell-%02d-bank-%d.rds",i,b))
            if(file.exists(path)) next
            start <- proc.time()[3]
            z <- po_run_bank(mechanism.cells[i,],design,b,seq_len(mechanism.banks[b]))
            saveRDS(z,paste0(path,".partial"),version=2)
            stopifnot(file.rename(paste0(path,".partial"),path))
            cat(sprintf("Completed cell %02d (%s n=%d), %s bank: %.1f seconds\n",i,
                mechanism.cells$scenario[i],mechanism.cells$n[i],names(mechanism.banks)[b],proc.time()[3]-start))
        }
        TRUE
    }
    workers <- as.integer(Sys.getenv("FASTGLM_STUDY_WORKERS","3")); stopifnot(is.finite(workers),workers>=1L)
    results <- if(.Platform$OS.type=="unix" && workers>1L)
        parallel::mclapply(seq_len(nrow(mechanism.cells)),job,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
        else lapply(seq_len(nrow(mechanism.cells)),job)
    stopifnot(all(vapply(results,isTRUE,logical(1))),identical(readRDS(manifest),identity))
    po_summarize_mechanisms(root)
} else if(action%in%c("audit","summarize")) {
    stopifnot(identical(readRDS(manifest),identity))
    po_summarize_mechanisms(root,audit.only=action=="audit")
} else stop("unknown action")
