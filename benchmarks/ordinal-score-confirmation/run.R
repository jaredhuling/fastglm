# Rscript benchmarks/ordinal-score-confirmation/run.R pilot|run|audit
args <- commandArgs(trailingOnly=TRUE); action <- if(length(args)) args[1] else "audit"
root <- "benchmarks/ordinal-score-confirmation"
library.path <- Sys.getenv("FASTGLM_CONFIRM_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
library(fastglm)
RNGkind("Mersenne-Twister","Inversion","Rejection")
source(file.path(root,"config.R")); source(file.path(root,"summarize.R"))
frozen <- c(file.path(root,c("PROTOCOL.md","config.R","run.R","summarize.R","validate.R")),
    "tests/testthat/test-ordinal-po-test.R",
    "benchmarks/ordinal-proportional-odds-development/analytic-score/score.R")
source.files <- c("DESCRIPTION","NAMESPACE",list.files("R",pattern="\\.R$",full.names=TRUE),
    list.files("src",pattern="\\.(cpp|h|hpp)$",full.names=TRUE))
installed <- system.file(package="fastglm")
installed.files <- list.files(installed,recursive=TRUE,full.names=TRUE)
identity <- list(protocol=tools::md5sum(frozen),source=tools::md5sum(source.files),
    installed=setNames(unname(tools::md5sum(installed.files)),substring(installed.files,nchar(installed)+2L)),
    version=as.character(packageVersion("fastglm")),base.revision="b5e08af",
    R=R.version.string,RNG=RNGkind())
manifest <- file.path(root,"manifest.rds")
if(file.exists(manifest)) stopifnot(identical(identity,readRDS(manifest))) else {
    stopifnot(action=="pilot")
    saveRDS(identity,manifest,version=2); dput(identity,file=file.path(root,"manifest.txt"))
    snapshot <- setNames(lapply(source.files,function(path) readBin(path,"raw",file.info(path)$size)),source.files)
    saveRDS(snapshot,file.path(root,"source-snapshot.rds"),version=2)
    capture.output(sessionInfo(),file=file.path(root,"session.txt"))
}
one <- function(d) {
    t0 <- proc.time()[3]; warning <- character(); error <- ""
    fit <- tryCatch(withCallingHandlers(fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,
        tol=1e-10,maxit=500L),warning=function(w) {warning <<- c(warning,conditionMessage(w)); invokeRestart("muffleWarning")}),
        error=function(e) {error <<- conditionMessage(e); NULL})
    fit.seconds <- unname(proc.time()[3]-t0)
    original.ok <- !is.null(fit) && isTRUE(fit$converged)
    do.call(rbind,lapply(c("joint","x1"),function(test) {
        t0 <- proc.time()[3]; problem <- error
        result <- if(original.ok) tryCatch(fastglm_ordinal_assess(fit,po.test="score",
            po.terms=if(test=="x1") "x1" else NULL)$proportional.odds.test,
            error=function(e) {problem <<- conditionMessage(e); NULL}) else NULL
        available <- !is.null(result) && isTRUE(result$available)
        reason <- if(!is.null(result)) result$reason else if(any(colSums(d$counts)==0)) "empty_categories" else
            if(grepl("rank|constant column",problem)) "rank_deficient_design" else
            if(!original.ok && !is.null(fit)) "original_fit_not_converged" else "fit_error"
        data.frame(test,available,statistic=if(available) result$statistic else NA_real_,
            df=if(test=="joint") ncol(d$x)*(d$K-2L) else d$K-2L,
            p.value=if(available) result$p.value else NA_real_,
            condition=if(available) result$information.rcond else NA_real_,
            iterations=if(!is.null(result)) result$null.iterations else NA_integer_,
            elapsed=unname(proc.time()[3]-t0)+fit.seconds,reason,
            detail=if(!is.null(result)) result$detail else if(nzchar(problem)) problem else
                if(!is.null(fit)) fit$convergence.reason else "unknown fit error",
            warning=paste(unique(c(warning,if(!is.null(result)) result$warning)),collapse=" | "))
    }))
}
run.cell <- function(i,indices) {
    cell <- confirmation.cells[i,]
    do.call(rbind,lapply(indices,function(replication) {
        seed <- 110361101L+10000L*i+replication; set.seed(seed)
        d <- confirmation.generate(cell$scenario,cell$n); z <- one(d)
        z$cell <- i; z$scenario <- cell$scenario; z$n <- cell$n; z$replication <- replication
        z$seed <- seed; z$null <- d$null; z$p <- ncol(d$x); z$K <- d$K
        z$counts <- paste(colSums(d$counts),collapse=","); z
    }))
}
if(action=="pilot") {
    z <- do.call(rbind,lapply(seq_len(nrow(confirmation.cells)),run.cell,indices=9001:9002))
    saveRDS(z,file.path(root,"pilot.rds"),version=2)
    timing <- aggregate(elapsed~scenario+n+test,z,mean)
    write.csv(timing,file.path(root,"pilot-timing.csv"),row.names=FALSE)
    cat("Pilot records:",nrow(z),"; unavailable:",sum(!z$available),"\n")
    cat("Conservative compute seconds:",sum(timing$elapsed)*confirmation.repetitions,"\n")
} else if(action=="run") {
    dir.create(file.path(root,"raw"),showWarnings=FALSE)
    job <- function(i) {
        path <- file.path(root,"raw",sprintf("cell-%02d.rds",i))
        if(file.exists(path)) stop("Cell already exists; audit before deciding how to resume: ",path)
        z <- run.cell(i,seq_len(confirmation.repetitions)); saveRDS(z,paste0(path,".partial"),version=2)
        stopifnot(file.rename(paste0(path,".partial"),path))
        cat(sprintf("Completed cell %02d: %s n=%d\n",i,confirmation.cells$scenario[i],confirmation.cells$n[i])); TRUE
    }
    results <- parallel::mclapply(seq_len(nrow(confirmation.cells)),job,mc.cores=3L,mc.preschedule=FALSE,mc.set.seed=FALSE)
    stopifnot(all(vapply(results,isTRUE,logical(1))))
    confirmation.summarize(root,confirmation.cells,confirmation.repetitions)
} else if(action=="audit") {
    confirmation.summarize(root,confirmation.cells,confirmation.repetitions)
} else stop("unknown action")
stopifnot(identical(identity,readRDS(manifest)),identical(identity$source,tools::md5sum(source.files)),
    identical(identity$protocol,tools::md5sum(frozen)))
