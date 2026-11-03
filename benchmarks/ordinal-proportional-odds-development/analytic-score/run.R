# Rscript benchmarks/ordinal-proportional-odds-development/analytic-score/run.R pilot|run|summarize|audit
args <- commandArgs(trailingOnly=TRUE); action <- if(length(args)) args[1] else "audit"
root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
library.path <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
RNGkind("Mersenne-Twister","Inversion","Rejection")
source(file.path(root,"../study/config.R")); source(file.path(root,"../prototype.R"))
source(file.path(root,"score.R")); source(file.path(root,"summarize.R"))
methods <- c("binary_meanBR_Wald","score_ML","score_meanBR")
frozen <- c("METHODS.md","PROTOCOL.md","score.R","validate.R","run.R","summarize.R","../study/config.R","../prototype.R")
installed <- system.file(package="fastglm"); installed.files <- list.files(installed,recursive=TRUE,full.names=TRUE)
identity <- list(files=setNames(unname(tools::md5sum(file.path(root,frozen))),frozen),
    installed=setNames(unname(tools::md5sum(installed.files)),substring(installed.files,nchar(installed)+2L)),
    packages=vapply(c("fastglm","numDeriv"),function(p) as.character(packageVersion(p)),character(1)),
    revision="b5e08af",R=R.version.string,RNG=RNGkind())
manifest <- file.path(root,"manifest.rds")
if(action %in% c("pilot","run")) {
    if(file.exists(manifest)) stopifnot(identical(identity,readRDS(manifest))) else {
        saveRDS(identity,manifest,version=2); dput(identity,file=file.path(root,"manifest.txt"))
        capture.output(sessionInfo(),file=file.path(root,"session.txt"))
    }
} else stopifnot(identical(identity,readRDS(manifest)))
one <- function(method,d) {
    warning <- character(); error <- ""; t0 <- proc.time()[3]
    fit <- tryCatch(withCallingHandlers({
        if(method=="binary_meanBR_Wald") {
            z <- ordinal_po_covariance(d$x,d$counts,firth=TRUE)
            list(statistic=z$statistic,df=z$df,condition=rcond(z$difference.covariance),
                 nuisance.statistic=NA_real_,iterations=NA_integer_)
        } else {
            z <- ordinal_score_test(d$x,d$y,ncol(d$counts),firth=method=="score_meanBR")
            list(statistic=z$statistic,df=z$df,condition=z$condition,
                 nuisance.statistic=z$nuisance.statistic,iterations=z$fit$iterations)
        }
    },warning=function(w) {warning <<- c(warning,conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) {error <<- conditionMessage(e); NULL})
    elapsed <- proc.time()[3]-t0; df <- ncol(d$x)*(ncol(d$counts)-2L)
    available <- !is.null(fit) && is.finite(fit$statistic) && fit$statistic>=0 && fit$df==df
    if(!available && !nzchar(error)) error <- "invalid_statistic"
    reason <- if(available) "available" else if(any(colSums(d$counts)==0)) "empty_category" else
        if(grepl("converged|convergence",error)) "fit_not_converged" else
        if(grepl("positive definite|singular|chol",error)) "information_factorization" else "other_error"
    data.frame(method,available,statistic=if(available) fit$statistic else NA_real_,df,
        p.value=if(available) pchisq(fit$statistic,df,lower.tail=FALSE) else NA_real_,
        condition=if(available) fit$condition else NA_real_,
        nuisance.statistic=if(available) fit$nuisance.statistic else NA_real_,
        iterations=if(available) fit$iterations else NA_integer_,elapsed=unname(elapsed),reason,error,
        warning=paste(unique(warning),collapse=" | "))
}
run.cell <- function(i,indices) {
    cell <- po_cells[i,]
    do.call(rbind,lapply(indices,function(rep) {
        seed <- 80361101L+10000L*i+rep; set.seed(seed); d <- po_generate(cell$scenario,cell$n)
        z <- do.call(rbind,lapply(methods,one,d=d)); z$cell <- i; z$scenario <- cell$scenario
        z$n <- cell$n; z$replication <- rep; z$seed <- seed; z$null <- d$null
        z$counts <- paste(colSums(d$counts),collapse=","); z$min.category <- min(colSums(d$counts)); z
    }))
}
if(action=="pilot") {
    z <- do.call(rbind,lapply(seq_len(nrow(po_cells)),run.cell,indices=9001:9002))
    saveRDS(z,file.path(root,"pilot.rds"),version=2)
    timing <- aggregate(elapsed~scenario+n+method,z,mean)
    write.csv(timing,file.path(root,"pilot-timing.csv"),row.names=FALSE)
    cat("Pilot records:",nrow(z),"; unavailable:",sum(!z$available),"\n")
    cat("Estimated compute seconds:",sum(timing$elapsed)*po_repetitions,"\n")
} else if(action=="run") {
    dir.create(file.path(root,"raw"),showWarnings=FALSE)
    job <- function(i) {
        path <- file.path(root,"raw",sprintf("cell-%02d.rds",i)); if(file.exists(path)) return(TRUE)
        z <- run.cell(i,seq_len(po_repetitions)); saveRDS(z,paste0(path,".partial"),version=2)
        stopifnot(file.rename(paste0(path,".partial"),path))
        cat(sprintf("Completed cell %02d: %s n=%d\n",i,po_cells$scenario[i],po_cells$n[i])); TRUE
    }
    results <- parallel::mclapply(seq_len(nrow(po_cells)),job,mc.cores=3L,mc.preschedule=FALSE,mc.set.seed=FALSE)
    stopifnot(all(vapply(results,isTRUE,logical(1))),identical(identity,readRDS(manifest)))
    summarize.analytic(root,po_cells,methods,po_repetitions)
} else if(action %in% c("audit","summarize")) {
    summarize.analytic(root,po_cells,methods,po_repetitions,audit.only=action=="audit")
} else stop("unknown action")
