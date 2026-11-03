args <- commandArgs(TRUE); action <- if(length(args)) args[1] else "audit"
root <- "benchmarks/ordinal-score-meanbr-confirmation"
lib <- Sys.getenv("FASTGLM_MEANBR_LIBRARY","")
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
library(fastglm)
RNGkind("Mersenne-Twister","Inversion","Rejection")
generator <- "benchmarks/ordinal-score-confirmation/config.R"
source(generator); source(file.path(root,"summarize.R"))
cells <- expand.grid(scenario=confirmation.scenarios,n=c(80L,300L,1200L),
    KEEP.OUT.ATTRS=FALSE,stringsAsFactors=FALSE)
repetitions <- 500L
sources <- c("DESCRIPTION","NAMESPACE",list.files("R",pattern="\\.R$",full.names=TRUE),
    list.files("src",pattern="\\.(cpp|h|hpp)$",full.names=TRUE))
protocol <- c(file.path(root,c("PROTOCOL.md","run.R","summarize.R")),generator,"tests/testthat/test-result-display.R")
installed <- system.file(package="fastglm"); installed.files <- list.files(installed,recursive=TRUE,full.names=TRUE)
identity <- list(source=tools::md5sum(sources),protocol=tools::md5sum(protocol),
    installed=setNames(unname(tools::md5sum(installed.files)),substring(installed.files,nchar(installed)+2L)),
    R=R.version.string,version=as.character(packageVersion("fastglm")),RNG=RNGkind())
manifest <- file.path(root,"manifest.rds")
if(file.exists(manifest)) stopifnot(identical(identity,readRDS(manifest))) else {
    stopifnot(action=="pilot")
    saveRDS(identity,manifest,version=2); dput(identity,file=file.path(root,"manifest.txt"))
    saveRDS(setNames(lapply(sources,function(p) readBin(p,"raw",file.info(p)$size)),sources),
        file.path(root,"source-snapshot.rds"),version=2)
    capture.output(sessionInfo(),file=file.path(root,"session.txt"))
}
reference <- function(p,K) {
    x <- outer(seq_len(20),seq_len(p),function(i,j) sin(i*j))
    colnames(x) <- paste0("x",seq_len(p))
    counts <- matrix(2,20,K,dimnames=list(NULL,as.character(seq_len(K))))
    fastglm_ordinal(x,counts,firth=TRUE)
}
run.cell <- function(i,indices) {
    cell <- cells[i,]; K <- if(cell$scenario=="correlated6") 6L else 4L
    p <- if(K==6L) 4L else 2L; template <- reference(p,K)
    do.call(rbind,lapply(indices,function(replication) {
        seed <- 133611101L+10000L*i+replication; set.seed(seed)
        d <- confirmation.generate(cell$scenario,cell$n)
        do.call(rbind,lapply(c("ML","meanBR"),function(estimation) do.call(rbind,lapply(c("joint","x1"),function(test) {
            t0 <- proc.time()[3]; error <- ""
            out <- tryCatch(fastglm_po_test(template,newdata=d$x,y=d$y,weights=d$weights,offset=d$offset,
                terms=if(test=="x1") "x1" else NULL,null.estimation=estimation),
                error=function(e) {error <<- conditionMessage(e); NULL})
            available <- !is.null(out) && out$available
            data.frame(cell=i,scenario=cell$scenario,n=cell$n,replication,seed,null=d$null,p,K,
                test,estimation,available,statistic=if(available) out$statistic else NA_real_,
                df=if(test=="joint") p*(K-2L) else K-2L,p.value=if(available) out$p.value else NA_real_,
                reason=if(is.null(out)) "error" else out$reason,detail=if(is.null(out)) error else out$detail,
                separated=!is.null(out$separation),condition=if(available) out$information.rcond else NA_real_,
                iterations=if(is.null(out)) NA_integer_ else out$null.iterations,
                warning=paste(out$warning,collapse=" | "),elapsed=unname(proc.time()[3]-t0),
                counts=paste(colSums(d$counts),collapse=","))
        })))))
    }))
}
if(action=="pilot") {
    z <- do.call(rbind,lapply(seq_len(nrow(cells)),run.cell,indices=9001:9002))
    saveRDS(z,file.path(root,"pilot.rds"),version=2)
    cat("Pilot records:",nrow(z),"; unavailable:",sum(!z$available),"\n")
    cat("Estimated single-process compute seconds:",sum(z$elapsed)*repetitions/2,"\n")
} else if(action=="run") {
    dir.create(file.path(root,"raw"),showWarnings=FALSE)
    job <- function(i) {
        path <- file.path(root,"raw",sprintf("cell-%02d.rds",i)); stopifnot(!file.exists(path))
        z <- run.cell(i,seq_len(repetitions)); saveRDS(z,paste0(path,".partial"),version=2)
        stopifnot(file.rename(paste0(path,".partial"),path))
        cat("Completed cell",i,"\n"); TRUE
    }
    result <- parallel::mclapply(seq_len(nrow(cells)),job,mc.cores=3L,mc.preschedule=FALSE,mc.set.seed=FALSE)
    stopifnot(all(vapply(result,isTRUE,logical(1))))
    summarize.meanbr(root,cells,repetitions)
} else if(action=="audit") summarize.meanbr(root,cells,repetitions) else stop("unknown action")
stopifnot(identical(identity,readRDS(manifest)),identical(identity$source,tools::md5sum(sources)),
    identical(identity$protocol,tools::md5sum(protocol)))
