# run generates missing stress cells; summarize/audit only inspect saved cells.
action <- commandArgs(trailingOnly=TRUE); action <- if(length(action)) action[1] else "audit"
root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
library.path <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
RNGkind("Mersenne-Twister","Inversion","Rejection")
source(file.path(root,"score.R")); source(file.path(root,"../prototype.R")); source(file.path(root,"summarize.R"))
original <- readRDS(file.path(root,"manifest.rds"))
stopifnot(identical(unname(tools::md5sum(file.path(root,names(original$files)))),unname(original$files)))
installed <- system.file(package="fastglm"); paths <- file.path(installed,names(original$installed))
stopifnot(identical(unname(tools::md5sum(paths)),unname(original$installed)))
identity <- list(original=original,files=tools::md5sum(file.path(root,c("STRESS-PROTOCOL.md","stress.R"))))
manifest <- file.path(root,"stress-manifest.rds")
if(file.exists(manifest)) stopifnot(identical(identity,readRDS(manifest))) else {
    stopifnot(action=="run"); saveRDS(identity,manifest,version=2); dput(identity,file=file.path(root,"stress-manifest.txt"))
}
cells <- expand.grid(n=c(80L,250L),scenario=c("normal5","many_predictors5","uncommon_binary5","leverage3"),stringsAsFactors=FALSE)
cells$cell <- seq_len(nrow(cells)); methods <- c("binary_meanBR_Wald","score_ML","score_meanBR")
generate <- function(cell) {
    n <- cell$n; scenario <- cell$scenario; alpha <- c(-1.8,-.6,.6,1.8)
    if(scenario=="normal5") {x <- matrix(rnorm(n*4L),n,4); beta <- c(.8,-.6,.4,-.3)}
    else if(scenario=="many_predictors5") {x <- matrix(rnorm(n*8L),n,8); beta <- c(.5,-.4,.3,-.2,0,0,0,0)}
    else if(scenario=="uncommon_binary5") {
        x <- cbind(rnorm(n),rbinom(n,1,.05),rbinom(n,1,.2),runif(n,-1,1)); beta <- c(.6,.8,-.5,.4)
    } else {x <- cbind(runif(n,-1,1)+15*rbinom(n,1,.03),runif(n,-1,1)); beta <- c(.25,-.4); alpha <- c(-.8,.8)}
    colnames(x) <- paste0("x",seq_len(ncol(x)))
    F <- plogis(outer(-drop(x%*%beta),alpha,"+")); y <- 1L+rowSums(runif(n)>F)
    counts <- matrix(0,n,length(alpha)+1L); counts[cbind(seq_len(n),y)] <- 1
    list(x=x,y=y,counts=counts)
}
one <- function(method,d) {
    warning <- character(); error <- ""; t0 <- proc.time()[3]
    fit <- tryCatch(withCallingHandlers({
        if(method==methods[1]) ordinal_po_covariance(d$x,d$counts,firth=TRUE)
        else ordinal_score_test(d$x,d$y,ncol(d$counts),firth=method=="score_meanBR")
    },warning=function(w) {warning <<- c(warning,conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) {error <<- conditionMessage(e); NULL})
    df <- ncol(d$x)*(ncol(d$counts)-2L)
    ok <- !is.null(fit) && is.finite(fit$statistic) && fit$statistic>=0 && fit$df==df
    if(!ok && !nzchar(error)) error <- "invalid_statistic"
    reason <- if(ok) "available" else if(any(colSums(d$counts)==0)) "empty_category" else
        if(grepl("rank|constant|full.rank",error)) "design_rank" else
        if(grepl("converged|convergence",error)) "fit_not_converged" else "other_error"
    data.frame(method,available=ok,statistic=if(ok) fit$statistic else NA_real_,df,
        p.value=if(ok) pchisq(fit$statistic,df,lower.tail=FALSE) else NA_real_,
        elapsed=unname(proc.time()[3]-t0),reason,error,warning=paste(unique(warning),collapse=" | "))
}
job <- function(i) {
    path <- file.path(root,"stress-raw",sprintf("cell-%02d.rds",i)); if(file.exists(path)) return(TRUE)
    z <- do.call(rbind,lapply(1:1000,function(rep) {
        seed <- 90361101L+10000L*i+rep; set.seed(seed); d <- generate(cells[i,])
        z <- do.call(rbind,lapply(methods,one,d=d)); z$cell <- i; z$replication <- rep
        z$seed <- seed; z$counts <- paste(colSums(d$counts),collapse=","); z$min.category <- min(colSums(d$counts))
        z$design.rank <- qr(cbind(1,d$x))$rank; z$design.columns <- ncol(d$x)+1L; z
    }))
    saveRDS(z,paste0(path,".partial"),version=2); stopifnot(file.rename(paste0(path,".partial"),path))
    cat(sprintf("Completed stress cell %d: %s n=%d\n",i,cells$scenario[i],cells$n[i])); TRUE
}
if(action=="run") {
    dir.create(file.path(root,"stress-raw"),showWarnings=FALSE)
    z <- parallel::mclapply(seq_len(nrow(cells)),job,mc.cores=3L,mc.preschedule=FALSE,mc.set.seed=FALSE)
    stopifnot(all(vapply(z,isTRUE,logical(1))))
} else stopifnot(action %in% c("audit","summarize"))
rows <- paired <- failures <- list()
for(i in seq_len(nrow(cells))) {
    d <- readRDS(file.path(root,"stress-raw",sprintf("cell-%02d.rds",i)))
    stopifnot(nrow(d)==3000L,all(d$cell==i),!anyDuplicated(paste(d$replication,d$method)),
        all(d$seed==90361101L+10000L*i+d$replication),all(d$available==is.finite(d$p.value)))
    for(method in methods) {
        z <- d[d$method==method,]; stopifnot(identical(z$replication,1:1000),all(!z$available | z$min.category>0))
        count <- lapply(strsplit(z$counts,",",fixed=TRUE),as.numeric)
        stopifnot(all(vapply(count,sum,numeric(1))==cells$n[i]),all(vapply(count,min,numeric(1))==z$min.category))
        n <- sum(z$available)
        for(level in c(.05,.01)) {
            k <- sum(z$p.value<=level,na.rm=TRUE); ci <- wilson(k,n)
            rows[[length(rows)+1L]] <- data.frame(cells[i,],method,level,available=n,attempted=1000L,
                rejected=k,rate=k/n,mc.lower=ci[1],mc.upper=ci[2],unconditional.lower=k/1000,
                unconditional.upper=(k+1000-n)/1000,median.ms=1000*median(z$elapsed[z$available]))
        }
        f <- as.data.frame(table(z$reason),stringsAsFactors=FALSE); names(f) <- c("reason","count")
        failures[[length(failures)+1L]] <- cbind(cells[rep(i,nrow(f)),],method,f)
    }
    for(a in 2:3) for(b in seq_len(a-1L)) for(level in c(.05,.01)) {
        za <- d[d$method==methods[a],]; zb <- d[d$method==methods[b],]; ok <- za$available & zb$available
        diff <- as.numeric(za$p.value[ok]<=level)-as.numeric(zb$p.value[ok]<=level)
        paired[[length(paired)+1L]] <- data.frame(cells[i,],method=methods[a],baseline=methods[b],level,
            shared=sum(ok),difference=mean(diff),mcse=sd(diff)/sqrt(length(diff)))
    }
}
capture.output(cat("PASS: 8000 stress datasets, 24000 method records; source/install identity, cell keys, seed mapping and category totals.\n"),file=file.path(root,"stress-audit.txt"))
if(action!="audit") {
    write.csv(do.call(rbind,rows),file.path(root,"stress-summary.csv"),row.names=FALSE)
    write.csv(do.call(rbind,paired),file.path(root,"stress-paired.csv"),row.names=FALSE)
    write.csv(do.call(rbind,failures),file.path(root,"stress-failures.csv"),row.names=FALSE)
}
