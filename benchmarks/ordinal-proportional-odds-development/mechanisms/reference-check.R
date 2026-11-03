root <- "benchmarks/ordinal-proportional-odds-development/mechanisms"
study.library <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(study.library)) .libPaths(c(study.library,.libPaths()))
RNGkind("Mersenne-Twister","Inversion","Rejection")
for(f in c("algebra.R","config.R","banks.R","summarize.R")) source(file.path(root,f))
original <- readRDS(file.path(root,"manifest.rds"))
stopifnot(identical(unname(tools::md5sum(file.path(root,names(original$files)))),unname(original$files)))
identity <- c(original,list(reference.files=tools::md5sum(file.path(root,c("REFERENCE-CHECK.md","reference-check.R")))))
manifest <- file.path(root,"reference-manifest.rds")
if(file.exists(manifest)) stopifnot(identical(readRDS(manifest),identity)) else {
    saveRDS(identity,manifest,version=2); dput(identity,file=file.path(root,"reference-manifest.txt"))
}
job <- function(i) {
    path <- file.path(root,"raw",sprintf("cell-%02d-bank-4.rds",i))
    if(!file.exists(path)) {
        z <- po_run_bank(mechanism.cells[i,],readRDS(file.path(root,sprintf("design-%02d.rds",i))),4L,1:20000)
        saveRDS(z,paste0(path,".partial"),version=2); stopifnot(file.rename(paste0(path,".partial"),path))
    }
    TRUE
}
stopifnot(all(vapply(parallel::mclapply(c(3L,8L),job,mc.cores=2L,mc.set.seed=FALSE),isTRUE,logical(1))))
rows <- list()
for(i in c(3L,8L)) {
    cell <- mechanism.cells[i,]
    banks <- lapply(1:4,function(b) readRDS(file.path(root,"raw",sprintf("cell-%02d-bank-%d.rds",i,b))))
    extra <- banks[[4]]
    stopifnot(nrow(extra$delta)==20000L,extra$bank==4L,identical(extra$meta$rep,1:20000),
        all(extra$meta$seed==40361100L+100000L*i+40000L+(1:20000)),all(extra$meta$fit.ok))
    for(population in c("all","observed")) {
        select <- function(z) z$meta$fit.ok & (population=="all" | z$meta$all.observed)
        C <- cov(banks[[1]]$delta[select(banks[[1]]),,drop=FALSE])
        Te <- apply(banks[[3]]$delta[select(banks[[3]]),,drop=FALSE],1,po_statistic,V=C)
        for(b in c(2L,4L)) {
            Tr <- apply(banks[[b]]$delta[select(banks[[b]]),,drop=FALSE],1,po_statistic,V=C)
            p <- vapply(Te,function(t) (1+sum(Tr>=t))/(1+length(Tr)),numeric(1))
            for(level in c(.05,.01)) {
                k <- sum(p<=level); ci <- po_wilson(k,length(p)); rank <- floor(level*(length(Tr)+1))
                rows[[length(rows)+1L]] <- data.frame(cell,population,reference.bank=b,reference.n=length(Tr),
                    evaluation.n=length(Te),level,critical.value=sort(Tr)[length(Tr)-rank+1L],
                    rejection=k,rate=k/length(p),mc.lower=ci[1],mc.upper=ci[2],
                    reference.tail.sd.approx=sqrt(level*(1-level)/(length(Tr)+2)))
            }
        }
    }
}
write.csv(do.call(rbind,rows),file.path(root,"reference-check.csv"),row.names=FALSE)
cat("PASS: additional reference-bank identities, seeds, dimensions, binary fits, and unchanged main sources.\n")
