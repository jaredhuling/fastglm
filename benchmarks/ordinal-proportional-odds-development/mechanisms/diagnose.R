# Post-study numerical replay and category-count diagnosis; no fitted method changes.
root <- "benchmarks/ordinal-proportional-odds-development/mechanisms"
study.library <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(study.library)) .libPaths(c(study.library,.libPaths()))
RNGkind("Mersenne-Twister","Inversion","Rejection")
source(file.path(root,"algebra.R")); source(file.path(root,"config.R"))
rows <- list()
for(i in seq_len(nrow(mechanism.cells))) {
    design <- readRDS(file.path(root,sprintf("design-%02d.rds",i)))
    stopifnot(identical(design,po_fixed_design(mechanism.cells[i,])))
    z <- readRDS(file.path(root,"raw",sprintf("cell-%02d-bank-3.rds",i)))
    # Include the largest coefficient contrast and an empty-category dataset, when present.
    indices <- unique(c(1L,which.max(rowSums(z$delta^2)),head(which(!z$meta$all.observed),1L)))
    for(r in indices) {
        set.seed(z$meta$seed[r]); y <- 1L+rowSums(runif(nrow(design$x))>design$F)
        fit <- po_mechanism_fit(design,y)
        delta.error <- max(abs(fit$delta-z$delta[r,]))
        stat <- vapply(fit$V,function(V) po_statistic(fit$delta,V),numeric(1))
        ok <- is.finite(stat) & is.finite(z$statistic[r,])
        stat.error <- max(abs(stat[ok]-z$statistic[r,ok])/(1+abs(stat[ok])))
        stopifnot(delta.error<1e-9,stat.error<1e-9,identical(is.finite(stat),is.finite(z$statistic[r,])))
        rows[[length(rows)+1L]] <- data.frame(cell=i,rep=r,delta.error=delta.error,statistic.relative.error=stat.error)
    }
}
write.csv(do.call(rbind,rows),file.path(root,"replay-checks.csv"),row.names=FALSE)
z <- readRDS(file.path(root,"raw/cell-03-bank-3.rds"))
count <- as.integer(sub(",.*","",z$meta$counts))
by.count <- lapply(0:5,function(k) {
    at <- if(k==5) count>=k else count==k
    data.frame(rare.category.count=if(k==5) "5+" else as.character(k),n=sum(at),
        contrast.mean=mean(z$delta[at,1]),contrast.sd=sd(z$delta[at,1]),
        empirical.ordinary.fpr=mean(z$statistic[at,"empirical_ordinary"]>qchisq(.95,1)),
        empirical.adjusted.fpr=mean(z$statistic[at,"empirical_adjusted"]>qchisq(.95,1)))
})
write.csv(do.call(rbind,by.count),file.path(root,"endpoint-count-diagnostics.csv"),row.names=FALSE)
z <- readRDS(file.path(root,"raw/cell-06-bank-3.rds")); at <- !z$meta$all.observed
interior <- data.frame(empty.interior=sum(at),max.absolute.contrast=max(abs(z$delta[at,1])),
    min.empirical.variance=min(z$variance.first[at,"empirical_ordinary"]),
    max.empirical.variance=max(z$variance.first[at,"empirical_ordinary"]),
    numerical.factorizations=sum(is.finite(z$statistic[at,"empirical_ordinary"])))
stopifnot(interior$max.absolute.contrast==0,interior$max.empirical.variance<1e-25)
write.csv(interior,file.path(root,"empty-interior-diagnostics.csv"),row.names=FALSE)
cat("PASS: regenerated fixed designs and selected full-fit replays, including extremes and empty categories.\n")
cat("Empty-interior ordinary covariance is theoretically zero; tiny positive roundoff is not genuine positive definiteness.\n")
