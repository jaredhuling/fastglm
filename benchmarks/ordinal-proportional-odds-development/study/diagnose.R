# Post-study implementation diagnosis; does not alter the frozen comparison.
root <- "benchmarks/ordinal-proportional-odds-development/study"
study.library <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(study.library)) .libPaths(c(study.library,.libPaths()))
source(file.path(root,"config.R")); source(file.path(root,"../prototype.R"))
RNGkind("Mersenne-Twister","Inversion","Rejection")
checks <- list(); k <- 0L
# First five available meanBR datasets in representative regular/sparse cells.
# Selection is diagnostic and independent of the magnitude of the test statistic.
for(cell in c(1L,4L,10L,11L,13L,19L)) {
    raw <- readRDS(file.path(root,"raw",sprintf("cell-%02d.rds",cell)))
    chosen <- head(raw[raw$method=="meanBR" & raw$available,],5L)
    for(r in seq_len(nrow(chosen))) {
        saved <- chosen[r,]; set.seed(saved$seed)
        d <- po_generate(saved$scenario,saved$n)
        fit <- ordinal_po_covariance(d$x,d$counts,firth=TRUE)
        x <- cbind(1,d$x); p <- ncol(x); q <- ncol(d$counts)-1L
        scores <- function(theta) do.call(cbind,lapply(seq_len(q),function(j) {
            index <- (j-1L)*p+seq_len(p)
            x*(as.numeric(d$y>j)-plogis(drop(x%*%theta[index])))
        }))
        theta <- as.vector(fit$coefficients)
        individual <- scores(theta)
        A <- -numDeriv::jacobian(function(b) colSums(scores(b)),theta)
        direct <- solve(A)%*%crossprod(individual)%*%t(solve(A))
        error <- max(abs(direct-fit$covariance))/(1+max(abs(direct)))
        replay <- abs(saved$statistic-fit$statistic)/(1+abs(fit$statistic))
        stopifnot(error<1e-7,replay<1e-12)
        k <- k+1L
        checks[[k]] <- data.frame(cell=cell,replication=saved$replication,
            covariance.relative.error=error,statistic.replay.error=replay)
    }
}
write.csv(do.call(rbind,checks),file.path(root,"implementation-checks.csv"),row.names=FALSE)
raw <- do.call(rbind,lapply(seq_len(nrow(po_cells)),function(i)
    readRDS(file.path(root,"raw",sprintf("cell-%02d.rds",i)))))
car <- raw[raw$method=="car",]
strata <- do.call(rbind,lapply(split(car,interaction(car$cell,car$empty.categories>0,drop=TRUE)),function(z) {
    ok <- z$available
    data.frame(scenario=z$scenario[1],n=z$n[1],empty=z$empty.categories[1]>0,
        datasets=nrow(z),available=sum(ok),rejected=sum(z$p.value[ok]<.05),warnings=sum(nzchar(z$warning)))
}))
write.csv(strata,file.path(root,"comparator-support-diagnostics.csv"),row.names=FALSE)
cat("PASS: 30 independent numerical-Jacobian covariance checks and deterministic statistic replays\n")
print(sapply(do.call(rbind,checks)[,3:4],max))
