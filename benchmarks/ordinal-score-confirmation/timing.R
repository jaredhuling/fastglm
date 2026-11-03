library.path <- Sys.getenv("FASTGLM_CONFIRM_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
library(fastglm)
root <- "benchmarks/ordinal-score-confirmation"
identity <- readRDS(file.path(root,"manifest.rds"))
stopifnot(identical(identity$source,tools::md5sum(names(identity$source))))
rows <- lapply(c(1000L,10000L,100000L),function(n) {
    set.seed(12190402L+n)
    x <- cbind(x1=runif(n,-1,1),x2=rnorm(n),x3=rbinom(n,1,.4)); offset <- .3*sin(x[,1])
    F <- plogis(outer(-drop(x%*%c(.6,-.4,.5))-offset,c(-1.4,-.3,.7,1.6),"+"))
    y <- ordered(1L+rowSums(runif(n)>F),levels=1:5)
    fit <- fastglm_ordinal(x,y,offset=offset,tol=1e-10,maxit=500)
    stopifnot(fit$converged)
    # Warm both public paths, then retain every paired timing repetition.
    invisible(fastglm_ordinal_assess(fit))
    stopifnot(fastglm_ordinal_assess(fit,po.test="score")$proportional.odds.test$available)
    do.call(rbind,lapply(seq_len(5),function(replication) {
        plain <- system.time(fastglm_ordinal_assess(fit))[["elapsed"]]
        score <- system.time(z <- fastglm_ordinal_assess(fit,po.test="score"))[["elapsed"]]
        stopifnot(z$proportional.odds.test$available)
        data.frame(n,p=3L,K=5L,replication,plain.seconds=plain,score.seconds=score,
            incremental.seconds=score-plain)
    }))
})
timing <- do.call(rbind,rows)
write.csv(timing,file.path(root,"timing.csv"),row.names=FALSE)
write.csv(aggregate(cbind(plain.seconds,score.seconds,incremental.seconds)~n+p+K,timing,median),
    file.path(root,"timing-summary.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(root,"timing-session.txt"))
