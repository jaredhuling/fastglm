# Unprofiled paired timings. Run from the repository root with the frozen library.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L,dir.exists(args[1L]))
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
destination <- "benchmarks/ordinal-fit-profile/results"
output <- file.path(destination,"unprofiled-timing.csv")
if(file.exists(output)) stop("Timing records already exist")
capture.output(sessionInfo(),file=file.path(destination,"timing-session.txt"))
set.seed(111801)
records <- list()
for(n in c(100000L,1000000L)) {
    p <- if(n==100000L) 6L else 3L; K <- if(p==6L) 5L else 4L
    x <- matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    y <- ordered(cut(.4*x[,1]-.2*x[,2]+rlogis(n),c(-Inf,seq(-1.2,1.2,length.out=K-1L),Inf)),labels=seq_len(K))
    for(firth in c(FALSE,TRUE)) {
        prepared <- fastglm_ordinal_prepare(x,firth=firth)
        ordinary <- function() fastglm_ordinal(x,y,firth=firth)
        reuse <- function() fastglm_ordinal(prepared,y)
        a <- ordinary();b <- reuse()
        stopifnot(a$converged,b$converged,max(abs(coef(a)-coef(b)))<1e-9)
        for(repetition in 1:5) for(operation in if(repetition%%2) c("ordinary","reuse") else c("reuse","ordinary")) {
            gc()
            elapsed <- system.time(fit <- get(operation)())[["elapsed"]]
            records[[length(records)+1L]] <- data.frame(n,p,K,estimation=if(firth) "meanBR" else "ML",
                operation,repetition,seconds=elapsed,converged=fit$converged,iterations=fit$iterations)
        }
        write.csv(do.call(rbind,records),output,row.names=FALSE)
    }
}
