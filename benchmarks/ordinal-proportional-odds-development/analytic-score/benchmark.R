# Descriptive timing only: one generated design/response per size, three repetitions.
root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
library.path <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
source(file.path(root,"score.R")); RNGkind("Mersenne-Twister","Inversion","Rejection")
rows <- list()
for(n in c(10000L,100000L)) {
    set.seed(95361101L+n); x <- matrix(runif(n*3L,-1,1),n,3L)
    F <- plogis(outer(-drop(x%*%c(.6,-.5,.4)),c(-1.8,-.6,.6,1.8),"+"))
    y <- ordered(1L+rowSums(runif(n)>F),levels=1:5)
    for(firth in c(FALSE,TRUE)) for(rep in 1:3) {
        start <- proc.time()[3]
        fit <- fastglm::fastglm_ordinal(x,y,firth=firth,tol=1e-10,maxit=500L)
        fit.seconds <- proc.time()[3]-start; stopifnot(isTRUE(fit$converged))
        start <- proc.time()[3]
        test <- ordinal_score_statistic(ordinal_score_parts(x,as.integer(y),fit$alpha,fit$beta))
        score.seconds <- proc.time()[3]-start
        rows[[length(rows)+1L]] <- data.frame(n,firth,rep,fit.seconds=unname(fit.seconds),
            score.seconds=unname(score.seconds),total.seconds=unname(fit.seconds+score.seconds),statistic=test$statistic)
    }
    other <- ordinal_score_statistic(ordinal_score_parts(x,as.integer(y),fit$alpha,fit$beta,8192L))
    stopifnot(abs(other$statistic-test$statistic)<1e-8*(1+test$statistic))
}
write.csv(do.call(rbind,rows),file.path(root,"benchmark.csv"),row.names=FALSE)
cat("PASS: representative timing cases, converged null fits, and block-size agreement.\n")
