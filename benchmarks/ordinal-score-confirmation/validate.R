# Execute before freezing confirmation. No confirmation outcomes are used.
library.path <- Sys.getenv("FASTGLM_CONFIRM_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
library(fastglm)
source("benchmarks/ordinal-proportional-odds-development/analytic-score/score.R")
validation <- do.call(rbind,lapply(seq_len(20),function(i) {
    set.seed(11120400L+i)
    x <- cbind(x1=runif(120,-1,1),x2=rnorm(120),x3=rbinom(120,1,.5))
    eta <- outer(-drop(x%*%c(.5,-.3,.2)),c(-1.2,0,1.4),"+")
    y <- 1L+rowSums(runif(120)>plogis(eta))
    prototype <- ordinal_score_test(x,y,4)
    production <- fastglm_ordinal_assess(prototype$fit,po.test="score")$proportional.odds.test
    stopifnot(production$available,production$df==prototype$df,
        abs(production$statistic-prototype$statistic)<1e-7)
    data.frame(seed=11120400L+i,prototype=prototype$statistic,production=production$statistic,
        difference=production$statistic-prototype$statistic)
}))
write.csv(validation,"benchmarks/ordinal-score-confirmation/validation.csv",row.names=FALSE)
cat("Twenty independent production/prototype comparisons passed; maximum absolute difference:",
    max(abs(validation$difference)),"\n")
