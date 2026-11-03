lib <- Sys.getenv("FASTGLM_MEANBR_LIBRARY","")
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
library(fastglm)
source("benchmarks/ordinal-score-confirmation/config.R")
root <- "benchmarks/ordinal-score-meanbr-confirmation"
old <- readRDS("benchmarks/ordinal-score-confirmation/raw/cell-05.rds")
old <- old[old$test=="joint" & !old$available,]
set.seed(133600000L)
reference <- fastglm_ordinal(cbind(x1=runif(100),x2=rep(0:1,50)),ordered(rep(1:4,25)),firth=TRUE)
rows <- lapply(seq_len(nrow(old)),function(i) {
    set.seed(old$seed[i]); d <- confirmation.generate(old$scenario[i],old$n[i])
    ml <- fastglm_po_test(reference,newdata=d$x,y=d$y,weights=d$weights,offset=d$offset)
    br <- fastglm_po_test(reference,newdata=d$x,y=d$y,weights=d$weights,offset=d$offset,null.estimation="meanBR")
    rare <- as.integer(d$y)[d$x[,2]==1]
    certificate <- length(rare)>0 && (all(rare==1) || all(rare==d$K))
    data.frame(replication=old$replication[i],seed=old$seed[i],original.reason=old$reason[i],
        rare.frequency=length(rare),certificate,ML.reason=ml$reason,
        meanBR.available=br$available,meanBR.reason=br$reason,meanBR.statistic=br$statistic,
        meanBR.p.value=br$p.value,meanBR.iterations=br$null.iterations)
})
z <- do.call(rbind,rows)
stopifnot(nrow(z)==116L,all(z$certificate[z$original.reason=="original_fit_not_converged"]))
write.csv(z,file.path(root,"diagnosis.csv"),row.names=FALSE)
print(with(z,table(original.reason,ML.reason,meanBR.available)))
