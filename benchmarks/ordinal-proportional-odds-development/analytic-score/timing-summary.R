# Paired timing ratios from the frozen main records; no additional model fits.
root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
rows <- list()
for(path in sort(list.files(file.path(root,"raw"),full.names=TRUE,pattern="[.]rds$"))) {
    z <- readRDS(path); base <- z[z$method=="binary_meanBR_Wald",]
    for(method in c("score_ML","score_meanBR")) {
        a <- z[z$method==method,]; stopifnot(identical(a$replication,base$replication))
        ok <- a$available & base$available; positive <- ok & base$elapsed>0
        rows[[length(rows)+1L]] <- data.frame(cell=a$cell[1],scenario=a$scenario[1],n=a$n[1],method,
            shared=sum(ok),positive.denominator=sum(positive),
            paired.total.time.ratio=sum(a$elapsed[ok])/sum(base$elapsed[ok]),
            median.individual.ratio=median(a$elapsed[positive]/base$elapsed[positive]))
    }
}
write.csv(do.call(rbind,rows),file.path(root,"timing-pairs.csv"),row.names=FALSE)
