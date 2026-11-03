confirmation.summarize <- function(root,cells,repetitions) {
    paths <- file.path(root,"raw",sprintf("cell-%02d.rds",seq_len(nrow(cells))))
    stopifnot(all(file.exists(paths)))
    records <- lapply(seq_along(paths),function(i) {
        z <- readRDS(paths[i]); cell <- cells[i,]
        stopifnot(nrow(z)==2L*repetitions,!anyDuplicated(z[c("replication","test")]),
            identical(sort(unique(z$replication)),seq_len(repetitions)),
            setequal(z$test,c("joint","x1")),all(z$cell==i),all(z$scenario==cell$scenario),
            all(z$n==cell$n),all(z$seed==110361101L+10000L*i+z$replication),
            all(z$null==!startsWith(cell$scenario,"alternative_")),all(z$df>0),
            all(z$df==ifelse(z$test=="joint",z$p,1L)*(z$K-2L)),
            all(z$available==is.finite(z$p.value)),all(z$available==is.finite(z$statistic)),
            all(z$statistic[z$available]>=0),all(z$p.value[z$available]>=0 & z$p.value[z$available]<=1),
            all(z$reason[z$available]=="available"),all(z$reason[!z$available]!="available"),
            all(z$elapsed>=0),all(is.finite(z$elapsed)))
        stopifnot(isTRUE(all.equal(z$p.value[z$available],
            pchisq(z$statistic[z$available],z$df[z$available],lower.tail=FALSE),tolerance=1e-14)))
        z
    })
    z <- do.call(rbind,records)
    wilson <- function(k,n) {
        if(!n) return(c(NA_real_,NA_real_))
        a <- qnorm(.975)^2; center <- (k+a/2)/(n+a)
        radius <- sqrt(a*(k*(n-k)/n+a/4))/(n+a)
        c(center-radius,center+radius)
    }
    groups <- split(z,interaction(z$cell,z$test,drop=TRUE))
    rates <- do.call(rbind,lapply(groups,function(g) do.call(rbind,lapply(c(.05,.01),function(level) {
        N <- nrow(g); A <- sum(g$available); k <- sum(g$p.value<level,na.rm=TRUE); ci <- wilson(k,A)
        data.frame(cell=g$cell[1],scenario=g$scenario[1],n=g$n[1],null=g$null[1],test=g$test[1],
            df=g$df[1],level,total=N,available=A,rejections=k,rate=if(A) k/A else NA_real_,
            lower=ci[1],upper=ci[2],availability=A/N,decision.lower=k/N,decision.upper=(k+N-A)/N)
    }))))
    statistics <- do.call(rbind,lapply(groups,function(g) {
        q <- quantile(g$statistic,c(.5,.9,.95,.99),na.rm=TRUE,names=FALSE)
        data.frame(cell=g$cell[1],scenario=g$scenario[1],n=g$n[1],test=g$test[1],
            median=q[1],q90=q[2],q95=q[3],q99=q[4],mean.seconds=mean(g$elapsed))
    }))
    failures <- aggregate(list(records=rep(1L,nrow(z))),z[c("cell","scenario","n","test","reason")],sum)
    warnings <- z[nzchar(z$warning),c("cell","scenario","n","replication","test","warning")]
    for(name in c("rates","statistics","failures","warnings"))
        write.csv(get(name),file.path(root,paste0(name,".csv")),row.names=FALSE,na="NA")
    cat("Audited",nrow(cells),"cells;",nrow(z),"records;",sum(!z$available),"unavailable tests.\n")
    invisible(z)
}
