summarize.meanbr <- function(root,cells,repetitions) {
    paths <- file.path(root,"raw",sprintf("cell-%02d.rds",seq_len(nrow(cells))))
    stopifnot(all(file.exists(paths)))
    z <- do.call(rbind,lapply(seq_along(paths),function(i) {
        d <- readRDS(paths[i])
        stopifnot(nrow(d)==4L*repetitions,!anyDuplicated(d[c("replication","test","estimation")]),
            identical(sort(unique(d$replication)),seq_len(repetitions)),all(d$cell==i),
            all(d$scenario==cells$scenario[i]),all(d$n==cells$n[i]),
            all(d$seed==133611101L+10000L*i+d$replication),setequal(d$test,c("joint","x1")),
            setequal(d$estimation,c("ML","meanBR")),all(d$null==!startsWith(cells$scenario[i],"alternative_")),
            all(d$df==ifelse(d$test=="joint",d$p,1L)*(d$K-2L)),
            all(d$available==is.finite(d$statistic)),all(d$available==is.finite(d$p.value)),
            all(d$statistic[d$available]>=0),all(d$reason[d$available]=="available"),
            all(d$reason[!d$available]!="available"),all(is.finite(d$elapsed)&d$elapsed>=0),
            isTRUE(all.equal(d$p.value[d$available],pchisq(d$statistic[d$available],d$df[d$available],lower.tail=FALSE),tolerance=1e-14)))
        d
    }))
    groups <- split(z,interaction(z$cell,z$test,z$estimation,drop=TRUE))
    rates <- do.call(rbind,lapply(groups,function(g) do.call(rbind,lapply(c(.05,.01),function(level) {
        N <- nrow(g); A <- sum(g$available); k <- sum(g$p.value<level,na.rm=TRUE)
        a <- qnorm(.975)^2; center <- (k+a/2)/(A+a); radius <- sqrt(a*(k*(A-k)/A+a/4))/(A+a)
        data.frame(cell=g$cell[1],scenario=g$scenario[1],n=g$n[1],null=g$null[1],test=g$test[1],estimation=g$estimation[1],
            level,total=N,available=A,rejections=k,rate=if(A) k/A else NA_real_,lower=center-radius,upper=center+radius,
            decision.lower=k/N,decision.upper=(k+N-A)/N,mean.seconds=mean(g$elapsed))
    }))))
    failures <- aggregate(list(records=rep(1L,nrow(z))),z[c("cell","scenario","n","test","estimation","reason")],sum)
    paired <- do.call(rbind,lapply(split(z,interaction(z$cell,z$test,drop=TRUE)),function(g) {
        a <- g[g$estimation=="ML",]; b <- g[g$estimation=="meanBR",]; b <- b[match(a$replication,b$replication),]
        good <- a$available & b$available
        data.frame(cell=a$cell[1],scenario=a$scenario[1],n=a$n[1],test=a$test[1],shared=sum(good),
            ML=mean(a$p.value[good]<.05),meanBR=mean(b$p.value[good]<.05),
            meanBR.only=sum(!a$available & b$available),ML.only=sum(a$available & !b$available))
    }))
    for(name in c("rates","failures","paired")) write.csv(get(name),file.path(root,paste0(name,".csv")),row.names=FALSE)
    cat("Audited",nrow(cells),"cells and",nrow(z),"records;",sum(!z$available),"unavailable.\n")
    invisible(z)
}
