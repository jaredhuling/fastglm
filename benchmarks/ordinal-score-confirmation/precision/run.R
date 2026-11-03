# No command-line arguments: the sourced runner first audits the original study.
stopifnot(!length(commandArgs(trailingOnly=TRUE)))
source("benchmarks/ordinal-score-confirmation/run.R")
precision.root <- file.path(root,"precision")
precision.cells <- c(21L,22L); precision.repetitions <- 5000L
precision.files <- file.path(precision.root,c("PROTOCOL.md","run.R"))
precision.identity <- list(parent=readRDS(manifest),files=tools::md5sum(precision.files),
    cells=precision.cells,repetitions=precision.repetitions)
precision.manifest <- file.path(precision.root,"manifest.rds")
if(file.exists(precision.manifest)) stopifnot(identical(precision.identity,readRDS(precision.manifest))) else {
    saveRDS(precision.identity,precision.manifest,version=2)
    dput(precision.identity,file=file.path(precision.root,"manifest.txt"))
}
dir.create(file.path(precision.root,"raw"),showWarnings=FALSE)
paths <- file.path(precision.root,"raw",sprintf("cell-%02d.rds",precision.cells))
if(any(!file.exists(paths))) {
    stopifnot(!any(file.exists(paths)))
    job <- function(i) {
        cell <- confirmation.cells[i,]
        z <- do.call(rbind,lapply(seq_len(precision.repetitions),function(replication) {
            seed <- 120361101L+10000L*i+replication; set.seed(seed)
            d <- confirmation.generate(cell$scenario,cell$n); z <- one(d)
            z$cell <- i; z$scenario <- cell$scenario; z$n <- cell$n; z$replication <- replication
            z$seed <- seed; z$null <- d$null; z$p <- ncol(d$x); z$K <- d$K
            z$counts <- paste(colSums(d$counts),collapse=","); z
        }))
        path <- file.path(precision.root,"raw",sprintf("cell-%02d.rds",i))
        saveRDS(z,paste0(path,".partial"),version=2); stopifnot(file.rename(paste0(path,".partial"),path))
        cat("Completed precision cell",i,"\n"); TRUE
    }
    ok <- parallel::mclapply(precision.cells,job,mc.cores=2L,mc.set.seed=FALSE)
    stopifnot(all(vapply(ok,isTRUE,logical(1))))
}
records <- lapply(seq_along(paths),function(j) {
    z <- readRDS(paths[j]); i <- precision.cells[j]
    stopifnot(nrow(z)==2L*precision.repetitions,!anyDuplicated(z[c("replication","test")]),
        identical(sort(unique(z$replication)),seq_len(precision.repetitions)),all(z$cell==i),
        all(z$scenario==confirmation.cells$scenario[i]),all(z$n==confirmation.cells$n[i]),all(z$null),
        all(z$seed==120361101L+10000L*i+z$replication),setequal(z$test,c("joint","x1")),
        all(z$df==ifelse(z$test=="joint",z$p,1L)*(z$K-2L)),
        all(z$available==is.finite(z$p.value)),all(z$available==is.finite(z$statistic)),
        all(z$statistic[z$available]>=0),all(z$reason[z$available]=="available"),
        all(z$reason[!z$available]!="available"),all(is.finite(z$elapsed)&z$elapsed>=0),
        isTRUE(all.equal(z$p.value[z$available],
            pchisq(z$statistic[z$available],z$df[z$available],lower.tail=FALSE),tolerance=1e-14)))
    z
})
z <- do.call(rbind,records)
rates <- do.call(rbind,lapply(split(z,interaction(z$cell,z$test,drop=TRUE)),function(g)
    do.call(rbind,lapply(c(.05,.01),function(level) {
        N <- nrow(g); A <- sum(g$available); k <- sum(g$p.value<level,na.rm=TRUE)
        a <- qnorm(.975)^2; center <- (k+a/2)/(A+a); radius <- sqrt(a*(k*(A-k)/A+a/4))/(A+a)
        data.frame(cell=g$cell[1],scenario=g$scenario[1],n=g$n[1],test=g$test[1],df=g$df[1],
            level,total=N,available=A,rejections=k,rate=k/A,lower=center-radius,upper=center+radius,
            availability=A/N,decision.lower=k/N,decision.upper=(k+N-A)/N)
    }))))
write.csv(rates,file.path(precision.root,"rates.csv"),row.names=FALSE)
failures <- aggregate(list(records=rep(1L,nrow(z))),z[c("cell","scenario","test","reason")],sum)
write.csv(failures,file.path(precision.root,"failures.csv"),row.names=FALSE)
stopifnot(identical(precision.identity,readRDS(precision.manifest)),
    identical(precision.identity$files,tools::md5sum(precision.files)),
    identical(identity$source,tools::md5sum(source.files)))
cat("Audited",nrow(z),"precision records;",sum(!z$available),"unavailable tests.\n")
