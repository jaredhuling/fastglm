wilson <- function(k,n) {
    if(!n) return(c(NA_real_,NA_real_))
    z <- qnorm(.975); p <- k/n; center <- (p+z*z/(2*n))/(1+z*z/n)
    half <- z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); c(center-half,center+half)
}
summarize.analytic <- function(root,cells,methods,repetitions,audit.only=FALSE) {
    rows <- paired <- failures <- list()
    for(i in seq_len(nrow(cells))) {
        d <- readRDS(file.path(root,"raw",sprintf("cell-%02d.rds",i)))
        stopifnot(nrow(d)==repetitions*length(methods),all(d$cell==i),all(d$n==cells$n[i]),
            all(d$scenario==cells$scenario[i]),!anyDuplicated(paste(d$replication,d$method)),
            all(d$seed==80361101L+10000L*i+d$replication),
            all(d$available==is.finite(d$p.value)),all(d$statistic[d$available]>=0))
        for(method in methods) {
            z <- d[d$method==method,]; stopifnot(identical(z$replication,seq_len(repetitions)))
            counts <- lapply(strsplit(z$counts,",",fixed=TRUE),as.numeric)
            stopifnot(all(vapply(counts,sum,numeric(1))==cells$n[i]),
                all(vapply(counts,min,numeric(1))==z$min.category),all(!z$available | z$min.category>0))
            n <- sum(z$available)
            for(level in c(.05,.01)) {
                k <- sum(z$p.value<=level,na.rm=TRUE); ci <- wilson(k,n)
                rows[[length(rows)+1L]] <- data.frame(cells[i,],method,level,null=z$null[1],
                    attempted=nrow(z),available=n,rejected=k,rate=if(n) k/n else NA_real_,
                    mc.lower=ci[1],mc.upper=ci[2],unconditional.lower=k/nrow(z),
                    unconditional.upper=(k+nrow(z)-n)/nrow(z),
                    median.ms=1000*median(z$elapsed[z$available]),
                    warnings=sum(nzchar(z$warning)),median.nuisance=if(method==methods[1]) NA_real_ else
                        median(z$nuisance.statistic[z$available]))
            }
            f <- as.data.frame(table(z$reason),stringsAsFactors=FALSE); names(f) <- c("reason","count")
            failures[[length(failures)+1L]] <- cbind(cells[rep(i,nrow(f)),],method,f)
        }
        for(a in 2:length(methods)) for(b in seq_len(a-1L)) {
            za <- d[d$method==methods[a],]; zb <- d[d$method==methods[b],]
            stopifnot(identical(za$replication,zb$replication)); ok <- za$available & zb$available
            for(level in c(.05,.01)) {
                diff <- as.numeric(za$p.value[ok]<=level)-as.numeric(zb$p.value[ok]<=level)
                paired[[length(paired)+1L]] <- data.frame(cells[i,],method=methods[a],baseline=methods[b],
                    level,shared=sum(ok),difference=mean(diff),mcse=sd(diff)/sqrt(length(diff)))
            }
        }
    }
    capture.output({cat("PASS: complete cells, unique method/replication keys, seed mapping, category totals, availability and finite statistics.\n")
        cat("Datasets:",nrow(cells)*repetitions,"; method records:",nrow(cells)*repetitions*length(methods),"\n")},file=file.path(root,"audit.txt"))
    if(!audit.only) {
        write.csv(do.call(rbind,rows),file.path(root,"summary.csv"),row.names=FALSE)
        write.csv(do.call(rbind,paired),file.path(root,"paired.csv"),row.names=FALSE)
        write.csv(do.call(rbind,failures),file.path(root,"failure-counts.csv"),row.names=FALSE)
    }
    invisible(TRUE)
}
