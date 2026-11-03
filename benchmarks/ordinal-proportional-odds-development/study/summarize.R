po_wilson <- function(success,total,alpha=.05) {
    if(!total) return(c(NA_real_,NA_real_))
    p <- success/total; z <- qnorm(1-alpha/2)
    center <- (p+z^2/(2*total))/(1+z^2/total)
    radius <- z*sqrt(p*(1-p)/total+z^2/(4*total^2))/(1+z^2/total)
    c(center-radius,center+radius)
}

po_summarize <- function(root,identity,audit.only=FALSE) {
    paths <- file.path(root,"raw",sprintf("cell-%02d.rds",seq_len(nrow(po_cells))))
    stopifnot(all(file.exists(paths)))
    records <- lapply(paths,readRDS)
    for(i in seq_along(records)) {
        z <- records[[i]]
        stopifnot(nrow(z)==po_repetitions*length(po_methods),all(z$cell==i),
            all(z$scenario==po_cells$scenario[i]),all(z$n==po_cells$n[i]),
            !anyDuplicated(z[,c("replication","method")]),
            setequal(z$replication,seq_len(po_repetitions)),setequal(z$method,po_methods),
            all(z$seed==20261108L+10000L*i+z$replication),
            all(is.finite(z$p.value[z$available])),all(is.na(z$p.value[!z$available])))
    }
    raw.hashes <- setNames(unname(tools::md5sum(paths)),basename(paths))
    audit <- c(sprintf("PASS: %d cells, %d datasets, %d method records",nrow(po_cells),
        nrow(po_cells)*po_repetitions,sum(vapply(records,nrow,integer(1)))),
        "PASS: unique complete method/replication grid, seed mapping, and availability fields",
        "PASS: frozen source and environment identity checked by dispatcher")
    writeLines(audit,file.path(root,"audit.txt")); cat(paste(audit,collapse="\n"),"\n")
    if(audit.only) return(invisible(NULL))
    raw <- do.call(rbind,records)
    summary <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$method,drop=TRUE)),function(z) {
        ok <- z$available; n <- sum(ok); p <- z$p.value[ok]
        reject <- sum(p<.05); reject01 <- sum(p<.01)
        rate <- if(n) reject/n else NA_real_; rate01 <- if(n) reject01/n else NA_real_
        interval <- po_wilson(reject,n); interval01 <- po_wilson(reject01,n)
        simultaneous <- po_wilson(reject,n,.05/72)
        errors <- z$difference[ok]-z$truth[ok]
        vv <- z$variance[ok]; valid <- is.finite(errors)&is.finite(vv)&vv>=0
        errors <- errors[valid]; se <- sqrt(vv[valid]); nd <- length(errors)
        sd.error <- if(nd>1) sd(errors) else NA_real_
        coverage <- if(nd) mean(abs(errors)<=qnorm(.975)*se) else NA_real_
        data.frame(scenario=z$scenario[1],n=z$n[1],method=z$method[1],null=z$null[1],
            datasets=nrow(z),available=n,availability=n/nrow(z),
            availability.mcse=sqrt((n/nrow(z))*(1-n/nrow(z))/nrow(z)),
            rejection=rate,rejection.mcse=if(n) sqrt(rate*(1-rate)/n) else NA_real_,
            rejection.lower=interval[1],rejection.upper=interval[2],
            rejection01=rate01,rejection01.mcse=if(n) sqrt(rate01*(1-rate01)/n) else NA_real_,
            rejection01.lower=interval01[1],rejection01.upper=interval01[2],
            rejected.all=reject/nrow(z),unavailable.upper=(reject+sum(!ok))/nrow(z),
            simultaneous.lower=simultaneous[1],simultaneous.upper=simultaneous[2],
            null.flag=if(!z$null[1]) "alternative" else if(is.na(rate)) "unavailable" else
                if(simultaneous[1]>.05) "inflated" else if(simultaneous[2]<.05) "conservative" else "unresolved",
            empty.fraction=mean(z$empty.categories>0),warnings=mean(nzchar(z$warning)),
            median.ms=1000*median(z$elapsed),p90.ms=1000*unname(quantile(z$elapsed,.9)),
            contrast.n=nd,bias=if(nd) mean(errors) else NA_real_,bias.mcse=if(nd) sd.error/sqrt(nd) else NA_real_,
            empirical.sd=sd.error,rmse=if(nd) sqrt(mean(errors^2)) else NA_real_,
            root.mean.variance=if(nd) sqrt(mean(se^2)) else NA_real_,coverage=coverage,
            coverage.mcse=if(nd) sqrt(coverage*(1-coverage)/nd) else NA_real_,
            mean.width=if(nd) mean(2*qnorm(.975)*se) else NA_real_,
            max.asymmetry=if(n) max(z$covariance.asymmetry[ok],na.rm=TRUE) else NA_real_)
    }))
    summary <- summary[order(match(summary$scenario,po_scenarios),summary$n,match(summary$method,po_methods)),]
    rownames(summary) <- NULL
    pairs <- do.call(rbind,lapply(records,function(z) do.call(rbind,lapply(combn(po_methods,2,simplify=FALSE),function(m) {
        a <- z[z$method==m[1],]; b <- z[z$method==m[2],]
        b <- b[match(a$replication,b$replication),]; ok <- a$available & b$available
        delta <- as.numeric(a$p.value[ok]<.05)-as.numeric(b$p.value[ok]<.05)
        data.frame(scenario=z$scenario[1],n=z$n[1],first=m[1],second=m[2],shared=sum(ok),
            difference=if(length(delta)) mean(delta) else NA_real_,
            mcse=if(length(delta)>1) sd(delta)/sqrt(length(delta)) else NA_real_)
    }))))
    failures <- aggregate(list(count=raw$replication),raw[,c("scenario","n","method","reason")],length)
    write.csv(summary,file.path(root,"summary.csv"),row.names=FALSE)
    write.csv(pairs,file.path(root,"paired.csv"),row.names=FALSE)
    write.csv(failures,file.path(root,"failure-counts.csv"),row.names=FALSE)
    dput(list(input=identity,raw=raw.hashes),file=file.path(root,"results-manifest.txt"))
    plot.results <- function() {
        old <- par(mfrow=c(4,2),mar=c(3.2,3.7,2.1,1),oma=c(0,0,1.5,0))
        on.exit(par(old))
        colors <- c("#2166ac","#b2182b","#4d9221")
        for(s in po_scenarios[!grepl("alternative",po_scenarios)]) {
            z <- summary[summary$scenario==s,]
            ylim <- c(0,max(.12,z$rejection.upper,na.rm=TRUE))
            plot(NA,xlim=c(65,1200),ylim=ylim,log="x",xaxt="n",xlab="Sample size",
                 ylab="Rejection probability",main=s)
            axis(1,at=c(80,250,1000),labels=c(80,250,1000)); abline(h=.05,lty=2,col="gray50")
            for(k in seq_along(po_methods)) {
                v <- z[z$method==po_methods[k],]; at <- v$n*exp((k-2)*.05)
                lines(at,v$rejection,type="b",col=colors[k],pch=14+k)
                arrows(at,v$rejection.lower,at,v$rejection.upper,angle=90,code=3,length=.03,col=colors[k])
            }
            if(s==po_scenarios[1]) legend("topright",legend=po_methods,col=colors,pch=15:17,lty=1,bty="n",cex=.8)
        }
        mtext("Nominal 5%; conditional on available results; separate 95% Monte Carlo intervals",outer=TRUE,cex=.8)
    }
    pdf(file.path(root,"null-rejection.pdf"),width=9,height=11); plot.results(); dev.off()
    png(file.path(root,"null-rejection.png"),width=1350,height=1650,res=150); plot.results(); dev.off()
    format.percent <- function(x) ifelse(is.na(x),"NA",sprintf("%.1f%%",100*x))
    rows <- vapply(seq_len(nrow(summary)),function(i) {
        z <- summary[i,]
        paste0("| ",paste(c(z$scenario,z$n,z$method,paste0(z$available,"/",z$datasets),
            format.percent(z$rejection),paste(format.percent(c(z$rejection.lower,z$rejection.upper)),collapse=" to "),
            format.percent(z$rejection01),z$null.flag),collapse=" | ")," |")
    },character(1))
    writeLines(c("# Frozen development results", "",
        "Generated by `summarize.R` from the complete raw cell files. See `results-manifest.txt` for hashes and environment identity.","",
        audit,"", "Null scenarios report false-positive rates; alternative scenarios report power.",
        "Rates condition on available tests. Intervals describe Monte Carlo uncertainty, not uncertainty for a fitted regression.",
        "The null flag uses Bonferroni-adjusted intervals across 72 comparisons; the displayed intervals are separate 95% intervals.","",
        "| Scenario | n | Method | Available | Rejection at 5% | 95% MC interval | Rejection at 1% | Null flag |",
        "|---|---:|---|---:|---:|---|---:|---|", rows,"",
        "`summary.csv` also reports missing-result bounds, contrast accuracy and uncertainty, warnings, and timings.",
        "`paired.csv` reports rejection differences on common available datasets with paired Monte Carlo standard errors.",
        "`failure-counts.csv` and the raw records retain unsuccessful fits. Timings compare different interface workloads and are descriptive.",
        "This study does not establish general finite-sample validity and does not change the public diagnostic."),file.path(root,"RESULTS.md"))
    invisible(summary)
}
