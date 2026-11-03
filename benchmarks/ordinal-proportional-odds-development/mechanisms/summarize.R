po_wilson <- function(k,n) {
    if(!n) return(c(NA_real_,NA_real_))
    z <- qnorm(.975); p <- k/n; c0 <- (p+z*z/(2*n))/(1+z*z/n)
    h <- z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    c(c0-h,c0+h)
}

po_summarize_mechanisms <- function(root,audit.only=FALSE) {
    rows <- paired <- moments <- failures <- diagnostics <- list()
    append.row <- function(z) {rows[[length(rows)+1L]] <<- z}
    for(i in seq_len(nrow(mechanism.cells))) {
        cell <- mechanism.cells[i,]; design <- readRDS(file.path(root,sprintf("design-%02d.rds",i)))
        banks <- lapply(seq_along(mechanism.banks),function(b)
            readRDS(file.path(root,"raw",sprintf("cell-%02d-bank-%d.rds",i,b))))
        names(banks) <- names(mechanism.banks)
        for(b in seq_along(banks)) {
            z <- banks[[b]]; nr <- mechanism.banks[b]
            stopifnot(identical(z$cell,cell),z$bank==b,nrow(z$delta)==nr,nrow(z$meta)==nr,
                identical(z$meta$rep,seq_len(nr)),!anyDuplicated(z$meta$seed),
                all(z$meta$seed==40361100L+100000L*i+10000L*b+seq_len(nr)),
                all(rowSums(is.finite(z$delta))==ifelse(z$meta$fit.ok,ncol(z$delta),0)),
                identical(colnames(z$statistic),mechanism.methods),
                all(z$count.V["all",]<=sum(z$meta$fit.ok)),
                all(z$count.V["observed",]<=sum(z$meta$fit.ok & z$meta$all.observed)))
            counts <- t(vapply(strsplit(z$meta$counts,",",fixed=TRUE),as.integer,integer(design$q+1L)))
            stopifnot(all(rowSums(counts)==cell$n),all(z$meta$all.observed==(rowSums(counts==0)==0)))
            failures[[length(failures)+1L]] <- data.frame(cell,bank=names(banks)[b],attempted=nr,
                binary.failed=sum(!z$meta$fit.ok),empty.categories=sum(!z$meta$all.observed),
                null.failed=sum(nzchar(z$meta$null.error)),warnings=sum(nzchar(z$meta$warning)))
            diagnostics[[length(diagnostics)+1L]] <- data.frame(cell,bank=names(banks)[b],
                score.max=max(z$meta$score.max,na.rm=TRUE),
                adjustment.median=median(z$meta$adjustment.size,na.rm=TRUE),
                adjustment.q95=unname(quantile(z$meta$adjustment.size,.95,na.rm=TRUE)),
                seconds=sum(z$meta$elapsed))
        }
        if(audit.only) next
        for(population in c("all","observed")) {
            select <- function(z) z$meta$fit.ok & (population=="all" | z$meta$all.observed)
            a <- banks$moments; e <- banks$evaluation; r <- banks$reference
            ma <- select(a); me <- select(e); mr <- select(r)
            C <- cov(a$delta[ma,,drop=FALSE]); mu <- colMeans(a$delta[ma,,drop=FALSE])
            stopifnot(sum(ma)>ncol(C)+1L,all(is.finite(C)))
            root.inverse <- backsolve(chol(C),diag(nrow(C)))
            average.V <- lapply(mechanism.methods,function(method) {
                count <- a$count.V[population,method]
                if(count) a$sum.V[[population]][[method]]/count else NULL
            }); names(average.V) <- mechanism.methods
            for(method in mechanism.methods) {
                V <- average.V[[method]]
                eig <- if(is.null(V)) NA_real_ else eigen(crossprod(root.inverse,V%*%root.inverse),symmetric=TRUE,only.values=TRUE)$values
                moments[[length(moments)+1L]] <- data.frame(cell,population,method,moment.n=sum(ma),covariance.n=a$count.V[population,method],
                    bias.first=mu[1],bias.mcse=sqrt(C[1,1]/sum(ma)),sd.first=sqrt(C[1,1]),
                    standardized.mean.norm=sqrt(po_statistic(mu,C)),
                    mean.variance.first=if(is.null(V)) NA_real_ else V[1,1],
                    variance.ratio.first=if(is.null(V)) NA_real_ else V[1,1]/C[1,1],
                    trace.ratio=if(is.null(V)) NA_real_ else sum(diag(V))/sum(diag(C)),
                    normalized.eigen.min=min(eig),normalized.eigen.max=max(eig))
            }
            quadratic <- function(d,V,center=numeric(ncol(d))) apply(d,1,function(x) {
                if(any(!is.finite(x))) NA_real_ else po_statistic(x-center,V)
            })
            T <- cbind(e$statistic,
                independent_mean_empirical=quadratic(e$delta,average.V$empirical_ordinary),
                independent_sampling_covariance=quadratic(e$delta,C),
                independent_sampling_centered=quadratic(e$delta,C,mu))
            P <- pchisq(T,ncol(C),lower.tail=FALSE)
            ref <- quadratic(r$delta[mr,,drop=FALSE],C)
            stopifnot(all(is.finite(ref)))
            mc <- vapply(T[,"independent_sampling_covariance"],function(t) {
                if(!is.finite(t)) NA_real_ else (1+sum(ref>=t))/(length(ref)+1)
            },numeric(1))
            P <- cbind(P,oracle_simulation=mc)
            # Eligibility counts include binary failures; availability excludes them.
            eligible <- population=="all" | e$meta$all.observed
            for(method in colnames(P)) for(level in c(.05,.01)) {
                ok <- eligible & is.finite(P[,method]); reject <- sum(P[ok,method]<=level)
                n <- sum(ok); total <- sum(eligible); ci <- po_wilson(reject,n)
                append.row(data.frame(cell,population,method,level,attempted=total,available=n,
                    unavailable=total-n,rejected=reject,rate=if(n) reject/n else NA_real_,
                    mc.lower=ci[1],mc.upper=ci[2],unconditional.lower=reject/total,
                    unconditional.upper=(reject+total-n)/total,moment.n=sum(ma),reference.n=sum(mr)))
                baseline <- "empirical_ordinary"; both <- ok & is.finite(P[,baseline])
                diff <- as.numeric(P[both,method]<=level)-as.numeric(P[both,baseline]<=level)
                paired[[length(paired)+1L]] <- data.frame(cell,population,method,level,
                    shared=sum(both),difference=if(length(diff)) mean(diff) else NA_real_,
                    mcse=if(length(diff)>1L) sd(diff)/sqrt(length(diff)) else NA_real_)
            }
        }
    }
    failures <- do.call(rbind,failures); diagnostics <- do.call(rbind,diagnostics)
    stopifnot(sum(failures$attempted)==nrow(mechanism.cells)*sum(mechanism.banks))
    capture.output({cat("PASS: dimensions, cells, bank identities, seeds, category counts, missingness and covariance counts.\n")
        cat("Total response datasets:",sum(failures$attempted),"\n")
        cat("Binary fit failures:",sum(failures$binary.failed),"\n")
        cat("Maximum adjusted-score residual:",max(diagnostics$score.max),"\n")
        cat("Source/dependency identity is checked by run.R before this audit.\n")},file=file.path(root,"audit.txt"))
    if(audit.only) return(invisible(TRUE))
    write.csv(do.call(rbind,rows),file.path(root,"summary.csv"),row.names=FALSE)
    write.csv(do.call(rbind,paired),file.path(root,"paired.csv"),row.names=FALSE)
    write.csv(do.call(rbind,moments),file.path(root,"moments.csv"),row.names=FALSE)
    write.csv(failures,file.path(root,"failure-counts.csv"),row.names=FALSE)
    write.csv(diagnostics,file.path(root,"fit-diagnostics.csv"),row.names=FALSE)
    invisible(TRUE)
}
