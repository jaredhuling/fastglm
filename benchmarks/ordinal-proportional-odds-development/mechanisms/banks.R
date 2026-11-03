po_run_bank <- function(cell,design,bank,indices) {
    m <- nrow(design$D); methods <- mechanism.methods; nr <- length(indices)
    delta <- matrix(NA_real_,nr,m)
    statistic <- variance.first <- variance.trace <- matrix(NA_real_,nr,length(methods),dimnames=list(NULL,methods))
    sum.V <- lapply(c("all","observed"),function(s) setNames(lapply(methods,function(a) matrix(0,m,m)),methods))
    names(sum.V) <- c("all","observed")
    count.V <- matrix(0L,2L,length(methods),dimnames=list(names(sum.V),methods))
    meta <- vector("list",nr)
    for(r in seq_along(indices)) {
        rep <- indices[r]; seed <- 40361100L+100000L*cell$cell+10000L*bank+rep
        set.seed(seed); u <- runif(cell$n); y <- 1L+rowSums(u>design$F)
        counts <- tabulate(y,nbins=design$q+1L); observed <- all(counts>0)
        error <- ""; t0 <- proc.time()[3]
        fit <- tryCatch(po_mechanism_fit(design,y),error=function(e) {error <<- conditionMessage(e); NULL})
        elapsed <- proc.time()[3]-t0
        if(!is.null(fit)) {
            delta[r,] <- fit$delta
            for(method in methods) {
                V <- fit$V[[method]]
                statistic[r,method] <- po_statistic(fit$delta,V)
                if(!is.null(V) && all(is.finite(V))) {
                    variance.first[r,method] <- V[1,1]; variance.trace[r,method] <- sum(diag(V))
                    for(population in c("all",if(observed) "observed")) {
                        sum.V[[population]][[method]] <- sum.V[[population]][[method]]+V
                        count.V[population,method] <- count.V[population,method]+1L
                    }
                }
            }
        }
        meta[[r]] <- data.frame(rep=rep,seed=seed,bank=bank,fit.ok=!is.null(fit),all.observed=observed,
            counts=paste(counts,collapse=","),min.category=min(counts),error=error,elapsed=elapsed,
            score.max=if(is.null(fit)) NA_real_ else fit$score.max,
            adjustment.size=if(is.null(fit)) NA_real_ else fit$adjustment.size,
            null.error=if(is.null(fit)) "binary_fit_failed" else fit$null.error,
            warning=if(is.null(fit)) "" else fit$warning)
    }
    list(cell=cell,bank=bank,meta=do.call(rbind,meta),delta=delta,statistic=statistic,
         variance.first=variance.first,variance.trace=variance.trace,sum.V=sum.V,count.V=count.V)
}
