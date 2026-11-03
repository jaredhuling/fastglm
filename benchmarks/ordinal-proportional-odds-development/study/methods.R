po_one_method <- function(method,d) {
    warnings <- character(); error <- ""; began <- proc.time()[3]
    result <- tryCatch(withCallingHandlers({
        if(method=="car") {
            data <- data.frame(y=ordered(d$y,levels=seq_len(ncol(d$counts))),d$x)
            fit <- MASS::polr(y~.,data=data,method="logistic")
            if(fit$convergence!=0L) stop("ordinal_fit_not_converged")
            z <- car::poTest(fit)
            V <- z$D%*%z$vcov%*%t(z$D)
            list(statistic=z$chisq,df=z$df,difference=NA_real_,variance=NA_real_,V=V)
        } else {
            z <- ordinal_po_covariance(d$x,d$counts,firth=method=="meanBR")
            list(statistic=z$statistic,df=z$df,difference=z$difference[1],
                 variance=z$difference.covariance[1,1],V=z$difference.covariance)
        }
    },warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart("muffleWarning")}),
    error=function(e) {error <<- conditionMessage(e); NULL})
    elapsed <- unname(proc.time()[3]-began)
    expected.df <- ncol(d$x)*(ncol(d$counts)-2L)
    available <- !is.null(result) && length(result$statistic)==1L &&
        is.finite(result$statistic) && result$statistic>=0 && identical(as.integer(result$df),as.integer(expected.df))
    if(!available && !nzchar(error)) error <- "invalid_test_statistic_or_degrees_of_freedom"
    reason <- if(available) "available" else if(any(colSums(d$counts)==0)) "empty_category" else
        if(grepl("converged|convergence",error)) "fit_not_converged" else
        if(grepl("positive definite|singular|chol",error)) "covariance_factorization" else "other_error"
    data.frame(method=method,available=available,statistic=if(available) result$statistic else NA_real_,
        df=expected.df,p.value=if(available) pchisq(result$statistic,expected.df,lower.tail=FALSE) else NA_real_,
        difference=if(available) result$difference else NA_real_,
        variance=if(available) result$variance else NA_real_,
        covariance.rcond=if(available) tryCatch(rcond(result$V),error=function(e) NA_real_) else NA_real_,
        covariance.asymmetry=if(available) max(abs(result$V-t(result$V)))/(1+max(abs(result$V))) else NA_real_,
        elapsed=elapsed,reason=reason,error=error,warning=paste(unique(warnings),collapse=" | "))
}

po_run_cell <- function(cell,repetitions,path=NULL) {
    out <- vector("list",length(repetitions))
    for(i in seq_along(repetitions)) {
        r <- repetitions[i]; seed <- 20261108L+10000L*cell$cell+r
        set.seed(seed); d <- po_generate(cell$scenario,cell$n)
        record <- do.call(rbind,lapply(po_methods,po_one_method,d=d))
        record$cell <- cell$cell; record$scenario <- cell$scenario; record$n <- cell$n
        record$replication <- r; record$seed <- seed; record$null <- d$null
        record$truth <- d$truth; record$min.category <- min(colSums(d$counts))
        record$empty.categories <- sum(colSums(d$counts)==0)
        out[[i]] <- record
    }
    out <- do.call(rbind,out)
    if(!is.null(path)) {
        temporary <- paste0(path,".tmp")
        saveRDS(out,temporary,version=2)
        stopifnot(file.rename(temporary,path))
    }
    out
}
