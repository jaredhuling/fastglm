# Rscript .../reproduce.R BASELINE_LIBRARY NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==2L)
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
out <- args[2L];if(dir.exists(out)) stop("Use a new result directory")
dir.create(out,recursive=TRUE)
set.seed(154005);n <- 96L
d <- list(x=cbind(a=runif(n,-1,1),b=runif(n,-1,1)),
    y=ordered(rep(1:4,length.out=n),levels=1:4),
    weights=rep(c(0,1,2,1,1),length.out=n),offset=cos(seq_len(n))/10)
saveRDS(d,file.path(out,"input.rds"))
source <- fastglm_source(function(k) {
    i <- seq.int((k-1L)*24L+1L,k*24L)
    list(X=d$x[i,,drop=FALSE],y=d$y[i],weights=d$weights[i],offset=d$offset[i])
},4L)
records <- list();fits <- list();states <- list()
write_exact <- function(x,file) {
    x <- as.matrix(x)
    writeLines(c(paste(colnames(x),collapse=","),apply(x,1L,function(row)
        paste(sprintf("%.17g",row),collapse=","))),file)
}
for(link in c("logit","probit","cloglog")) {
    caps <- switch(link,logit=c(1L,8L,30L,100L,200L),probit=c(30L,200L),
                   cloglog=c(1L,8L,30L,42L,200L))
    for(cap in caps) for(mode in c("dense",if(cap%in%c(30L,200L)) "source")) {
        key <- paste(link,cap,mode,sep="-");warnings <- character()
        fit <- withCallingHandlers(tryCatch(if(mode=="dense")
            fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,nonparallel=1:2,
                firth=TRUE,link=link,maxit=cap) else
            fastglm_ordinal_streaming(source,nonparallel=1:2,firth=TRUE,link=link,maxit=cap),
            error=identity),warning=function(w) {warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
        if(inherits(fit,"error")) stop(conditionMessage(fit))
        fits[[key]] <- fit
        records[[length(records)+1L]] <- data.frame(key=key,link=link,cap=cap,mode=mode,
            converged=fit$converged,reason=fit$convergence.reason,iterations=fit$iterations,
            evaluations=fit$evaluations,fallbacks=fit$fallbacks,scaled.score=fit$scaled.score,
            predictor.step=fit$predictor.step,minimum.gap=fit$minimum.gap,
            information.rcond=fit$information.rcond,loglik=fit$loglik,
            last.rejection=fit$last.rejected.evaluation,warnings=paste(warnings,collapse=" | "))
        write.csv(do.call(rbind,records),file.path(out,"fits.csv"),row.names=FALSE)
        saveRDS(fits,file.path(out,"fits.rds"))
        if(mode=="dense") {
            xx <- fastglm:::.ordinal_prediction_scale(d$x,fit$working$center,fit$working$scale)
            xx[d$weights==0,] <- 0;off <- d$offset;off[d$weights==0] <- 0
            state <- fastglm:::ordinal_partial_native_evaluate(matrix(numeric(),n,0L),xx,
                as.integer(d$y),matrix(numeric(),0L,0L),d$weights,off,4L,
                fit$working$coefficients,TRUE,condition_tol=1e-14,
                link=match(link,c("logit","probit","cloglog"))-1L)
            state$coefficients <- fit$working$coefficients;states[[key]] <- state
            write_exact(cbind(weights=d$weights,offset=off,y=as.integer(d$y),xx),
                file.path(out,paste0(key,"-input.csv")))
            write_exact(matrix(state$coefficients,1L,dimnames=list(NULL,paste0("b",1:9))),
                file.path(out,paste0(key,"-coefficients.csv")))
            write_exact(cbind(ordinary=as.numeric(state$ordinary.score),adjustment=as.numeric(state$adjustment),
                adjusted=as.numeric(state$score),step=as.numeric(state$step)),
                file.path(out,paste0(key,"-native-vectors.csv")))
            colnames(state$information) <- colnames(state$cov.unscaled) <- paste0("b",1:9)
            write_exact(state$information,file.path(out,paste0(key,"-native-information.csv")))
            write_exact(state$cov.unscaled,file.path(out,paste0(key,"-native-inverse.csv")))
        }
        cat(key,fit$converged,fit$convergence.reason,fit$minimum.gap,fit$scaled.score,"\n")
    }
}
saveRDS(states,file.path(out,"states.rds"))
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
files <- c(system.file("DESCRIPTION",package="fastglm"),list.files(system.file("libs",package="fastglm"),full.names=TRUE),
    "benchmarks/ordinal-partial-boundary/development-protocol.txt","benchmarks/ordinal-partial-boundary/reproduce.R")
write.csv(data.frame(path=files,md5=unname(tools::md5sum(files))),file.path(out,"source-hashes.csv"),row.names=FALSE)
