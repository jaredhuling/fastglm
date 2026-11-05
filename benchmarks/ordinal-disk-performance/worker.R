args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=5L) stop("Supply library, input root, cell RDS, result RDS, and version label")
library(fastglm,lib.loc=args[1L]);input.root<-normalizePath(args[2L])
cell<-readRDS(args[3L]);manifest<-readRDS(file.path(input.root,"inputs.rds"))
dataset<-manifest$datasets[[paste(cell$case,cell$link,sep="-")]]
stopifnot(!is.null(dataset),!file.exists(args[4L]))
make_reader<-function(paths,bytes) {
    force(paths);force(bytes)
    state<-new.env(parent=emptyenv());state$reads<-0L;state$seconds<-0
    state$bytes<-0;state$maximum.rows<-0L
    callback<-function(k) {
        started<-proc.time()[["elapsed"]]
        value<-readRDS(paths[k])
        state$seconds<-state$seconds+proc.time()[["elapsed"]]-started
        state$reads<-state$reads+1L;state$bytes<-state$bytes+bytes[k]
        state$maximum.rows<-max(state$maximum.rows,nrow(value$X))
        value
    }
    list(callback=callback,state=state)
}
reader<-make_reader(file.path(input.root,dataset$files),dataset$bytes)
stopifnot(!any(vapply(as.list(environment(reader$callback)),is.matrix,logical(1))))
selected<-if(cell$model=="partial") seq.int(3L,2L+cell$r) else NULL
warnings<-character();started<-proc.time()
value<-tryCatch(withCallingHandlers({
    if(cell$representation=="source") {
        source<-fastglm_source(reader$callback,length(dataset$files))
        fit<-fastglm_ordinal_streaming(source,nonparallel=selected,link=cell$link,
            firth=cell$firth,tol=manifest$configuration$tol,maxit=manifest$configuration$maxit)
        stopifnot(is.null(fit$x),is.null(fit$y),fit$maximum.block.rows<=cell$block)
    } else {
        # Preallocate dense input, avoiding an artificial second full copy of
        # every block during assembly. This branch intentionally retains n rows.
        x<-matrix(0,cell$n,2L+cell$r);y<-matrix(0,cell$n,cell$K)
        w<-off<-numeric(cell$n);first<-1L
        for(k in seq_along(dataset$files)) {
            block<-reader$callback(k);at<-seq.int(first,length.out=nrow(block$X))
            x[at,]<-block$X;y[at,]<-block$y;w[at]<-block$weights;off[at]<-block$offset
            first<-first+length(at)
        }
        colnames(x)<-colnames(block$X);colnames(y)<-colnames(block$y)
        fit<-fastglm_ordinal(x,y,weights=w,offset=off,nonparallel=selected,
            link=cell$link,firth=cell$firth,tol=manifest$configuration$tol,maxit=manifest$configuration$maxit)
    }
    fit
},warning=function(w) {warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
error=function(e) e)
elapsed<-proc.time()-started
result<-list(cell=cell,version=args[5L],library=find.package("fastglm"),
    elapsed=unname(elapsed["elapsed"]),user=unname(elapsed["user.self"]),system=unname(elapsed["sys.self"]),
    reader=as.list(reader$state),warnings=warnings,session=sessionInfo(),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS")),
    input.hashes=dataset$md5,binary.hash=tools::md5sum(list.files(system.file("libs",package="fastglm"),full.names=TRUE,recursive=TRUE)))
result$header.hashes<-tools::md5sum(file.path(system.file("include",package="fastglm"),c("ordinal.h","ordinal_partial.h")))
if(inherits(value,"error")) {result$status<-"error";result$error<-conditionMessage(value)} else {
    keys<-c("coefficients","cov.unscaled","information","ordinary.score","adjustment","score",
        "scaled.score","predictor.step","minimum.gap","information.rcond","loglik","history",
        "iterations","evaluations","fallbacks","converged","convergence.reason","last.rejected.evaluation")
    result$fit<-value[keys];result$status<-if(value$converged) "ok" else "nonconverged"
    result$source.passes<-if(cell$representation=="source") value$source.passes else NA_integer_
    result$d<-length(coef(value));result$retained.design<-!is.null(value$x)
}
saveRDS(result,args[4L])
cat(result$status,"elapsed",result$elapsed,"reads",result$reader$reads,"callback",result$reader$seconds,"\n")
