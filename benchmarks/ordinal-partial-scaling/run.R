args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L || dir.exists(args[1L]))
    stop("Supply one new output directory")
if(!file.exists("DESCRIPTION") || !dir.exists("benchmarks/ordinal-partial-scaling"))
    stop("Run from the fastglm repository root")
dir.create(args[1L],recursive=TRUE)
output <- normalizePath(args[1L])
library(fastglm)
config <- list(study="computational development",n=1200L,common=2L,
    categories=c(3L,5L,9L),varying=c(1L,4L,8L),links=c("logit","probit","cloglog"),
    seed=154100L,repetitions=5L,source.block=200L,tol=1e-8,maxit=200L)
saveRDS(config,file.path(output,"configuration.rds"))
paths <- c("R/ordinal-partial.R","R/ordinal-core.R","R/ordinal-prepared.R",
    "R/ordinal-streaming.R","inst/include/ordinal_partial.h","inst/include/ordinal.h",
    "src/fit_streaming_ordinal_partial.cpp","src/fit_ordinal_partial.cpp",
    "benchmarks/ordinal-partial-scaling/run.R")
saveRDS(list(time=Sys.time(),version=as.character(packageVersion("fastglm")),
    package.path=find.package("fastglm"),hashes=tools::md5sum(paths),session=sessionInfo(),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))),
    file.path(output,"environment.rds"))

fixture <- function(K,r,link) {
    set.seed(config$seed+100L*K+r)
    x <- matrix(runif(config$n*(config$common+r),-1,1),config$n)
    colnames(x) <- paste0("x",seq_len(ncol(x)))
    q <- K-1L;alpha <- seq(-1.5,1.5,length.out=q)
    beta <- c(.2,-.15)
    gamma <- outer(rep(.1/r,r),rep(1,q))+outer(rep(.025/r,r),sin(seq_len(q)))
    eta <- matrix(alpha,config$n,q,byrow=TRUE)-drop(x[,1:2]%*%beta)-x[,-(1:2),drop=FALSE]%*%gamma
    G <- switch(link,logit=plogis,probit=pnorm,cloglog=function(t)-expm1(-exp(t)))
    F <- G(eta);probability <- cbind(F,1)-cbind(0,F)
    stopifnot(all(probability>0))
    counts <- 1+round(100*probability)
    colnames(counts) <- paste0("category",seq_len(K))
    list(x=x,counts=counts,selected=seq.int(3L,ncol(x)),
        theta=c(alpha,beta,as.numeric(gamma)),K=K,r=r,link=link)
}

native <- getFromNamespace("ordinal_partial_native_evaluate","fastglm")
evaluate <- function(z,firth) native(z$x[,1:2,drop=FALSE],z$x[,z$selected,drop=FALSE],
    integer(),z$counts,rep(1,nrow(z$x)),numeric(nrow(z$x)),z$K,z$theta,
    firth=firth,link=match(z$link,config$links)-1L)
fit_dense <- function(z,firth) fastglm_ordinal(z$x,z$counts,nonparallel=z$selected,
    link=z$link,firth=firth,tol=config$tol,maxit=config$maxit)
timed <- function(fun) {
    fun()
    pilot <- unname(system.time(fun())["elapsed"])
    batch <- max(1L,min(20L,ceiling(.03/max(pilot,.001))))
    times <- replicate(config$repetitions,
        unname(system.time(for(j in seq_len(batch)) fun())["elapsed"])/batch)
    list(median=median(times),minimum=min(times),maximum=max(times),times=times,batch=batch)
}

# Profile before inspecting timings. Native operations appear together as a
# call; use the recorded result to decide whether deeper profiling is needed.
profile.data <- fixture(max(config$categories),max(config$varying),"logit")
Rprof(file.path(output,"profile.out"),interval=.001,memory.profiling=TRUE)
tryCatch(for(i in seq_len(3L)) fit_dense(profile.data,TRUE),finally=Rprof(NULL))
saveRDS(summaryRprof(file.path(output,"profile.out"),memory="both"),file.path(output,"profile.rds"))

grid <- expand.grid(K=config$categories,r=config$varying,link=config$links,
    firth=c(FALSE,TRUE),stringsAsFactors=FALSE)
records <- vector("list",nrow(grid))
for(i in seq_len(nrow(grid))) {
    cell <- grid[i,];z <- fixture(cell$K,cell$r,cell$link)
    message("Cell ",i,"/",nrow(grid),": K=",cell$K," r=",cell$r," ",cell$link," firth=",cell$firth)
    records[[i]] <- tryCatch({
        fit <- fit_dense(z,cell$firth)
        saveRDS(fit[c("coefficients","converged","convergence.reason","iterations",
            "evaluations","fallbacks","scaled.score","predictor.step","minimum.gap",
            "information.rcond")],file.path(output,sprintf("cell-%03d-fit.rds",i)))
        if(!fit$converged) stop("nonconvergence: ",fit$convergence.reason)
        native.time <- timed(function() evaluate(z,cell$firth))
        dense.time <- timed(function() fit_dense(z,cell$firth))
        result <- c(as.list(cell),list(status="ok",d=length(coef(fit)),
            evaluations=fit$evaluations,iterations=fit$iterations,source.passes=NA_integer_,
            fallbacks=fit$fallbacks,scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,
            minimum.gap=fit$minimum.gap,information.rcond=fit$information.rcond,
            native.seconds=native.time$median,dense.seconds=dense.time$median,
            prepared.seconds=NA_real_,stream.seconds=NA_real_,
            coefficient.difference=NA_real_,covariance.difference=NA_real_,
            dense.object.bytes=as.numeric(object.size(fit)),maximum.block.rows=NA_integer_,
            timing.repetitions=list(native=native.time,dense=dense.time)))
        if(cell$K%in%range(config$categories) && cell$r%in%range(config$varying)) {
            design <- fastglm_ordinal_prepare(z$x,nonparallel=z$selected,link=z$link,firth=cell$firth)
            prepared <- function() fastglm_ordinal(design,z$counts,tol=config$tol,maxit=config$maxit)
            prepared.fit <- prepared()
            blocks <- split(seq_len(nrow(z$x)),ceiling(seq_len(nrow(z$x))/config$source.block))
            source <- fastglm_source(function(k) {
                at <- blocks[[k]]
                list(X=z$x[at,,drop=FALSE],y=z$counts[at,,drop=FALSE])
            },length(blocks))
            streamed <- function() fastglm_ordinal_streaming(source,nonparallel=z$selected,
                link=z$link,firth=cell$firth,tol=config$tol,maxit=config$maxit)
            stream.fit <- streamed()
            stopifnot(prepared.fit$converged,stream.fit$converged,is.null(stream.fit$x),
                is.null(stream.fit$y),stream.fit$maximum.block.rows<=config$source.block)
            bd <- max(abs(coef(fit)-coef(prepared.fit)),abs(coef(fit)-coef(stream.fit)))
            vd <- max(abs(vcov(fit)-vcov(prepared.fit)),abs(vcov(fit)-vcov(stream.fit)))
            if(bd>1e-6 || vd>1e-6) stop("dense/prepared/source estimates disagree")
            pt <- timed(prepared);st <- timed(streamed)
            result$prepared.seconds <- pt$median;result$stream.seconds <- st$median
            result$coefficient.difference <- bd;result$covariance.difference <- vd
            result$source.passes <- stream.fit$source.passes
            result$maximum.block.rows <- stream.fit$maximum.block.rows
            result$timing.repetitions$prepared <- pt;result$timing.repetitions$stream <- st
        }
        result
    },error=function(e) c(as.list(cell),list(status="error",message=conditionMessage(e))))
    saveRDS(records[[i]],file.path(output,sprintf("cell-%03d.rds",i)))
}
saveRDS(records,file.path(output,"records.rds"))
fields <- unique(unlist(lapply(records,names)))
fields <- setdiff(fields,"timing.repetitions")
table <- do.call(rbind,lapply(records,function(x) {
    out <- setNames(rep(list(NA),length(fields)),fields)
    out[names(x)[names(x)!="timing.repetitions"]] <- x[names(x)!="timing.repetitions"]
    as.data.frame(out,stringsAsFactors=FALSE)
}))
write.csv(table,file.path(output,"records.csv"),row.names=FALSE)
