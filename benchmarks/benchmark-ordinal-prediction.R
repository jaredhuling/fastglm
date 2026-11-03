# Run from the repository root with two immutable installed libraries:
# Rscript benchmarks/benchmark-ordinal-prediction.R --baseline-library=OLD --candidate-library=NEW
# Identical saved fits isolate prediction from fitting/preparation changes.
options(warn=1)
args<-commandArgs(TRUE)
arg<-function(name,default="") {
    value<-args[startsWith(args,paste0("--",name,"="))]
    if(length(value))sub(paste0("^--",name,"="),"",value[1L]) else default
}
script<-normalizePath(sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1L]))
out<-arg("output","benchmarks/ordinal-prediction-results")
phase<-arg("phase","main");label<-arg("label");work<-arg("work")
Sys.setenv(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
hash<-function(x) {path<-tempfile();on.exit(unlink(path));saveRDS(x,path,version=2);unname(tools::md5sum(path))}
functions<-c(".ordinal_prediction_design",".ordinal_prediction_scale",".ordinal_probability_block",".ordinal_gradient","predict.fastglm_ordinal")
if(phase=="main") {
    baseline<-normalizePath(arg("baseline-library"));candidate<-normalizePath(arg("candidate-library"))
    if(file.exists(file.path(out,"metadata.rds")))stop("use a new output directory to preserve existing results")
    dir.create(out,recursive=TRUE,showWarnings=FALSE);out<-normalizePath(out)
    work<-tempfile("ordinal-prediction-");dir.create(work)
    .libPaths(c(baseline,.libPaths()));library(fastglm)
    stopifnot(normalizePath(find.package("fastglm"))==normalizePath(file.path(baseline,"fastglm")))
    cases<-expand.grid(n=c(20000L,100000L),p=c(3L,12L));cases$K<-ifelse(cases$p==3L,4L,8L)
    cases$case<-paste0("n",cases$n,"-p",cases$p,"-K",cases$K)
    cases$seed<-98200L+seq_len(nrow(cases));cases$repetitions<-ifelse(cases$n==20000L,7L,5L)
    fixtures<-lapply(seq_len(nrow(cases)),function(i) {
        ca<-cases[i,];set.seed(ca$seed);n<-2000L
        x<-matrix(rnorm(n*ca$p,sd=.5),n,dimnames=list(NULL,paste0("x",seq_len(ca$p))))
        offset<-runif(n,-.2,.2);beta<-c(.5,-.4,.3,rep(0,ca$p-3L))
        y<-ordered(findInterval(drop(x%*%beta)+offset+rlogis(n),qlogis(seq_len(ca$K-1L)/ca$K))+1L,levels=seq_len(ca$K))
        fit<-fastglm_ordinal(x,y,offset=offset,firth=TRUE,tol=1e-8)
        stopifnot(fit$converged);fit
    })
    names(fixtures)<-cases$case
    saveRDS(list(cases=cases,fits=fixtures),file.path(out,"fixtures.rds"),compress="xz")
    metadata<-list(timestamp_utc=format(Sys.time(),tz="UTC",usetz=TRUE),study="paired prediction development benchmark",
        baseline_library=baseline,candidate_library=candidate,script_md5=unname(tools::md5sum(script)),
        fixture_md5=unname(tools::md5sum(file.path(out,"fixtures.rds"))),cases=cases,R=R.version.string,
        protocol="Both implementations predict from the same serialized fitted objects and regenerated inputs. Seven repetitions at20k rows and five at100k. Preflight output equivalence precedes timing. R allocation recording is a separate warmed call, outside timing. All cases, errors, and warnings are retained.",
        allocation_scope="Cumulative R-managed allocation bytes, allocation count, and largest single R allocation; not process RSS or native peak memory.",
        work_directory=work)
    saveRDS(metadata,file.path(out,"metadata.rds"))
    write.csv(cases,file.path(out,"cases.csv"),row.names=FALSE)
    for(label in c("baseline","candidate")) {
        lib<-if(label=="baseline")baseline else candidate
        status<-system2(file.path(R.home("bin"),"Rscript"),
            c(shQuote(script),"--phase=measure",shQuote(paste0("--label=",label)),
              shQuote(paste0("--library=",lib)),shQuote(paste0("--output=",out)),shQuote(paste0("--work=",work))))
        if(status!=0L)stop(label," measurement exited with status ",status)
    }
    before<-read.csv(file.path(out,"baseline-summary.csv"));after<-read.csv(file.path(out,"candidate-summary.csv"))
    comparison<-merge(before,after,by=c("case","n","p","K","prediction"),suffixes=c("_before","_after"))
    comparison$elapsed_ratio_after_before<-comparison$median_ms_after/comparison$median_ms_before
    comparison$allocated_bytes_ratio_after_before<-comparison$allocated_bytes_after/comparison$allocated_bytes_before
    comparison$allocation_reduction_percent<-100*(1-comparison$allocated_bytes_ratio_after_before)
    write.csv(comparison,file.path(out,"comparison.csv"),row.names=FALSE)
    rows<-rbind(before,after);write.csv(rows,file.path(out,"summary.csv"),row.names=FALSE)
    for(name in c("raw-timings","accuracy","conditions")) {
        a<-read.csv(file.path(out,paste0("baseline-",name,".csv")))
        b<-read.csv(file.path(out,paste0("candidate-",name,".csv")))
        write.csv(rbind(a,b),file.path(out,paste0(name,".csv")),row.names=FALSE)
    }
    eligible<-comparison$status_before=="measured"&comparison$status_after=="measured"
    note<-c("Ordinal prediction: before/after allocation benchmark",
        sprintf("All %d matched cases are retained; %d pairs passed numerical checks and completed timing.",nrow(comparison),sum(eligible)),
        "Fitted parameters, centered computation, offsets, category mapping, and inference targets are held fixed. Only the prediction implementation differs.",
        metadata$allocation_scope,
        "Timings are a bounded comparison on one machine; allocation totals are the direct implementation evidence.")
    for(mode in unique(comparison$prediction)) {
        d<-comparison[comparison$prediction==mode & eligible,]
        if(nrow(d))note<-c(note,sprintf("%s: elapsed ratios %.3f-%.3f; R allocation reductions %.1f-%.1f%%.",
            mode,min(d$elapsed_ratio_after_before),max(d$elapsed_ratio_after_before),
            min(d$allocation_reduction_percent),max(d$allocation_reduction_percent)))
    }
    writeLines(note,file.path(out,"interpretation.txt"))
    metadata$completed_utc<-format(Sys.time(),tz="UTC",usetz=TRUE)
    metadata$complete_pairs<-sum(eligible)
    saveRDS(metadata,file.path(out,"metadata.rds"))
    print(comparison[,c("case","prediction","median_ms_before","median_ms_after","allocation_reduction_percent")],row.names=FALSE)
} else {
    lib<-normalizePath(arg("library"));.libPaths(c(lib,.libPaths()));library(fastglm)
    stopifnot(label%in%c("baseline","candidate"),normalizePath(find.package("fastglm"))==normalizePath(file.path(lib,"fastglm")))
    package<-find.package("fastglm");ns<-asNamespace("fastglm")
    metadata<-list(label=label,installed_path=package,version=as.character(packageVersion("fastglm")),
        R=R.version.string,system=Sys.info(),threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")),
        binary_md5=unname(tools::md5sum(file.path(package,"libs",paste0("fastglm",.Platform$dynlib.ext)))),
        r_database_md5=unname(tools::md5sum(file.path(package,"R","fastglm.rdb"))),
        prediction_function_md5=vapply(functions,function(name)if(exists(name,ns,inherits=FALSE))hash(deparse(get(name,ns))) else NA_character_,character(1)))
    saveRDS(metadata,file.path(out,paste0(label,"-metadata.rds")))
    fixture<-readRDS(file.path(out,"fixtures.rds"));cases<-fixture$cases
    rows<-raw<-accuracy<-list()
    conditions<-data.frame(label=character(),case=character(),prediction=character(),stage=character(),kind=character(),message=character())
    capture<-function(fun,case,prediction,stage) {
        tryCatch(withCallingHandlers(fun(),warning=function(w) {
            conditions[nrow(conditions)+1L,]<<-list(label,case,prediction,stage,"warning",conditionMessage(w));invokeRestart("muffleWarning")
        }),error=function(e) {
            conditions[nrow(conditions)+1L,]<<-list(label,case,prediction,stage,"error",conditionMessage(e));e
        })
    }
    modes<-c("probability","cumulative","class","probability_se","cumulative_se")
    for(i in seq_len(nrow(cases))) {
        ca<-cases[i,];fit<-fixture$fits[[i]];set.seed(ca$seed+10000L)
        x<-matrix(rnorm(ca$n*ca$p,sd=.5),ca$n,dimnames=list(NULL,paste0("x",seq_len(ca$p))))
        offset<-runif(ca$n,-.2,.2);input.hash<-hash(list(x=x,offset=offset))
        for(mode in modes) {
            type<-sub("_se$","",mode);se<-grepl("_se$",mode)
            run<-function()predict(fit,x,offset=offset,type=type,se.fit=se)
            value<-capture(run,ca$case,mode,"preflight")
            baseline.file<-file.path(work,paste0(ca$case,"-",mode,".rds"))
            valid<-!inherits(value,"error");max.fit<-max.se<-NA_real_
            if(valid && label=="baseline") {saveRDS(value,baseline.file);reference<-value} else
                reference<-if(file.exists(baseline.file))readRDS(baseline.file) else NULL
            if(valid) {
                valid<-!is.null(reference)&&isTRUE(all.equal(value,reference,tolerance=2e-12,check.attributes=TRUE))
                if(type=="class") max.fit<-if(identical(value,reference))0 else Inf else {
                    a<-if(se)value$fit else value;b<-if(se)reference$fit else reference
                    max.fit<-max(abs(a-b));if(se)max.se<-max(abs(value$se.fit-reference$se.fit))
                }
                valid<-valid && is.finite(max.fit) && max.fit<=2e-12 && (!se || (is.finite(max.se)&&max.se<=2e-12))
            }
            row<-data.frame(label,case=ca$case,n=ca$n,p=ca$p,K=ca$K,prediction=mode,
                status=if(valid)"eligible" else "accuracy failure",median_ms=NA_real_,allocated_bytes=NA_real_,
                allocations=NA_integer_,largest_allocation_bytes=NA_real_,output_bytes=if(inherits(value,"error"))NA_real_ else as.numeric(object.size(value)))
            accuracy[[length(accuracy)+1L]]<-data.frame(label,case=ca$case,prediction=mode,input_md5=input.hash,
                passed=valid,max_fit_difference=max.fit,max_se_difference=max.se)
            if(valid) {
                timing<-capture(function()microbenchmark::microbenchmark(run(),times=ca$repetitions)$time/1e6,ca$case,mode,"timing")
                if(!inherits(timing,"error")) {
                    row$median_ms<-median(timing)
                    raw[[length(raw)+1L]]<-data.frame(label,case=ca$case,prediction=mode,repetition=seq_along(timing),elapsed_ms=timing)
                    gc();path<-file.path(work,paste0(label,"-",ca$case,"-",mode,"-allocations.out"))
                    Rprofmem(path);allocation.result<-capture(run,ca$case,mode,"allocation");Rprofmem(NULL)
                    if(!inherits(allocation.result,"error")) {
                        lines<-readLines(path,warn=FALSE);bytes<-as.numeric(sub(" .*","",lines[grepl("^[0-9]+ :",lines)]))
                        row$allocated_bytes<-sum(bytes);row$allocations<-length(bytes)
                        row$largest_allocation_bytes<-if(length(bytes))max(bytes) else 0
                        row$status<-"measured"
                    } else row$status<-"allocation failure"
                } else row$status<-"timing failure"
            }
            rows[[length(rows)+1L]]<-row
            write.csv(do.call(rbind,rows),file.path(out,paste0(label,"-summary.csv")),row.names=FALSE)
            write.csv(do.call(rbind,accuracy),file.path(out,paste0(label,"-accuracy.csv")),row.names=FALSE)
            write.csv(conditions,file.path(out,paste0(label,"-conditions.csv")),row.names=FALSE)
            if(length(raw))write.csv(do.call(rbind,raw),file.path(out,paste0(label,"-raw-timings.csv")),row.names=FALSE)
        }
        cat(label,ca$case,"complete\n")
    }
    capture.output(sessionInfo(),file=file.path(out,paste0(label,"-session-info.txt")))
}
