# Run from the repository root. Reference package version is intentionally pinned:
# Rscript benchmarks/benchmark-ordinal.R --library=/path/to/fastglm-library \
#   --reference-library=/path/to/WeightIt-2.1.0-library --output=/path/to/new-results
# Four balanced repetitions, each timing ten fits, follow accuracy checks.
options(warn=1)
args<-commandArgs(TRUE)
argument<-function(name,default="") {
    value<-args[startsWith(args,paste0("--",name,"="))]
    if(length(value)) sub(paste0("^--",name,"="),"",value[1L]) else default
}
out<-argument("output","benchmarks/ordinal-results")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(out,"metadata.rds"))) stop("use a new output directory to preserve earlier results")
lib<-argument("library");reference.lib<-argument("reference-library")
.libPaths(c(c(lib,reference.lib)[nzchar(c(lib,reference.lib))],.libPaths()))
repetitions<-as.integer(argument("repetitions","4"));batch<-as.integer(argument("batch","10"))
stopifnot(length(repetitions)==1L,!is.na(repetitions),repetitions>=2L,repetitions%%2L==0L,
          length(batch)==1L,!is.na(batch),batch>=1L)
Sys.setenv(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
conditions<-data.frame(case=character(),implementation=character(),stage=character(),
                       run=integer(),kind=character(),message=character())
capture<-function(fun,case,implementation,stage,run=0L) {
    messages<-character();error<-""
    value<-tryCatch(withCallingHandlers(fun(),warning=function(w) {
        messages<<-c(messages,conditionMessage(w));invokeRestart("muffleWarning")
    }),error=function(e){error<<-conditionMessage(e);NULL})
    for(j in seq_along(messages)) conditions[nrow(conditions)+1L,]<<-
        list(case,implementation,stage,run,"warning",messages[j])
    if(nzchar(error)) conditions[nrow(conditions)+1L,]<<-list(case,implementation,stage,run,"error",error)
    list(value=value,warnings=messages,error=error)
}
for(package in c("fastglm","MASS","WeightIt")) {
    loaded<-capture(function()loadNamespace(package),"",package,"startup")
    if(nzchar(loaded$error)) {
        write.csv(conditions,file.path(out,"conditions.csv"),row.names=FALSE)
        stop(loaded$error)
    }
}
stopifnot(as.character(packageVersion("WeightIt"))=="2.1.0")
if(nzchar(lib)) stopifnot(normalizePath(find.package("fastglm"))==normalizePath(file.path(lib,"fastglm")))
if(nzchar(reference.lib)) stopifnot(normalizePath(find.package("WeightIt"))==normalizePath(file.path(reference.lib,"WeightIt")))
weightit.fit<-getFromNamespace(".ordinal_weightit.fit","WeightIt")
coefficient.tol<-1e-6;probability.tol<-2e-7
cases<-expand.grid(n=c(2000L,20000L),method=c("ML","meanBR"),stringsAsFactors=FALSE)
cases$p<-3L;cases$K<-4L;cases$seed<-84200L+cases$n
cases$case<-paste(cases$method,cases$n,sep="-")
metadata<-list(timestamp.utc=format(Sys.time(),tz="UTC",usetz=TRUE),cases=cases,
    repetitions=repetitions,batch=batch,coefficient.tolerance=coefficient.tol,
    probability.tolerance=probability.tol,
    controls=list(fastglm=list(tol=1e-8,maxit=200L,block.size=2048L),
                  MASS=list(Hess=FALSE,model=TRUE,maxit=1000L,reltol=1e-12),
                  WeightIt=list(hess=FALSE,maxit=1000L,reltol=1e-12,br.maxit=400L,br.tol=1e-10)),
    versions=vapply(c("fastglm","MASS","WeightIt"),function(z)as.character(packageVersion(z)),character(1)),
    installed.paths=vapply(c("fastglm","MASS","WeightIt"),find.package,character(1)),
    system=Sys.info(),R=R.version.string,RNG.kind=RNGkind(),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")),
    protocol="ML: public formula interfaces. meanBR: fastglm public matrix interface versus released WeightIt internal matrix fitter. fastglm predict() is included inside the timed call so all methods produce fitted category probabilities. Accuracy before timing; alternating implementation order; every timed fit checked outside the clock. No timing summary if any fit fails, warns, or misses accuracy checks.")
hashes<-data.frame(kind=character(),path=character(),md5=character())
add.hash<-function(paths,kind) {
    paths<-paths[file.exists(paths)]
    if(length(paths)) hashes<<-rbind(hashes,data.frame(kind=kind,path=paths,md5=unname(tools::md5sum(paths))))
}
add.hash(c("benchmarks/benchmark-ordinal.R","DESCRIPTION","NAMESPACE","R/ordinal.R",
           "R/ordinal-core.R","R/ordinal-categories.R","R/RcppExports.R",
           "src/fit_ordinal.cpp","src/RcppExports.cpp","src/Makevars"),"repository source")
for(package in names(metadata$versions)) {
    path<-find.package(package)
    add.hash(c(file.path(path,"DESCRIPTION"),file.path(path,"R",paste0(package,".rdb")),
               list.files(file.path(path,"libs"),full.names=TRUE,recursive=TRUE)),paste("installed",package))
}
saveRDS(metadata,file.path(out,"metadata.rds"))
write.csv(hashes,file.path(out,"hashes.csv"),row.names=FALSE)
write.csv(cases,file.path(out,"cases.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"session-info.txt"))
validation<-list();timings<-list();estimates<-list();summaries<-list()
fastglm.probabilities<-function(fit) {
    # The references produce fitted probabilities during fitting. Include that
    # output in the timed fastglm call as well, through its public predict method.
    fit$benchmark.probability<-predict(fit)
    fit
}
save.results<-function() {
    write.csv(conditions,file.path(out,"conditions.csv"),row.names=FALSE)
    for(name in c("validation","timings","summaries")) {
        value<-get(name)
        if(length(value)) write.csv(do.call(rbind,value),file.path(out,paste0(name,".csv")),row.names=FALSE)
    }
    saveRDS(estimates,file.path(out,"estimates.rds"))
}

for(k in seq_len(nrow(cases))) {
    case<-cases[k,];set.seed(case$seed)
    x<-matrix(rnorm(case$n*3L),case$n,3L,dimnames=list(NULL,paste0("x",1:3)))
    off<-runif(case$n,-.2,.2);w<-rep(c(0,1,2,1),length.out=case$n)
    y<-ordered(findInterval(drop(x%*%c(.5,-.4,.3))+off+rlogis(case$n),c(-1,0,1))+1L,levels=1:4)
    data<-data.frame(y,x,off=off,w=w)
    reference<-if(case$method=="ML") "MASS::polr" else "WeightIt2.1.0 internal"
    if(case$method=="ML") procedures<-list(
        fastglm=function()fastglm.probabilities(fastglm::fastglm_ordinal(y~x1+x2+x3+offset(off),data=data,weights=w,
            firth=FALSE,tol=1e-8,maxit=200L,block.size=2048L)),
        reference=function()MASS::polr(y~x1+x2+x3+offset(off),data=data,weights=w,
            method="logistic",Hess=FALSE,model=TRUE,control=list(maxit=1000L,reltol=1e-12)))
    else procedures<-list(
        fastglm=function()fastglm.probabilities(fastglm::fastglm_ordinal(x,y,weights=w,offset=off,
            firth=TRUE,tol=1e-8,maxit=200L,block.size=2048L)),
        reference=function()weightit.fit(x,y,weights=w,offset=off,link="logit",br=TRUE,hess=FALSE,
            control=list(maxit=1000L,reltol=1e-12,br.maxit=400L,br.tol=1e-10)))
    names(procedures)<-c("fastglm",reference)
    parameter.names<-c(paste(levels(y)[-4L],levels(y)[-1L],sep="|"),colnames(x))
    # WeightIt does not return the BR solver flag. Reconstruct its published
    # stopping criterion from the returned adjusted score on its scaled design.
    inspect<-function(fit,implementation) {
        if(implementation=="fastglm") {
            b<-c(fit$alpha,fit$beta);pr<-fit$benchmark.probability;ok<-isTRUE(fit$converged)
            score<-fit$scaled.score;code<-fit$convergence.reason
        } else if(case$method=="ML") {
            b<-c(fit$zeta,coef(fit));pr<-predict(fit,type="probs")
            ok<-identical(as.integer(fit$convergence),0L);score<-NA_real_;code<-as.character(fit$convergence)
        } else {
            b<-fit$coefficients[parameter.names];pr<-fit$fitted.values
            stopifnot(identical(colnames(fit$gradient),c(colnames(x),parameter.names[1:3])),
                      identical(as.numeric(fit$weights),as.numeric(w)))
            score<-max(abs(colSums(fit$gradient)/c(apply(x,2,sd),rep(1,3))))
            ok<-is.finite(score) && score<=1e-10*max(1,sum(w)) && min(pr)>1e-10
            code<-paste0("BR score/usable-probability check; initial ML code ",fit$solve$convergence)
        }
        b<-setNames(as.numeric(b),parameter.names);pr<-as.matrix(pr)
        cumulative<-plogis(outer(-drop(x%*%b[4:6])-off,b[1:3],"+"))
        independent<-cbind(cumulative,1)-cbind(0,cumulative)
        list(coefficients=b,probability=pr,converged=ok,score=score,code=code,
             probability.formula.error=max(abs(pr-independent)))
    }
    assess<-function(result,implementation,stage,run,target=NULL) {
        checked<-if(is.null(result$value)) list(value=NULL,error=result$error,warnings=character()) else
            capture(function()inspect(result$value,implementation),case$case,implementation,paste0(stage," validation"),run)
        a<-checked$value;error<-paste(c(result$error,checked$error)[nzchar(c(result$error,checked$error))],collapse="; ")
        coefficient.error<-probability.error<-NA_real_
        if(!is.null(a) && !is.null(target)) {
            coefficient.error<-max(abs(a$coefficients-target$coefficients))
            probability.error<-max(abs(a$probability-target$probability))
        }
        valid<-!is.null(a) && isTRUE(a$converged) && all(is.finite(a$coefficients)) &&
            all(is.finite(a$probability)) && all(diff(a$coefficients[1:3])>0) &&
            is.finite(a$probability.formula.error) && a$probability.formula.error<1e-12 &&
            (is.null(target) || (coefficient.error<=coefficient.tol && probability.error<=probability.tol))
        validation[[length(validation)+1L]]<<-data.frame(case=case$case,implementation,stage,run,
            converged=if(is.null(a)) FALSE else a$converged,valid=valid,
            warnings=length(result$warnings)+length(checked$warnings),coefficient.error,probability.error,
            probability.formula.error=if(is.null(a)) NA_real_ else a$probability.formula.error,
            score=if(is.null(a)) NA_real_ else a$score,convergence.code=if(is.null(a)) "error" else a$code,error=error)
        list(value=a,valid=valid,clean=valid && !length(result$warnings) && !length(checked$warnings))
    }
    warm<-lapply(names(procedures),function(name)capture(procedures[[name]],case$case,name,"preflight"))
    base<-assess(warm[[1L]],names(procedures)[1L],"preflight",0L)
    other<-assess(warm[[2L]],names(procedures)[2L],"preflight",0L,base$value)
    eligible<-base$clean && other$clean
    estimates[[case$case]]<-lapply(list(base$value,other$value),function(z)
        if(is.null(z)) NULL else z[c("coefficients","converged","score","code","probability.formula.error")])
    names(estimates[[case$case]])<-names(procedures)
    if(eligible) for(repetition in seq_len(repetitions)) {
        order<-if(repetition%%2L) 1:2 else 2:1
        for(position in seq_along(order)) {
            name<-names(procedures)[order[position]];gc()
            elapsed<-system.time(runs<-lapply(seq_len(batch),function(j)
                capture(procedures[[name]],case$case,name,"timed fit",(repetition-1L)*batch+j)))[["elapsed"]]
            checked<-lapply(seq_len(batch),function(j)assess(runs[[j]],name,"timed fit",
                                                           (repetition-1L)*batch+j,base$value))
            timings[[length(timings)+1L]]<-data.frame(case=case$case,implementation=name,repetition,
                position,batch,elapsed.batch.seconds=elapsed,seconds.per.fit=elapsed/batch,
                all.valid=all(vapply(checked,function(z)z$valid,logical(1))),
                all.clean=all(vapply(checked,function(z)z$clean,logical(1))))
            rm(runs,checked);save.results()
        }
    }
    raw<-if(length(timings)) do.call(rbind,timings) else NULL
    for(name in names(procedures)) {
        rows<-if(is.null(raw)) NULL else raw[raw$case==case$case & raw$implementation==name,,drop=FALSE]
        valid<-eligible && nrow(rows)==repetitions && all(rows$all.clean)
        summaries[[length(summaries)+1L]]<-data.frame(case,implementation=name,
            eligible=valid,repetitions.completed=if(is.null(rows)) 0L else nrow(rows),
            median.seconds=if(valid) median(rows$seconds.per.fit) else NA_real_,
            min.seconds=if(valid) min(rows$seconds.per.fit) else NA_real_,
            max.seconds=if(valid) max(rows$seconds.per.fit) else NA_real_)
    }
    save.results();cat(case$case,"complete; preflight eligible:",eligible,"\n")
}
capture.output(sessionInfo(),file=file.path(out,"session-info.txt"))
print(do.call(rbind,summaries),row.names=FALSE)
