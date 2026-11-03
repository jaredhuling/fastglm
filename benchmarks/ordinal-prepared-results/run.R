# Run baseline first, then candidate, using the same output directory.
# Rscript run.R baseline /path/to/baseline-library /path/to/output
# Rscript run.R candidate /path/to/candidate-library /path/to/output
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=3L || !args[1L]%in%c('baseline','candidate'))
    stop('usage: Rscript run.R baseline|candidate library output-directory')
mode<-args[1L];lib<-normalizePath(args[2L]);root<-args[3L]
script<-sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1L])
fixture<-file.path(dirname(normalizePath(script)),'fixtures.rds')
dir.create(root,recursive=TRUE,showWarnings=FALSE)
if(file.exists(file.path(root,paste0('benchmark-',mode,'.rds'))))
    stop('output already exists; use a fresh output directory')
.libPaths(c(lib,.libPaths()));library(fastglm)
cases<-readRDS(fixture)
reference<-if(mode=='candidate')readRDS(file.path(root,'benchmark-baseline.rds'))
measure<-function(fn,repetitions=3L) {
    elapsed<-numeric(repetitions)
    for(j in seq_len(repetitions)) {gc();elapsed[j]<-system.time(value<-fn())[['elapsed']]}
    path<-tempfile(tmpdir=root);Rprofmem(path)
    value<-tryCatch(fn(),finally=Rprofmem(NULL))
    lines<-readLines(path,warn=FALSE)
    bytes<-suppressWarnings(as.numeric(sub(' .*','',lines)))
    list(seconds=elapsed,median.seconds=median(elapsed),R.allocation.bytes=sum(bytes,na.rm=TRUE),
         memory.profile=path)
}
results<-list();rows<-list()
for(i in seq_along(cases)) for(br in c(FALSE,TRUE)) {
    d<-cases[[i]];key<-paste(d$n,d$p,d$K,if(br)'BR'else'ML',sep='_')
    direct<-function()lapply(d$y,function(y)fastglm_ordinal(d$x,y,weights=d$weights,
        offset=d$offset,firth=br))
    check<-direct()
    stopifnot(all(vapply(check,function(f)isTRUE(f$converged),logical(1))))
    error<-0
    if(mode=='candidate') {
        baseline<-reference$results[[key]]$estimates
        error<-max(vapply(seq_along(check),function(j)
            max(abs(coef(check[[j]])-baseline[[j]]$coefficients),
                abs(vcov(check[[j]])-baseline[[j]]$covariance)),numeric(1)))
        stopifnot(error<1e-10)
        design<-fastglm_ordinal_prepare(d$x,weights=d$weights,offset=d$offset,firth=br)
        initial<-fastglm_ordinal(design,d$y[[1L]])
        reused<-function()fastglm_ordinal_batch(design,d$y)
        prepared.check<-reused()
        error<-max(error,max(vapply(seq_along(check),function(j)
            max(abs(coef(check[[j]])-coef(prepared.check[[j]])),
                abs(vcov(check[[j]])-vcov(prepared.check[[j]]))),numeric(1))))
        stopifnot(error<1e-10,design$cache$builds==1L)
        prepare<-function()fastglm_ordinal_prepare(d$x,weights=d$weights,offset=d$offset,firth=br)
        first<-function(){newdesign<-prepare();fastglm_ordinal(newdesign,d$y[[1L]])}
        measurements<-list(direct=measure(direct),warm_reuse=measure(reused),
            first_fit_including_prepare=measure(first),prepare_only=measure(prepare))
        stopifnot(design$cache$builds==1L)
    } else measurements<-list(direct=measure(direct))
    results[[key]]<-list(estimates=lapply(check,function(f)list(coefficients=coef(f),
        covariance=vcov(f),iterations=f$iterations,converged=f$converged,loglik=as.numeric(logLik(f)))),
        error=error,measurements=measurements)
    for(nm in names(measurements)) {
        m<-measurements[[nm]]
        rows[[length(rows)+1L]]<-data.frame(case=key,n=d$n,p=d$p,K=d$K,firth=br,
            mode=mode,operation=nm,responses=if(nm%in%c('direct','warm_reuse'))length(d$y) else 1L,
            median.seconds=m$median.seconds,min.seconds=min(m$seconds),max.seconds=max(m$seconds),
            R.allocation.bytes=m$R.allocation.bytes,max.error=error)
    }
    cat(key,'complete\n');flush.console()
    saveRDS(list(results=results,table=do.call(rbind,rows),session=sessionInfo(),
        library=find.package('fastglm'),dll.md5=tools::md5sum(getLoadedDLLs()[['fastglm']][['path']]),
        R.md5=tools::md5sum(file.path(find.package('fastglm'),'R',c('fastglm.rdb','fastglm.rdx'))),
        fixture.md5=tools::md5sum(fixture)),file.path(root,paste0('benchmark-',mode,'.rds')))
    write.csv(do.call(rbind,rows),file.path(root,paste0('benchmark-',mode,'.csv')),row.names=FALSE)
}
