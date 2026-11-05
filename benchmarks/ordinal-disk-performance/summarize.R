args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=4L) stop("Supply disk results, baseline components, current components, and report filename")
options(width=160)
root<-normalizePath(args[1L]);execution<-read.csv(file.path(root,"execution.csv"))
configuration<-readRDS(file.path(root,"configuration.rds"))
stopifnot(nrow(execution)==nrow(configuration$grid)*configuration$configuration$repetitions*2L,
    !anyDuplicated(execution[c("cell","repetition","version")]),all(execution$exit==0L))
resource<-function(path) {
    lines<-readLines(path,warn=FALSE)
    number<-function(pattern,colon=FALSE) {
        value<-lines[grepl(pattern,lines,fixed=TRUE)]
        if(!length(value)) return(NA_real_)
        if(colon) as.numeric(trimws(sub(".*:","",value[1L]))) else
            suppressWarnings(as.numeric(strsplit(trimws(value[1L]),"[[:space:]]+")[[1L]][1L]))
    }
    linux<-any(grepl("Maximum resident set size",lines,fixed=TRUE))
    list(peak.bytes=if(linux) 1024*number("Maximum resident set size",TRUE) else number("maximum resident set size"),
        input.operations=if(linux) number("File system inputs",TRUE) else number("block input operations"),
        output.operations=if(linux) number("File system outputs",TRUE) else number("block output operations"),
        major.faults=if(linux) number("Major (requiring I/O) page faults",TRUE) else number("page faults"),
        minor.faults=if(linux) number("Minor (reclaiming a frame) page faults",TRUE) else number("page reclaims"))
}
objects<-lapply(execution$result,function(file) readRDS(file.path(root,file)))
for(i in seq_along(objects)) {
    expected.cell<-configuration$grid[execution$cell[i],,drop=FALSE]
    expected.cell$repetition<-execution$repetition[i]
    stopifnot(identical(objects[[i]]$cell,expected.cell),
        identical(objects[[i]]$version,execution$version[i]))
}
expected.provenance<-configuration$package.provenance
if(is.null(expected.provenance)) expected.provenance<-lapply(c("baseline","current"),function(version) {
    first<-objects[[which(execution$version==version)[1L]]]
    list(binary=first$binary.hash,headers=first$header.hashes)
})
matches.provenance<-function(value,expected)
    identical(unname(value$binary.hash),unname(expected$binary)) &&
    identical(unname(value$header.hashes),unname(expected$headers))
rows<-lapply(seq_along(objects),function(i) {
    z<-objects[[i]];r<-resource(file.path(root,sub("[.]rds$",".resources",execution$result[i])))
    data.frame(cell=execution$cell[i],z$cell,version=z$version,status=z$status,
        elapsed=z$elapsed,user=z$user,system=z$system,process.elapsed=execution$process.elapsed[i],
        callback.seconds=z$reader$seconds,reads=z$reader$reads,logical.bytes=z$reader$bytes,
        maximum.block.rows=z$reader$maximum.rows,source.passes=if(is.null(z$source.passes)) NA else z$source.passes,
        peak.bytes=r$peak.bytes,input.operations=r$input.operations,output.operations=r$output.operations,
        major.faults=r$major.faults,minor.faults=r$minor.faults,
        d=if(is.null(z$d)) NA else z$d,retained.design=if(is.null(z$retained.design)) NA else z$retained.design)
})
rows<-do.call(rbind,rows);write.csv(rows,file.path(root,"measurements.csv"),row.names=FALSE)
stopifnot(all(is.finite(rows$peak.bytes) & rows$peak.bytes>0),
    all(is.finite(rows$input.operations) & rows$input.operations>=0),
    all(is.finite(rows$output.operations) & rows$output.operations>=0),
    all(is.finite(rows$major.faults) & rows$major.faults>=0),
    all(is.finite(rows$minor.faults) & rows$minor.faults>=0))
checks<-list()
for(i in seq_len(nrow(configuration$grid))) for(rep in seq_len(configuration$configuration$repetitions)) {
    ids<-which(execution$cell==i & execution$repetition==rep)
    b<-objects[[ids[execution$version[ids]=="baseline"]]]
    a<-objects[[ids[execution$version[ids]=="current"]]]
    same.status<-identical(a$status,b$status)
    same.fit<-if(a$status=="error" || b$status=="error") identical(a$error,b$error) else identical(a$fit,b$fit)
    checks[[length(checks)+1L]]<-data.frame(cell=i,repetition=rep,
        same.status=same.status,same.fit=same.fit,same.reads=identical(a$reader$reads,b$reader$reads),
        same.inputs=identical(a$input.hashes,b$input.hashes),
        different.binary=!identical(unname(a$binary.hash),unname(b$binary.hash)),
        frozen.builds=matches.provenance(b,expected.provenance[[1L]]) &&
            matches.provenance(a,expected.provenance[[2L]]),
        single.thread=length(a$threads)==3L && length(b$threads)==3L &&
            all(a$threads=="1") && all(b$threads=="1"),
        bounded.source=if(a$cell$representation=="source")
            !a$retained.design && !b$retained.design &&
                a$reader$maximum.rows<=a$cell$block && b$reader$maximum.rows<=b$cell$block else TRUE)
}
checks<-do.call(rbind,checks);write.csv(checks,file.path(root,"checks.csv"),row.names=FALSE)
representation.checks<-list()
for(i in which(configuration$grid$representation=="source")) {
    source.cell<-configuration$grid[i,,drop=FALSE]
    match.fields<-c("case","link","firth","model")
    j<-which(configuration$grid$representation=="dense" &
        Reduce(`&`,lapply(match.fields,function(key) configuration$grid[[key]]==source.cell[[key]])))
    stopifnot(length(j)==1L)
    for(rep in seq_len(configuration$configuration$repetitions)) for(version in c("baseline","current")) {
        a<-objects[[which(execution$cell==i & execution$repetition==rep & execution$version==version)]]
        b<-objects[[which(execution$cell==j & execution$repetition==rep & execution$version==version)]]
        normalized<-function(key) {
            av<-a$fit[[key]];bv<-b$fit[[key]]
            if(is.null(av) || is.null(bv) || length(av)!=length(bv)) return(NA_real_)
            max(abs(av-bv))/(1+max(abs(c(av,bv))))
        }
        representation.checks[[length(representation.checks)+1L]]<-data.frame(
            source.cell[match.fields],repetition=rep,version=version,
            same.status=identical(a$status,b$status),
            coefficient.error=normalized("coefficients"),covariance.error=normalized("cov.unscaled"),
            loglik.error=normalized("loglik"))
    }
}
representation.checks<-do.call(rbind,representation.checks)
write.csv(representation.checks,file.path(root,"representations.csv"),row.names=FALSE)
summary<-do.call(rbind,lapply(seq_len(nrow(configuration$grid)),function(i) {
    a<-rows[rows$cell==i & rows$version=="current",];b<-rows[rows$cell==i & rows$version=="baseline",]
    a<-a[order(a$repetition),];b<-b[order(b$repetition),]
    data.frame(configuration$grid[i,],d=a$d[1L],all.ok=all(c(a$status,b$status)=="ok"),
        baseline.ms=1000*median(b$elapsed),current.ms=1000*median(a$elapsed),
        median.paired.ratio=median(a$elapsed/b$elapsed),min.paired.ratio=min(a$elapsed/b$elapsed),
        max.paired.ratio=max(a$elapsed/b$elapsed),baseline.callback.ms=1000*median(b$callback.seconds),
        current.callback.ms=1000*median(a$callback.seconds),reads=a$reads[1L],
        baseline.peak.MiB=median(b$peak.bytes)/1024^2,current.peak.MiB=median(a$peak.bytes)/1024^2)
}))
write.csv(summary,file.path(root,"summary.csv"),row.names=FALSE)
b<-readRDS(file.path(args[2L],"records.rds"));a<-readRDS(file.path(args[3L],"records.rds"))
stopifnot(length(a)==length(b))
components<-do.call(rbind,lapply(seq_along(a),function(i) {
    stopifnot(identical(a[[i]]$cell,b[[i]]$cell),identical(a[[i]]$link,b[[i]]$link),identical(a[[i]]$firth,b[[i]]$firth))
    av<-apply(a[[i]]$seconds,2,median);bv<-apply(b[[i]]$seconds,2,median)
    data.frame(a[[i]]$cell,d=a[[i]]$d,link=a[[i]]$link,firth=a[[i]]$firth,
        operation=names(av),baseline.ms=1000*bv,current.ms=1000*av,ratio=av/bv,
        same.equations=identical(a[[i]]$evaluation,b[[i]]$evaluation),
        evaluator.agreement=a[[i]]$passed && b[[i]]$passed,row.names=NULL)
}))
write.csv(components,file.path(root,"components.csv"),row.names=FALSE)
fmt<-function(x) format(signif(x,4),trim=TRUE)
range.text<-function(x) paste(fmt(range(x,na.rm=TRUE)),collapse=" to ")
report<-capture.output({
    cat("Ordinal row-kernel and genuine disk-source performance\n")
    cat("=====================================================\n\n")
    cat("Generated by benchmarks/ordinal-disk-performance/summarize.R from\n",args[1L],"\n",sep="")
    cat("Baseline: 2619731. This is bounded development evidence, not a calibration\n")
    cat("study or universal speed claim. Protocol/configuration/inputs were fixed\n")
    cat("before timing. All runs used one OpenMP/OpenBLAS/vecLib thread.\n\n")
    if(!all(checks$different.binary)) {
        cat("INVALID NATIVE COMPARISON: the recorded baseline and candidate binaries\n")
        cat("have identical hashes despite different installed candidate headers.\n")
        cat("These whole-fit records cannot establish the native candidate's effect.\n")
        cat("The tables are retained for provenance diagnosis only; the component\n")
        cat("wrappers were compiled separately from their recorded installed headers.\n\n")
    }
    cat("Candidate and numerical evidence\n--------------------------------\n")
    cat("Each compensated row gap is evaluated once; its prior value supplies the\n")
    cat("unchanged left+right neighboring-gap identity. Link-specific logit caches\n")
    cat("are skipped only where unused. Source passes and all statistical equations,\n")
    cat("ordering checks, compensated arithmetic, and convergence criteria are\n")
    cat("unchanged. No observation cache or factorization change was introduced.\n")
    cat("The rebuilt ordinal/display regression suite passed 4075 expectations,\n")
    cat("including the new gap regressions, with zero failures/errors/warnings\n")
    cat("and two expected worker skips. Its log and result object are retained\n")
    cat("under validation/. The earlier purported candidate check used a stale\n")
    cat("binary, as documented below; it is not the patched-binary correctness gate.\n")
    cat(sum(rows$status=="ok"),"of",nrow(rows),"isolated fits converged;",sum(rows$status!="ok"),"failed.\n")
    cat(sum(checks$same.fit),"of",nrow(checks),"baseline/current pairs have bit-identical\n")
    cat("recorded fit state, including covariance, equations, history and diagnostics.\n")
    cat("Identical input/read checks: ",sum(checks$same.inputs & checks$same.reads)," of ",nrow(checks),".\n",sep="")
    cat("Distinct, frozen build checks: ",sum(checks$different.binary & checks$frozen.builds),
        " of ",nrow(checks),".\n",sep="")
    cat("Binary MD5: baseline ",unname(expected.provenance[[1L]]$binary),";\n",sep="")
    cat("candidate ",unname(expected.provenance[[2L]]$binary),".\n",sep="")
    cat("Component equation/evaluator checks: ",all(components$same.equations)," / ",all(components$evaluator.agreement),".\n\n",sep="")
    cat("Across source and dense representations, maximum normalized coefficient,\n")
    cat("model covariance, and log-likelihood differences are ",
        paste(fmt(vapply(representation.checks[c("coefficient.error","covariance.error","loglik.error")],
            max,numeric(1),na.rm=TRUE)),collapse=", "),".\n",sep="")
    cat("These secondary comparisons use max absolute difference divided by one\n")
    cat("plus the largest absolute entry in either result. Raw differences and\n")
    cat("convergence status are retained in representations.csv.\n\n")
    cat("Native-operation timing\n-----------------------\n")
    cat("Fresh component cases isolate operations using immutable installed headers.\n")
    cat("Five complete repetitions are retained; tables show operation medians.\n")
    info<-components$ratio[components$operation=="information"]
    adjustment<-components$ratio[components$operation=="adjustment" & components$firth]
    cat("Measured information-time reduction range: ",range.text(100*(1-info)),"%;\n",sep="")
    cat("meanBR adjustment-time reduction range: ",range.text(100*(1-adjustment)),"%.\n",sep="")
    print(components[components$operation%in%c("information","adjustment") &
        (components$operation!="adjustment" | components$firth),
        c("id","n","d","link","firth","operation","baseline.ms","current.ms","ratio")],row.names=FALSE,digits=4)
    cat("\nWhole-fit workflow timing\n-------------------------\n")
    cat("Times include disk input reads/assembly and fitting, including its default\n")
    cat("model covariance, but no additional post-fit assessment or inference. Each\n")
    cat("repetition has its own R process; version order alternates. Three paired\n")
    cat("repetitions are retained.\n")
    print(summary[,c("case","link","firth","model","representation","d","baseline.ms","current.ms",
        "median.paired.ratio","reads")],row.names=FALSE,digits=4)
    for(representation in c("dense","source")) {
        take<-summary$representation==representation & summary$model=="partial"
        cat("\nPartial",representation,"paired-ratio medians:",range.text(summary$median.paired.ratio[take]),
            "; median across cells",fmt(median(summary$median.paired.ratio[take])),".\n")
    }
    cat("Ratios above one indicate slower measured candidate execution; negative\n")
    cat("gains and control cases are retained. Short timings, three repetitions,\n")
    cat("process/thermal variation and R timing resolution limit small differences.\n\n")
    cat("Disk reads and memory scope\n---------------------------\n")
    cat("The RDS reader retains paths, file sizes, and scalar counters only.\n")
    cat("Input generation runs separately. Dense controls preallocate their full\n")
    cat("inputs in the fitting process; source fits retain no x/y fields and\n")
    cat("process blocks no larger than their declared size. Structural source\n")
    cat("checks passed: ",all(checks$bounded.source),".\n",sep="")
    memory<-summary[summary$case%in%c("regular","large") & summary$model=="partial" &
        summary$link=="logit" & summary$firth,
        c("case","n","block","representation","baseline.callback.ms","current.callback.ms","reads",
            "baseline.peak.MiB","current.peak.MiB")]
    print(memory,row.names=FALSE,digits=5)
    cat("\nThese are median peak process RSS values, including R/package startup,\n")
    cat("input storage, temporary fitting allocations, and the returned model.\n")
    cat("They are not isolated native-allocation measurements. Callback elapsed\n")
    cat("time measures readRDS/deserialization using R's elapsed clock. Logical\n")
    cat("requested bytes, process CPU times, faults and OS I/O counters remain in\n")
    cat("measurements.csv; OS-reported input operation range:",range.text(rows$input.operations),".\n")
    cat("Generation, hash verification, and repeated reads warm the OS cache.\n")
    cat("No cache was flushed. Logical bytes are not physical device traffic;\n")
    cat("zero OS I/O counters do not establish that data never came from files.\n\n")
    cat("Bounded reader/design retention does not establish constant or lower peak\n")
    cat("process RSS. Repeated block reads and validation allocate temporary R\n")
    cat("objects; garbage collection and allocator behavior can affect the process\n")
    cat("peak. This panel measures that total without attributing its components.\n\n")
    cat("Retained operational failure\n----------------------------\n")
    cat("The initial disk-v1 worker saved a successful fit, but the restricted\n")
    cat("macOS resource collector could not query sysctl kern.clockrate and returned\n")
    cat("exit 1. That incomplete attempt remains intact and is excluded from these\n")
    cat("summaries. The same fixed panel was restarted in disk-v2 with permission\n")
    cat("for the resource query; the worker, candidate, inputs, and settings did not\n")
    cat("change. The result parser recognizes the macOS resource labels and rejects\n")
    cat("missing resource records.\n\n")
    cat("Retained build-provenance failure\n--------------------------------\n")
    cat("The complete disk-v2 retry loaded identical baseline and candidate native\n")
    cat("binaries despite distinct installed headers: the package build had reused\n")
    cat("objects after a header-only edit. Its records and invalid report remain\n")
    cat("available but cannot measure the candidate. The two affected translation\n")
    cat("units were then rebuilt. The runner now rejects identical binaries before\n")
    cat("timing, and the result analysis verifies distinct hashes in every pair.\n")
    cat("The separately compiled component panel used the intended headers.\n\n")
    cat("Decision limits\n---------------\n")
    control<-summary$median.paired.ratio[summary$model=="common"]
    cat("The unchanged common-path control ratios range from ",range.text(control),".\n",sep="")
    cat("That variation reinforces why small individual source-fit differences\n")
    cat("should not be attributed to the patch. Across the full partial panel,\n")
    cat("the larger and consistent dense-fit reductions support the row-work\n")
    cat("optimization; source gains are smaller and more sensitive to other costs.\n")
    cat("This candidate reduces row work, not the number of source reads. Whole-fit\n")
    cat("effects must be read alongside component timings and callback costs. The\n")
    cat("component wrapper is a separate translation unit, so its operation ratios\n")
    cat("do not by themselves establish an installed whole-fit speedup. The\n")
    cat("unchanged common path is a process-noise control. Sparse boundary behavior\n")
    cat("is addressed by separate diagnostics and regression tests, not inferred\n")
    cat("from these positive grouped-count timing fixtures.\n")
    cat("A subsequent source-performance investigation should profile validation,\n")
    cat("standardization, block conversion, and allocation alongside readRDS. The\n")
    cat("present panel does not separate those costs, justify weaker source checks,\n")
    cat("or demonstrate a reduction in peak memory from this arithmetic change.\n")
})
writeLines(report,args[4L])
saveRDS(list(configuration=configuration,checks=checks,representation.checks=representation.checks,
    summary=summary,components=components),file.path(root,"analysis.rds"))
if(!all(checks$same.status & checks$same.fit & checks$same.inputs & checks$same.reads &
    checks$different.binary & checks$frozen.builds & checks$single.thread & checks$bounded.source) ||
   !all(components$same.equations & components$evaluator.agreement)) stop("A numerical/provenance check failed; inspect retained records")
cat(paste(report,collapse="\n"),"\n")
