# Run from the repository root:
# Rscript benchmarks/benchmark-streaming-qr.R --library=/path/to/library
# Optional memory check (source rows are generated one block at a time):
# /usr/bin/time -l Rscript benchmarks/benchmark-streaming-qr.R --library=... --memory-n=1000000
options(warn=1)
args <- commandArgs(TRUE)
argument <- function(name,default=NULL) {
    value <- args[startsWith(args,paste0("--",name,"="))]
    if(length(value)) sub(paste0("^--",name,"="),"",value[1L]) else default
}
lib <- argument("library","")
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
library(fastglm)
stopifnot(!nzchar(lib) || normalizePath(find.package("fastglm"))==normalizePath(file.path(lib,"fastglm")))
out <- argument("output","benchmarks/streaming-qr-results")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
memory_n <- argument("memory-n")
# Preserve provenance of separately measured memory results when fitting times
# are refreshed. Those measurements must not inherit the new binary's hashes.
memory_provenance <- file.path(out,"memory-metadata.rds")
if(file.exists(file.path(out,"memory.csv")) && !file.exists(memory_provenance) &&
   file.exists(file.path(out,"metadata.rds"))) {
    old <- readRDS(file.path(out,"metadata.rds"))
    saveRDS(list(timestamp_utc=old$timestamp_utc,library=old$memory_library,
        native_sha256=old$memory_native_sha256,
        note=paste("Earlier, separately measured ordinary Gaussian QR memory results;",
                   "not remeasured with the current Firth stopping implementation.",old$memory_note)),
        memory_provenance)
}
metadata <- list(timestamp_utc=format(Sys.time(),tz="UTC",usetz=TRUE),
    source_commit=system2("git",c("rev-parse","HEAD"),stdout=TRUE),
    version=as.character(packageVersion("fastglm")),installed_path=find.package("fastglm"),
    R=R.version.string,platform=Sys.info()[c("sysname","release","machine")],
    threads=Sys.getenv(c("OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")),
    source_md5=tools::md5sum(c("inst/include/streaming_qr.h","src/fit_streaming_glm.cpp","R/streaming.R")),
    script_md5=unname(tools::md5sum("benchmarks/benchmark-streaming-qr.R")),
    installed_binary_md5=unname(tools::md5sum(file.path(find.package("fastglm"),"libs",paste0("fastglm",.Platform$dynlib.ext)))),
    installed_r_database_md5=unname(tools::md5sum(file.path(find.package("fastglm"),"R","fastglm.rdb"))),
    protocol="Full inference; tol=1e-8; maxit=100; three timed repetitions after accuracy/convergence evaluation. Failed and nonconverged fits remain in the summary with no runtime claim. QR reference uses a well-conditioned reparameterization for near-collinear cases.",
    stopping_criteria="QR Firth verifies the remaining fitted-predictor step, joint adjusted-score RMS, and dispersion stationarity. LLT Firth and the dense reference retain their coefficient-based stopping rules. Equal tolerance values therefore do not imply identical stopping criteria.",
    memory_provenance=if(file.exists(memory_provenance)) basename(memory_provenance) else NULL)
metadata_file <- file.path(out,if(is.null(memory_n)) "metadata.rds" else paste0("memory-",memory_n,"-metadata.rds"))
saveRDS(metadata,metadata_file)
if(!is.null(memory_n)) {
    n <- as.integer(memory_n); block <- 2000L; p <- 8L
    reader <- function(k) {
        first <- (k-1L)*block+1L; m <- min(block,n-first+1L)
        set.seed(91500L+k)
        X <- cbind(1,matrix(rnorm(m*(p-1L)),m))
        list(X=X,y=.3+rowSums(X[,-1L,drop=FALSE])*.1+rnorm(m))
    }
    elapsed <- system.time(fit <- fastglm_streaming(reader,ceiling(n/block),method="qr"))["elapsed"]
    stopifnot(fit$converged)
    write.csv(data.frame(n,p,block,elapsed_seconds=unname(elapsed),passes=fit$passes),
              file.path(out,paste0("memory-",n,".csv")),row.names=FALSE)
} else {
    stopifnot(requireNamespace("microbenchmark",quietly=TRUE))
    cases <- expand.grid(n=c(1000L,10000L,100000L),p=c(6L,24L),
                         model=c("Gaussian","Logistic","Firth Gamma"),stringsAsFactors=FALSE)
    cases$condition <- "well conditioned"
    extra <- data.frame(n=2000L,p=6L,model=c("Gaussian","Logistic","Firth logistic","Firth Gamma"),
                        condition="near collinear")
    cases <- rbind(cases,extra)
    rows <- list();raw <- list();warning_rows <- list()
    scalar <- function(x,default=NA_real_) if(is.null(x) || length(x)!=1L) default else unname(x)
    save_warnings <- function(messages,case,method,stage) {
        for(message in messages) warning_rows[[length(warning_rows)+1L]] <<-
            data.frame(case,method,stage,message,row.names=NULL)
        empty <- data.frame(n=integer(),p=integer(),model=character(),condition=character(),
                            method=character(),stage=character(),message=character())
        write.csv(if(length(warning_rows)) do.call(rbind,warning_rows) else empty,
                  file.path(out,"warnings.csv"),row.names=FALSE)
    }
    for(k in seq_len(nrow(cases))) {
        case <- cases[k,];set.seed(91520L+k)
        n<-case$n;p<-case$p;Z<-cbind(1,matrix(rnorm(n*(p-1L),sd=.3),n));X<-Z
        transform<-diag(p)
        if(case$condition=="near collinear") {
            X[,p]<-Z[,p-1L]+1e-7*Z[,p]
            transform[p-1L,p]<-1;transform[p,p]<-1e-7
        }
        offset<-runif(n,-.05,.05);w<-rep(c(.5,1,2),length.out=n)
        family<-if(grepl("logistic|Logistic",case$model)) binomial() else
            if(case$model=="Firth Gamma") Gamma("log") else gaussian()
        if(family$family=="binomial") w<-rep(c(1,1,2),length.out=n)
        w[seq_len(min(3,n))]<-0
        eta<-.3+rowSums(Z[,-1L,drop=FALSE])*.12+offset
        y<-switch(family$family,gaussian=eta+rnorm(n,sd=.5),
            binomial=rbinom(n,1,plogis(eta)),Gamma=rgamma(n,shape=2,scale=exp(eta)/2))
        firth<-grepl("Firth",case$model)
        reference_warnings <- character()
        reference <- tryCatch(withCallingHandlers(
            fastglm(Z,y,family,weights=w,offset=offset,method=0,firth=firth,tol=1e-8),
            warning=function(w) {
                reference_warnings <<- c(reference_warnings,conditionMessage(w))
                invokeRestart("muffleWarning")
            }),error=identity)
        save_warnings(reference_warnings,case,"dense QR reference","reference")
        reference_ok <- !inherits(reference,"error") && isTRUE(reference$converged)
        eta_reference<-if(reference_ok) drop(Z%*%coef(reference))+offset else NULL
        inverse_transform<-solve(transform)
        covariance_reference<-if(reference_ok) inverse_transform%*%reference$cov.unscaled%*%t(inverse_transform) else NULL
        block<-min(2000L,n);groups<-split(seq_len(n),ceiling(seq_len(n)/block))
        reader<-function(j){i<-groups[[j]];list(X=X[i,,drop=FALSE],y=y[i],weights=w[i],offset=offset[i])}
        for(method in c("llt","qr")) {
            warnings<-character()
            run<-function()fastglm_streaming(reader,length(groups),family,method=method,firth=firth,tol=1e-8)
            fit<-tryCatch(withCallingHandlers(run(),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),error=identity)
            row<-data.frame(case,method,seed=91520L+k,block,converged=FALSE,status="error",iterations=NA_integer_,
                convergence_reason="",reference_converged=reference_ok,
                linear_predictor_error=NA_real_,covariance_relative_error=NA_real_,dispersion_relative_error=NA_real_,
                standardized_adjusted_score=NA_real_,remaining_predictor_step=NA_real_,adjusted_score_rms=NA_real_,
                remaining_dispersion_step=NA_real_,adjusted_dispersion_score_rms=NA_real_,
                refined_convergence_verification=NA,convergence_certificate_passed=NA,
                accuracy_passed=NA,median_ms=NA_real_,passes=NA_integer_,
                timing_warnings="",message="",row.names=NULL)
            if(inherits(fit,"error")) row$message<-conditionMessage(fit) else {
                eta_fit<-if(method=="qr") fastglm:::streaming_qr_eta(X,coef(fit),offset) else drop(X%*%coef(fit))+offset
                row$converged<-fit$converged
                row$convergence_reason<-scalar(fit$convergence.reason,if(fit$converged) "converged" else "not converged")
                row$status<-if(fit$converged) "converged" else row$convergence_reason
                row$iterations<-fit$iter;row$passes<-fit$passes
                if(reference_ok) {
                    row$linear_predictor_error<-max(abs(eta_fit-eta_reference))
                    row$covariance_relative_error<-max(abs(fit$cov.unscaled-covariance_reference))/max(abs(covariance_reference))
                    row$dispersion_relative_error<-abs(fit$dispersion-reference$dispersion)/reference$dispersion
                }
                if(firth) row$standardized_adjusted_score<-max(abs(fit$standardized.adjusted.score))
                row$remaining_predictor_step<-scalar(fit$remaining.predictor.step)
                row$adjusted_score_rms<-scalar(fit$adjusted.score.rms)
                row$remaining_dispersion_step<-scalar(fit$remaining.dispersion.step)
                row$adjusted_dispersion_score_rms<-scalar(fit$adjusted.dispersion.score.rms)
                row$refined_convergence_verification<-scalar(fit$refined.convergence.verification,NA)
                if(method=="qr" && firth) {
                    certificate<-c(row$remaining_predictor_step,row$adjusted_score_rms)
                    if(family$family=="Gamma") certificate<-c(certificate,row$remaining_dispersion_step,
                                                            row$adjusted_dispersion_score_rms)
                    row$convergence_certificate_passed<-all(is.finite(certificate)) && all(certificate<=1e-8)
                }
                row$message<-paste(unique(warnings),collapse="; ")
                # Near collinearity magnifies coefficient-coordinate error;
                # predictions and covariance are the meaningful comparisons.
                row$accuracy_passed <- all(is.finite(c(row$linear_predictor_error,
                    row$covariance_relative_error,row$dispersion_relative_error))) &&
                    row$linear_predictor_error<2e-5 && row$covariance_relative_error<2e-4 &&
                    row$dispersion_relative_error<2e-5
                if(!reference_ok) {
                    row$status<-"reference failure"
                    row$message<-paste(c(row$message,if(inherits(reference,"error")) conditionMessage(reference) else
                        "Dense reference did not converge; accuracy and timing were not evaluated."),collapse="; ")
                } else if(!row$accuracy_passed) {
                    row$status <- "accuracy failure"
                    row$message <- paste(c(row$message,"Accuracy check failed; inspect the saved error columns."),collapse="; ")
                }
                if(fit$converged && identical(row$convergence_certificate_passed,FALSE)) {
                    row$status<-"convergence certificate failure"
                    row$message<-paste(c(row$message,"Reported convergence did not pass the saved exact-current criteria."),collapse="; ")
                }
                if(fit$converged && row$accuracy_passed && !identical(row$convergence_certificate_passed,FALSE)) {
                    timing_warnings <- character()
                    timing <- tryCatch(withCallingHandlers(
                        microbenchmark::microbenchmark(run(),times=3L)$time/1e6,
                        warning=function(w) {
                            timing_warnings <<- c(timing_warnings,conditionMessage(w))
                            invokeRestart("muffleWarning")
                        }),error=identity)
                    row$timing_warnings <- paste(timing_warnings,collapse="; ")
                    if(inherits(timing,"error")) {
                        row$status<-"timing failure"
                        row$message<-paste(c(row$message,conditionMessage(timing)),collapse="; ")
                    } else {
                        row$median_ms<-median(timing)
                        raw[[length(raw)+1L]]<-data.frame(case,method,repetition=seq_along(timing),elapsed_ms=timing,row.names=NULL)
                    }
                }
            }
            save_warnings(warnings,case,method,"preflight")
            if(nzchar(row$timing_warnings)) save_warnings(timing_warnings,case,method,"timing")
            rows[[length(rows)+1L]]<-row
            write.csv(do.call(rbind,rows),file.path(out,"summary.csv"),row.names=FALSE)
            if(length(raw))write.csv(do.call(rbind,raw),file.path(out,"raw-timings.csv"),row.names=FALSE)
        }
        cat(case$model,case$n,case$p,case$condition,"complete\n")
    }
    summary <- do.call(rbind,rows)
    metadata$completed_utc <- format(Sys.time(),tz="UTC",usetz=TRUE)
    metadata$result_counts <- table(summary$status)
    metadata$warning_counts <- if(length(warning_rows)) table(do.call(rbind,warning_rows)$stage) else integer()
    metadata$warning_note <- "All reference, preflight-fit, and timing warnings are attributed by case and stage in warnings.csv. Timer precision notices are not fitting failures."
    saveRDS(metadata,metadata_file)
    near <- summary[summary$condition=="near collinear" & summary$method=="qr",]
    interpretation <- c(
        sprintf("The matched 44-cell panel contains %d converged and timed fits; %d returned fits pass the accuracy checks.",
                sum(is.finite(summary$median_ms)),sum(summary$accuracy_passed,na.rm=TRUE)),
        "Failures remain in summary.csv with their messages and without timing claims.",
        "The near-collinear QR cases are retained individually:",
        vapply(seq_len(nrow(near)),function(i) sprintf("  %s: %s, %d iterations, predictor error %.3g.",
            near$model[i],near$status[i],near$iterations[i],near$linear_predictor_error[i]),character(1)),
        metadata$stopping_criteria,
        "These are three-repetition timings on one machine. They describe these designs and do not establish a universal solver ranking.",
        if(file.exists(memory_provenance)) "memory.csv retains earlier ordinary-Gaussian measurements with separate provenance in memory-metadata.rds; it was not rerun in this timing refresh.")
    writeLines(interpretation,file.path(out,"interpretation.txt"))
    capture.output(sessionInfo(),file=file.path(out,"session-info.txt"))
}
