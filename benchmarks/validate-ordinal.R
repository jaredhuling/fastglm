# Development study; run from the repository root. No fitting defaults are tuned.
# Rscript benchmarks/validate-ordinal.R --stage=pilot --library=/path/to/library
# Rscript benchmarks/validate-ordinal.R --stage=run --reps=200 --library=/path/to/library
# Rscript benchmarks/validate-ordinal.R --stage=reference --library=... --reference-library=...
# Rscript benchmarks/validate-ordinal.R --stage=profile --library=...
# Rscript benchmarks/validate-ordinal.R --stage=summarize --run-id=development-v1
# Rscript benchmarks/validate-ordinal.R --stage=audit --run-id=development-v1
options(warn=1)
script.path <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1L])
args <- commandArgs(TRUE)
argument <- function(name,default="") {
    a <- args[startsWith(args,paste0("--",name,"="))]
    if(length(a)) sub(paste0("^--",name,"="),"",a[1L]) else default
}
stage <- argument("stage","pilot")
stopifnot(stage %in% c("pilot","run","reference","profile","summarize","audit"))
root <- argument("output","benchmarks/ordinal-validation")
run.id <- argument("run-id",switch(stage,pilot="pilot-v1",run="development-v1",reference="reference-v1",
                                  profile="profiles-v1",summarize="development-v1",audit="development-v1"))
out <- file.path(root,run.id)
lib <- argument("library"); ref.lib <- argument("reference-library")
source.root <- argument("source-root",".")
.libPaths(c(c(lib,ref.lib)[nzchar(c(lib,ref.lib))],.libPaths()))
Sys.setenv(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
reps <- as.integer(argument("reps",if(stage=="pilot") "3" else "200"))
stopifnot(length(reps)==1L,is.finite(reps),reps>0L)

specification <- function() {
    add <- function(id,n,p,K,rho,alpha,beta,role,regular=TRUE,kind="individual")
        list(id=id,n=n,p=p,K=K,rho=rho,alpha=alpha,beta=beta,role=role,regular=regular,kind=kind)
    list(
        add("ordinary_small",40L,2L,4L,0,c(-1,0,1),c(.5,-.4),"Small-sample bias and Wald approximation"),
        add("ordinary_medium",200L,2L,4L,0,c(-1,0,1),c(.5,-.4),"Reference sampling behavior"),
        add("rare_endpoints",80L,2L,5L,0,c(-3,-.7,.7,3),c(.5,-.4),"Sampling-empty endpoint categories under a finite model"),
        add("rare_interior",80L,2L,5L,0,c(-1,-.98,.6,1.3),c(.5,-.4),"Sampling-empty interior category and collapse frequency"),
        add("empty_interior",80L,2L,5L,0,c(-1,-1,.6,1.3),c(.5,-.4),"Structural zero category; common slopes identified in the reduced model",FALSE),
        add("strong_signal",60L,2L,3L,0,c(-.5,.8),c(3,-2),"Finite-model separation stress; retain empty-cell and directional-separation frequencies"),
        add("deterministic_separation",60L,1L,3L,0,c(-.5,.5),NA_real_,"Deliberate nonregular stress: no finite slope target and no slope coverage interpretation",FALSE,"deterministic"),
        add("correlated",150L,6L,4L,.95,c(-1,0,1),c(.6,-.4,.25,0,0,-.2),"Weak predictor contrasts under high correlation"),
        add("larger_K",300L,4L,8L,.2,qlogis((1:7)/8),c(.4,-.3,.2,.15),"More thresholds and smaller category cells"),
        add("larger_p",300L,12L,4L,.3,c(-1,0,1),c(.6,-.4,.3,.2,rep(0,8)),"More slopes including null effects"),
        add("frequency_offset",100L,3L,4L,.2,c(-1,0,1),c(.5,-.4,.3),"Independent grouped outcomes, effective integer frequencies, zero rows and offsets",TRUE,"grouped"),
        add("empty_endpoint",80L,2L,4L,0,c(-Inf,0,.8),c(.5,-.4),"Structural zero endpoint: nonregular full scale; common-slope diagnostic only",FALSE))
}
scenarios <- specification()
scenario.table <- do.call(rbind,lapply(seq_along(scenarios),function(i) {
    s<-scenarios[[i]]
    data.frame(scenario=s$id,n=s$n,p=s$p,K=s$K,rho=s$rho,alpha=paste(s$alpha,collapse=";"),
               beta=paste(s$beta,collapse=";"),regular_full_model=s$regular,kind=s$kind,
               design_seed=120000L+i,test_seed=130000L+i,role=s$role)
}))
probabilities <- function(x,offset,s) {
    location <- if(s$kind=="deterministic") x[,1L] else drop(x%*%s$beta)+offset
    if(s$kind=="deterministic") {
        y<-findInterval(location,s$alpha)+1L
        z<-matrix(0,nrow(x),s$K);z[cbind(seq_len(nrow(x)),y)]<-1;return(z)
    }
    cumulative<-plogis(outer(-location,s$alpha,"+"))
    pr<-cbind(cumulative,1)-cbind(0,cumulative)
    stopifnot(all(pr>=0),max(abs(rowSums(pr)-1))<1e-14)
    pr
}
make.design <- function(s,seed,n=s$n) {
    set.seed(seed)
    Sigma<-outer(seq_len(s$p),seq_len(s$p),function(i,j)s$rho^abs(i-j))
    x<-matrix(rnorm(n*s$p),n,s$p)%*%chol(Sigma)
    colnames(x)<-paste0("x",seq_len(s$p))
    offset<-if(s$kind=="grouped") runif(n,-.5,.5) else numeric(n)
    list(x=x,offset=offset,probability=probabilities(x,offset,s))
}
make.data <- function(s,design,seed) {
    set.seed(seed);n<-nrow(design$x);K<-s$K
    totals<-if(s$kind=="grouped") rep(c(0L,2L,4L,8L),length.out=n) else rep(1L,n)
    effective<-matrix(0,n,K)
    for(i in seq_len(n)) if(totals[i]>0L)
        effective[i,]<-as.vector(rmultinom(1L,totals[i],design$probability[i,]))
    if(s$kind=="grouped") {
        weights<-rep(c(0,.5,2,4),length.out=n)
        response<-effective
        response[weights>0,]<-response[weights>0,,drop=FALSE]/weights[weights>0]
        colnames(response)<-as.character(seq_len(K))
    } else {
        response<-ordered(max.col(effective,ties.method="first"),levels=seq_len(K))
        weights<-rep(1,n)
    }
    category.total<-colSums(effective);empty<-which(category.total==0)
    # A sufficient directional flag, not a general separation test. Truth is
    # used only for this diagnostic and never supplied to the fitting method.
    direction<-if(s$kind=="deterministic") design$x[,1L] else drop(design$x%*%s$beta)
    separated<-defined<-logical(K-1L)
    for(j in seq_len(K-1L)) {
        lower<-rowSums(effective[,seq_len(j),drop=FALSE])>0
        upper<-rowSums(effective[,(j+1L):K,drop=FALSE])>0
        defined[j]<-any(lower)&&any(upper)
        separated[j]<-defined[j]&&max(direction[lower])<min(direction[upper])
    }
    list(x=design$x,y=response,weights=weights,offset=design$offset,effective=effective,
         empty=empty,empty.ends=sum(empty %in% c(1L,K)),empty.interior=sum(!empty %in% c(1L,K)),
         separated.splits=sum(separated),all.splits.separated=all(defined & separated),frequency=sum(totals))
}
serial.hash <- function(x) {
    file<-tempfile();on.exit(unlink(file));saveRDS(x,file,version=2);unname(tools::md5sum(file))
}
mean.safe <- function(x) if(length(x)&&any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
mcse.mean <- function(x) {x<-x[is.finite(x)];if(length(x)>1L)sd(x)/sqrt(length(x)) else NA_real_}
proportion.se <- function(p,n) if(n>0&&is.finite(p))sqrt(p*(1-p)/n) else NA_real_

summarize <- function(directory) {
    payload<-readRDS(file.path(directory,"replications.rds"));fits<-payload$fits;slopes<-payload$slopes
    spec<-read.csv(file.path(directory,"scenarios.csv"),stringsAsFactors=FALSE)
    fits$regular_inference_success<-fits$primary_success & fits$regular_full_model
    groups<-split(seq_len(nrow(fits)),interaction(fits$scenario,fits$method,drop=TRUE))
    summary<-do.call(rbind,lapply(groups,function(ix) {
        d<-fits[ix,];s<-spec[match(d$scenario[1L],spec$scenario),];R<-nrow(d)
        ps<-mean(d$primary_success);us<-mean(d$usable_success)
        mse<-d$probability_mse[d$primary_success];umse<-d$probability_mse[d$usable_success]
        data.frame(scenario=s$scenario,n=s$n,p=s$p,K=s$K,method=d$method[1L],regular_full_model=s$regular_full_model,
            attempted=R,converged=sum(d$converged),usable_success=sum(d$usable_success),primary_success=sum(d$primary_success),
            regular_inference_success=sum(d$regular_inference_success),convergence_rate=mean(d$converged),
            returned_not_converged_rate=mean(!d$converged & d$status!="error"),
            rejected_input_rate=mean(d$status=="error"),reduced_model_success_rate=mean(d$usable_success & d$collapsed),
            primary_success_rate=ps,primary_success_mcse=proportion.se(ps,R),nonprimary_rate=1-ps,
            usable_success_rate=us,usable_success_mcse=proportion.se(us,R),collapse_rate=mean(d$collapsed),
            empty_endpoint_rate=mean(d$empty_endpoints>0),empty_interior_rate=mean(d$empty_interior>0),
            directional_separation_rate=mean(d$all_splits_separated),
            primary_probability_rmse=sqrt(mean.safe(mse)),primary_probability_rmse_mcse=mcse.mean(mse)/(2*sqrt(mean.safe(mse))),
            usable_probability_rmse=sqrt(mean.safe(umse)),
            median_fit_seconds=median(d$fit_seconds),max_fit_seconds=max(d$fit_seconds),
            median_iterations=median(d$iterations,na.rm=TRUE),stringsAsFactors=FALSE)
    }))
    groups<-split(seq_len(nrow(slopes)),interaction(slopes$scenario,slopes$method,slopes$term,drop=TRUE))
    slope.summary<-do.call(rbind,lapply(groups,function(ix) {
        d<-slopes[ix,];s<-spec[match(d$scenario[1L],spec$scenario),];finite.target<-is.finite(d$truth[1L]);R<-nrow(d)
        primary<-d$primary_success & d$inference_available & finite.target
        usable<-d$usable_success & d$inference_available & finite.target
        e<-d$estimate-d$truth;cover<-d$covered
        conditional<-if(any(primary))mean(cover[primary]) else NA_real_
        joint<-if(finite.target)mean(primary & !is.na(cover) & cover) else NA_real_
        data.frame(scenario=s$scenario,method=d$method[1L],term=d$term[1L],truth=d$truth[1L],
            regular_full_model=s$regular_full_model,attempted=R,primary_inference=sum(primary),usable_inference=sum(usable),
            bias=mean.safe(e[primary]),bias_mcse=mcse.mean(e[primary]),
            rmse=sqrt(mean.safe(e[primary]^2)),rmse_mcse=mcse.mean(e[primary]^2)/(2*sqrt(mean.safe(e[primary]^2))),
            empirical_sd=if(sum(primary)>1L)sd(d$estimate[primary]) else NA_real_,
            mean_model_se=mean.safe(d$standard_error[primary]),
            coverage_conditional_success=conditional,coverage_conditional_mcse=proportion.se(conditional,sum(primary)),
            coverage_and_primary_success=joint,coverage_joint_mcse=proportion.se(joint,R),
            regular_coverage_conditional_success=if(s$regular_full_model)conditional else NA_real_,
            regular_coverage_and_success=if(s$regular_full_model)joint else NA_real_,
            interval_above_truth=if(any(primary))mean(d$lower[primary]>d$truth[primary]) else NA_real_,
            interval_below_truth=if(any(primary))mean(d$upper[primary]<d$truth[primary]) else NA_real_,
            mean_width=mean.safe(d$width[primary]),width_mcse=mcse.mean(d$width[primary]),
            usable_bias=mean.safe(e[usable]),usable_rmse=sqrt(mean.safe(e[usable]^2)),
            usable_coverage=if(any(usable))mean(cover[usable]) else NA_real_,
            usable_coverage_and_success=if(finite.target)mean(usable & !is.na(cover) & cover) else NA_real_,
            stringsAsFactors=FALSE)
    }))
    slope.summary$coverage_scope<-ifelse(slope.summary$regular_full_model,"regular finite-model development diagnostic",
        ifelse(is.finite(slope.summary$truth),"nonregular common-slope diagnostic; no regular full-model coverage claim","no finite slope target"))
    a<-slopes[slopes$method=="ML",];b<-slopes[slopes$method=="meanBR",]
    pairs<-merge(a,b,by=c("scenario","replication","term"),suffixes=c("_ML","_BR"))
    pair.summary<-do.call(rbind,lapply(split(pairs,interaction(pairs$scenario,pairs$term,drop=TRUE)),function(d) {
        good<-d$primary_success_ML & d$primary_success_BR & is.finite(d$truth_ML)
        delta<-(d$estimate_BR-d$truth_BR)^2-(d$estimate_ML-d$truth_ML)^2
        data.frame(scenario=d$scenario[1L],term=d$term[1L],attempted_pairs=nrow(d),both_primary_success=sum(good),
                   mean_squared_error_difference_BR_minus_ML=mean.safe(delta[good]),paired_difference_mcse=mcse.mean(delta[good]))
    }))
    write.csv(fits,file.path(directory,"fits.csv"),row.names=FALSE)
    write.csv(slopes,file.path(directory,"slopes.csv"),row.names=FALSE)
    write.csv(summary,file.path(directory,"summary.csv"),row.names=FALSE)
    write.csv(slope.summary,file.path(directory,"slope-summary.csv"),row.names=FALSE)
    write.csv(pair.summary,file.path(directory,"paired-summary.csv"),row.names=FALSE)
    meta<-readRDS(file.path(directory,"metadata.rds"))
    report<-c("Ordinal estimation: development screen",paste("Run:",meta$run_id),
        "This study diagnoses sampling behavior and numerical boundaries; it is not formal confirmation and was not used to tune an estimator or an interval.",
        "Each scenario uses one fixed training design and one fixed held-out design, with independent fresh response seeds. ML and mean bias reduction use the same responses.",
        "Noncollapsed usable success (primary_success) requires convergence, finite slope covariance and no category collapse. Regular-inference success additionally requires a regular finite generating model. These measures are distinct from numerical convergence.",
        "All attempts remain in the denominators. Collapsed-category fits are reported as reduced-model successes when usable, not as numerical failures; they do not count as full-scale regular-inference successes.",
        "Conditional coverage uses primary successes only; joint coverage counts an unsuccessful attempt as not covering. Usable/reduced-model summaries are separate. Nonregular scenarios have no regular full-model coverage interpretation. The 95% Wald intervals use an estimate plus or minus 1.96 model-based standard errors.",
        "The separation flag checks the generating direction only. It is sufficient when all nonempty cumulative splits separate, but is not a general separation detector.",
        sprintf("Completed %d attempts across %d scenarios and two estimators.",nrow(fits),length(unique(fits$scenario))),
        "Runtime in this statistical panel is descriptive and may include concurrent lightweight development work; standalone profile runs are separate.",
        "See summary.csv for failure/probability metrics, slope-summary.csv for bias/RMSE/coverage and Monte Carlo errors, and paired-summary.csv for paired MSE differences with their common-success denominators.")
    first<-slope.summary[slope.summary$term=="x1",]
    for(id in c("ordinary_small","strong_signal","correlated")) {
        ml<-first[first$scenario==id & first$method=="ML",];br<-first[first$scenario==id & first$method=="meanBR",]
        paired<-pair.summary[pair.summary$scenario==id & pair.summary$term=="x1",]
        report<-c(report,sprintf("%s, first slope: ML/meanBR bias %.4f/%.4f and RMSE %.4f/%.4f. The paired squared-error difference (meanBR minus ML) is %.4f, Monte Carlo SE %.4f, using %d common noncollapsed successes.",
            id,ml$bias,br$bias,ml$rmse,br$rmse,paired$mean_squared_error_difference_BR_minus_ML,
            paired$paired_difference_mcse,paired$both_primary_success))
    }
    interior<-summary[summary$scenario=="rare_interior" & summary$method=="meanBR",]
    report<-c(report,sprintf("The rare-interior scenario collapsed a category in %.1f%% of datasets. The fit still converged in %d/%d meanBR attempts; reduced-model and noncollapsed-model inference are kept separate.",
        100*interior$collapse_rate,interior$converged,interior$attempted))
    endpoint<-summary[summary$scenario=="rare_endpoints" & summary$method=="meanBR",]
    report<-c(report,sprintf("The finite rare-endpoint scenario had empty endpoints in %.1f%% of these datasets. The separate structural-empty-endpoint case supplies a deliberate boundary test; it is not evidence about a regular finite-threshold sampling distribution.",100*endpoint$empty_endpoint_rate))
    for(method in c("ML","meanBR")) {
        z<-slope.summary[slope.summary$regular_full_model & slope.summary$method==method & slope.summary$primary_inference>=100,]
        if(nrow(z))report<-c(report,sprintf("Across regular scenarios with at least 100 noncollapsed successes, %s marginal Wald coverage ranges from %.3f to %.3f; associated Monte Carlo standard errors range from %.3f to %.3f. These selected ranges are not simultaneous coverage guarantees.",
            method,min(z$coverage_conditional_success),max(z$coverage_conditional_success),
            min(z$coverage_conditional_mcse),max(z$coverage_conditional_mcse)))
    }
    report<-c(report,"The structural-zero and deterministic-separation scenarios are nonregular. Their finite-fit, probability, and explicitly defined common-slope summaries do not support regular full-model interval claims.",
        "No interval correction or estimator setting was selected from these coverage results. The observed shortfalls motivate targeted inference validation, especially under correlated predictors, rather than a numerical adjustment to achieve 95%.")
    writeLines(report,file.path(directory,"assessment.txt"))
    analysis.metadata<-list(generator_md5=unname(tools::md5sum(script.path)),
        raw_results_md5=unname(tools::md5sum(file.path(directory,"replications.rds"))),
        protocol_md5=unname(tools::md5sum(file.path(directory,"scenarios.csv"))),run_id=meta$run_id)
    reference.run<-argument("reference-run");profile.run<-argument("profile-run")
    if(nzchar(reference.run)) {
        path<-file.path(dirname(directory),reference.run,"reference-validation.csv")
        reference<-read.csv(path)
        report<-c(report,sprintf("Pinned external checks: %d/%d comparisons with MASS 7.3.65 ML and WeightIt 2.1.0 meanBR pass. Maximum coefficient/probability discrepancies are %.3g/%.3g; grouped and frequency-expanded coefficients differ by at most %.3g.",
            sum(reference$valid),nrow(reference),max(reference$coefficient_error),max(reference$probability_error),max(reference$grouping_coefficient_error)))
        analysis.metadata$reference_results_md5<-tools::md5sum(path)
    }
    if(nzchar(profile.run)) {
        folder<-file.path(dirname(directory),profile.run)
        timing<-read.csv(file.path(folder,"timing.csv"));allocation<-read.csv(file.path(folder,"allocation-summary.csv"))
        profile<-readRDS(file.path(folder,"profiles.rds"))
        profile.note<-"Standalone profiles use 20,000 rows with 3 predictors and 4 categories, or 12 predictors and 8 categories. Timings describe this machine and are separate from statistical-panel elapsed times."
        for(case in unique(timing$case)) {
            t<-timing[timing$case==case,]
            get.time<-function(method,operation)1000*t$median_seconds[t$method==method & t$operation==operation]
            profile.note<-c(profile.note,sprintf("%s: ML/meanBR fitting %.2f/%.2f ms; probability prediction %.2f ms. At the meanBR estimate, ordinary expected-information evaluation takes %.2f ms and adjusted evaluation takes %.2f ms.",
                case,get.time("ML","fit"),get.time("meanBR","fit"),get.time("meanBR","prediction"),
                get.time("meanBR","expected_information"),get.time("meanBR","adjusted_information")))
        }
        native.share<-vapply(profile,function(x)x$by.total['"ordinal_native_fit"','total.pct'],numeric(1))
        prediction.share<-vapply(profile,function(x)x$by.total['"predict.fastglm_ordinal"','total.pct'],numeric(1))
        profile.note<-c(profile.note,sprintf("The native fitter accounts for %.1f-%.1f%% of sampled fit-plus-prediction time; probability prediction accounts for %.1f-%.1f%%.",
            min(native.share),max(native.share),min(prediction.share),max(prediction.share)),
            "R-managed allocation concentrations occur in initialization rank-QR/centering copies and prediction's cumulative-probability outer products and block intermediates. These are concrete candidates for allocation reduction; no package changes were made in this study.",
            "Rprof samples R call stacks and does not identify individual C++ instructions. Rprofmem includes R-managed allocations above 64 KiB, excluding Eigen/C++ allocations. R vector-heap maxima include retained designs and benchmark fits; they are not incremental fit memory, process RSS or native peak memory.")
        writeLines(profile.note,file.path(folder,"interpretation.txt"))
        report<-c(report,profile.note)
        analysis.metadata$profile_results_md5<-tools::md5sum(file.path(folder,c("timing.csv","allocation-summary.csv","profiles.rds","memory.csv")))
    }
    if(nzchar(reference.run)||nzchar(profile.run)) {
        commands<-c("Reproduction from the repository root:",
            "  Rscript benchmarks/validate-ordinal.R --stage=run --run-id=NEW --reps=200 --library=LIB --source-root=SOURCE",
            "  Rscript benchmarks/validate-ordinal.R --stage=audit --run-id=development-v1",
            "  Rscript benchmarks/validate-ordinal.R --stage=summarize --run-id=development-v1 --reference-run=reference-v2 --profile-run=profiles-v1",
            "Existing run IDs are never overwritten by simulation stages. Summarize and audit read saved records without fitting models. Pilot-v1 retains the initial forced-garbage-collection timing overhead; pilot-v2 uses the corrected recorder. Each execution stores its exact runner and installed-library fingerprints; analysis provenance is recorded separately.")
        writeLines(c(report,commands),file.path(dirname(directory),"study-note.txt"))
    }
    saveRDS(analysis.metadata,file.path(directory,"analysis-metadata.rds"))
    if(normalizePath(script.path)!=normalizePath(file.path(directory,"analysis-runner.R"),mustWork=FALSE))
        file.copy(script.path,file.path(directory,"analysis-runner.R"),overwrite=TRUE)
    invisible(list(summary=summary,slopes=slope.summary,pairs=pair.summary))
}

if(stage %in% c("summarize","audit")) {
    stopifnot(file.exists(file.path(out,"replications.rds")))
    if(stage=="summarize") summarize(out)
    payload<-readRDS(file.path(out,"replications.rds"));meta<-readRDS(file.path(out,"metadata.rds"))
    stopifnot(nrow(payload$fits)==meta$expected_fits,!anyDuplicated(payload$fits[c("scenario","replication","method")]),
              all(table(payload$fits$scenario,payload$fits$method)==meta$replications),
              nrow(payload$slopes)==meta$expected_slopes,
              all(payload$fits$primary_success <= payload$fits$usable_success),
              all(!payload$fits$primary_success | !payload$fits$collapsed))
    paired.hashes<-split(payload$fits$data_md5,interaction(payload$fits$scenario,payload$fits$replication,drop=TRUE))
    stopifnot(all(vapply(paired.hashes,function(z)length(z)==2L&&length(unique(z))==1L,logical(1))))
    write.csv(data.frame(run_id=meta$run_id,expected_fits=meta$expected_fits,observed_fits=nrow(payload$fits),
                         expected_slopes=meta$expected_slopes,observed_slopes=nrow(payload$slopes),complete=TRUE),
              file.path(out,"audit.csv"),row.names=FALSE)
    cat("Audit passed:",nrow(payload$fits),"fit attempts\n")
    quit(save="no")
}

if(file.exists(file.path(out,"metadata.rds"))) stop("Use a new run-id to preserve existing evidence")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
loadNamespace("fastglm")
if(nzchar(lib))stopifnot(normalizePath(find.package("fastglm"))==normalizePath(file.path(lib,"fastglm")))
source.files<-c("R/ordinal.R","R/ordinal-core.R","R/ordinal-categories.R","src/fit_ordinal.cpp","R/RcppExports.R","src/RcppExports.cpp")
metadata<-list(run_id=run.id,study_type="development",stage=stage,timestamp_utc=format(Sys.time(),tz="UTC",usetz=TRUE),
    replications=reps,expected_fits=length(scenarios)*reps*2L,expected_slopes=sum(vapply(scenarios,`[[`,integer(1),"p"))*reps*2L,
    seed_rule=if(stage=="pilot") "response=200000 + 1000*scenario + replication" else "response=500000 + 1000*scenario + replication",
    fitting_controls=list(tol=1e-8,maxit=200L,block.size=2048L,control=list()),
    uncertainty="Two-sided95% Wald intervals using expected Fisher covariance; no rescaling, clipping, or coverage calibration",
    installed_path=find.package("fastglm"),version=as.character(packageVersion("fastglm")),R=R.version.string,
    source_commit=system2("git",c("rev-parse","HEAD"),stdout=TRUE),source_root=normalizePath(source.root),
    source_md5=tools::md5sum(file.path(source.root,source.files)),
    script_md5=unname(tools::md5sum(script.path)),system=Sys.info(),RNG_kind=RNGkind(),
    threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")),
    installed_binary_md5=unname(tools::md5sum(file.path(find.package("fastglm"),"libs",paste0("fastglm",.Platform$dynlib.ext)))),
    installed_r_database_md5=unname(tools::md5sum(file.path(find.package("fastglm"),"R","fastglm.rdb"))),
    scheduler_state="not_armed; short run followed synchronously",elapsed_context="statistical development panel; no dedicated benchmark ranking")
saveRDS(metadata,file.path(out,"metadata.rds"))
file.copy(script.path,file.path(out,"runner.R"),overwrite=FALSE)
write.csv(scenario.table,file.path(out,"scenarios.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,"session-info.txt"))

capture.fit <- function(fun) {
    warnings<-character();error<-""
    elapsed<-system.time(value<-tryCatch(withCallingHandlers(fun(),warning=function(w) {
        warnings<<-c(warnings,conditionMessage(w));invokeRestart("muffleWarning")
    }),error=function(e){error<<-conditionMessage(e);NULL}),gcFirst=FALSE)[["elapsed"]]
    list(value=value,warnings=warnings,error=error,seconds=elapsed)
}
fit.ordinal <- function(d,br) fastglm::fastglm_ordinal(d$x,d$y,weights=d$weights,offset=d$offset,
    firth=br,tol=1e-8,maxit=200L,block.size=2048L)

if(stage %in% c("pilot","run")) {
    fits<-slopes<-list();design.hashes<-list()
    start<-proc.time()[["elapsed"]]
    for(k in seq_along(scenarios)) {
        s<-scenarios[[k]];design<-make.design(s,120000L+k);test<-make.design(s,130000L+k,2000L)
        design.hashes[[k]]<-data.frame(scenario=s$id,training_md5=serial.hash(design),heldout_md5=serial.hash(test))
        for(replication in seq_len(reps)) {
            seed<-if(stage=="pilot") 200000L+1000L*k+replication else 500000L+1000L*k+replication
            d<-make.data(s,design,seed);data.hash<-serial.hash(d)
            for(method in if(replication%%2L) c("ML","meanBR") else c("meanBR","ML")) {
                result<-capture.fit(function()fit.ordinal(d,method=="meanBR"));fit<-result$value
                converged<-!is.null(fit)&&isTRUE(fit$converged)
                collapsed<-length(d$empty)>0L && d$empty.interior>0L
                inference<-FALSE;se<-rep(NA_real_,s$p);estimate<-se;prob.mse<-NA_real_
                postfit.errors<-postfit.warnings<-character()
                if(!is.null(fit)) {
                    estimate<-unname(fit$beta)
                    if(converged) {
                        variance.check<-capture.fit(function()stats::vcov(fit,component="location"))
                        prediction.check<-capture.fit(function()stats::predict(fit,test$x,offset=test$offset))
                        v<-variance.check$value;pr<-prediction.check$value
                        postfit.errors<-c(variance.check$error,prediction.check$error)
                        postfit.warnings<-c(variance.check$warnings,prediction.check$warnings)
                        if(!is.null(v)&&all(is.finite(v))&&all(diag(v)>0)) {
                            se<-sqrt(diag(v));inference<-all(is.finite(se)&se>0)
                        }
                        if(!is.null(pr)&&identical(dim(pr),dim(test$probability))&&all(is.finite(pr)))
                            prob.mse<-mean((pr-test$probability)^2)
                    }
                }
                usable<-converged&&inference&&all(is.finite(estimate))&&is.finite(prob.mse)
                primary<-usable&&!collapsed
                status<-if(is.null(fit)) "error" else if(!converged) fit$convergence.reason else
                    if(!usable) "inference_or_prediction_failure" else if(collapsed) "converged_reduced" else "converged_full"
                fits[[length(fits)+1L]]<-data.frame(scenario=s$id,replication,seed,method,data_md5=data.hash,
                    regular_full_model=s$regular,converged,usable_success=usable,primary_success=primary,
                    collapsed,empty_endpoints=d$empty.ends,empty_interior=d$empty.interior,
                    separated_cumulative_splits=d$separated.splits,all_splits_separated=d$all.splits.separated,
                    frequency=d$frequency,status,iterations=if(is.null(fit))NA_integer_ else fit$iterations,
                    scaled_score=if(is.null(fit))NA_real_ else fit$scaled.score,
                    predictor_step=if(is.null(fit))NA_real_ else fit$predictor.step,
                    information_rcond=if(is.null(fit))NA_real_ else fit$information.rcond,
                    min_threshold_gap=if(is.null(fit))NA_real_ else min(diff(c(-Inf,fit$alpha,Inf))),
                    max_abs_slope=if(is.null(fit))NA_real_ else max(abs(fit$beta)),
                    probability_mse=prob.mse,fit_seconds=result$seconds,
                    warnings=paste(c(result$warnings,postfit.warnings),collapse="; "),
                    error=paste(c(result$error,postfit.errors)[nzchar(c(result$error,postfit.errors))],collapse="; "),row.names=NULL)
                truth<-s$beta;z<-qnorm(.975);lower<-estimate-z*se;upper<-estimate+z*se
                slopes[[length(slopes)+1L]]<-data.frame(scenario=s$id,replication,method,term=colnames(d$x),truth,
                    estimate,standard_error=se,lower,upper,width=2*z*se,
                    covered=ifelse(is.finite(truth)&inference,lower<=truth & truth<=upper,NA),
                    inference_available=inference,usable_success=usable,primary_success=primary,collapsed,row.names=NULL)
            }
        }
        payload<-list(fits=do.call(rbind,fits),slopes=do.call(rbind,slopes))
        saveRDS(payload,file.path(out,"replications.rds"),compress="xz")
        write.csv(do.call(rbind,design.hashes),file.path(out,"design-hashes.csv"),row.names=FALSE)
        write.csv(data.frame(run_id=run.id,completed_fits=nrow(payload$fits),expected_fits=metadata$expected_fits,
                            elapsed_seconds=proc.time()[["elapsed"]]-start,last_scenario=s$id),
                  file.path(out,"status.csv"),row.names=FALSE)
        cat(s$id,"complete:",nrow(payload$fits),"of",metadata$expected_fits,"fits\n")
    }
    result<-summarize(out)
    metadata$elapsed_seconds<-proc.time()[["elapsed"]]-start
    metadata$completed_utc<-format(Sys.time(),tz="UTC",usetz=TRUE)
    saveRDS(metadata,file.path(out,"metadata.rds"))
    write.csv(data.frame(run_id=run.id,expected_fits=metadata$expected_fits,observed_fits=nrow(payload$fits),
                         expected_slopes=metadata$expected_slopes,observed_slopes=nrow(payload$slopes),
                         complete=nrow(payload$fits)==metadata$expected_fits && nrow(payload$slopes)==metadata$expected_slopes),
              file.path(out,"audit.csv"),row.names=FALSE)
    if(stage=="pilot") {
        by.case<-aggregate(fit_seconds~scenario,payload$fits,sum)
        by.case$projected_200_seconds<-by.case$fit_seconds*200/reps
        projected<-sum(by.case$projected_200_seconds)
        overhead<-max(0,metadata$elapsed_seconds-sum(payload$fits$fit_seconds))*200/reps
        estimate<-data.frame(target_reps=200L,projected_fit_seconds=projected,projected_other_seconds=overhead,
                            safety_multiplier=1.5,estimated_total_seconds=1.5*(projected+overhead),
                            coverage_mcse_at_95=sqrt(.95*.05/200),worst_proportion_mcse=sqrt(.25/200))
        write.csv(by.case,file.path(out,"runtime-by-scenario.csv"),row.names=FALSE)
        write.csv(estimate,file.path(out,"runtime-estimate.csv"),row.names=FALSE)
        print(estimate)
    }
}

if(stage=="reference") {
    metadata$replications<-metadata$expected_fits<-metadata$expected_slopes<-NULL
    metadata$expected_reference_comparisons<-8L
    for(package in c("MASS","WeightIt"))loadNamespace(package)
    stopifnot(as.character(packageVersion("MASS"))=="7.3.65",as.character(packageVersion("WeightIt"))=="2.1.0")
    metadata$reference_versions<-vapply(c("MASS","WeightIt"),function(p)as.character(packageVersion(p)),character(1))
    metadata$reference_paths<-vapply(c("MASS","WeightIt"),find.package,character(1))
    metadata$reference_database_md5<-vapply(c("MASS","WeightIt"),function(p)
        unname(tools::md5sum(file.path(find.package(p),"R",paste0(p,".rdb")))),character(1))
    metadata$reference_controls<-list(ML=list(maxit=1000L,reltol=1e-12),
        meanBR=list(maxit=1000L,reltol=1e-12,br.maxit=400L,br.tol=1e-10))
    metadata$reference_scope<-"Point estimates and category probabilities; covariance formulas are not equated because MASS uses the observed likelihood Hessian."
    saveRDS(metadata,file.path(out,"metadata.rds"))
    weightit.fit<-getFromNamespace(".ordinal_weightit.fit","WeightIt")
    rows<-list()
    for(k in c(2L,8L,9L,11L)) {
        s<-scenarios[[k]];design<-make.design(s,120000L+k);d<-make.data(s,design,710000L+k)
        index<-rep(seq_len(s$n),each=s$K);xx<-d$x[index,,drop=FALSE]
        yy<-ordered(rep(seq_len(s$K),times=s$n),levels=seq_len(s$K))
        ww<-as.vector(t(d$effective));oo<-d$offset[index]
        expanded<-list(x=xx,y=yy,weights=ww,offset=oo)
        dat<-data.frame(.y=yy,xx,.w=ww,.offset=oo)
        formula<-reformulate(c(colnames(xx),"offset(.offset)"),response=".y")
        cut.names<-paste(seq_len(s$K-1L),2:s$K,sep="|")
        parameter.names<-c(cut.names,colnames(xx))
        for(method in c("ML","meanBR")) {
            ours<-capture.fit(function()fit.ordinal(d,method=="meanBR"))
            grouped.check<-capture.fit(function()fit.ordinal(expanded,method=="meanBR"))
            reference<-capture.fit(if(method=="ML") function()
                MASS::polr(formula,data=dat,weights=.w,method="logistic",Hess=FALSE,
                    control=list(maxit=1000L,reltol=1e-12)) else function()
                weightit.fit(xx,yy,weights=ww,offset=oo,link="logit",br=TRUE,hess=FALSE,
                    control=list(maxit=1000L,reltol=1e-12,br.maxit=400L,br.tol=1e-10)))
            good<-!is.null(ours$value)&&!is.null(reference$value)
            b<-rep(NA_real_,length(parameter.names));reference.ok<-FALSE;score<-NA_real_
            if(!is.null(reference$value)) {
                f<-reference$value
                if(method=="ML") {b<-c(f$zeta,coef(f));reference.ok<-identical(as.integer(f$convergence),0L)}
                else {
                    b<-f$coefficients[parameter.names]
                    score<-max(abs(colSums(f$gradient)/c(apply(xx,2,sd),rep(1,s$K-1L))))
                    reference.ok<-is.finite(score)&&score<=1e-10*max(1,sum(ww))&&min(f$fitted.values)>1e-10
                }
            }
            coefficient.error<-probability.error<-grouping.error<-NA_real_
            if(good) {
                coefficient.error<-max(abs(c(ours$value$alpha,ours$value$beta)-b))
                cumulative<-plogis(outer(-drop(d$x%*%tail(b,s$p))-d$offset,head(b,s$K-1L),"+"))
                pr<-cbind(cumulative,1)-cbind(0,cumulative)
                probability.error<-max(abs(predict(ours$value)-pr))
            }
            if(!is.null(ours$value)&&!is.null(grouped.check$value))
                grouping.error<-max(abs(ours$value$coefficients-grouped.check$value$coefficients))
            valid<-good&&isTRUE(ours$value$converged)&&reference.ok&&is.finite(coefficient.error)&&
                coefficient.error<2e-6&&probability.error<5e-7&&is.finite(grouping.error)&&grouping.error<1e-7
            rows[[length(rows)+1L]]<-data.frame(scenario=s$id,method,seed=710000L+k,
                reference=if(method=="ML")"MASS7.3.65::polr" else "WeightIt2.1.0 internal meanBR",
                valid,fastglm_converged=!is.null(ours$value)&&isTRUE(ours$value$converged),
                reference_converged=reference.ok,reference_scaled_score=score,coefficient_error=coefficient.error,
                probability_error=probability.error,grouping_coefficient_error=grouping.error,
                warnings=paste(c(ours$warnings,reference$warnings,grouped.check$warnings),collapse="; "),
                errors=paste(c(ours$error,reference$error,grouped.check$error)[
                    nzchar(c(ours$error,reference$error,grouped.check$error))],collapse="; "))
        }
    }
    validation<-do.call(rbind,rows)
    write.csv(validation,file.path(out,"reference-validation.csv"),row.names=FALSE)
    capture.output(sessionInfo(),file=file.path(out,"session-info.txt"))
    print(validation)
}

if(stage=="profile") {
    metadata$replications<-metadata$expected_fits<-metadata$expected_slopes<-NULL
    metadata$profile_cases<-data.frame(case=c("common","many_parameters"),n=20000L,p=c(3L,12L),K=c(4L,8L))
    rows<-list();allocations<-list();profiles<-list()
    for(case in c("common","many_parameters")) {
        p<-if(case=="common")3L else 12L;K<-if(case=="common")4L else 8L
        s<-list(id=case,n=20000L,p=p,K=K,rho=.2,alpha=qlogis(seq_len(K-1L)/K),
                beta=c(.5,-.4,.3,rep(0,p-3L)),kind="individual")
        design<-make.design(s,810000L+p);d<-make.data(s,design,820000L+p)
        for(method in c("ML","meanBR")) {
            br<-method=="meanBR";warm<-fit.ordinal(d,br);stopifnot(warm$converged)
            ns<-asNamespace("fastglm")
            operations<-list(
                fit=function()fit.ordinal(d,br),
                prediction=function()predict(warm),
                expected_information=function()get("ordinal_native_evaluate",ns)(d$x,as.integer(d$y),matrix(numeric(),0,0),
                    d$weights,d$offset,K,c(warm$alpha,warm$beta),FALSE,2048L,1e-14),
                adjusted_information=function()get("ordinal_native_evaluate",ns)(d$x,as.integer(d$y),matrix(numeric(),0,0),
                    d$weights,d$offset,K,c(warm$alpha,warm$beta),TRUE,2048L,1e-14))
            for(operation in names(operations)) {
                repetitions<-if(case=="common")20L else 5L
                elapsed<-rep(NA_real_,5L)
                for(j in seq_along(elapsed))elapsed[j]<-system.time(for(r in seq_len(repetitions))operations[[operation]]())[["elapsed"]]/repetitions
                rows[[length(rows)+1L]]<-data.frame(case,n=s$n,p,K,method,operation,repetitions,
                    median_seconds=median(elapsed),min_seconds=min(elapsed),max_seconds=max(elapsed))
            }
            label<-paste(case,method,sep="-");prof<-file.path(out,paste0(label,"-Rprof.out"))
            Rprof(prof,interval=.001,memory.profiling=TRUE)
            for(r in seq_len(5L)){value<-fit.ordinal(d,br);pred<-predict(value)}
            Rprof(NULL)
            prof.summary<-summaryRprof(prof,memory="both")
            profiles[[label]]<-prof.summary
            path<-file.path(out,paste0(label,"-allocations.out"))
            gc(reset=TRUE);Rprofmem(path,threshold=65536)
            value<-fit.ordinal(d,br);pred<-predict(value)
            Rprofmem(NULL);heap<-gc()
            lines<-readLines(path,warn=FALSE);matches<-grepl("^[0-9]+ :",lines)
            bytes<-as.numeric(sub(" .*","",lines[matches]));calls<-sub("^[0-9]+ :", "",lines[matches])
            if(length(bytes)) {
                a<-aggregate(bytes,list(call=calls),sum);names(a)[2]<-"allocated_bytes";a<-a[order(-a$allocated_bytes),]
                a$case<-case;a$method<-method;allocations[[label]]<-a
            }
            rows[[length(rows)+1L]]<-data.frame(case,n=s$n,p,K,method,operation="R_heap_max_vector_bytes",
                repetitions=1L,median_seconds=heap["Vcells","max used"]*8,min_seconds=NA_real_,max_seconds=NA_real_)
        }
    }
    timing<-do.call(rbind,rows)
    write.csv(timing[timing$operation!="R_heap_max_vector_bytes",],file.path(out,"timing.csv"),row.names=FALSE)
    memory<-timing[timing$operation=="R_heap_max_vector_bytes",c("case","n","p","K","method","median_seconds")]
    names(memory)[ncol(memory)]<-"maximum_R_vector_heap_bytes"
    write.csv(memory,file.path(out,"memory.csv"),row.names=FALSE)
    if(length(allocations))write.csv(do.call(rbind,allocations),file.path(out,"allocation-summary.csv"),row.names=FALSE)
    saveRDS(profiles,file.path(out,"profiles.rds"))
    metadata$profile_limits<-"Rprof identifies R call stacks; native routines appear as one call. Rprofmem records R-managed allocations above64KiB, not Eigen/C++ allocations. R heap maxima include retained designs and benchmark fits; they are neither incremental fit memory, process RSS nor native peak memory. Timing comparisons are separate from the statistical panel."
    saveRDS(metadata,file.path(out,"metadata.rds"))
    writeLines(metadata$profile_limits,file.path(out,"interpretation.txt"))
    print(timing)
}
