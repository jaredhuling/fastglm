# Bounded development comparison. Production methods and interval rules are fixed.
# See README.md. Run from the repository root; no packages are installed here.
options(warn=1)
arg <- function(name,default="") {
    z <- commandArgs(TRUE); z <- z[startsWith(z,paste0("--",name,"="))]
    if(length(z)) substring(z[1L],nchar(name)+4L) else default
}
stage <- arg("stage","pilot")
stopifnot(stage %in% c("pilot","run","summarize","audit"))
root <- arg("output","benchmarks/ordinal-uncertainty-validation")
run.id <- arg("run-id",if(stage=="pilot") "pilot-v1" else "development-v1")
directory <- file.path(root,run.id)
library.path <- arg("library","/private/tmp/fastglm-ordinal-reuse/lib")
source.root <- arg("source-root","/private/tmp/fastglm-ordinal-reuse/source")
repetitions <- as.integer(arg("reps",if(stage=="pilot") "3" else "200"))
B <- as.integer(arg("bootstrap",if(stage=="pilot") "39" else "199"))
stopifnot(repetitions>0L,B>=2L)
Sys.setenv(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",MKL_NUM_THREADS="1")
.libPaths(c(library.path,.libPaths()))
library(fastglm)
RNGkind("Mersenne-Twister","Inversion","Rejection")
script <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1L])
md5 <- function(path) unname(tools::md5sum(path))
object.hash <- function(x) { f<-tempfile();on.exit(unlink(f));saveRDS(x,f,version=2);md5(f) }
scenarios <- data.frame(
    scenario=c("regular","small","rare_endpoint","correlated_predictors","independent_clusters"),
    n=c(250L,60L,120L,120L,150L),rho=c(.25,.25,.25,.90,.25),
    alpha1=c(-1,-1,-4,-1,-1),alpha2=0,alpha3=1,beta1=.5,beta2=-.35,
    clusters=c(0L,0L,0L,0L,30L),cluster.size=c(1L,1L,1L,1L,5L),
    covariate.icc=c(0,0,0,0,.5),latent.normal.icc=c(0,0,0,0,.6),
    purpose=c("Regular reference","Small-sample approximation","Sample-empty rare endpoint and uncertainty availability",
              "Weakly identified predictor contrasts","Independent clusters with exact marginal proportional odds"))
parameters <- c("threshold:1|2","threshold:2|3","threshold:3|4","x1","x2")
truth <- function(s) setNames(as.numeric(s[1,c("alpha1","alpha2","alpha3","beta1","beta2")]),parameters)
generate <- function(s,seed) {
    set.seed(seed); n<-s$n
    sigma<-matrix(c(1,s$rho,s$rho,1),2L)
    if(s$clusters>0L) {
        cluster<-rep(seq_len(s$clusters),each=s$cluster.size)
        common<-matrix(rnorm(s$clusters*2L),s$clusters,2L)
        z<-sqrt(s$covariate.icc)*common[cluster,,drop=FALSE]+sqrt(1-s$covariate.icc)*matrix(rnorm(n*2L),n,2L)
        latent<-sqrt(s$latent.normal.icc)*rnorm(s$clusters)[cluster]+sqrt(1-s$latent.normal.icc)*rnorm(n)
        # Every latent normal has N(0,1) marginal distribution, independent of X.
        # qlogis(Phi(latent)) therefore has a standard logistic marginal.
        error<-qlogis(pnorm(latent))
    } else { cluster<-NULL; z<-matrix(rnorm(n*2L),n,2L); error<-rlogis(n) }
    x<-z%*%chol(sigma);colnames(x)<-c("x1","x2")
    offset<-runif(n,-.2,.2)
    location<-drop(x%*%c(s$beta1,s$beta2))+offset
    code<-findInterval(location+error,c(s$alpha1,s$alpha2,s$alpha3))+1L
    y<-ordered(code,levels=1:4)
    counts<-tabulate(code,nbins=4L)
    list(x=x,y=y,offset=offset,cluster=cluster,category.counts=counts,
         hash=object.hash(list(x=x,y=y,offset=offset,cluster=cluster)))
}
capture <- function(expr) {
    warning<-character();start<-proc.time()[[3L]]
    value<-tryCatch(withCallingHandlers(expr,warning=function(w) {
        warning<<-c(warning,conditionMessage(w));invokeRestart("muffleWarning")
    }),error=function(e)e)
    list(value=if(inherits(value,"error")) NULL else value,
         error=if(inherits(value,"error")) conditionMessage(value) else "",
         warnings=warning,elapsed=proc.time()[[3L]]-start)
}
fit.one <- function(data,s,br,boot.seed,B) {
    method<-if(br) "meanBR" else "ML"
    fitted<-capture(fastglm_ordinal(data$x,data$y,offset=data$offset,firth=br))
    fit<-fitted$value
    theta<-setNames(rep(NA_real_,5L),parameters)
    if(!is.null(fit)) theta[]<-coef(fit)[parameters]
    intervals<-setNames(vector("list",4L),c("model","HC0","HC1","bootstrap"))
    for(kind in names(intervals)) {
        intervals[[kind]]<-list(lower=theta*NA_real_,upper=theta*NA_real_,se=theta*NA_real_,
            error=if(is.null(fit)) fitted$error else "",warnings=character(),elapsed=0)
    }
    if(!is.null(fit)) for(kind in c("model","HC0","HC1")) {
        calc<-capture(if(kind=="model") vcov(fit) else fastglm_vcov(fit,type=kind,cluster=data$cluster,cadjust=TRUE))
        interval<-intervals[[kind]]
        if(!is.null(calc$value)) {
            interval$se<-sqrt(diag(calc$value))[parameters]
            interval$lower<-theta-qnorm(.975)*interval$se
            interval$upper<-theta+qnorm(.975)*interval$se
        }
        interval$error<-calc$error;interval$warnings<-calc$warnings;interval$elapsed<-calc$elapsed
        intervals[[kind]]<-interval
    }
    boot<-NULL;boot.error<-"";boot.warning<-character();boot.elapsed<-0
    if(!is.null(fit)) {
        calc<-capture(fastglm_bootstrap(fit,R=B,type=if(is.null(data$cluster)) "parametric" else "cluster",
            cluster=data$cluster,seed=boot.seed,workers=1L))
        boot<-calc$value;boot.error<-calc$error;boot.warning<-calc$warnings;boot.elapsed<-calc$elapsed
        intervals$bootstrap$error<-calc$error;intervals$bootstrap$warnings<-calc$warnings
        intervals$bootstrap$elapsed<-calc$elapsed
        if(!is.null(boot)) {
            intervals$bootstrap$se<-boot$std.error[parameters]
            intervals$bootstrap$lower<-boot$conf.int[parameters,"lower"]
            intervals$bootstrap$upper<-boot$conf.int[parameters,"upper"]
        }
    }
    # Keep every finite point result, even when its original fit did not converge.
    list(estimator=method,estimate=theta,converged=!is.null(fit)&&isTRUE(fit$converged),
        fit.error=fitted$error,fit.warnings=fitted$warnings,fit.elapsed=fitted$elapsed,
        iterations=if(is.null(fit)) NA_integer_ else fit$iterations,
        rank=if(is.null(fit)) NA_integer_ else fit$rank,
        scaled.score=if(is.null(fit)) NA_real_ else fit$scaled.score,
        predictor.step=if(is.null(fit)) NA_real_ else fit$predictor.step,
        convergence.reason=if(is.null(fit)) "fit_error" else fit$convergence.reason,
        boundary=if(is.null(fit)) NA else fit$boundary,
        empty.categories=which(data$category.counts==0L),intervals=intervals,
        bootstrap=boot,bootstrap.error=boot.error,bootstrap.warnings=boot.warning,bootstrap.elapsed=boot.elapsed)
}
manifest <- function() {
    pkg<-find.package("fastglm")
    binary<-list.files(file.path(pkg,"libs"),recursive=TRUE,full.names=TRUE)
    installed<-c(file.path(pkg,c("DESCRIPTION","NAMESPACE","R/fastglm.rdb","R/fastglm.rdx")),binary)
    source.files<-c("DESCRIPTION","NAMESPACE",list.files(file.path(source.root,"R"),full.names=FALSE))
    sources<-c(file.path(source.root,c("DESCRIPTION","NAMESPACE")),
               list.files(file.path(source.root,"R"),full.names=TRUE),file.path(source.root,"src/fit_ordinal.cpp"))
    sources<-sources[file.exists(sources)]
    list(package.version=as.character(packageVersion("fastglm")),package.path=pkg,
         installed=data.frame(path=installed,md5=as.character(tools::md5sum(installed))),
         sources=data.frame(path=sources,md5=as.character(tools::md5sum(sources))),
         script.md5=md5(script),R=R.version.string,session=capture.output(sessionInfo()),
         RNGkind=RNGkind(),threads=Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")))
}
mean.finite <- function(x) if(any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
mcse <- function(x) {x<-x[is.finite(x)];if(length(x)>1L)sd(x)/sqrt(length(x)) else NA_real_}
prop.se <- function(p,n) if(n>0L&&is.finite(p))sqrt(p*(1-p)/n) else NA_real_
records <- function(directory) {
    f<-list.files(file.path(directory,"records"),pattern="\\.rds$",full.names=TRUE)
    if(!length(f)) stop("no replication records")
    lapply(f,readRDS)
}
summarize <- function(directory) {
    payload<-records(directory); rows<-fits<-boot.diagnostics<-list();i<-j<-k<-0L
    for(record in payload) for(fit in record$fits) {
        j<-j+1L
        fits[[j]]<-data.frame(scenario=record$scenario,replicate=record$replicate,estimator=fit$estimator,
            data.seed=record$data.seed,bootstrap.seed=record$bootstrap.seed,data.hash=record$data.hash,
            converged=fit$converged,fit.error=fit$fit.error,fit.warning=paste(fit$fit.warnings,collapse="; "),
            original.empty=length(fit$empty.categories)>0L,boundary=fit$boundary,iterations=fit$iterations,
            scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,convergence.reason=fit$convergence.reason,
            fit.seconds=fit$fit.elapsed,bootstrap.seconds=fit$bootstrap.elapsed,
            bootstrap.error=fit$bootstrap.error,bootstrap.warning=paste(fit$bootstrap.warnings,collapse="; "),
            bootstrap.available=if(is.null(fit$bootstrap)) FALSE else fit$bootstrap$inference.available)
        for(kind in names(fit$intervals)) {
            ci<-fit$intervals[[kind]];available<-is.finite(ci$lower)&is.finite(ci$upper)
            covered<-available&ci$lower<=record$truth&record$truth<=ci$upper
            i<-i+1L;rows[[i]]<-data.frame(scenario=record$scenario,replicate=record$replicate,
                estimator=fit$estimator,interval=kind,parameter=parameters,truth=record$truth,
                estimate=fit$estimate,original.converged=fit$converged,lower=ci$lower,upper=ci$upper,
                std.error=ci$se,available=available,covered=covered,
                warning=paste(ci$warnings,collapse="; "),error=ci$error,row.names=NULL)
        }
        if(!is.null(fit$bootstrap)) {
            k<-k+1L;b<-fit$bootstrap$diagnostics
            b$scenario<-record$scenario;b$dataset<-record$replicate;b$estimator<-fit$estimator
            for(nm in names(b)) if(is.list(b[[nm]])) b[[nm]]<-vapply(b[[nm]],paste,character(1),collapse="|")
            boot.diagnostics[[k]]<-b
        }
    }
    rows<-do.call(rbind,rows);fits<-do.call(rbind,fits)
    boot.diagnostics<-if(length(boot.diagnostics))do.call(rbind,boot.diagnostics) else data.frame()
    write.csv(rows,file.path(directory,"interval-records.csv"),row.names=FALSE)
    write.csv(fits,file.path(directory,"fit-records.csv"),row.names=FALSE)
    saveRDS(boot.diagnostics,file.path(directory,"bootstrap-diagnostics.rds"),compress="xz")
    groups<-split(rows,interaction(rows$scenario,rows$estimator,rows$interval,rows$parameter,drop=TRUE))
    summaries<-lapply(groups,function(z) {
        n<-nrow(z);a<-z$available;nav<-sum(a);p<-if(nav)mean(z$covered[a]) else NA_real_
        width<-z$upper[a]-z$lower[a]
        data.frame(scenario=z$scenario[1],estimator=z$estimator[1],interval=z$interval[1],parameter=z$parameter[1],
            datasets=n,available=nav,availability=mean(a),availability.mcse=prop.se(mean(a),n),
            conditional.coverage=p,conditional.coverage.mcse=prop.se(p,nav),
            available.and.covered=mean(z$covered),joint.mcse=prop.se(mean(z$covered),n),
            miss.below=if(nav)mean(z$upper[a]<z$truth[a]) else NA_real_,
            miss.above=if(nav)mean(z$lower[a]>z$truth[a]) else NA_real_,
            mean.width=mean.finite(width),width.mcse=mcse(width),median.width=if(nav)median(width) else NA_real_,
            mean.se=mean.finite(z$std.error[a]))
    })
    summary<-do.call(rbind,summaries);rownames(summary)<-NULL
    write.csv(summary,file.path(directory,"summary.csv"),row.names=FALSE)
    unique.points<-rows[rows$interval=="model",]
    point.groups<-split(unique.points,interaction(unique.points$scenario,unique.points$estimator,unique.points$parameter,drop=TRUE))
    point.summary<-do.call(rbind,lapply(point.groups,function(z) {
        good<-z$original.converged&is.finite(z$estimate);err<-z$estimate[good]-z$truth[good]
        allerr<-z$estimate[is.finite(z$estimate)]-z$truth[is.finite(z$estimate)]
        data.frame(scenario=z$scenario[1],estimator=z$estimator[1],parameter=z$parameter[1],datasets=nrow(z),
            converged.finite=sum(good),bias=mean.finite(err),bias.mcse=mcse(err),rmse=sqrt(mean.finite(err^2)),
            empirical.sd=if(length(err)>1L)sd(err) else NA_real_,all.finite=length(allerr),
            all.finite.bias=mean.finite(allerr),all.finite.rmse=sqrt(mean.finite(allerr^2)))
    }))
    write.csv(point.summary,file.path(directory,"point-summary.csv"),row.names=FALSE)
    # Paired contrasts retain both the common-available and complete-data views.
    paired<-list();h<-0L
    for(s in unique(rows$scenario)) for(e in unique(rows$estimator)) for(p in parameters) {
        z<-rows[rows$scenario==s&rows$estimator==e&rows$parameter==p,]
        ref<-z[z$interval=="model",];ref<-ref[order(ref$replicate),]
        for(kind in c("HC0","HC1","bootstrap")) {
            alt<-z[z$interval==kind,];alt<-alt[match(ref$replicate,alt$replicate),]
            common<-ref$available&alt$available
            dc<-as.numeric(alt$covered[common])-as.numeric(ref$covered[common])
            dj<-as.numeric(alt$covered)-as.numeric(ref$covered)
            dw<-(alt$upper-alt$lower-ref$upper+ref$lower)[common]
            h<-h+1L;paired[[h]]<-data.frame(scenario=s,estimator=e,parameter=p,comparison=paste0(kind,"-model"),
                common.available=sum(common),conditional.coverage.difference=mean.finite(dc),conditional.difference.mcse=mcse(dc),
                available.covered.difference=mean(dj),joint.difference.mcse=mcse(dj),
                width.difference=mean.finite(dw),width.difference.mcse=mcse(dw))
        }
    }
    write.csv(do.call(rbind,paired),file.path(directory,"paired-differences.csv"),row.names=FALSE)
    # Simultaneous inclusion of all five parameters is descriptive: these are
    # marginal 95% intervals, not a simultaneous 95% confidence procedure.
    joint.groups<-split(rows,interaction(rows$scenario,rows$estimator,rows$interval,drop=TRUE))
    simultaneous<-do.call(rbind,lapply(joint.groups,function(z) {
        by.rep<-split(z,z$replicate);a<-vapply(by.rep,function(v)all(v$available),logical(1))
        c<-vapply(by.rep,function(v)all(v$covered),logical(1));pc<-if(any(a))mean(c[a]) else NA_real_
        data.frame(scenario=z$scenario[1],estimator=z$estimator[1],interval=z$interval[1],datasets=length(a),
            all.available=sum(a),conditional.all.parameter.inclusion=pc,conditional.mcse=prop.se(pc,sum(a)),
            available.and.all.included=mean(c),joint.mcse=prop.se(mean(c),length(a)))
    }))
    write.csv(simultaneous,file.path(directory,"simultaneous-diagnostics.csv"),row.names=FALSE)
    invisible(summary)
}
audit <- function(directory) {
    meta<-readRDS(file.path(directory,"manifest.rds"));data<-records(directory)
    keys<-vapply(data,function(z)paste(z$scenario,z$replicate),character(1))
    stopifnot(!anyDuplicated(keys),length(data)==nrow(scenarios)*meta$repetitions,
        all(table(vapply(data,`[[`,character(1),"scenario"))==meta$repetitions))
    for(z in data) {
        stopifnot(identical(z$run.signature,meta$signature),length(z$fits)==2L,
                  identical(names(z$fits),c("ML","meanBR")))
        for(f in z$fits) if(!is.null(f$bootstrap)) {
            b<-f$bootstrap;stopifnot(nrow(b$replicates)==meta$bootstrap,nrow(b$diagnostics)==meta$bootstrap)
            if(any(!b$diagnostics$converged|!b$diagnostics$statistic.ok) ||
               length(b$original.empty.categories) || any(b$diagnostics$has.empty.categories,na.rm=TRUE))
                stopifnot(!b$inference.available,all(is.na(b$conf.int)),all(is.na(b$std.error)))
        }
    }
    writeLines(c(paste("Complete datasets:",length(data)),paste("Original fits:",2L*length(data)),
        paste("Planned bootstrap replicates per fitted original:",meta$bootstrap),
        "Unique keys, manifest signatures, complete method pairing, and unavailable-summary policy passed."),
        file.path(directory,"audit.txt"))
    invisible(TRUE)
}
if(stage %in% c("summarize","audit")) {
    if(stage=="summarize") summarize(directory) else audit(directory)
    quit(status=0L)
}
if(stage=="run" && repetitions!=200L) stop("development-v1 requires the prespecified 200 repetitions")
if(stage=="run" && B!=199L) stop("development-v1 requires the prespecified 199 bootstrap refits")
identity<-manifest();configuration<-list(repetitions=repetitions,bootstrap=B,stage=stage,scenarios=scenarios,
    seed.base=if(stage=="pilot")210000000L else 310000000L,nominal=.95,
    estimators=c("ML","meanBR"),intervals=c("model","HC0","HC1","bootstrap"))
signature<-object.hash(list(configuration=configuration,installed=identity$installed,script=identity$script.md5))
meta<-c(configuration,list(signature=signature,identity=identity,started=format(Sys.time(),tz="UTC")))
dir.create(directory,recursive=TRUE,showWarnings=FALSE);dir.create(file.path(directory,"records"),showWarnings=FALSE)
if(file.exists(file.path(directory,"manifest.rds"))) {
    old<-readRDS(file.path(directory,"manifest.rds"))
    if(!identical(old$signature,signature)) stop("refusing incompatible resume; retain the old run and choose a new run-id")
} else {
    saveRDS(meta,file.path(directory,"manifest.rds"))
    write.csv(scenarios,file.path(directory,"scenarios.csv"),row.names=FALSE)
    write.csv(identity$installed,file.path(directory,"installed-hashes.csv"),row.names=FALSE)
    write.csv(identity$sources,file.path(directory,"source-hashes.csv"),row.names=FALSE)
    writeLines(identity$session,file.path(directory,"session-info.txt"))
}
selection<-arg("scenario","all")
for(si in seq_len(nrow(scenarios))) {
    s<-scenarios[si,,drop=FALSE]
    if(selection!="all" && s$scenario!=selection) next
    for(r in seq_len(repetitions)) {
        file<-file.path(directory,"records",sprintf("%s-%03d.rds",s$scenario,r))
        if(file.exists(file)) {
            z<-readRDS(file);if(!identical(z$run.signature,signature))stop("incompatible replication record")
            next
        }
        seed<-configuration$seed.base+si*100000L+r;boot.seed<-seed+5000000L
        start<-proc.time()[[3L]];data<-generate(s,seed)
        fitted<-list(ML=fit.one(data,s,FALSE,boot.seed,B),meanBR=fit.one(data,s,TRUE,boot.seed,B))
        record<-list(scenario=s$scenario,replicate=r,data.seed=seed,bootstrap.seed=boot.seed,
            data.hash=data$hash,truth=truth(s),category.counts=data$category.counts,fits=fitted,
            elapsed=proc.time()[[3L]]-start,run.signature=signature)
        saveRDS(record,file,compress="gzip")
        cat(sprintf("%s replicate %d/%d, %.3f seconds\n",s$scenario,r,repetitions,record$elapsed));flush.console()
    }
}
if(selection=="all") { summarize(directory);audit(directory) }
