# Numerical implementation checks, without inferential calibration.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop("supply a new output directory")
out <- args[1L]
if(dir.exists(out)) stop("output directory already exists; evidence is not overwritten")
if(!requireNamespace("ordinal",quietly=TRUE) || !requireNamespace("pkgload",quietly=TRUE))
    stop("the developer comparison requires ordinal and pkgload")
pkgload::load_all(".",compile=FALSE,quiet=TRUE)
dir.create(out,recursive=TRUE)
file.copy("benchmarks/ordinal-profile-validation/run.R",file.path(out,"runner.R"))
file.copy("benchmarks/ordinal-profile-validation/protocol.txt",file.path(out,"protocol.txt"))
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
files <- c("R/ordinal-profile.R","R/ordinal.R","R/ordinal-inference.R","R/ordinal-po-test.R",
    "R/firth-profile.R","R/wald-test.R","R/RcppExports.R","inst/include/ordinal.h",
    "src/fit_ordinal.cpp","src/RcppExports.cpp","src/fastglm.so",
    "benchmarks/ordinal-profile-validation/run.R","benchmarks/ordinal-profile-validation/protocol.txt")
hashes <- tools::md5sum(files)
write.csv(data.frame(file=names(hashes),md5=unname(hashes)),file.path(out,"source-hashes.csv"),row.names=FALSE)
writeLines(c(paste("time:",format(Sys.time(),tz="UTC")),
    paste("revision:",system2("git",c("rev-parse","HEAD"),stdout=TRUE)),
    "loading: current source via pkgload, compile=FALSE",
    paste("ordinal:",packageVersion("ordinal"))),file.path(out,"metadata.txt"))

make_input <- function(link,large,seed) {
    set.seed(seed);n <- if(large) 60L else 24L;p <- if(large) 4L else 2L
    K <- if(large) 5L else 3L
    x <- matrix(rnorm(n*p,sd=.7),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    beta <- c(.3,-.4,.15,-.2)[seq_len(p)];alpha <- seq(-1,1,length.out=K-1L)
    offset <- .2*sin(x[,1L]);weights <- rep(c(0,1,2,1),length.out=n)
    inverse <- switch(link,logit=plogis,probit=pnorm,cloglog=function(z) -expm1(-exp(z)))
    F <- inverse(outer(-drop(x%*%beta)-offset,alpha,"+"))
    P <- cbind(F,1)-cbind(0,F)
    counts <- t(vapply(seq_len(n),function(i) as.numeric(rmultinom(1L,20L,P[i,])),numeric(K)))
    rows <- rep(seq_len(n),each=K)
    data <- data.frame(y=ordered(rep(seq_len(K),n),levels=seq_len(K)),x[rows,,drop=FALSE],
        off=offset[rows],weight=as.vector(t(counts))*rep(weights,each=K))
    list(x=x,counts=counts,weights=weights,offset=offset,data=data,link=link,seed=seed)
}

cases <- expand.grid(link=c("logit","probit","cloglog"),large=c(FALSE,TRUE),stringsAsFactors=FALSE)
cases$seed <- 113600L+seq_len(nrow(cases));cases$id <- paste(cases$link,ifelse(cases$large,"larger","smaller"),sep="-")
write.csv(cases,file.path(out,"cases.csv"),row.names=FALSE)
inputs <- lapply(seq_len(nrow(cases)),function(i) make_input(cases$link[i],cases$large[i],cases$seed[i]))
names(inputs) <- cases$id;saveRDS(inputs,file.path(out,"inputs.rds"))
records <- list();endpoints <- list();score.records <- list()
for(i in seq_len(nrow(cases))) {
    input <- inputs[[i]];id <- cases$id[i];warnings <- character()
    result <- withCallingHandlers(tryCatch({
        fit <- fastglm_ordinal(input$x,input$counts,weights=input$weights,offset=input$offset,
            link=input$link,tol=1e-10)
        form <- reformulate(c(colnames(input$x),"offset(off)"),response="y")
        reference <- ordinal::clm(form,data=input$data,weights=weight,link=input$link)
        restricted.data <- input$data;restricted.data$off <- restricted.data$off+.15*restricted.data$x1
        reduced.form <- reformulate(c(colnames(input$x)[-1L],"offset(off)"),response="y")
        restricted <- ordinal::clm(reduced.form,data=restricted.data,weights=weight,link=input$link)
        test <- fastglm_profile_test(fit,"x1",null=.15)
        ci <- confint(fit,parm="x1",method="profile")
        ref.ci <- confint(reference,type="profile")["x1",,drop=FALSE]
        diagnostic <- attr(ci,"diagnostics")
        endpoints[[length(endpoints)+1L]] <- data.frame(case=id,parameter="x1",side=c("lower","upper"),
            fastglm=as.numeric(ci),ordinal=as.numeric(ref.ci),error=as.numeric(ci)-as.numeric(ref.ci),
            target.error=diagnostic$lr.error,nuisance.score=diagnostic$score,
            predictor.step=diagnostic$predictor.step,converged=diagnostic$converged)
        L <- matrix(0,2,length(coef(fit)),dimnames=list(c("threshold gap","slope contrast"),names(coef(fit))))
        L[1,1:2] <- c(-1,1);L[2,c("x1","x2")] <- c(1,-.4)
        null <- c(.8,.15);A <- matrix(c(2,-.3,.4,1.2),2)
        score.ok <- TRUE
        for(estimation in c("ML","meanBR")) {
            score <- fastglm_score_test(fit,contrast=L,null=null,null.estimation=estimation)
            transformed <- fastglm_score_test(fit,contrast=A%*%L,null=drop(A%*%null),null.estimation=estimation)
            residual <- if(score$available) max(abs(drop(L%*%score$diagnostics$restricted$coefficients)-null)) else Inf
            difference <- abs(unname(score$statistic-transformed$statistic))
            passed <- isTRUE(score$available && transformed$available && residual<1e-10 && difference<1e-7 &&
                score$diagnostics$restricted$score<1e-7 && score$diagnostics$restricted$predictor.step<1e-7)
            score.ok <- score.ok && passed
            score.records[[length(score.records)+1L]] <- data.frame(case=id,estimation=estimation,
                available=score$available,statistic=unname(score$statistic),constraint.residual=residual,
                transformed.difference=difference,passed=passed,message=score$diagnostics$message)
        }
        full.error <- as.numeric(logLik(fit)-logLik(reference))
        restricted.error <- unname(test$loglik["restricted"])-as.numeric(logLik(restricted))
        passed <- isTRUE(fit$converged && test$available && all(diagnostic$converged) &&
            abs(full.error)<2e-6 && abs(restricted.error)<2e-6 &&
            max(abs(as.numeric(ci)-as.numeric(ref.ci)))<5e-4 && max(abs(diagnostic$lr.error))<1e-5 &&
            max(diagnostic$score)<1e-7 && max(diagnostic$predictor.step)<1e-7 && score.ok)
        data.frame(case=id,passed=passed,full.loglik.error=full.error,restricted.loglik.error=restricted.error,
            profile.max.error=max(abs(as.numeric(ci)-as.numeric(ref.ci))),message="")
    },error=function(e) data.frame(case=id,passed=FALSE,full.loglik.error=NA_real_,restricted.loglik.error=NA_real_,
        profile.max.error=NA_real_,message=conditionMessage(e))),warning=function(w) {
            warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")
        })
    result$warnings <- paste(unique(warnings),collapse="; ");records[[i]] <- result
}
housing <- list()
for(link in c("logit","probit","cloglog")) housing[[link]] <- tryCatch({
    fit <- fastglm_ordinal(Sat~Infl+Type+Cont,MASS::housing,weights=Freq,link=link,tol=1e-10)
    reference <- ordinal::clm(Sat~Infl+Type+Cont,data=MASS::housing,weights=Freq,link=link)
    ci <- confint(fit,parm=c("InflHigh","ContHigh"),method="profile")
    ref.ci <- confint(reference,type="profile")[c("InflHigh","ContHigh"),,drop=FALSE]
    data.frame(link=link,parameter=rep(rownames(ci),2),side=rep(c("lower","upper"),each=2),
        fastglm=as.numeric(ci),ordinal=as.numeric(ref.ci),error=as.numeric(ci)-as.numeric(ref.ci),
        passed=abs(as.numeric(ci)-as.numeric(ref.ci))<5e-4,message="")
},error=function(e) data.frame(link=link,parameter=NA_character_,side=NA_character_,fastglm=NA_real_,
    ordinal=NA_real_,error=NA_real_,passed=FALSE,message=conditionMessage(e)))
records <- do.call(rbind,records);housing <- do.call(rbind,housing)
write.csv(records,file.path(out,"comparisons.csv"),row.names=FALSE)
write.csv(do.call(rbind,endpoints),file.path(out,"profile-endpoints.csv"),row.names=FALSE)
write.csv(do.call(rbind,score.records),file.path(out,"score-constraints.csv"),row.names=FALSE)
write.csv(housing,file.path(out,"housing-profile-reference.csv"),row.names=FALSE)
print(records,row.names=FALSE)
cat("Housing endpoint maximum error:",max(abs(housing$error)),"\n")
if(!all(records$passed) || !all(housing$passed)) stop("implementation comparison failed; inspect every recorded outcome")
