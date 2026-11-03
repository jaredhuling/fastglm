# Analytic versus score-difference information for count-model inference.
# Run from the repository root:
# Rscript benchmarks/benchmark-count-information.R /private/tmp/count-information
# Optional FASTGLM_BENCH_LIB selects an installed package library. During R-only
# development, FASTGLM_INFORMATION_SOURCE can name a candidate count-inference.R.
# This measures information and complete sandwich covariance calculations,
# excluding coefficient fitting. Sparse/source cases retain the same fitted
# coefficients and data in different supported inference representations.

args<-commandArgs(trailingOnly=TRUE)
output<-if(length(args)) args[1L] else tempfile("count-information-")
dir.create(output,recursive=TRUE,showWarnings=FALSE)
lib<-Sys.getenv("FASTGLM_BENCH_LIB")
if(nzchar(lib)) .libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(fastglm))
stopifnot(requireNamespace("Matrix",quietly=TRUE))
ns<-asNamespace("fastglm")
analytic<-new.env(parent=ns)
candidate<-Sys.getenv("FASTGLM_INFORMATION_SOURCE")
if(nzchar(candidate)) sys.source(candidate,analytic)
for(nm in c(".count_information",".count_robust_vcov","fastglm_vcov")) {
    f<-get(nm,analytic,inherits=TRUE);environment(f)<-analytic;assign(nm,f,analytic)
}
numerical<-new.env(parent=analytic)
numerical$.count_information_block<-function(object,z,parameters) {
    p<-length(parameters);hessian<-matrix(0,p,p)
    for(j in seq_len(p)) {
        h<-.Machine$double.eps^(1/3)*(1+abs(parameters[j]))
        a<-b<-parameters;a[j]<-a[j]+h;b[j]<-b[j]-h
        hessian[,j]<--(colSums(.count_score(object,z,a))-colSums(.count_score(object,z,b)))/(2*h)
    }
    (hessian+t(hessian))/2
}
environment(numerical$.count_information_block)<-numerical
for(nm in c(".count_information",".count_robust_vcov","fastglm_vcov")) {
    f<-get(nm,analytic);environment(f)<-numerical;assign(nm,f,numerical)
}

elapsed<-function(fun) {
    first<-system.time(fun())[["elapsed"]]
    if(first>=.025) return(first)
    iterations<-min(30L,max(2L,ceiling(.03/max(first,.001))))
    system.time(for(i in seq_len(iterations)) fun())[["elapsed"]]/iterations
}
relative.error<-function(a,b) max(abs(a-b))/max(1,max(abs(b)))
set.seed(8510)
results<-list();fits<-list()
for(size in 1:2) {
    n<-c(2500L,10000L)[size];p<-c(4L,16L)[size]
    x<-cbind(intercept=1,matrix(rbinom(n*(p-1L),1,.15),n,p-1L))
    colnames(x)<-c("intercept",paste0("x",seq_len(p-1L)))
    weights<-rep(c(0,1,1,2,1),length.out=n);offset<-.05*sin(seq_len(n))
    mu<-exp(.6+.3*rowSums(x[,-1L,drop=FALSE])/sqrt(p)+offset)
    y.nb<-rnbinom(n,mu=mu,size=1.2)
    y.zi<-ifelse(rbinom(n,1,plogis(-1+.3*x[,2])),0,y.nb)
    y.positive<-rbinom(n,1,.6)*rgamma(n,shape=2,scale=mu/2)
    for(kind in c("nb","truncated","hurdle_poisson","hurdle_nb","zi","twopart")) {
        cat("Preparing",kind,"n =",n,"p =",p,"\n")
        f<-switch(kind,
            nb=fastglm_nb(x,y.nb,weights=weights,offset=offset),
            truncated=fastglm_truncated(x[y.nb>0,,drop=FALSE],y.nb[y.nb>0],
                weights=weights[y.nb>0],offset=offset[y.nb>0],dist="negbin"),
            hurdle_poisson=fastglm_hurdle(x,response=y.nb,weights=weights,offset=offset,dist="poisson"),
            hurdle_nb=fastglm_hurdle(x,response=y.nb,weights=weights,offset=offset,dist="negbin"),
            zi={
                d<-data.frame(y=y.zi,x[,-1L,drop=FALSE])
                fastglm_zi(y~.,d,weights=weights,offset=offset,dist="negbin",em.maxit=1000L)
            },
            twopart=fastglm_twopart(x,response=y.positive,weights=weights,offset=offset))
        if(!isTRUE(f$converged)) stop("benchmark fit did not converge: ",kind)
        pars<-get(".count_parameters",analytic)(f)
        two<-get(".count_two_component",analytic)(f)
        design<-get(if(two) ".count_design" else ".prediction_design",analytic)(f)
        xx<-if(two) design$x$count else design$x
        zz<-if(two) design$x$zero else xx
        yy<-f$y;ww<-if(two) f$weights else f$prior.weights
        oo<-if(two) design$offset else list(count=design$offset,zero=rep(0,length(yy)))
        groups<-rep(seq_len(50L),length.out=length(yy))
        block.size<-1024L
        for(backend in if(kind %in% c("nb","hurdle_nb","twopart")) c("dense","sparse","source") else "dense") {
            object<-f
            if(backend=="sparse") object$x<-if(two)
                list(count=as(Matrix::Matrix(xx,sparse=TRUE),"dgCMatrix"),
                     zero=as(Matrix::Matrix(zz,sparse=TRUE),"dgCMatrix")) else as(Matrix::Matrix(xx,sparse=TRUE),"dgCMatrix")
            if(backend=="source") {
                callback<-local({ x0<-xx;z0<-zz;y0<-yy;w0<-ww;o0<-oo
                    function(k) {
                        i<-seq.int((k-1L)*block.size+1L,min(k*block.size,length(y0)))
                        list(X=x0[i,,drop=FALSE],Z=z0[i,,drop=FALSE],y=y0[i],weights=w0[i],offset=lapply(o0,`[`,i))
                    }
                })
                object$source<-fastglm_source(callback,ceiling(length(yy)/block.size));object$fetch<-callback
                object$x<-object$y<-NULL
            }
            cluster<-if(backend=="source") function(k)
                groups[seq.int((k-1L)*block.size+1L,min(k*block.size,length(groups)))] else groups
            cases<-list(
                information=function(e) e$.count_information(object,block.size)$information,
                HC0=function(e) e$fastglm_vcov(object,type="HC0",full=TRUE,block.size=block.size),
                cluster=function(e) e$fastglm_vcov(object,type="HC1",cluster=cluster,full=TRUE,block.size=block.size))
            for(target in names(cases)) {
                calculation<-cases[[target]]
                a<-calculation(analytic);b<-calculation(numerical)
                discrepancy<-relative.error(a,b)
                if(!is.finite(discrepancy) || discrepancy>2e-5) stop("accuracy gate failed: ",kind," ",backend," ",target)
                times<-matrix(NA_real_,5L,2L,dimnames=list(NULL,c("analytic","numerical")))
                invisible(gc())
                for(j in seq_len(nrow(times))) for(which in sample(colnames(times)))
                    times[j,which]<-elapsed(function() calculation(if(which=="analytic") analytic else numerical))
                aa<-median(times[,"analytic"]);bb<-median(times[,"numerical"])
                results[[length(results)+1L]]<-data.frame(model=kind,backend=backend,target=target,n=n,p=p,
                    parameters=length(pars),analytic_seconds=aa,numerical_seconds=bb,speedup=bb/aa,
                    relative_error=discrepancy,analytic_min=min(times[,1]),analytic_max=max(times[,1]),
                    numerical_min=min(times[,2]),numerical_max=max(times[,2]))
                cat(kind,backend,target,"speedup",round(bb/aa,2),"error",format(discrepancy,digits=3),"\n")
            }
        }
        fits[[paste(kind,n,p,sep="-")]]<-f
    }
}
results<-do.call(rbind,results)
write.csv(results,file.path(output,"timings.csv"),row.names=FALSE)
saveRDS(list(results=results,fits=fits,session=sessionInfo(),package_path=find.package("fastglm"),
    package_version=as.character(packageVersion("fastglm")),candidate_source=candidate,
    candidate_md5=if(nzchar(candidate)) tools::md5sum(candidate) else NA_character_),file.path(output,"results.rds"))
writeLines(c(capture.output(sessionInfo()),paste("Installed package:",find.package("fastglm")),
    paste("Candidate R source:",candidate),if(nzchar(candidate)) paste("Candidate MD5:",tools::md5sum(candidate))),
    file.path(output,"session.txt"))
print(results)
