args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% c(2L,3L),args[1L] %in% c("baseline","candidate"),dir.exists(args[2L]))
implementation <- args[1L]
.libPaths(c(normalizePath(args[2L]),.libPaths()))
library(fastglm)
ns <- asNamespace("fastglm")
destination <- if(length(args)==3L) args[3L] else "benchmarks/ordinal-score-performance/confirmation"
if(dir.exists(destination) && length(list.files(destination,pattern=paste0("^",implementation,"-"))))
    stop("Results already exist; supply a fresh output directory as the third argument.")
dir.create(destination,recursive=TRUE,showWarnings=FALSE)
path <- function(suffix) file.path(destination,paste0(implementation,"-",suffix))
capture.output(sessionInfo(),file=path("session.txt"))
capture.output(list(library=find.package("fastglm"),date=Sys.time(),machine=Sys.info()),file=path("environment.txt"))
functions <- c("fastglm_po_test",".ordinal_assessment_po_test",".ordinal_po_expected_score")
if(implementation=="candidate") functions <- c(functions,".ordinal_po_score_moments",".ordinal_po_score_statistic")
dump(functions,file=path("functions.R"),envir=ns)
if(implementation=="candidate") {
    files <- c("DESCRIPTION","NAMESPACE",list.files("R",full.names=TRUE),
        list.files("src",pattern="\\.(cpp|h|hpp)$",full.names=TRUE))
    write.csv(data.frame(file=files,md5=unname(tools::md5sum(files))),path("source-md5.csv"),row.names=FALSE)
}
hash <- function(object) {
    file <- tempfile(); on.exit(unlink(file)); saveRDS(object,file,version=2)
    unname(tools::md5sum(file))
}
configurations <- data.frame(n=c(5000L,30000L,30000L,1000000L),p=c(3L,6L,12L,3L),K=c(4L,5L,8L,4L))
timings <- results <- inputs <- coefficients <- allocations <- list()
for(case in seq_len(nrow(configurations))) {
    n <- configurations$n[case]; p <- configurations$p[case]; K <- configurations$K[case]
    set.seed(111501+case)
    x <- matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    y <- ordered(cut(.35*x[,1]-.2*x[,2]+rlogis(n),c(-Inf,seq(-1.4,1.4,length.out=K-1L),Inf)),labels=seq_len(K))
    input.hash <- hash(list(x=x,y=y))
    for(estimation in c("ML","meanBR")) {
        fit <- fastglm_ordinal(x,y,firth=estimation=="meanBR",tol=1e-10,maxit=500)
        inputs[[length(inputs)+1L]] <- data.frame(case,n,p,K,estimation,input.hash,
            coefficient.hash=hash(coef(fit)),converged=fit$converged,iterations=fit$iterations)
        coefficients[[length(coefficients)+1L]] <- data.frame(case,estimation,
            coefficient=names(coef(fit)),estimate=unname(coef(fit)))
        response <- get(".ordinal_assessment_response",ns)(y,NULL,fit$levels,n)
        kernel <- function() get(".ordinal_po_expected_score",ns)(x,response,numeric(n),fit,seq_len(p),2048L)
        joint <- function() fastglm_po_test(fit,null.estimation=estimation)
        terms <- if(implementation=="candidate") function() fastglm_po_test(fit,null.estimation=estimation,by.term=TRUE)$tests else
            function() c(list(Joint=joint()),setNames(lapply(colnames(x),function(term)
                fastglm_po_test(fit,terms=term,null.estimation=estimation)),colnames(x)))
        for(operation in c("kernel","joint","terms")) {
            fun <- get(operation); invisible(fun())
            for(repetition in seq_len(5L)) {
                gc()
                elapsed <- system.time(value <- fun())[["elapsed"]]
                timings[[length(timings)+1L]] <- data.frame(implementation,case,n,p,K,estimation,operation,repetition,seconds=elapsed)
                tests <- if(operation=="terms") value else list(Joint=value)
                for(hypothesis in names(tests)) {
                    z <- tests[[hypothesis]]
                    results[[length(results)+1L]] <- data.frame(implementation,case,estimation,operation,repetition,hypothesis,
                        available=if(operation=="kernel") is.finite(z$statistic) else z$available,
                        df=if(operation=="kernel") p*(K-2L) else z$df,statistic=z$statistic,
                        p.value=if(operation=="kernel") pchisq(z$statistic,p*(K-2L),lower.tail=FALSE) else z$p.value,
                        reason=if(operation=="kernel") "available" else z$reason)
                }
            }
        }
        if(case==3L) {
            file <- tempfile(); gc(); Rprofmem(file); invisible(kernel()); Rprofmem(NULL)
            lines <- readLines(file); unlink(file)
            bytes <- suppressWarnings(as.numeric(sub(" .*","",lines)))
            allocations[[length(allocations)+1L]] <- data.frame(implementation,case,estimation,
                allocated.bytes=sum(bytes,na.rm=TRUE),largest.bytes=max(bytes,na.rm=TRUE))
        }
        cat(implementation,"case",case,estimation,"finished\n"); flush.console()
        write.csv(do.call(rbind,timings),path("timing.csv"),row.names=FALSE)
        write.csv(do.call(rbind,results),path("results.csv"),row.names=FALSE)
        write.csv(do.call(rbind,inputs),path("inputs.csv"),row.names=FALSE)
        write.csv(do.call(rbind,coefficients),path("coefficients.csv"),row.names=FALSE)
        if(length(allocations)) write.csv(do.call(rbind,allocations),path("allocations.csv"),row.names=FALSE)
    }
}
