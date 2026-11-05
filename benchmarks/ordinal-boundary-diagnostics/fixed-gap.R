# Rscript fixed-gap.R INSTALLED_LIBRARY PRIOR_REPRODUCTION NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==3L)
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
stopifnot(requireNamespace("nleqslv",quietly=TRUE))
input <- args[2L];out <- args[3L]
if(dir.exists(out)) stop("Use a new output directory")
dir.create(out,recursive=TRUE)
d <- readRDS(file.path(input,"input.rds"));fits <- readRDS(file.path(input,"fits.rds"))
saveRDS(d,file.path(out,"input.rds"))
write.exact <- function(x,file) {
    x <- as.matrix(x)
    writeLines(c(paste(colnames(x),collapse=","),apply(x,1L,function(row)
        paste(sprintf("%.17g",row),collapse=","))),file)
}
records <- states <- list()
for(link in c("logit","cloglog")) {
    ref <- fits[[paste(link,"200-dense",sep="-")]]
    x <- fastglm:::.ordinal_prediction_scale(d$x,ref$working$center,ref$working$scale)
    good <- d$weights>0;x[!good,] <- 0;off <- d$offset;off[!good] <- 0
    normal <- c(-1,1,0,x[12,],-x[12,],0,0)
    normal.square <- sum(normal^2)
    tangent <- qr.Q(qr(matrix(normal,ncol=1)),complete=TRUE)[,-1L,drop=FALSE]
    evaluate <- function(b) fastglm:::ordinal_partial_native_evaluate(matrix(numeric(),nrow(x),0L),x,
        as.integer(d$y),matrix(numeric(),0L,0L),d$weights,off,4L,b,TRUE,
        condition_tol=1e-14,link=match(link,c("logit","probit","cloglog"))-1L)
    previous <- as.numeric(ref$working$coefficients);scale <- NULL
    for(delta in 10^seq(-3,-8)) {
        key <- paste(link,format(delta,scientific=TRUE),sep="-")
        base <- previous+normal*((delta-sum(normal*previous))/normal.square)
        decode <- function(value) base+drop(tangent%*%value)
        initial <- evaluate(base)
        if(is.null(scale)) scale <- sqrt(diag(crossprod(tangent,initial$information%*%tangent)))
        calls <- invalid <- 0L
        residual <- function(value) {
            calls <<- calls+1L
            evaluated <- tryCatch(evaluate(decode(value)),error=identity)
            if(inherits(evaluated,"error") || any(!is.finite(evaluated$score))) {
                invalid <<- invalid+1L;return(rep(NaN,8L))
            }
            drop(crossprod(tangent,evaluated$score))/scale
        }
        answer <- tryCatch(nleqslv::nleqslv(numeric(8L),residual,method="Newton",global="dbldog",
            xscalm="auto",control=list(maxit=50L,ftol=1e-11,xtol=1e-12,allowSingular=TRUE)),error=identity)
        b <- if(inherits(answer,"error")) base else decode(answer$x)
        final <- tryCatch(evaluate(b),error=identity)
        if(inherits(final,"error")) stop("Final constrained point invalid: ",conditionMessage(final))
        previous <- b
        tangent.score <- drop(crossprod(tangent,final$score))
        normal.score <- sum(normal*final$score)/normal.square
        row <- data.frame(key=key,link=link,target.gap=delta,actual.gap=sum(normal*b),
            gap.error=abs(sum(normal*b)-delta),minimum.gap=final$minimum.gap,
            calls=calls,invalid=invalid,iterations=if(inherits(answer,"error")) NA_integer_ else answer$iter,
            termination=if(inherits(answer,"error")) NA_integer_ else answer$termcd,
            tangent.maximum=max(abs(tangent.score)),normal.multiplier=normal.score,
            raw.maximum=max(abs(final$score)),standardized.score=final$scaled.score,
            information.rcond=final$information.rcond,
            message=if(inherits(answer,"error")) conditionMessage(answer) else answer$message)
        records[[key]] <- row
        states[[key]] <- list(coefficients=b,normal=normal,tangent=tangent,answer=answer,state=final)
        write.csv(do.call(rbind,records),file.path(out,"attempts.csv"),row.names=FALSE)
        saveRDS(states,file.path(out,"states.rds"))
        write.exact(cbind(weights=d$weights,offset=off,y=as.integer(d$y),x),file.path(out,paste0(key,"-input.csv")))
        write.exact(matrix(b,1L,dimnames=list(NULL,paste0("b",1:9))),file.path(out,paste0(key,"-coefficients.csv")))
        write.exact(cbind(ordinary=as.numeric(final$ordinary.score),adjustment=as.numeric(final$adjustment),
            adjusted=as.numeric(final$score),step=as.numeric(final$step)),file.path(out,paste0(key,"-native-vectors.csv")))
        colnames(final$information) <- colnames(final$cov.unscaled) <- paste0("b",1:9)
        write.exact(final$information,file.path(out,paste0(key,"-native-information.csv")))
        write.exact(final$cov.unscaled,file.path(out,paste0(key,"-native-inverse.csv")))
        print(row,row.names=FALSE)
    }
}
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
files <- c(file.path(input,c("input.rds","fits.rds")),
    "benchmarks/ordinal-boundary-diagnostics/development-protocol.txt",
    "benchmarks/ordinal-boundary-diagnostics/fixed-gap.R",
    "benchmarks/ordinal-partial-boundary/high-precision.py",
    list.files(system.file("libs",package="fastglm"),full.names=TRUE))
write.csv(data.frame(path=files,md5=unname(tools::md5sum(files))),file.path(out,"source-hashes.csv"),row.names=FALSE)
