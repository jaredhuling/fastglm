# Equation-preserving root attempts in feasibility coordinates; not a new bias target.
# Rscript .../independent-roots.R LIBRARY REPRODUCTION_DIRECTORY NEW_OUTPUT
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==3L)
.libPaths(c(normalizePath(args[1L]),.libPaths()));library(fastglm)
stopifnot(requireNamespace("nleqslv",quietly=TRUE))
input <- args[2L];out <- args[3L];if(dir.exists(out)) stop("Use a new output directory")
dir.create(out,recursive=TRUE)
fits <- readRDS(file.path(input,"fits.rds"));d <- readRDS(file.path(input,"input.rds"))
records <- answers <- list()
for(link in c("logit","probit","cloglog")) {
    ref <- fits[[paste(link,200L,"dense",sep="-")]]
    x <- fastglm:::.ordinal_prediction_scale(d$x,ref$working$center,ref$working$scale)
    good <- d$weights>0;x[!good,] <- 0;off <- d$offset;off[!good] <- 0
    maximum <- function(b) {
        gamma <- matrix(b[4:9],2L,3L)
        apply(x[good,,drop=FALSE]%*%(gamma[,-1L,drop=FALSE]-gamma[,-3L,drop=FALSE]),2L,max)
    }
    # This invertible (piecewise smooth) algorithmic map enforces strict domain
    # gaps. Residuals remain the original full-model adjusted scores; no bias
    # adjustment is derived in these nonlinear coordinates.
    encode <- function(b) c(b[1L],log(diff(b[1:3])-maximum(b)),b[4:9])
    decode <- function(a) {b<-a;b[1:3]<-cumsum(c(a[1L],maximum(a)+exp(a[2:3])));b}
    evaluate <- function(b) fastglm:::ordinal_partial_native_evaluate(matrix(numeric(),nrow(x),0L),x,
        as.integer(d$y),matrix(numeric(),0L,0L),d$weights,off,4L,b,TRUE,
        condition_tol=1e-14,link=match(link,c("logit","probit","cloglog"))-1L)
    counts <- tabulate(as.integer(d$y)[rep(seq_len(nrow(x)),d$weights)],4L)
    inverse <- switch(link,logit=qlogis,probit=qnorm,cloglog=function(p) log(-log1p(-p)))
    intercept <- c(inverse(cumsum(counts)[1:3]/sum(counts))+weighted.mean(off,d$weights),rep(0,6L))
    starts <- list(intercept=intercept,early=fits[[paste(link,if(link=="probit") 30L else 8L,"dense",sep="-")]]$working$coefficients,
                   cap30=fits[[paste(link,30L,"dense",sep="-")]]$working$coefficients,
                   final=ref$working$coefficients)
    for(label in names(starts)) {
        b <- as.numeric(starts[[label]]);state <- evaluate(b)
        scale <- sqrt(diag(evaluate(intercept)$information))
        invalid <- 0L;calls <- 0L
        fn <- function(a) {
            calls <<- calls+1L
            value <- tryCatch(evaluate(decode(a)),error=identity)
            if(inherits(value,"error")) {invalid<<-invalid+1L;return(rep(NaN,9L))}
            drop(value$score)/scale
        }
        result <- tryCatch(nleqslv::nleqslv(encode(b),fn,method="Newton",global="dbldog",xscalm="auto",
            control=list(maxit=200L,ftol=1e-10,xtol=1e-12,allowSingular=TRUE)),error=identity)
        final <- if(!inherits(result,"error")) tryCatch(evaluate(decode(result$x)),error=identity) else result
        valid <- !inherits(final,"error")
        row <- data.frame(link=link,start=label,calls=calls,invalid.evaluations=invalid,
            termination=if(inherits(result,"error")) NA_integer_ else result$termcd,
            iterations=if(inherits(result,"error")) NA_integer_ else result$iter,
            native.criteria=valid && final$scaled.score<1e-8 && final$predictor.step<1e-8,
            score=if(valid) final$scaled.score else NA_real_,step=if(valid) final$predictor.step else NA_real_,
            minimum.gap=if(valid) final$minimum.gap else NA_real_,
            rcond=if(valid) final$information.rcond else NA_real_,
            message=if(inherits(result,"error")) conditionMessage(result) else result$message)
        records[[length(records)+1L]] <- row
        answers[[paste(link,label)]] <- list(result=result,state=final,
            coefficients=if(!inherits(result,"error")) decode(result$x) else NULL)
        write.csv(do.call(rbind,records),file.path(out,"attempts.csv"),row.names=FALSE)
        saveRDS(answers,file.path(out,"attempts.rds"));print(row,row.names=FALSE)
    }
}
writeLines(capture.output(sessionInfo()),file.path(out,"session.txt"))
