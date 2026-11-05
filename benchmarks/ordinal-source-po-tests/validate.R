# Deterministic agreement and probability-derivative audit; not a timing or
# calibration experiment. The baseline mode loads an immutable installation.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)<2L || !args[1L]%in%c("baseline","current"))
    stop("usage: validate.R baseline LIBRARY OUTPUT.rds | current REFERENCE.rds")
baseline <- args[1L]=="baseline"
if(baseline) {
    if(length(args)!=3L) stop("baseline mode requires a library and output file")
    library(fastglm,lib.loc=args[2L])
} else {
    pkgload::load_all(".",quiet=TRUE,compile=FALSE)
    if(!requireNamespace("numDeriv",quietly=TRUE)) stop("numDeriv is required for the independent audit")
}
fixture <- function(link) {
    i <- seq_len(30L);X <- cbind(x=(i-15.5)/15,z=cos(i*.61))
    off <- sin(i*.27)*.12
    eta <- matrix(c(-1.15,-.1,1.05),30,3,byrow=TRUE)-.21*X[,1]-off-
        outer(X[,2],c(-.12,.03,.11))
    F <- switch(link,logit=plogis(eta),probit=pnorm(eta),cloglog=-expm1(-exp(eta)))
    y <- 1+floor(41*(cbind(F,1)-cbind(0,F)))
    colnames(y) <- c("first","second","third","fourth")
    list(X=X,y=y,offset=off,weights=rep(c(1,0,2,1,1),6))
}
records <- list();derivative.errors <- numeric()
for(link in c("logit","probit","cloglog")) {
    d <- fixture(link)
    for(partial in c(FALSE,TRUE)) {
        if(baseline) fit <- fastglm_ordinal(d$X,d$y,weights=d$weights,offset=d$offset,
            nonparallel=if(partial)"z",link=link,tol=1e-10) else {
            blocks <- list(1:4,5:13,14:21,22:30)
            source <- fastglm_source(function(k) {
                at <- blocks[[k]]
                list(X=d$X[at,,drop=FALSE],y=d$y[at,,drop=FALSE],
                    weights=d$weights[at],offset=d$offset[at])
            },length(blocks))
            fit <- fastglm_ordinal_streaming(source,nonparallel=if(partial)"z",link=link,tol=1e-10)
        }
        for(estimation in c("ML","meanBR")) {
            out <- fastglm_po_test(fit,"z",null.estimation=estimation)
            stopifnot(isTRUE(out$available))
            key <- paste(link,partial,"score",estimation,sep="/")
            records[[key]] <- c(statistic=out$statistic,df=out$df,frequency=out$frequency)
            if(!baseline && partial) {
                probability <- function(b) {
                    eta <- matrix(b[1:3],30,3,byrow=TRUE)-d$X[,1]*b[4]-d$offset-
                        outer(d$X[,2],b[5:7])
                    F <- switch(link,logit=plogis(eta),probit=pnorm(eta),cloglog=-expm1(-exp(eta)))
                    cbind(F,1)-cbind(0,F)
                }
                b <- out$null.coefficients;pr <- probability(b)
                J <- numDeriv::jacobian(function(b)as.vector(probability(b)),b)
                U <- drop(crossprod(J,as.vector(d$y*d$weights/pr)))
                I <- crossprod(J,J*as.vector(rowSums(d$y)*d$weights/pr))
                L <- out$contrast;Q <- qr.Q(qr(t(L)),complete=TRUE)
                B <- Q[,(nrow(L)+1L):ncol(L),drop=FALSE]
                e <- U-I%*%B%*%solve(crossprod(B,I%*%B),crossprod(B,U))
                independent <- drop(crossprod(e,solve(I,e)))
                derivative.errors[key] <- abs(independent-out$statistic)
            }
        }
        if(partial) {
            out <- fastglm_po_test(fit,"z",method="lr")
            stopifnot(isTRUE(out$available))
            records[[paste(link,partial,"lr","ML",sep="/")]] <-
                c(statistic=out$statistic,df=out$df,frequency=out$frequency)
        }
    }
}
if(baseline) {
    saveRDS(list(revision="2619731",records=records,session=sessionInfo()),args[3L])
    cat("Saved",length(records),"dense reference tests from baseline 2619731.\n")
} else {
    reference <- readRDS(args[2L]);stopifnot(identical(reference$revision,"2619731"))
    stopifnot(identical(names(records),names(reference$records)))
    difference <- vapply(names(records),function(key)max(abs(records[[key]]-reference$records[[key]])),numeric(1))
    print(data.frame(test=names(records),maximum.absolute.difference=difference),row.names=FALSE)
    cat("Maximum independent derivative discrepancy:",max(derivative.errors),"\n")
    stopifnot(all(difference<1e-6),all(derivative.errors<2e-6))
    cat("PASS: dense/source agreement and independent ordinary-score calculations.\n")
}
