# Post-study diagnosis: expected versus observed likelihood information.
# Replays four specified cells without changing the frozen candidates or records.
root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
library.path <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
source(file.path(root,"score.R")); source(file.path(root,"../study/config.R")); source(file.path(root,"summarize.R"))
observed.information <- function(x,y,alpha,beta) {
    n <- nrow(x); p <- ncol(x); q <- length(alpha); d <- q*(p+1L)
    eta <- outer(-drop(x%*%beta),alpha,"+"); f <- plogis(eta)*plogis(-eta)
    t <- exp(-diff(alpha))/(-expm1(-diff(alpha))); H <- matrix(0,d,d)
    derivative <- function(j) {
        Z <- matrix(0,n,d); Z[,j] <- 1; Z[,q+seq_len(p)] <- -x
        if(j>1L) Z[,q+p+(j-2L)*p+seq_len(p)] <- -x
        Z
    }
    for(k in seq_len(q+1L)) {
        w <- as.numeric(y==k)
        if(k<=q) {up <- derivative(k); H <- H+crossprod(up,(w*f[,k])*up)}
        if(k>1L) {lo <- derivative(k-1L); H <- H+crossprod(lo,(w*f[,k-1L])*lo)}
        if(k>1L && k<=q) {gap <- up-lo; H <- H+crossprod(gap,(w*t[k-1L]*(1+t[k-1L]))*gap)}
    }
    H
}
# Published SAS/STAT 12.1 Cheese example, expanded frequency table.
counts <- rbind(c(0,0,1,7,8,8,19,8,1),c(6,9,12,11,7,6,1,0,0),
                c(1,1,6,8,23,7,5,1,0),c(0,0,0,1,3,7,14,16,11))
y <- unlist(lapply(1:4,function(i) rep(1:9,counts[i,])))
group <- rep(1:4,rowSums(counts)); x <- vapply(1:3,function(g) as.numeric(group==g),numeric(length(y)))
fit <- ordinal_score_test(x,y,9); parts <- ordinal_score_parts(x,y,fit$fit$alpha,fit$fit$beta)
H <- observed.information(x,y,fit$fit$alpha,fit$fit$beta)
q <- 8L; p <- 3L; theta <- c(fit$fit$alpha,fit$fit$beta,numeric(21L))
nll <- function(b) {
    deviation <- cbind(numeric(p),matrix(b[-seq_len(q+p)],p,q-1L))
    eta <- outer(-drop(x%*%b[q+seq_len(p)]),b[seq_len(q)],"+")-x%*%deviation
    logF <- plogis(eta,log.p=TRUE); logS <- plogis(eta,lower.tail=FALSE,log.p=TRUE)
    logP <- cbind(logF[,1],logF[,-1,drop=FALSE]+logS[,-q,drop=FALSE]+
        log(-expm1(eta[,-q,drop=FALSE]-eta[,-1,drop=FALSE])),logS[,q])
    -sum(logP[cbind(seq_along(y),y)])
}
numeric.H <- numDeriv::hessian(function(step) nll(theta+step),numeric(length(theta)),
    method.args=list(eps=.01,d=.01))
stopifnot(max(abs(H-numeric.H))<2e-5)
ee <- eigen(H,symmetric=TRUE); keep <- ee$values>1e-8*max(ee$values)
u <- drop(crossprod(ee$vectors,parts$score)); generalized <- sum(u[keep]^2/ee$values[keep])
stopifnot(abs(generalized-17.2866)<.001,fit$df==21L)
write.csv(data.frame(expected.statistic=fit$statistic,observed.generalized.statistic=generalized,
    published.SAS.statistic=17.2866,expected.rank=ncol(H),observed.rank=sum(keep),df=fit$df,
    maximum.Hessian.error=max(abs(H-numeric.H))),file.path(root,"published-example-check.csv"),row.names=FALSE)
# The generalized inverse above is only an external-example diagnostic. No
# pseudo-inverse or rank substitution is allowed in the inferential comparisons.
generate.stress <- function(scenario,n) {
    if(scenario=="many_predictors5") {
        x <- matrix(rnorm(n*8L),n,8L); beta <- c(.5,-.4,.3,-.2,0,0,0,0); alpha <- c(-1.8,-.6,.6,1.8)
    } else {x <- cbind(runif(n,-1,1)+15*rbinom(n,1,.03),runif(n,-1,1)); beta <- c(.25,-.4); alpha <- c(-.8,.8)}
    colnames(x) <- paste0("x",seq_len(ncol(x))); F <- plogis(outer(-drop(x%*%beta),alpha,"+"))
    list(x=x,y=1L+rowSums(runif(n)>F),K=length(alpha)+1L)
}
cells <- data.frame(bank=c("raw","raw","stress-raw","stress-raw"),cell=c(4L,10L,3L,7L),
    scenario=c("regular5","rare_lower3","many_predictors5","leverage3"))
records <- list()
for(i in seq_len(nrow(cells))) {
    old <- readRDS(file.path(root,cells$bank[i],sprintf("cell-%02d.rds",cells$cell[i])))
    old <- old[old$method=="score_ML",]
    for(r in seq_len(nrow(old))) {
        stat <- NA_real_; error <- ""; discrepancy <- NA_real_
        if(old$available[r]) {
            set.seed(old$seed[r])
            d <- if(cells$bank[i]=="raw") po_generate(cells$scenario[i],80L) else generate.stress(cells$scenario[i],80L)
            K <- if(is.null(d$K)) ncol(d$counts) else d$K
            f <- suppressWarnings(ordinal_score_test(d$x,d$y,K)); discrepancy <- abs(f$statistic-old$statistic[r])
            stopifnot(discrepancy<1e-7*(1+old$statistic[r]))
            pp <- ordinal_score_parts(d$x,d$y,f$fit$alpha,f$fit$beta)
            pp$information <- observed.information(d$x,d$y,f$fit$alpha,f$fit$beta)
            stat <- tryCatch(ordinal_score_statistic(pp)$statistic,error=function(e) {error <<- conditionMessage(e); NA_real_})
        } else error <- "original_score_unavailable"
        records[[length(records)+1L]] <- data.frame(scenario=cells$scenario[i],replication=r,seed=old$seed[r],
            df=old$df[r],expected.statistic=old$statistic[r],observed.statistic=stat,replay.error=discrepancy,error)
    }
}
z <- do.call(rbind,records); saveRDS(z,file.path(root,"information-audit.rds"),version=2)
rows <- list()
for(scenario in cells$scenario) for(level in c(.05,.01)) {
    d <- z[z$scenario==scenario,]; ok <- is.finite(d$expected.statistic)&is.finite(d$observed.statistic)
    for(method in c("expected","observed")) {
        pp <- pchisq(d[[paste0(method,".statistic")]][ok],d$df[ok],lower.tail=FALSE)
        k <- sum(pp<=level); ci <- wilson(k,length(pp))
        rows[[length(rows)+1L]] <- data.frame(scenario,method,level,shared=sum(ok),rejection=k,
            rate=k/length(pp),mc.lower=ci[1],mc.upper=ci[2])
    }
}
write.csv(do.call(rbind,rows),file.path(root,"information-comparison.csv"),row.names=FALSE)
cat("PASS: observed Hessian numerical check, SAS example diagnostic, and 4000 dataset records with exact available-score replays.\n")
