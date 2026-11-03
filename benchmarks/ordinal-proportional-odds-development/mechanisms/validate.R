root <- "benchmarks/ordinal-proportional-odds-development/mechanisms"
study.library <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(study.library)) .libPaths(c(study.library,.libPaths()))
source(file.path(root,"algebra.R")); source(file.path(root,"config.R"))
source(file.path(root,"../prototype.R"))
set.seed(30361000)
z <- cbind(1,matrix(rnorm(300),100,3))
cases <- list(c(-.5,.3,-.2,.4),c(-4,.3,-.2,.4),c(4,.3,-.2,.4))
for(b in cases) {
    a <- po_information(z,b)
    J <- numDeriv::jacobian(function(theta) po_information(z,theta,adjusted=FALSE)$adjustment,b)
    stopifnot(max(abs((a$A-a$H)-J))<1e-7,max(abs(a$H-t(a$H)))<1e-10)
}
design <- po_fixed_design(mechanism.cells[8,]); y <- 1L+rowSums(design$F<runif(nrow(design$x)))
fit <- po_mechanism_fit(design,y)
counts <- matrix(0,length(y),design$q+1L); counts[cbind(seq_along(y),y)] <- 1
old <- ordinal_po_covariance(design$x,counts,firth=TRUE)
stopifnot(max(abs(fit$delta-old$difference))<1e-7,
          max(abs(fit$V$empirical_ordinary-old$difference.covariance))<1e-7,
          abs(po_statistic(fit$delta,fit$V$empirical_ordinary)-old$statistic)<1e-7)
# Category-enumerated score covariance agrees with the nested-event formula.
z <- cbind(1,design$x); q <- design$q
probs <- cbind(design$F,1)-cbind(0,design$F); surv <- 1-design$F
B <- matrix(0,ncol(z)*q,ncol(z)*q)
for(k in seq_len(q+1L)) {
    scores <- do.call(cbind,lapply(seq_len(q),function(j) z*(as.numeric(k>j)-surv[,j])))
    B <- B+crossprod(scores,probs[,k]*scores)
}
stopifnot(max(abs(B-design$B.true))<1e-10)
# Empty endpoint and interior responses still permit a fixed-dimensional slope vector.
small <- po_fixed_design(mechanism.cells[1,])
for(y in list(rep(2L,80),rep(c(1L,3L),40))) {
    answer <- po_mechanism_fit(small,y)
    stopifnot(length(answer$delta)==1L,all(is.finite(answer$delta)))
}
cat("PASS: analytic adjusted-score derivative, Hessian symmetry, original covariance agreement,\n")
cat("      category-enumerated score variance, and empty-category fixed-dimension fits.\n")
