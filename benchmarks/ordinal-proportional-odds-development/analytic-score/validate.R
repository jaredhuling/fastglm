root <- "benchmarks/ordinal-proportional-odds-development/analytic-score"
library.path <- Sys.getenv("FASTGLM_STUDY_LIBRARY","")
if(nzchar(library.path)) .libPaths(c(library.path,.libPaths()))
source(file.path(root,"score.R"))
set.seed(71361101)
x <- cbind(x1=runif(60,-1,1),x2=rnorm(60)); alpha <- c(-1.2,.2,1.4); beta <- c(.4,-.3)
F <- plogis(outer(-drop(x%*%beta),alpha,"+")); y <- 1L+rowSums(runif(60)>F)
q <- length(alpha); p <- ncol(x); theta <- c(alpha,beta,numeric(p*(q-1L)))
probability <- function(theta) {
    deviation <- cbind(numeric(p),matrix(theta[-seq_len(q+p)],p,q-1L))
    eta <- outer(-drop(x%*%theta[q+seq_len(p)]),theta[seq_len(q)],"+")-x%*%deviation
    F <- plogis(eta); cbind(F,1)-cbind(0,F)
}
parts <- ordinal_score_parts(x,y,alpha,beta)
numeric.score <- numDeriv::grad(function(b) sum(log(probability(b)[cbind(seq_along(y),y)])),theta)
stopifnot(max(abs(numeric.score-parts$score))<1e-6)
# Expected negative Hessian, taking expectation with the probability held at truth.
P <- probability(theta)
expected <- -numDeriv::hessian(function(b) sum(P*log(probability(b))),theta,
    method.args=list(eps=.01,d=.01))
stopifnot(max(abs(expected-parts$information))<2e-5)
stopifnot(max(abs(ordinal_score_parts(x,y,alpha,beta,7L)$information-parts$information))<1e-10)
# Exact finite-sample first two moments of the score on a tiny design.
xx <- matrix(c(-1,-.4,.1,.5,1),ncol=1); aa <- c(-.8,.9); bb <- .5
PP <- plogis(outer(-drop(xx%*%bb),aa,"+")); PP <- cbind(PP,1)-cbind(0,PP)
outcomes <- as.matrix(expand.grid(rep(list(1:3),5)))
U <- t(apply(outcomes,1,function(yy) ordinal_score_parts(xx,yy,aa,bb)$score))
mass <- apply(outcomes,1,function(yy) prod(PP[cbind(1:5,yy)]))
II <- ordinal_score_parts(xx,outcomes[1,],aa,bb)$information
stopifnot(abs(sum(mass)-1)<1e-12,max(abs(colSums(U*mass)))<1e-12,
          max(abs(crossprod(U,mass*U)-II))<1e-10)
for(firth in c(FALSE,TRUE)) {
    fit <- ordinal_score_test(x,y,4L,firth)
    parts <- ordinal_score_parts(x,y,fit$fit$alpha,fit$fit$beta)
    at <- parts$nuisance
    stopifnot(max(abs(parts$score[at]-fit$fit$ordinary.score))<1e-7,
              max(abs(parts$information[at,at]-fit$fit$information))<1e-7)
    # Full-score norm decomposes into efficient tested and nuisance norms.
    full <- sum(forwardsolve(t(chol(parts$information)),parts$score)^2)
    stopifnot(abs(full-fit$statistic-fit$nuisance.statistic)<1e-8)
    reverse <- ordinal_score_test(x,5L-y,4L,firth)
    transformed <- ordinal_score_test(sweep(sweep(x,2,c(3,-2),"*"),2,c(10,-4),"+"),y,4L,firth)
    stopifnot(abs(reverse$statistic-fit$statistic)<1e-6,
              abs(transformed$statistic-fit$statistic)<1e-6)
}
cat("PASS: numerical likelihood score and expected Hessian; exact enumerated score moments;\n")
cat("      block accumulation; production null-score/information identities; nuisance projection;\n")
cat("      outcome reversal and affine predictor invariance at ML and meanBR null fits.\n")
