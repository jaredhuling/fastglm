# Historical exploration, retained as negative as well as positive evidence.
# The halving branch accepts its last trial after exhausting the search; it is
# not a valid safeguarded solver and was rejected. It is not package code.
# Rscript benchmarks/firth-noncanonical/development/initial-prototype.R BASELINE_LIBRARY
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
library(fastglm,lib.loc=args[1L])
d <- MASS::housing; x <- model.matrix(~Infl+Type+Cont,d)
counts <- cbind(d$Freq*(d$Sat=="Low"),d$Freq*(d$Sat!="Low"))
m <- rowSums(counts); keep <- m>0; x <- x[keep,]; counts <- counts[keep,]; m <- m[keep]
y <- counts[,1]/m; family <- binomial("cloglog")
state <- function(beta,eta=drop(x%*%beta)) {
    mu <- family$linkinv(eta); dm <- family$mu.eta(eta); variance <- mu*(1-mu)
    w <- m*dm^2/variance; q <- qr(x*sqrt(w)); h <- rowSums(qr.Q(q)^2)
    score <- drop(crossprod(x,m*(y-mu)*dm/variance+.5*h*(1-exp(eta))))
    I <- crossprod(x,x*w)
    step <- solve(I,score)
    projected <- solve(I,crossprod(x,w*(eta+(y-mu)/dm)+.5*h*(1-exp(eta))))
    list(beta=beta,step=step,next.beta=drop(projected),norm=max(abs(x%*%step)),score=score,eta=eta)
}
for(mode in c("bounded","halving")) {
    b <- numeric(ncol(x)); s <- state(b,family$linkfun((counts[,1]+.5)/(m+1)))
    for(iteration in 1:200) {
        direction <- s$next.beta-b
        fraction <- if(mode=="bounded") min(1,1/max(abs(x%*%direction))) else 1
        for(half in 0:50) {
            candidate <- b+fraction*direction
            next.state <- tryCatch(state(candidate),error=function(e) NULL)
            if(!is.null(next.state) && all(is.finite(next.state$step)) &&
               (mode=="bounded" || next.state$norm<s$norm)) break
            fraction <- fraction/2
        }
        if(is.null(next.state)) break
        if(iteration<8) cat(mode,iteration,"fraction",fraction,"remaining",next.state$norm,"\n")
        b <- candidate;s <- next.state
        if(max(abs(s$step))<1e-9) break
    }
    cat(mode,"iterations",iteration,"remaining",s$norm,"\n");print(b)
}
reference <- brglm2::brglmFit(x,counts,family=family,control=brglm2::brglmControl(type="AS_mean",epsilon=1e-10,maxit=500))
print(reference$coefficients);print(reference$converged)
