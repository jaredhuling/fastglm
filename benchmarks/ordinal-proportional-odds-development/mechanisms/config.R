mechanism.cells <- data.frame(scenario=c("regular3","regular3","rare_lower3","rare_lower3",
    "rare_lower3","rare_interior3","rare_interior3","regular5","regular5"),
    n=c(80L,1000L,80L,250L,1000L,80L,1000L,80L,1000L),cell=seq_len(9L))
mechanism.banks <- c(moments=1000L,reference=1000L,evaluation=2000L)
mechanism.methods <- c("empirical_ordinary","empirical_adjusted","true_ordinary","true_adjusted",
                       "null_ordinary","null_adjusted","true_linear")
po_fixed_design <- function(cell) {
    set.seed(30361100L+cell$cell)
    five <- cell$scenario=="regular5"; n <- cell$n
    x <- if(five) cbind(x1=runif(n,-1,1),x2=runif(n,-1,1),x3=rbinom(n,1,.4)) else cbind(x1=runif(n,-1,1))
    alpha <- if(five) c(-1.8,-.6,.6,1.8) else if(cell$scenario=="rare_lower3") c(-3.8,.4) else
        if(cell$scenario=="rare_interior3") c(-.08,.08) else c(-.8,.8)
    beta <- if(five) c(.6,-.5,.4) else .7
    F <- plogis(outer(-drop(x%*%beta),alpha,"+")); q <- length(alpha)
    z <- cbind(1,x); d <- ncol(z); B <- po_score_variance(z,1-F)
    Ainv <- matrix(0,d*q,d*q)
    for(j in seq_len(q)) {
        at <- (j-1L)*d+seq_len(d)
        Ainv[at,at] <- chol2inv(chol(po_information(z,c(-alpha[j],beta),adjusted=FALSE)$A))
    }
    D <- po_contrasts(ncol(x),q)
    list(x=x,F=F,q=q,D=D,B.true=B,V.linear=po_sandwich(Ainv,B,D),alpha=alpha,beta=beta)
}
