# Run from the repository root using the frozen baseline installation.
library(fastglm)
env <- new.env(parent=asNamespace("fastglm"))
sys.source("benchmarks/ordinal-score-performance/development.R",env)
set.seed(110401)
out <- list()
for(case in list(c(20000,3,4),c(20000,12,8),c(100000,6,5))) {
    n<-case[1];p<-case[2];K<-case[3]
    x<-matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0("x",seq_len(p))))
    y<-ordered(cut(.3*x[,1]+rlogis(n),c(-Inf,seq(-1.5,1.5,length.out=K-1),Inf)),labels=seq_len(K))
    fit<-fastglm_ordinal(x,y)
    response<-fastglm:::.ordinal_assessment_response(y,NULL,fit$levels,n)
    args<-list(x,response,numeric(n),fit,seq_len(p),2048L)
    a<-do.call(fastglm:::.ordinal_po_expected_score,args)
    b<-do.call(env$structured_score,args)
    stopifnot(isTRUE(all.equal(a,b,tolerance=1e-8)))
    for(method in c("baseline","structured")) {
        fun<-if(method=="baseline") fastglm:::.ordinal_po_expected_score else env$structured_score
        elapsed<-replicate(7,system.time(do.call(fun,args))[["elapsed"]])
        out[[length(out)+1L]]<-data.frame(n=n,p=p,K=K,method=method,replicate=seq_along(elapsed),seconds=elapsed,
            statistic=b$statistic,statistic.difference=b$statistic-a$statistic)
    }
}
write.csv(do.call(rbind,out),"benchmarks/ordinal-score-performance/development-timing.csv",row.names=FALSE)
capture.output(sessionInfo(),file="benchmarks/ordinal-score-performance/development-session.txt")
print(aggregate(seconds~n+p+K+method,do.call(rbind,out),median))
