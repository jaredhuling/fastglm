disk_configuration <- function() list(
    baseline="2619731",seed=155701L,repetitions=3L,
    cases=data.frame(id=c("regular","large","wide"),n=c(50000L,500000L,12000L),
        K=c(5L,5L,10L),r=c(4L,4L,12L),block=c(5000L,5000L,3000L)),
    components=data.frame(id=c("rows","larger_rows","parameters"),
        n=c(40000L,120000L,6000L),K=c(7L,4L,12L),r=c(5L,3L,10L)),
    links=c("logit","probit","cloglog"),tol=1e-8,maxit=200L)

disk_fixture <- function(n,K,r,link,seed,first=1L) {
    set.seed(seed)
    x<-matrix(runif(n*(2L+r),-1,1),n,2L+r)
    colnames(x)<-c("common1","common2",paste0("vary",seq_len(r)))
    q<-K-1L;alpha<-seq(-1.4,1.3,length.out=q);beta<-c(.17,-.12)
    gamma<-outer(rep(.06/r,r),rep(1,q))+outer(rep(.02/r,r),cos(seq_len(q)))
    offset<-.05*cos(seq.int(first,length.out=n))
    eta<-matrix(alpha,n,q,byrow=TRUE)-drop(x[,1:2,drop=FALSE]%*%beta)-offset-
        x[,-(1:2),drop=FALSE]%*%gamma
    inverse<-switch(link,logit=plogis,probit=pnorm,cloglog=function(a)-expm1(-exp(a)))
    F<-inverse(eta);probability<-cbind(F,1)-cbind(0,F)
    stopifnot(all(probability>0))
    counts<-1+floor(20*probability);colnames(counts)<-paste0("category",seq_len(K))
    list(X=x,y=counts,weights=rep(c(1,2,1,1,0),length.out=n),offset=offset,
        theta=c(alpha,beta,gamma))
}

disk_grid <- function(config) {
    cells<-list()
    for(i in seq_len(nrow(config$cases))) {
        case<-config$cases[i,]
        links<-if(case$id=="large") "logit" else config$links
        for(link in links) for(firth in c(FALSE,TRUE)) {
            models<-if(case$id=="regular" && link=="logit") c("partial","common") else "partial"
            for(model in models) for(representation in c("source","dense"))
                cells[[length(cells)+1L]]<-data.frame(case=case$id,link=link,firth=firth,
                    model=model,representation=representation,n=case$n,K=case$K,r=case$r,block=case$block)
        }
    }
    do.call(rbind,cells)
}
