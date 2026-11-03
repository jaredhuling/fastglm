set.seed(8201)
cases <- lapply(list(c(20000L,3L,4L),c(20000L,12L,8L)),function(d) {
    n<-d[1L];p<-d[2L];K<-d[3L]
    x<-matrix(rnorm(n*p),n,p,dimnames=list(NULL,paste0('x',seq_len(p))))
    weights<-rep(c(1,1,2,3),length.out=n);weights[seq.int(50L,n,50L)]<-0
    offset<-runif(n,-.2,.2);beta<-seq(.15,.35,length.out=p)/sqrt(p)
    cuts<-qlogis(seq_len(K-1L)/K)
    y<-lapply(1:2,function(j)ordered(cut(drop(x%*%beta)+offset+rlogis(n),
        c(-Inf,cuts,Inf),labels=paste0('category',seq_len(K)))))
    list(x=x,weights=weights,offset=offset,y=y,n=n,p=p,K=K)
})
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=1L) stop('usage: Rscript generate-fixtures.R output-file.rds')
if(file.exists(args[1L])) stop('output already exists')
saveRDS(cases,args[1L])
