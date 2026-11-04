ordinal_cov_source_fixture <- function() {
    n<-24L
    x<-cbind(x=seq(-1,1,length.out=n),z=sin(seq_len(n)))
    counts<-cbind(low=4L+(seq_len(n)%%3L),middle=5L+(seq_len(n)%%2L),high=3L+(seq_len(n)%%4L))
    list(x=x,y=counts,weights=c(rep(0,4),rep(c(1,2),10)),
        offset=.1*cos(seq_len(n)),cluster=rep(1:5,length.out=n))
}

ordinal_cov_test_source <- function(data,size=5L) {
    rows<-split(seq_len(nrow(data$x)),ceiling(seq_len(nrow(data$x))/size))
    source<-fastglm_source(function(k) {
        at<-rows[[k]]
        list(X=data$x[at,,drop=FALSE],y=if(is.matrix(data$y)) data$y[at,,drop=FALSE] else data$y[at],
            weights=data$weights[at],offset=data$offset[at])
    },length(rows))
    list(source=source,rows=rows,cluster=function(k) data$cluster[rows[[k]]])
}

test_that("ordinal source sandwiches agree with dense, prepared, and expanded fits", {
    a<-ordinal_cov_source_fixture();a$cluster[a$weights==0]<-NA
    effective<-a$y*a$weights
    index<-rep(seq_len(nrow(a$x)),rowSums(effective))
    expanded<-list(x=a$x[index,,drop=FALSE],
        y=ordered(unlist(lapply(seq_len(nrow(a$x)),function(i) rep(1:3,effective[i,]))),levels=1:3),
        weights=rep(1,length(index)),offset=a$offset[index],cluster=a$cluster[index])
    blocks<-ordinal_cov_test_source(a,4L)
    expanded.blocks<-ordinal_cov_test_source(expanded,31L)
    for(link in c("logit","probit","cloglog")) for(br in c(FALSE,TRUE)) for(partial in c(FALSE,TRUE)) {
        selection<-if(partial) "z" else NULL
        dense<-fastglm_ordinal(a$x,a$y,weights=a$weights,offset=a$offset,
            nonparallel=selection,link=link,firth=br,tol=1e-10)
        design<-fastglm_ordinal_prepare(a$x,weights=a$weights,offset=a$offset,
            nonparallel=selection,link=link,firth=br)
        prepared<-fastglm_ordinal(design,a$y,tol=1e-10)
        source<-fastglm_ordinal_streaming(blocks$source,nonparallel=selection,link=link,firth=br,tol=1e-10)
        individuals<-fastglm_ordinal_streaming(expanded.blocks$source,
            nonparallel=selection,link=link,firth=br,tol=1e-10)
        expect_true(source$converged);expect_true(individuals$converged)
        for(type in c("HC0","HC1")) {
            reference<-fastglm_vcov(dense,type,block.size=3L)
            actual<-fastglm_vcov(source,type,block.size=2L)
            expect_equal(actual,reference,tolerance=2e-8)
            expect_equal(actual,fastglm_vcov(prepared,type,block.size=3L),tolerance=2e-8)
            expect_equal(c(actual),c(fastglm_vcov(individuals,type,block.size=11L)),tolerance=2e-8)
            clustered<-fastglm_vcov(source,type,cluster=blocks$cluster,block.size=2L)
            expected<-fastglm_vcov(dense,type,cluster=a$cluster,block.size=3L)
            expect_equal(clustered,expected,tolerance=2e-8)
            expect_equal(c(clustered),c(fastglm_vcov(individuals,type,
                cluster=expanded.blocks$cluster,block.size=11L)),tolerance=2e-8)
            expect_equal(attr(clustered,"clusters"),5L)
            expect_equal(attr(clustered,"frequency.total"),length(index))
        }
        expected<-fastglm_vcov(source,cluster=blocks$cluster,cadjust=FALSE)
        corrected<-vcov(source,type="HC1",cluster=blocks$cluster)
        expect_equal(c(corrected),c(expected)*(5/4)*(source$frequency-1)/(source$frequency-source$rank),tolerance=1e-12)
        tab<-summary(source,vcov.=corrected)
        standard.errors<-if(partial) tab$coefficients[,2L] else c(tab$thresholds[,2L],tab$location[,2L])
        expect_equal(standard.errors,sqrt(diag(corrected)),ignore_attr=TRUE)
        if(partial) expect_equal(c(vcov(source,type="HC0",component="nonparallel")),
            c(fastglm_vcov(dense)[4:5,4:5]),tolerance=2e-8)
        if(br) expect_match(attr(corrected,"covariance.scope"),"first-order ordinary-score")
        expect_null(source$x);expect_null(source$y)
    }
})

test_that("source cluster callbacks retain original rows and exact cross-block identifiers", {
    a<-ordinal_cov_source_fixture();a$x[a$weights==0,]<-1e200
    a$offset[a$weights==0]<-1e200
    a$cluster<-c(1,1+1e-15,2,3,4)[a$cluster]
    a$cluster[a$weights==0]<-NA
    blocks<-ordinal_cov_test_source(a,4L)
    fit<-fastglm_ordinal_streaming(blocks$source,nonparallel="z",firth=TRUE,tol=1e-10)
    dense<-fastglm_ordinal(a$x,a$y,weights=a$weights,offset=a$offset,nonparallel="z",firth=TRUE,tol=1e-10)
    calls<-integer();before<-fit$state$passes
    actual<-fastglm_vcov(fit,cluster=function(k) {
        calls<<-c(calls,k)
        blocks$cluster(k)
    },block.size=1L)
    expect_identical(calls,seq_len(fit$source$n_chunks))
    expect_identical(fit$state$passes,before+1L)
    expect_identical(attr(actual,"clusters"),5L)
    expect_equal(actual,fastglm_vcov(dense,cluster=a$cluster),tolerance=1e-9)
    expect_equal(fastglm_vcov(fit,cluster=blocks$cluster,block.size=100L),actual,tolerance=1e-12)
    labels<-c("cluster:1","a:b","NA","",".")[match(a$cluster,unique(a$cluster[!is.na(a$cluster)]))]
    expected<-fastglm_vcov(dense,cluster=labels)
    factors<-function(k) factor(labels[blocks$rows[[k]]],levels=rev(unique(labels[!is.na(labels)])))
    expect_equal(fastglm_vcov(fit,cluster=factors),expected,tolerance=1e-9)
    # Integer and double blocks use the same numeric identifier semantics.
    identifiers<-rep(1:5,length.out=nrow(a$x));identifiers[a$weights==0]<-NA
    mixed<-function(k) if(k%%2L) identifiers[blocks$rows[[k]]] else as.numeric(identifiers[blocks$rows[[k]]])
    expect_equal(fastglm_vcov(fit,cluster=mixed),fastglm_vcov(dense,cluster=identifiers),tolerance=1e-9)
})

test_that("source cluster labels and covariance controls fail explicitly", {
    a<-ordinal_cov_source_fixture();blocks<-ordinal_cov_test_source(a,4L)
    fit<-fastglm_ordinal_streaming(blocks$source,nonparallel="z",firth=TRUE)
    expect_error(fastglm_vcov(fit,cluster=a$cluster),"function of block index")
    expect_error(fastglm_vcov(fit,cluster=function(k) 1:3),"original block rows")
    expect_error(fastglm_vcov(fit,cluster=function(k) matrix(1:4,4)),"vector of identifiers")
    expect_error(fastglm_vcov(fit,cluster=function(k) as.list(1:4)),"vector of identifiers")
    expect_error(fastglm_vcov(fit,cluster=function(k) complex(length=4)),"vector of identifiers")
    expect_error(fastglm_vcov(fit,cluster=function(k) rep(NA,4)),"positive-frequency rows")
    expect_error(fastglm_vcov(fit,cluster=function(k) if(k==2L) 1:4 else letters[1:4]),"types changed")
    expect_error(fastglm_vcov(fit,cluster=function(k) rep(1,4)),"at least two")
    expect_error(fastglm_vcov(fit,"HC2"),"HC0.*HC1")
    expect_error(fastglm_vcov(fit,cadjust=1),"cadjust")
    expect_error(fastglm_vcov(fit,block.size=0),"block.size")
    expect_error(vcov(fit,cluster=blocks$cluster),"require.*HC0")
    fit$converged<-FALSE
    expect_error(fastglm_vcov(fit),"converged fit")
    x<-matrix(numeric(),2L,0L)
    source<-fastglm_source(function(k) list(X=x[k,,drop=FALSE],y=matrix(c(3,0,4),1,
        dimnames=list(NULL,c("low","middle","high")))),2L)
    fit<-suppressWarnings(fastglm_ordinal_streaming(source,firth=TRUE))
    expect_error(fastglm_vcov(fit),"every declared category")
})

test_that("formula source covariance preserves term coding and reread schema checks", {
    a<-ordinal_cov_source_fixture()
    data<-data.frame(x=a$x[,1],g=factor(rep(letters[1:3],8)),off=a$offset,w=a$weights)
    data$counts<-a$y
    rows<-split(seq_len(nrow(data)),ceiling(seq_len(nrow(data))/5))
    source<-fastglm_source(function(k) data[rows[[k]],],length(rows),
        formula=counts~x+g+offset(off),prototype=data,weights="w")
    fit<-fastglm_ordinal_streaming(source,nonparallel=~g,firth=TRUE,link="probit",tol=1e-10)
    dense<-fastglm_ordinal(counts~x+g+offset(off),data,weights=w,
        nonparallel=~g,firth=TRUE,link="probit",tol=1e-10)
    groups<-function(k) a$cluster[rows[[k]]]
    expect_equal(fastglm_vcov(fit),fastglm_vcov(dense),tolerance=2e-8)
    expect_equal(fastglm_vcov(fit,"HC1",cluster=groups),
        fastglm_vcov(dense,"HC1",cluster=a$cluster),tolerance=2e-8)
    data$x[6]<-NA_real_
    expect_error(fastglm_vcov(fit),"finite|missing|NA")
    data$x[6]<-a$x[6,1]
    colnames(data$counts)<-c("changed","middle","high")
    expect_error(fastglm_vcov(fit),"count columns|column names|category")
})

test_that("source covariance preserves multiplication before frequency normalization", {
    x<-matrix(numeric(),2L,0L)
    source<-fastglm_source(function(k) list(X=x[k,,drop=FALSE],
        y=matrix(rep(1e308,3),1,dimnames=list(NULL,letters[1:3])),
        weights=if(k==1L) 0 else 1e-308),2L)
    fit<-fastglm_ordinal_streaming(source,firth=TRUE)
    dense<-fastglm_ordinal(matrix(numeric(),1L,0L),matrix(rep(1,3),1,
        dimnames=list(NULL,letters[1:3])),firth=TRUE)
    expect_equal(fastglm_vcov(fit),fastglm_vcov(dense),tolerance=1e-10)
})
