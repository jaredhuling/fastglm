# Ordinary cumulative-logit scores and their observed negative Jacobian.
# For an interior category k, with eta_s = alpha_s - x beta - offset,
# log(pi_k) = log(plogis(eta_k)) + log(plogis(-eta_{k-1}))
#             + log(1 - exp(alpha_{k-1} - alpha_k)).
# Its negative Hessian is g_k z_k z_k' + g_{k-1} z_{k-1} z_{k-1}'
# plus t(1+t) dd', where t=1/expm1(alpha_k-alpha_{k-1}) and
# d selects the threshold difference. This is observed, not expected, information.
# Only block-sized arrays are formed; category scores have at most two nonzero
# threshold entries, so no observations-by-categories stacked design is needed.
.ordinal_sandwich_block <- function(x,alpha,beta,offset,codes,counts,weights,
                                    frequency=1,independent=TRUE,link="logit") {
    if(link!="logit") return(.ordinal_link_sandwich(x,alpha,beta,offset,codes,counts,weights,frequency,independent,link))
    n<-nrow(x);p<-ncol(x);q<-length(alpha);d<-q+p
    eta<-outer(-streaming_qr_eta(x,beta,offset),alpha,"+")
    cumulative<-plogis(eta);survival<-plogis(-eta)
    derivative<-cumulative*survival
    gap<-diff(alpha)
    if(any(!is.finite(gap)|gap<=0)) stop("ordinal thresholds must be finite and strictly ordered",call.=FALSE)
    tgap<-exp(-gap)/(-expm1(-gap));curvature<-tgap*(1+tgap)
    if(any(!is.finite(curvature))) stop("ordinal threshold curvature is not finite",call.=FALSE)
    information<-meat<-matrix(0,d,d)
    row.threshold<-information.weight<-matrix(0,n,q)
    row.location<-numeric(n)
    if(independent) {cross.weight<-matrix(0,n,q);location.weight<-numeric(n)}
    grouped<-length(counts)>0L
    for(k in seq_len(q+1L)) {
        # Multiplication precedes division, preserving tiny prior weights paired
        # with large raw counts. No unweighted count row sum is formed.
        mass<-if(grouped) weights*counts[,k]/frequency else (weights/frequency)*(codes==k)
        if(!any(mass>0)) next
        upper<-if(k<=q) survival[,k]+if(k>1L) tgap[k-1L] else 0 else numeric(n)
        lower<-if(k>1L) -cumulative[,k-1L]-if(k<=q) tgap[k-1L] else 0 else numeric(n)
        # Gap contributions cancel from the location score. Compute this directly
        # to avoid subtracting two large threshold scores for a narrow category.
        location<-(if(k>1L) cumulative[,k-1L] else 0)-(if(k<=q) survival[,k] else 0)
        row.location<-row.location+mass*location
        if(k<=q) {
            row.threshold[,k]<-row.threshold[,k]+mass*upper
            information.weight[,k]<-information.weight[,k]+mass*derivative[,k]
            if(independent) {
                meat[k,k]<-meat[k,k]+sum(mass*upper^2)
                cross.weight[,k]<-cross.weight[,k]+mass*upper*location
            }
        }
        if(k>1L) {
            j<-k-1L
            row.threshold[,j]<-row.threshold[,j]+mass*lower
            information.weight[,j]<-information.weight[,j]+mass*derivative[,j]
            if(independent) {
                meat[j,j]<-meat[j,j]+sum(mass*lower^2)
                cross.weight[,j]<-cross.weight[,j]+mass*lower*location
            }
        }
        if(k>1L && k<=q) {
            j<-k-1L;h<-sum(mass)*curvature[j]
            information[cbind(c(j,k),c(j,k))]<-information[cbind(c(j,k),c(j,k))]+h
            information[j,k]<-information[j,k]-h;information[k,j]<-information[j,k]
            if(independent) {
                meat[j,k]<-meat[j,k]+sum(mass*lower*upper);meat[k,j]<-meat[j,k]
            }
        }
        if(independent) location.weight<-location.weight+mass*location^2
    }
    ii<-seq_len(q)
    information[cbind(ii,ii)]<-information[cbind(ii,ii)]+colSums(information.weight)
    if(p) {
        jj<-q+seq_len(p)
        information[ii,jj]<- -crossprod(information.weight,x)
        information[jj,ii]<-t(information[ii,jj,drop=FALSE])
        information[jj,jj]<-crossprod(x,x*rowSums(information.weight))
        if(independent) {
            meat[ii,jj]<-crossprod(cross.weight,x)
            meat[jj,ii]<-t(meat[ii,jj,drop=FALSE])
            meat[jj,jj]<-crossprod(x,x*location.weight)
        }
    }
    score<-c(colSums(row.threshold),if(p) drop(crossprod(x,row.location)))
    rows<-if(independent) NULL else cbind(row.threshold,x*row.location)
    if(any(!is.finite(information)) || any(!is.finite(meat)) || any(!is.finite(score)) ||
       (!is.null(rows) && any(!is.finite(rows))))
        stop("ordinal score or observed information is not finite",call.=FALSE)
    list(information=information,meat=meat,score=score,rows=rows)
}

.ordinal_cluster_vector <- function(cluster,object) {
    if(is.null(cluster)) return(NULL)
    if(!is.atomic(cluster) || !is.null(dim(cluster)) || is.complex(cluster))
        stop("ordinal clusters must be a vector of identifiers",call.=FALSE)
    n<-nrow(object$x)
    if(length(cluster)!=n) {
        omitted<-as.integer(object$na.action)
        if(!length(omitted) || length(cluster)!=n+length(omitted) || anyNA(omitted) ||
           any(omitted<1L|omitted>length(cluster)) || anyDuplicated(omitted))
            stop("ordinal cluster identifiers must match retained rows, or rows before NA omission after any subset",call.=FALSE)
        cluster<-cluster[-omitted]
    }
    if(anyNA(cluster[object$category$totals>0]))
        stop("ordinal cluster identifiers are missing for positive-frequency rows",call.=FALSE)
    # Numeric labels that differ beyond printed precision must remain distinct.
    # Use exact matching before converting the integer keys to character below.
    match(cluster,unique(cluster))
}

.ordinal_observed_inverse <- function(information) {
    scale<-sqrt(diag(information))
    if(any(!is.finite(information)) || any(!is.finite(scale)|scale<=0))
        stop("ordinal observed information is not positive definite; sandwich covariance is unavailable",call.=FALSE)
    normalized<-information/outer(scale,scale)
    factor<-tryCatch(chol(normalized),error=function(e) NULL)
    condition<-if(is.null(factor)) 0 else rcond(normalized)
    if(is.null(factor) || !is.finite(condition) || condition<1e-14)
        stop("ordinal observed information is numerically singular; sandwich covariance is unavailable",call.=FALSE)
    chol2inv(factor)/outer(scale,scale)
}

.ordinal_robust_vcov <- function(object,type="HC0",cluster=NULL,cadjust=TRUE,block.size=8192L) {
    .ordinal_check_inference(object)
    type<-match.arg(type,c("HC0","HC1"))
    if(!is.logical(cadjust) || length(cadjust)!=1L || is.na(cadjust))
        stop("'cadjust' must be TRUE or FALSE",call.=FALSE)
    if(!is.numeric(block.size) || is.complex(block.size) || length(block.size)!=1L ||
       !is.finite(block.size) || block.size<1 || block.size>.Machine$integer.max || block.size!=floor(block.size))
        stop("'block.size' must be a positive integer",call.=FALSE)
    if(inherits(object,"fastglm_ordinal_stream"))
        return(.ordinal_stream_robust_vcov(object,type,cluster,cadjust,as.integer(block.size)))
    category<-object$category
    if(is.null(category) || !is.matrix(object$x) || nrow(object$x)!=length(category$totals))
        stop("ordinal sandwich covariance requires the retained design and response",call.=FALSE)
    if(isTRUE(object$boundary) || any(category$original.category.totals==0))
        stop("ordinal sandwich covariance requires every declared category to have positive frequency; empty-category inference is not implemented",call.=FALSE)
    cluster<-.ordinal_cluster_vector(cluster,object)
    accumulator<-.ordinal_sandwich_accumulator(object,sum(category$totals),type,!is.null(cluster),cadjust)
    .design_blocks(object$x,function(x,i) {
        counts<-if(category$grouped) category$counts[i,,drop=FALSE] else matrix(numeric(),0,0)
        accumulator$add(x,object$offset[i],if(category$grouped) integer() else category$codes[i],
            counts,category$weights[i],category$totals[i],if(!is.null(cluster)) cluster[i],rows=i)
    },as.integer(block.size))
    accumulator$finish()
}

# Dense and source fits share the same finite-sample aggregation. Information
# and independent meat are divided by N; cluster scores are divided by N before
# their outer products. Only the readers and cluster-key construction differ.
.ordinal_sandwich_accumulator <- function(object,N,type,clustered,cadjust) {
    partial<-inherits(object,"fastglm_ordinal_partial")
    q<-length(object$working$alpha);p<-length(object$beta)
    d<-length(object$working$coefficients)
    if(!is.finite(N) || N<=0) stop("ordinal total frequency must be finite and positive",call.=FALSE)
    if(type=="HC1" && N<=d) stop("ordinal HC1 covariance requires total frequency greater than the number of parameters",call.=FALSE)
    information<-meat<-matrix(0,d,d)
    clusters<-if(clustered) new.env(hash=TRUE,parent=emptyenv()) else NULL
    add<-function(x,offset,codes,counts,weights,totals,groups=NULL,rows=NULL) {
        positive<-totals>0
        if(!any(positive)) return(invisible(NULL))
        x<-x[positive,,drop=FALSE];offset<-offset[positive];weights<-weights[positive]
        if(length(counts)) counts<-counts[positive,,drop=FALSE] else codes<-codes[positive]
        if(!is.null(rows)) rows<-rows[positive]
        if(clustered) groups<-groups[positive]
        if(!partial && p) x<-sweep(sweep(x,2,object$working$center,"-"),2,object$working$scale,"/")
        z<-if(partial) .ordinal_partial_sandwich_block(object,x,offset,codes,counts,weights,
            frequency=N,independent=!clustered,rows=rows) else
            .ordinal_sandwich_block(x,object$working$alpha,object$working$beta,offset,
                codes,counts,weights,frequency=N,independent=!clustered,link=.ordinal_link(object))
        information<<-information+z$information
        if(!clustered) meat<<-meat+z$meat else {
            summed<-rowsum(z$rows,as.character(groups),reorder=FALSE)
            for(j in seq_len(nrow(summed))) {
                key<-paste0("cluster:",rownames(summed)[j]);value<-summed[j,]
                if(exists(key,clusters,inherits=FALSE)) value<-value+get(key,clusters,inherits=FALSE)
                assign(key,value,clusters)
            }
        }
        invisible(NULL)
    }
    finish<-function() {
    G<-NA_integer_;cluster.factor<-1
    if(clustered) {
        keys<-ls(clusters,all.names=TRUE);G<-length(keys)
        if(G<2L) stop("ordinal cluster covariance requires at least two positive-frequency clusters",call.=FALSE)
        for(key in keys) meat<-meat+tcrossprod(get(key,clusters,inherits=FALSE))
        if(cadjust) cluster.factor<-G/(G-1)
    }
    hc1.factor<-if(type=="HC1") (if(!clustered) N else N-1)/(N-d) else 1
    inverse<-.ordinal_observed_inverse(information)
    # H and independent meat were divided by N. Cluster rows were divided by N
    # before aggregation, avoiding overflow from squaring large frequencies.
    v<-inverse%*%meat%*%inverse
    if(!clustered) v<-v/N
    v<-v*hc1.factor*cluster.factor
    transform<-if(partial) object$working$T else diag(d)
    if(!partial && p) {
        transform[seq_len(q),q+seq_len(p)]<-matrix(object$working$center/object$working$scale,q,p,byrow=TRUE)
        transform[q+seq_len(p),q+seq_len(p)]<-diag(1/object$working$scale,nrow=p)
    }
    v<-transform%*%v%*%t(transform);v<-(v+t(v))/2
    if(any(!is.finite(v))) stop("ordinal sandwich covariance cannot be represented in the coefficient units",call.=FALSE)
    dimnames(v)<-rep(list(names(coef(object))),2L)
    attr(v,"covariance.type")<-type
    attr(v,"covariance.scope")<-if(isTRUE(object$firth))
        "first-order ordinary-score sandwich at mean-bias-reduced estimate" else "ordinary-score sandwich"
    attr(v,"estimating.equations")<-paste0("ordinary cumulative-",.ordinal_link(object)," likelihood scores")
    attr(v,"bread")<-"inverse observed negative score Jacobian"
    attr(v,"frequency.total")<-N
    attr(v,"clusters")<-G
    attr(v,"hc1.factor")<-hc1.factor
    attr(v,"cluster.factor")<-cluster.factor
    v
    }
    list(add=add,finish=finish)
}
