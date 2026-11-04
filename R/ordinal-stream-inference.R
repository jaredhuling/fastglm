# A source cluster callback returns identifiers in original block row order.
# The registry retains only distinct positive-frequency labels. R's exact
# matching preserves numeric labels beyond their printed precision; converting
# those labels directly to environment names would merge distinct clusters.
.ordinal_source_cluster_registry <- function() {
    labels<-NULL;kind<-NULL
    function(value,totals) {
        if(!is.atomic(value) || !is.null(dim(value)) || is.complex(value) ||
           !typeof(value)%in%c("logical","integer","double","character","raw"))
            stop("ordinal source clusters must be a vector of identifiers",call.=FALSE)
        if(length(value)!=length(totals))
            stop("ordinal source cluster identifiers must match the original block rows",call.=FALSE)
        positive<-totals>0
        if(anyNA(value[positive]))
            stop("ordinal cluster identifiers are missing for positive-frequency rows",call.=FALSE)
        out<-rep(NA_integer_,length(value))
        if(!any(positive)) return(out)
        value<-value[positive]
        if(is.factor(value)) value<-as.character(value)
        if(!is.object(value) && is.integer(value)) value<-as.numeric(value)
        current.kind<-list(type=typeof(value),class=class(value))
        if(is.null(kind)) {kind<<-current.kind;labels<<-value[FALSE]}
        if(!identical(current.kind,kind))
            stop("ordinal source cluster identifier types changed between positive-frequency blocks",call.=FALSE)
        index<-match(value,labels)
        if(anyNA(index)) {
            labels<<-c(labels,unique(value[is.na(index)]))
            index<-match(value,labels)
        }
        out[positive]<-index
        out
    }
}

.ordinal_stream_robust_vcov <- function(object,type,cluster,cadjust,block.size) {
    category<-object$category
    if(is.null(category) || isTRUE(object$boundary) || any(category$original.category.totals==0))
        stop("ordinal sandwich covariance requires every declared category to have positive frequency; empty-category inference is not implemented",call.=FALSE)
    if(!is.function(object$fetch) || is.null(object$source$n_chunks))
        stop("ordinal source covariance requires its recorded block reader",call.=FALSE)
    if(!is.null(cluster) && !is.function(cluster))
        stop("ordinal source clusters require a function of block index",call.=FALSE)
    accumulator<-.ordinal_sandwich_accumulator(object,object$frequency,type,!is.null(cluster),cadjust)
    register<-if(!is.null(cluster)) .ordinal_source_cluster_registry()
    first<-0
    for(k in seq_len(object$source$n_chunks)) {
        z<-object$fetch(k)
        groups<-if(!is.null(cluster)) register(cluster(k),z$totals)
        .design_blocks(z$X,function(x,i) {
            counts<-if(category$grouped) z$counts[i,,drop=FALSE] else matrix(numeric(),0,0)
            accumulator$add(x,z$offset[i],if(category$grouped) integer() else z$codes[i],
                counts,z$weights[i],z$totals[i],if(!is.null(groups)) groups[i],rows=first+i)
        },block.size)
        first<-first+nrow(z$X)
    }
    accumulator$finish()
}
