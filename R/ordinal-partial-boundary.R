# A sufficient certificate for a saturated cumulative-logit model only.
# The mutable state retains at most pn+1 predictor patterns, never input rows.
.ordinal_partial_boundary_start <- function(common,varying,categories,firth,link) {
    if(!isTRUE(firth) || !identical(link,"logit") || common!=0L || varying<1L || categories<3L)
        return(NULL)
    state <- new.env(parent=emptyenv())
    state$active <- TRUE;state$varying <- varying;state$categories <- categories
    state$patterns <- matrix(numeric(),0L,varying)
    state$offsets <- numeric();state$counts <- matrix(numeric(),0L,categories)
    state
}

.ordinal_partial_boundary_add <- function(state,z,codes,counts,weights,offset) {
    if(is.null(state) || !isTRUE(state$active)) return(invisible(state))
    abandon <- function() {
        state$active <- FALSE;state$patterns <- state$offsets <- state$counts <- NULL
        invisible(state)
    }
    grouped <- length(counts)>0L
    pending <- weights>0
    if(grouped) pending <- pending & rowSums(counts*weights)>0
    while(any(pending)) {
        first <- match(TRUE,pending)
        pattern <- z[first,]
        group <- 0L
        if(nrow(state$patterns)) for(j in seq_len(nrow(state$patterns)))
            if(all(pattern==state$patterns[j,])) {group <- j;break}
        if(!group) {
            if(nrow(state$patterns)>=state$varying+1L) return(abandon())
            state$patterns <- rbind(state$patterns,pattern)
            state$offsets <- c(state$offsets,offset[first])
            state$counts <- rbind(state$counts,numeric(state$categories))
            group <- nrow(state$patterns)
        }
        matched <- pending
        for(j in seq_len(state$varying)) matched <- matched & z[,j]==pattern[j]
        # Equality is exact: approximate matching could change the model.
        if(any(offset[matched]!=state$offsets[group])) return(abandon())
        addition <- if(grouped) colSums(counts[matched,,drop=FALSE]*weights[matched]) else
            vapply(seq_len(state$categories),function(k) sum(weights[matched & codes==k]),numeric(1))
        state$counts[group,] <- state$counts[group,]+addition
        if(any(!is.finite(state$counts[group,]))) return(abandon())
        pending[matched] <- FALSE
    }
    invisible(state)
}

.ordinal_partial_boundary_finish <- function(state) {
    if(is.null(state) || !isTRUE(state$active) || nrow(state$patterns)!=state$varying+1L)
        return(NULL)
    # Anchor before scaling. Dividing large, nearly equal coordinates first
    # can destroy exact affine dependence and falsely suggest saturation.
    centered <- sweep(state$patterns,2L,state$patterns[1L,],"-")
    scale <- apply(abs(centered),2L,max)
    if(any(!is.finite(scale)) || any(scale<=0)) return(NULL)
    design <- cbind(1,sweep(centered,2L,scale,"/"))
    if(qr(design,tol=sqrt(.Machine$double.eps))$rank!=ncol(design)) return(NULL)
    empty <- which(state$counts[,seq.int(2L,state$categories-1L),drop=FALSE]==0,arr.ind=TRUE)
    if(!nrow(empty)) return(NULL)
    empty[,2L] <- empty[,2L]+1L
    colnames(empty) <- c("pattern","category")
    list(message=paste0("this fully flexible logit model has no mean-bias-reduced solution with all category probabilities positive: ",
        "an intermediate outcome category is absent within a predictor pattern"),
        patterns=nrow(state$patterns),empty.pattern.categories=empty)
}
