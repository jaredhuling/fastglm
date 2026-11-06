# Keep the declared ordinal scale when unused interior categories are removed
# from the numerical fit. See Kosmidis (2014), section 8.3, remark 1. A tied
# threshold is a boundary extension of the reduced model: convergence and rank
# refer to the reduced coordinates, not separate full-model threshold scores.
.ordinal_response_data <- function(y,weights=NULL,levels=NULL,firth=FALSE) {
    if(!is.logical(firth) || length(firth)!=1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE",call.=FALSE)
    grouped<-is.matrix(y)
    n<-if(grouped) nrow(y) else length(y)
    if(n<1L) stop("the ordinal response must contain observations",call.=FALSE)
    if(is.null(weights)) weights<-rep(1,n)
    if(!is.numeric(weights) || !is.null(dim(weights)) || length(weights)!=n ||
       any(!is.finite(weights)|weights<0))
        stop("ordinal frequency weights must be finite and nonnegative",call.=FALSE)
    weights<-as.numeric(weights)
    integer.frequency<-function(x) all(abs(x-round(x))<=pmin(1e-7,16*.Machine$double.eps*pmax(1,abs(x))))
    if(grouped) {
        if(!is.numeric(y) || any(!is.finite(y)|y<0))
            stop("ordinal category counts must be finite and nonnegative",call.=FALSE)
        if(is.null(levels)) levels<-colnames(y)
        if(is.null(levels)) levels<-as.character(seq_len(ncol(y)))
        if(length(levels)!=ncol(y)) stop("category labels must match the count columns",call.=FALSE)
        if(!is.null(colnames(y)) && !identical(colnames(y),as.character(levels)))
            stop("ordinal count column names must match 'levels' in order",call.=FALSE)
    } else {
        if(is.null(levels)) {
            if(!is.factor(y)) stop("supply a factor with ordered levels or explicit 'levels'",call.=FALSE)
            levels<-base::levels(y)
        }
    }
    levels<-as.character(levels);K<-length(levels)
    if(K<2L || anyNA(levels) || any(!nzchar(levels)) || anyDuplicated(levels))
        stop("at least two distinct, nonmissing ordinal category labels are required",call.=FALSE)
    original.codes<-integer()
    if(grouped) {
        effective<-y*weights
        if(any(!is.finite(effective))) stop("effective ordinal category counts overflow",call.=FALSE)
        if(!integer.frequency(effective)) stop("ordinal fitting requires integer effective category counts",call.=FALSE)
        totals<-rowSums(effective);category.totals<-colSums(effective)
    } else {
        if(!is.null(dim(y))) stop("invalid ordinal response dimensions",call.=FALSE)
        if(is.numeric(y)) {
            if(any(!is.finite(y)|y!=floor(y)|y<1|y>K))
                stop("numeric ordinal responses must be category codes from 1 to length(levels)",call.=FALSE)
            original.codes<-as.integer(y)
        } else if(is.factor(y) && identical(base::levels(y),levels)) {
            original.codes<-as.integer(y)
        } else original.codes<-match(as.character(y),levels)
        if(anyNA(original.codes)) stop("unknown or missing ordinal response category",call.=FALSE)
        if(!integer.frequency(weights)) stop("ordinal fitting requires integer frequency weights",call.=FALSE)
        totals<-weights
        category.totals<-ordinal_category_totals(original.codes,weights,K)
    }
    if(any(!is.finite(totals)) || any(!is.finite(category.totals)) || !is.finite(sum(totals)))
        stop("total ordinal frequencies overflow",call.=FALSE)
    if(!any(totals>0)) stop("positive total ordinal frequency is required",call.=FALSE)
    empty<-which(category.totals==0)
    empty.ends<-intersect(empty,c(1L,K))
    if(!firth && length(empty.ends))
        stop("maximum likelihood has infinite thresholds for empty endpoint categories; use firth = TRUE or explicitly redefine the response scale",call.=FALSE)
    collapsed<-setdiff(empty,c(1L,K))
    retained<-setdiff(seq_len(K),collapsed)
    # Endpoints are always retained, so every original finite threshold maps to
    # a finite reduced threshold. The same map expands covariance by repeating
    # its rows/columns; no inverse of that singular expanded matrix is needed.
    threshold.map<-findInterval(seq_len(K-1L),retained)
    codes<-integer();counts<-matrix(numeric(),0L,0L)
    if(grouped) {
        counts<-y[,retained,drop=FALSE];storage.mode(counts)<-"double"
        colnames(counts)<-levels[retained]
    } else {
        codes<-match(original.codes,retained)
        # A removed category can occur only in zero-frequency observations.
        # Use a valid code for native validation; these rows contribute nothing.
        if(anyNA(codes)) {
            if(any(weights[is.na(codes)]>0)) stop("internal ordinal category mapping error",call.=FALSE)
            codes[is.na(codes)]<-1L
        }
        codes<-as.integer(codes)
    }
    list(codes=codes,counts=counts,weights=weights,totals=as.numeric(totals),
         category.totals=as.numeric(category.totals[retained]),levels=levels[retained],
         original.levels=levels,original.category.totals=as.numeric(category.totals),
         original.codes=original.codes,retained.categories=retained,threshold.map=as.integer(threshold.map),
         collapsed.categories=collapsed,empty.categories=empty,empty.end.categories=empty.ends,
         boundary=length(collapsed)>0L,grouped=grouped,n=n,
         response.kind=if(grouped) "counts" else "codes")
}
