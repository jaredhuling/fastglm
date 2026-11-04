.ordinal_partial_require_common <- function(object,operation) {
    if(inherits(object,"fastglm_ordinal_partial"))
        stop(operation," is not yet available for nonparallel ordinal models",call.=FALSE)
    invisible(NULL)
}

.ordinal_partial_test_columns <- function(object,terms) {
    columns <- object$design.columns
    if(!is.null(object$terms)) {
        labels <- attr(object$terms,"term.labels")
        if(is.null(terms)) terms <- labels
        if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
           any(!terms%in%labels)) stop("unknown or duplicated ordinal formula term",call.=FALSE)
        assignment <- object$design.assign
        if(length(assignment)!=length(columns))
            stop("ordinal term tests require retained design assignments; supply a named contrast",call.=FALSE)
        selected <- which(assignment%in%match(terms,labels))
    } else {
        if(is.null(terms)) terms <- columns
        if(!is.character(terms) || !length(terms) || anyNA(terms) || anyDuplicated(terms) ||
           any(!terms%in%columns)) stop("unknown or duplicated ordinal predictor column",call.=FALSE)
        selected <- match(terms,columns)
    }
    q <- length(object$alpha)
    design.index <- c(rep(NA_integer_,q),object$common,rep(object$nonparallel,q))
    index <- which(design.index%in%selected)
    list(index=index,terms=terms)
}
