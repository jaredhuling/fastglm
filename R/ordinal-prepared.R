#' Prepare predictors for repeated ordinal fits
#'
#' Retains a dense location design and its prediction coding for use with
#' [fastglm_ordinal()] or [fastglm_ordinal_batch()]. Repeated fits reuse weighted
#' centering, scaling, rank checks, and the transformed design when the
#' predictors and effective row frequencies agree. Model scores, information,
#' category mappings, and starting thresholds are recalculated for each fit.
#' @param x a dense numeric matrix without an intercept, or a one-sided formula.
#' @param data data for a formula.
#' @param weights,offset default fitting frequencies and location offsets.
#'   Formula offsets and an additional `offset` argument are combined during
#'   preparation. Predictors must be complete; missing rows are not removed.
#' @param levels optional default outcome labels in increasing order.
#' @param firth default for mean bias reduction in subsequent fits.
#' @param link default cumulative link, as in [fastglm_ordinal()].
#' @details The cache contains ordinary R objects and can be saved with
#'   `saveRDS()`. Changing the design, effective row totals, or the
#'   `standardize` optimizer control rebuilds the relevant setup. For grouped
#'   counts, effective totals are counts times their fitting weights. Changing
#'   category allocations while retaining those totals permits reuse.
#'
#'   Fitting with a prepared design preserves the formula's factor coding and
#'   transformations for prediction. A replacement fitting offset is a total
#'   location offset. After changing prepared formula offsets, predictions on
#'   new data require an explicit replacement `offset`.
#' @returns A `fastglm_ordinal_design` object. Its `cache` environment is shared
#'   by repeated fits; each fitted model retains its own response and the common
#'   original design, without copying the cached transformed design into the fit.
#' @export
fastglm_ordinal_prepare <- function(x,data=NULL,weights=NULL,offset=NULL,levels=NULL,firth=FALSE,link="logit") {
    link <- c("logit","probit","cloglog")[.ordinal_link_code(link)+1L]
    if(!is.logical(firth) || length(firth)!=1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE",call.=FALSE)
    call <- match.call(); tt <- mf <- NULL; coding <- assignment <- NULL
    if(inherits(x,"formula")) {
        if(length(x)!=2L) stop("ordinal preparation requires a one-sided formula",call.=FALSE)
        frame.call <- call[c(1L,match(c("x","data","weights","offset"),names(call),0L))]
        names(frame.call)[names(frame.call)=="x"] <- "formula"
        frame.call[[1L]] <- quote(stats::model.frame)
        frame.call$na.action <- quote(stats::na.fail)
        frame.call$drop.unused.levels <- FALSE
        mf <- eval(frame.call,parent.frame())
        tt <- attr(mf,"terms")
        x <- model.matrix(tt,mf)
        coding <- attr(x,"contrasts")
        assignment <- attr(x,"assign"); assignment <- assignment[assignment!=0L]
        x <- x[,attr(x,"assign")!=0L,drop=FALSE]
        weights <- model.weights(mf)
        offset <- model.offset(mf)
    }
    if(!is.matrix(x) || !is.numeric(x) || is.complex(x) || !nrow(x) || any(!is.finite(x)))
        stop("ordinal preparation requires a finite dense numeric matrix without a location intercept",call.=FALSE)
    if(is.integer(x)) storage.mode(x) <- "double"
    if(ncol(x)>0L && is.null(colnames(x))) colnames(x) <- paste0("x",seq_len(ncol(x)))
    if(anyNA(colnames(x)) || any(!nzchar(colnames(x))) || anyDuplicated(colnames(x)))
        stop("ordinal predictor column names must be distinct and nonmissing",call.=FALSE)
    if(is.null(weights)) weights <- rep(1,nrow(x))
    if(is.null(offset)) offset <- numeric(nrow(x))
    if(!is.numeric(weights) || is.complex(weights) || !is.null(dim(weights)) ||
       length(weights)!=nrow(x) || any(!is.finite(weights)|weights<0) || !any(weights>0))
        stop("prepared ordinal weights must be finite, nonnegative, and match the design rows",call.=FALSE)
    if(!is.numeric(offset) || is.complex(offset) || !is.null(dim(offset)) ||
       length(offset)!=nrow(x) || any(!is.finite(offset)))
        stop("prepared ordinal offsets must be finite and match the design rows",call.=FALSE)
    if(!is.null(levels)) {
        levels <- as.character(levels)
        if(length(levels)<2L || anyNA(levels) || any(!nzchar(levels)) || anyDuplicated(levels))
            stop("prepared ordinal levels must contain distinct nonmissing category labels",call.=FALSE)
    }
    offset <- as.numeric(offset); weights <- as.numeric(weights)
    structure(list(x=x,weights=weights,offset=offset,levels=levels,firth=firth,link=link,
        terms=tt,design.assign=assignment,xlevels=if(!is.null(tt)) stats::.getXlevels(tt,mf),contrasts=coding,
        design.columns=colnames(x),offset.expression=if(!is.null(tt)) call$offset,
        offset.reference=offset,cache=new.env(parent=emptyenv()),call=call),
        class="fastglm_ordinal_design")
}

#' @rdname fastglm_ordinal
#' @export
fastglm_ordinal.fastglm_ordinal_design <- function(x,y,weights=NULL,offset=NULL,levels=NULL,
                                                  firth=NULL,tol=1e-8,maxit=200L,
                                                  block.size=2048L,control=list(),link=NULL,...) {
    if(length(list(...))) stop("unused ordinal fitting arguments",call.=FALSE)
    if(missing(y)) stop("an ordinal response is required",call.=FALSE)
    design <- x
    if(!is.null(design$terms) && !identical(colnames(design$x),design$design.columns))
        stop("prepared ordinal formula columns changed; prepare the revised formula again",call.=FALSE)
    if(is.null(weights)) weights <- design$weights
    original.offset <- is.null(offset) && identical(design$offset,design$offset.reference)
    if(is.null(offset)) offset <- design$offset
    if(is.null(levels)) levels <- design$levels
    if(is.null(firth)) firth <- design$firth
    if(is.null(link)) link <- .ordinal_link(design)
    if(!is.environment(design$cache)) design$cache <- new.env(parent=emptyenv())
    fit <- .ordinal_fit_matrix(design$x,y,weights,offset,levels,firth,tol,maxit,block.size,
                               control,design$cache,link)
    for(key in c("terms","design.assign","xlevels","contrasts","offset.expression")) fit[[key]] <- design[[key]]
    fit$prediction.offset.required <- !is.null(design$terms) && !original.offset
    fit$prepared <- TRUE
    fit$call <- match.call(); fit$call[[1L]] <- quote(fastglm::fastglm_ordinal)
    fit
}

#' Fit several ordinal responses using shared predictors
#'
#' Fits responses sequentially using a prepared ordinal design. The design
#' setup is reused whenever effective row frequencies match. Each response uses
#' its own empirical starting thresholds and category mapping; estimates from
#' an earlier response are not used as starting values.
#' @param design an object from [fastglm_ordinal_prepare()], or a dense numeric
#'   matrix without a location intercept.
#' @param y a list of ordinal responses. Each element may be an ordered factor,
#'   category codes with declared `levels`, or a matrix of category counts.
#'   Use a list even for a single count matrix, so its category columns are not
#'   mistaken for separate responses.
#' @param weights,offset common fitting vectors, or matrices with one column per
#'   response. `NULL` uses the prepared defaults. Replacement offsets are total
#'   location offsets.
#' @param levels common outcome labels or a list with one category order per
#'   response. `NULL` uses the prepared default or each response's own labels.
#' @param firth default to the prepared design's setting, or `FALSE` for a matrix.
#' @param link default to the prepared link, or `"logit"` for a matrix.
#' @param tol,maxit,block.size,control ordinal fitting controls; see [fastglm_ordinal()].
#' @returns A list of `fastglm_ordinal` fits, with the response list's names.
#' @export
fastglm_ordinal_batch <- function(design,y,weights=NULL,offset=NULL,levels=NULL,firth=NULL,
                                  tol=1e-8,maxit=200L,block.size=2048L,control=list(),link=NULL) {
    if(!inherits(design,"fastglm_ordinal_design")) design <- fastglm_ordinal_prepare(design)
    if(!is.list(y) || is.data.frame(y))
        stop("'y' must be a list of ordinal responses; a count matrix is one response",call.=FALSE)
    if(!length(y)) return(list())
    for(value in list(weights,offset)) if(is.matrix(value) && ncol(value)!=length(y))
        stop("batch weights and offsets must have one column per response",call.=FALSE)
    if(is.list(levels) && length(levels)!=length(y))
        stop("batch category orders must have one element per response",call.=FALSE)
    take <- function(value,j) if(is.matrix(value)) value[,j] else value
    out <- lapply(seq_along(y),function(j) {
        fit <- fastglm_ordinal.fastglm_ordinal_design(design,y[[j]],
            weights=take(weights,j),offset=take(offset,j),
            levels=if(is.list(levels)) levels[[j]] else levels,firth=firth,
            tol=tol,maxit=maxit,block.size=block.size,control=control,link=link)
        fit$batch.response <- j
        fit
    })
    names(out) <- names(y)
    out
}
