#' Read a large dataset in repeatable blocks of rows
#'
#' A source describes how to obtain a block of observations when a fitting
#' function needs it. This permits repeated model-fitting passes without
#' loading the complete dataset into memory. The supplied function, called
#' a callback, receives the block number and returns its observations.
#'
#' When a formula is supplied, `prototype` is an example data frame used to
#' establish the design columns, factor levels, and transformations. This
#' recorded specification, called the schema, is reused for every block so
#' that a column has the same meaning throughout the analysis. For splines
#' and other data-dependent transformations, the prototype must establish
#' the intended basis, including knots and boundary knots.
#' @param callback function of chunk index returning a data frame, or a list
#'   with `X`, `y`, and optional `weights` and `offset`. Two-component models
#'   may also supply `Z` for the binary design and a list of `count` and `zero`
#'   offsets; otherwise `Z` defaults to `X` and a vector offset applies to
#'   the count or positive-outcome component.
#' @param n_chunks positive number of chunks.
#' @param reset optional function called before chunk one on every pass.
#' @param formula optional model formula for data-frame blocks. A formula
#'   such as `y ~ x | z` specifies separate count/positive and binary designs.
#' @param prototype example data frame defining all intended factor levels
#'   and transformations; it need not contain all observations.
#' @param weights,offset optional column names in data-frame blocks. For
#'   a two-component formula, two offset column names supply count/positive
#'   and binary offsets, respectively.
#' @returns A `fastglm_source` object for [fastglm_streaming()],
#'   [fastglm_ordinal_streaming()], or the count and two-part fitting functions.
#' @export
fastglm_source <- function(callback, n_chunks, reset = NULL, formula = NULL,
                           prototype = NULL, weights = NULL, offset = NULL) {
    .check_iterations(1e-8,n_chunks)
    if (!is.function(callback) || (!is.null(reset) && !is.function(reset))) stop("callback and reset must be functions",call.=FALSE)
    if(!is.null(weights) && (!is.character(weights) || length(weights)!=1L || is.na(weights)))
        stop("weights must name one data column",call.=FALSE)
    if(!is.null(offset) && (!is.character(offset) || !length(offset) || length(offset)>2L || anyNA(offset)))
        stop("offset must name one or two data columns",call.=FALSE)
    schema <- NULL
    if (!is.null(formula)) {
        if (is.null(prototype)) stop("formula sources require a prototype defining the model schema",call.=FALSE)
        f <- Formula::Formula(formula)
        if(length(f)[1L]!=1L || length(f)[2L]>2L)
            stop("source formulas require one response and at most two right-hand sides",call.=FALSE)
        build <- function(j) {
            mf <- model.frame(stats::formula(f,lhs=1L,rhs=j),prototype,na.action=na.fail)
            tt <- attr(mf,"terms");xx <- model.matrix(tt,mf)
            list(terms=tt,xlevels=.getXlevels(tt,mf),contrasts=attr(xx,"contrasts"),
                 columns=colnames(xx),assign=attr(xx,"assign"))
        }
        schema <- build(1L)
        if(length(f)[2L]==2L) {
            parts <- list(count=schema,zero=build(2L))
            schema <- lapply(names(schema),function(nm) lapply(parts,`[[`,nm))
            names(schema) <- names(parts$count)
        } else if(length(offset)>1L) stop("two offset columns require a two-component formula",call.=FALSE)
        schema$formula <- f
        schema$offset <- offset
    }
    fetch <- function(k) {
        if (k == 1L && !is.null(reset)) reset()
        z <- callback(k)
        if (is.null(schema)) return(z)
        column <- function(nm) {
            if(!nm %in% names(z)) stop("source block lacks column '",nm,"'",call.=FALSE)
            z[[nm]]
        }
        part <- function(tt,xlevels,contrasts,columns) {
            mf <- model.frame(tt,z,xlev=xlevels,na.action=na.fail)
            xx <- model.matrix(tt,mf,contrasts.arg=contrasts)
            if(!identical(colnames(xx),columns)) stop("chunk model columns do not match the source schema",call.=FALSE)
            off <- model.offset(mf);if(is.null(off)) off <- rep(0,nrow(xx))
            list(X=xx,y=model.response(mf),offset=off)
        }
        if(inherits(schema$terms,"terms")) {
            out <- part(schema$terms,schema$xlevels,schema$contrasts,schema$columns)
            out$offset.zero <- out$offset
            if(length(offset)) out$offset <- out$offset+column(offset)
        } else {
            p <- lapply(c("count","zero"),function(k)
                part(schema$terms[[k]],schema$xlevels[[k]],schema$contrasts[[k]],schema$columns[[k]]))
            out <- list(X=p[[1L]]$X,Z=p[[2L]]$X,y=p[[1L]]$y,
                        offset=list(count=p[[1L]]$offset,zero=p[[2L]]$offset))
            for(j in seq_along(offset)) out$offset[[j]] <- out$offset[[j]]+column(offset[[j]])
        }
        out$weights <- if(!is.null(weights)) column(weights)
        out
    }
    structure(list(callback=fetch,n_chunks=as.integer(n_chunks),schema=schema),class="fastglm_source")
}

#' Create a source from indexed database queries
#'
#' Each query must return a stable, disjoint chunk in a deterministic order.
#' The database snapshot must remain unchanged throughout fitting and inference.
#' @param connection a DBI connection.
#' @param queries character vector of read-only SELECT queries, one per chunk.
#' @param formula,prototype,weights,offset passed to [fastglm_source()].
#' @returns A resettable `fastglm_source`.
#' @export
fastglm_source_dbi <- function(connection, queries, formula, prototype,
                               weights = NULL, offset = NULL) {
    if (!requireNamespace("DBI",quietly=TRUE)) stop("install DBI to use database sources",call.=FALSE)
    if (!is.character(queries) || !length(queries) || any(!grepl("^\\s*(SELECT|WITH)\\b",queries,ignore.case=TRUE))) stop("queries must be read-only SELECT statements",call.=FALSE)
    fastglm_source(function(k) DBI::dbGetQuery(connection,queries[[k]]),length(queries),
                    formula=formula,prototype=prototype,weights=weights,offset=offset)
}

#' Create a source from Parquet partitions
#'
#' Each file is one bounded chunk. The formula schema is supplied explicitly
#' so that factor contrasts and transformations remain fixed across files.
#' @param files character vector of Parquet paths.
#' @param formula,prototype,weights,offset passed to [fastglm_source()].
#' @returns A `fastglm_source`.
#' @export
fastglm_source_arrow <- function(files, formula, prototype, weights=NULL, offset=NULL) {
    if (!requireNamespace("arrow",quietly=TRUE)) stop("install arrow to use Parquet sources",call.=FALSE)
    if (!is.character(files) || !length(files)) stop("supply at least one Parquet file",call.=FALSE)
    fastglm_source(function(k) as.data.frame(arrow::read_parquet(files[[k]])),length(files),
                    formula=formula,prototype=prototype,weights=weights,offset=offset)
}
