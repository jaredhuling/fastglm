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
#'   and binary offsets, respectively. For a one-part formula, multinomial
#'   fits ([fastglm_multinom()]) accept one offset column name per linear
#'   predictor, in order (or one name, applied to every linear predictor);
#'   other models accept one name, and report an error at the start of the fit
#'   otherwise.
#' @param cache `FALSE` (default) reads every block from `callback` on every
#'   pass. `TRUE`, or the path of a directory, writes each block, after the
#'   formula has been applied, to an uncompressed file the first time it is
#'   read, and later passes read that file instead. This helps when a block is
#'   costly to obtain, as from a database query or a Parquet file, because
#'   fitting reads the data several times. Memory use is unchanged (one block
#'   at a time); the files take about as much disk space as the design matrix.
#'   A cache in a temporary directory (`TRUE`) is deleted when the source is
#'   garbage collected or the session ends; a named directory is kept. The
#'   cached blocks are those of the first pass, so later passes see the same
#'   data even if the underlying source changes.
#' @returns A `fastglm_source` object for [fastglm_streaming()],
#'   [fastglm_ordinal_streaming()], [fastglm_multinom()], or the count and
#'   two-part fitting functions.
#' @export
fastglm_source <- function(callback, n_chunks, reset = NULL, formula = NULL,
                           prototype = NULL, weights = NULL, offset = NULL, cache = FALSE) {
    .check_iterations(1e-8,n_chunks)
    if (!is.function(callback) || (!is.null(reset) && !is.function(reset))) stop("callback and reset must be functions",call.=FALSE)
    if(!is.null(weights) && (!is.character(weights) || length(weights)!=1L || is.na(weights)))
        stop("weights must name one data column",call.=FALSE)
    # One-part formulas accept one offset column per linear predictor (any
    # number, for vector GLMs); otherwise one or two columns, as before. The
    # formula is inspected only when more than two names are given.
    if(!is.null(offset) && (!is.character(offset) || !length(offset) || anyNA(offset) ||
                            (length(offset)>2L && (is.null(formula) ||
                                length(Formula::Formula(formula))[2L]!=1L))))
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
            if(j==1L) response <<- model.response(mf)
            list(terms=tt,xlevels=.getXlevels(tt,mf),contrasts=attr(xx,"contrasts"),
                 columns=colnames(xx),assign=attr(xx,"assign"))
        }
        response <- NULL
        schema <- build(1L)
        if(length(f)[2L]==2L) {
            parts <- list(count=schema,zero=build(2L))
            schema <- lapply(names(schema),function(nm) lapply(parts,`[[`,nm))
            names(schema) <- names(parts$count)
        }
        schema$formula <- f
        schema$offset <- offset
        # Levels of a factor response in the prototype (no extra work), used
        # by models with a categorical response.
        if(is.factor(response)) schema$response.levels <- base::levels(response)
    }
    # Several offset columns with a one-part formula are per-linear-predictor
    # offsets (a block offset with one column each). Only readers that ask for
    # them (vector GLMs) receive them; other models keep the message.
    per.predictor <- !is.null(schema) && inherits(schema$terms,"terms") && length(offset)>1L
    if(per.predictor) schema$offset.per.predictor <- TRUE
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
            if(per.predictor) out$offset <- out$offset+matrix(vapply(offset,function(nm) as.numeric(column(nm)),
                numeric(nrow(out$X))),nrow(out$X),dimnames=list(NULL,offset))
            else if(length(offset)) out$offset <- out$offset+column(offset)
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
    cached <- .source_cache(fetch,as.integer(n_chunks),cache)
    if(per.predictor) {
        inner <- cached$fetch
        cached$fetch <- function(k,per.predictor=FALSE) {
            # Models with one linear predictor keep their messages.
            if(!per.predictor) stop(if(length(offset)>2L) "offset must name one or two data columns" else
                "two offset columns require a two-component formula",call.=FALSE)
            inner(k)
        }
    }
    structure(list(callback=cached$fetch,n_chunks=as.integer(n_chunks),schema=schema,
                   cache=cached$directory),class="fastglm_source")
}

# Wrap a block reader so that each block, as returned, is written to an
# uncompressed file on its first read and read from that file afterwards.
# Only one block is held in memory at a time.
.source_cache <- function(fetch,n_chunks,cache) {
    if(isFALSE(cache)) return(list(fetch=fetch,directory=NULL))
    temporary <- isTRUE(cache)
    if(!temporary && !(is.character(cache) && length(cache)==1L && !is.na(cache) && nzchar(cache)))
        stop("'cache' must be TRUE, FALSE, or a directory path",call.=FALSE)
    directory <- if(temporary) tempfile("fastglm-cache-") else cache
    dir.create(directory,recursive=TRUE,showWarnings=FALSE)
    if(!dir.exists(directory)) stop("cannot create the cache directory",call.=FALSE)
    files <- file.path(directory,sprintf("block-%07d.rds",seq_len(n_chunks)))
    state <- new.env(parent=emptyenv()); state$cached <- logical(n_chunks)
    if(temporary) reg.finalizer(state,function(e) unlink(directory,recursive=TRUE),onexit=TRUE)
    cached.fetch <- function(k) {
        if(state$cached[k]) return(readRDS(files[k]))
        z <- fetch(k)
        saveRDS(z,files[k],compress=FALSE)
        state$cached[k] <- TRUE
        z
    }
    list(fetch=cached.fetch,directory=directory)
}

#' Create a source from indexed database queries
#'
#' Each query must return a stable, disjoint chunk in a deterministic order.
#' The database snapshot must remain unchanged throughout fitting and inference.
#' @param connection a DBI connection.
#' @param queries character vector of read-only SELECT queries, one per chunk.
#' @param formula,prototype,weights,offset,cache passed to [fastglm_source()].
#' @returns A resettable `fastglm_source`.
#' @export
fastglm_source_dbi <- function(connection, queries, formula, prototype,
                               weights = NULL, offset = NULL, cache = FALSE) {
    if (!requireNamespace("DBI",quietly=TRUE)) stop("install DBI to use database sources",call.=FALSE)
    if (!is.character(queries) || !length(queries) || any(!grepl("^\\s*(SELECT|WITH)\\b",queries,ignore.case=TRUE))) stop("queries must be read-only SELECT statements",call.=FALSE)
    out <- fastglm_source(function(k) DBI::dbGetQuery(connection,queries[[k]]),length(queries),
                    formula=formula,prototype=prototype,weights=weights,offset=offset,cache=cache)
    out$origin <- "database queries (DBI)"
    out
}

#' Create a source from Parquet partitions
#'
#' Each file is one bounded chunk. The formula schema is supplied explicitly
#' so that factor contrasts and transformations remain fixed across files.
#' @param files character vector of Parquet paths.
#' @param formula,prototype,weights,offset,cache passed to [fastglm_source()].
#' @returns A `fastglm_source`.
#' @export
fastglm_source_arrow <- function(files, formula, prototype, weights=NULL, offset=NULL, cache=FALSE) {
    if (!requireNamespace("arrow",quietly=TRUE)) stop("install arrow to use Parquet sources",call.=FALSE)
    if (!is.character(files) || !length(files)) stop("supply at least one Parquet file",call.=FALSE)
    out <- fastglm_source(function(k) as.data.frame(arrow::read_parquet(files[[k]])),length(files),
                    formula=formula,prototype=prototype,weights=weights,offset=offset,cache=cache)
    out$origin <- "Parquet files (arrow)"
    out
}

#' @export
print.fastglm_source <- function(x, ...) {
    .fastglm_print_header("Data source for blockwise fitting")
    .fastglm_print_field("Origin",if(is.null(x$origin)) "callback" else x$origin)
    .fastglm_print_field("Blocks",x$n_chunks)
    s <- x$schema
    if(is.null(s)) {
        .fastglm_print_field("Design","model matrices returned by the callback")
    } else {
        .fastglm_print_field("Formula",paste(deparse(stats::formula(s$formula)),collapse=" "))
        if(is.list(s$columns)) {
            .fastglm_print_field("Design columns",sprintf("%d (count), %d (zero)",
                length(s$columns$count),length(s$columns$zero)))
        } else .fastglm_print_field("Design columns",length(s$columns))
        .fastglm_print_field("Offset columns",s$offset)
    }
    .fastglm_print_field("Block cache",if(is.null(x$cache)) "none" else x$cache)
    invisible(x)
}
