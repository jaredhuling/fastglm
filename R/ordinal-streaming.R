#' Fit an ordinal model from repeatable blocks of observations
#'
#' Fits the same cumulative-link likelihood and mean bias-reduction equations
#' as [fastglm_ordinal()] while retaining one source block and matrices whose
#' dimensions depend on the number of coefficients. The source is read again
#' for successive fitting evaluations. Mean bias reduction first combines the
#' information from every block, then rereads the blocks to evaluate its
#' adjustment using that global information.
#' @param source a [fastglm_source()] or a function returning a list with
#'   `X`, `y`, and optional `weights` and `offset` for a block index.
#'   Matrix blocks must omit the location intercept. Formula sources remove
#'   their model-matrix intercept automatically.
#' @param n_chunks number of blocks when `source` is a function.
#' @param levels category labels in increasing order. May be inferred from an
#'   ordered factor or the column names of a category-count matrix in the
#'   first block. Every block must retain the same category specification,
#'   including categories absent from that block.
#' @param firth,link,tol,maxit,block.size,control as in [fastglm_ordinal()].
#'   `block.size` limits temporary calculations within a supplied block;
#'   the source determines the size of the block loaded into memory.
#' @param nonparallel predictors whose slopes may differ across thresholds.
#'   Select complete model terms with a one-sided formula for a formula source,
#'   or columns by name or index for a matrix source. All declared categories
#'   must have positive total frequency for these models.
#' @details The rows and values returned for each block must remain unchanged
#'   during fitting and later prediction or residual calculations. Empty
#'   blocks are not accepted; blocks with zero total frequency are allowed.
#'   Category reduction and rank checks use the complete source, so individual
#'   blocks may omit categories or have deficient predictor rank.
#'
#'   Storage depends on the largest source block, the number of coefficients,
#'   and block-size metadata. The source itself must also obtain blocks without
#'   retaining the complete dataset if bounded memory is required. Formula
#'   sources use their recorded predictor schema and fixed transformation basis.
#'
#'   Coefficients, model-based covariance, summaries, diagnostics, and Wald
#'   intervals use the usual ordinal methods. Robust and clustered covariance
#'   through [fastglm_vcov()] rereads the source and accumulates the same
#'   ordinary-score contributions as retained dense fits. Predictions, residuals, and
#'   weights can be returned one block at a time through a callback. Collecting
#'   their complete output requires memory proportional to the number of rows.
#'   The fitted design and response are not retained. [fastglm_ordinal_assess()]
#'   accumulates prediction scores and calibration over the original source;
#'   [fastglm_po_test()] refits common-slope hypotheses through source passes.
#'   Optional boundary rows from [fastglm_diagnostics()] also use one source pass.
#'   Profile intervals, general constrained coefficient-score tests, and
#'   refitting bootstrap calculations require retained observations.
#'   Average prediction contrasts can receive explicit evaluation data.
#' @returns A `fastglm_ordinal_stream` object inheriting from
#'   `fastglm_ordinal`, with source and reader references, source pass counts,
#'   and the usual ordinal estimates and convergence diagnostics.
#' @examples
#' set.seed(104)
#' x <- matrix(rnorm(120),ncol=1,dimnames=list(NULL,"exposure"))
#' y <- ordered(cut(.4*x[,1]+rlogis(120),c(-Inf,-.5,.7,Inf)))
#' reader <- function(k) {
#'     i <- ((k-1)*30+1):(k*30)
#'     list(X=x[i,,drop=FALSE],y=y[i])
#' }
#' fit <- fastglm_ordinal_streaming(reader,4,firth=TRUE)
#' summary(fit)
#' head(predict(fit))
#' @md
#' @export
fastglm_ordinal_streaming <- function(source,n_chunks=NULL,levels=NULL,firth=FALSE,
    link="logit",tol=1e-8,maxit=200L,block.size=2048L,control=list(),nonparallel=NULL) {
    call <- match.call();call[[1L]] <- quote(fastglm::fastglm_ordinal_streaming)
    source <- if(inherits(source,"fastglm_source")) source else fastglm_source(source,n_chunks)
    if(!is.null(source$schema$terms) && !inherits(source$schema$terms,"terms"))
        stop("ordinal fitting requires a source with one predictor formula",call.=FALSE)
    if(!is.logical(firth) || length(firth)!=1L || is.na(firth)) stop("'firth' must be TRUE or FALSE",call.=FALSE)
    link.code <- .ordinal_link_code(link);link <- c("logit","probit","cloglog")[link.code+1L]
    .check_iterations(tol,maxit);.check_iterations(tol,block.size)
    defaults <- list(score.tol=tol,step.tol=tol,max.step=4,max.halves=35L,
        condition.tol=1e-14,standardize=TRUE,trace=FALSE)
    if(!is.list(control) || (length(control) && (is.null(names(control)) || anyNA(names(control)) ||
       any(!nzchar(names(control))) || anyDuplicated(names(control)))) || length(setdiff(names(control),names(defaults))))
        stop("unknown or duplicated ordinal optimizer control",call.=FALSE)
    ctl <- utils::modifyList(defaults,control)
    for(nm in c("score.tol","step.tol","max.step","max.halves","condition.tol")) {
        z <- ctl[[nm]]
        if(!is.numeric(z) || is.complex(z) || length(z)!=1L || !is.finite(z) || z<=0 ||
           (nm=="max.halves" && (z!=floor(z) || z>.Machine$integer.max)))
            stop("invalid ordinal optimizer control: ",nm,call.=FALSE)
    }
    for(nm in c("standardize","trace"))
        if(!is.logical(ctl[[nm]]) || length(ctl[[nm]])!=1L || is.na(ctl[[nm]]))
            stop("invalid ordinal optimizer flag: ",nm,call.=FALSE)
    state <- new.env(parent=emptyenv())
    state$passes <- state$reads <- 0L;state$rows <- integer(source$n_chunks)
    state$p <- NULL;state$columns <- NULL;state$levels <- levels
    state$grouped <- NULL;state$factor.levels <- NULL;state$count.columns <- NULL
    state$maximum.block.rows <- 0L
    read <- .ordinal_stream_reader(source,state)
    frequency <- positive.rows <- saturated <- 0;center <- NULL;offset.mean <- 0;category.totals <- NULL
    selected <- boundary.check <- NULL
    for(k in seq_len(source$n_chunks)) {
        z <- read(k,summaries=TRUE);total <- sum(z$totals);good <- z$totals>0
        if(is.null(center)) {
            center <- numeric(ncol(z$X));category.totals <- numeric(length(state$levels))
            selected <- if(!is.null(source$schema$terms))
                .ordinal_nonparallel_terms(nonparallel,source$schema$terms,
                    source$schema$assign[source$schema$assign!=0L]) else
                list(columns=.ordinal_nonparallel_columns(nonparallel,state$columns),terms=character())
            boundary.check <- .ordinal_partial_boundary_start(state$p-length(selected$columns),
                length(selected$columns),length(state$levels),firth,link)
        }
        if(!is.null(boundary.check)) .ordinal_partial_boundary_add(boundary.check,
            z$X[,selected$columns,drop=FALSE],z$codes,z$counts,z$weights,z$offset)
        category.totals <- category.totals+z$category.totals
        if(total>0) {
            next.frequency <- frequency+total
            block.mean <- colSums(z$X[good,,drop=FALSE]*(z$totals[good]/total))
            center <- center*(frequency/next.frequency)+block.mean*(total/next.frequency)
            offset.mean <- offset.mean*(frequency/next.frequency)+sum(z$offset[good]*(z$totals[good]/total))*(total/next.frequency)
            frequency <- next.frequency
        }
        positive.rows <- positive.rows+sum(good);saturated <- saturated+z$saturated
    }
    if(!is.finite(frequency) || frequency<=0) stop("ordinal fitting requires finite positive total frequency",call.=FALSE)
    partial <- length(selected$columns)>0L
    if(partial && any(category.totals<=0))
        stop("nonparallel ordinal fitting requires every declared category to have positive frequency",call.=FALSE)
    response <- .ordinal_response_data(matrix(category.totals,nrow=1,dimnames=list(NULL,state$levels)),
        levels=state$levels,firth=firth || partial)
    response$grouped <- state$grouped;response$response.kind <- if(state$grouped) "counts" else "codes"
    response$n <- sum(state$rows);response$weights <- response$totals <- NULL
    response$counts <- matrix(numeric(),0L,0L);response$codes <- response$original.codes <- integer()
    p <- state$p;q <- length(response$levels)-1L;d <- p+q
    common <- setdiff(seq_len(p),selected$columns)
    if(partial) d <- q+length(common)+q*length(selected$columns)
    scale <- rep(1,p)
    if(p) {
        variance <- numeric(p)
        for(k in seq_len(source$n_chunks)) {
            z <- read(k);good <- z$totals>0
            if(any(good)) variance <- variance+colSums(sweep(z$X[good,,drop=FALSE],2,center,"-")^2*(z$totals[good]/frequency))
        }
        scale <- sqrt(variance)
        if(any(!is.finite(scale)|scale<=0)) stop("the ordinal location design contains a redundant constant column or cannot be scaled",call.=FALSE)
        qr.state <- NULL
        for(k in seq_len(source$n_chunks)) {
            z <- read(k);good <- z$totals>0
            if(any(good)) {
                xx <- sweep(sweep(z$X[good,,drop=FALSE],2,center,"-"),2,scale,"/")
                xx <- cbind(1,xx)
                qr.state <- streaming_qr_update(xx,numeric(nrow(xx)),rep(1,nrow(xx)),qr.state)
            }
        }
        tryCatch(streaming_qr_finish(qr.state),error=function(e)
            stop("the ordinal source location design must be full rank with a separate threshold intercept",call.=FALSE))
        if(!ctl$standardize) {center <- numeric(p);scale <- rep(1,p)}
    }
    certificate <- .ordinal_partial_boundary_finish(boundary.check)
    if(!is.null(certificate)) stop(certificate$message,": ",
        paste(unique(response$levels[certificate$empty.pattern.categories[,"category"]]),collapse=", "),
        ".",call.=FALSE)
    fetch <- function(k) {
        z <- read(k);good <- z$totals>0
        if(p) z$X <- sweep(sweep(z$X,2,center,"-"),2,scale,"/")
        z$X[!good,] <- 0;z$offset[!good] <- 0
        if(state$grouped) z$counts <- z$counts[,response$retained.categories,drop=FALSE] else {
            z$codes <- match(z$codes,response$retained.categories)
            z$codes[is.na(z$codes) & !good] <- 1L
            z$codes <- as.integer(z$codes)
        }
        if(partial) {
            z$Z <- z$X[,selected$columns,drop=FALSE]
            z$X <- z$X[,common,drop=FALSE]
        }
        z
    }
    cumulative <- cumsum(response$category.totals)[seq_len(q)]
    remaining <- rev(cumsum(rev(response$category.totals)))[-1L]
    alpha <- .ordinal_start_thresholds(cumulative,remaining,link)+offset.mean
    if(partial) {
        fit <- ordinal_partial_stream_native_fit(fetch,source$n_chunks,length(common),
            length(selected$columns),q+1L,c(alpha,numeric(d-q)),firth,
            ctl$score.tol,ctl$step.tol,as.integer(maxit),as.integer(block.size),ctl$max.step,
            as.integer(ctl$max.halves),ctl$condition.tol,ctl$trace,link.code)
        object <- .ordinal_partial_result(fit,center,scale,response,frequency,positive.rows,
            firth,link,ctl,tol,maxit,block.size,state$columns,selected$columns)
        class(object) <- c("fastglm_ordinal_stream",class(object))
        object$nonparallel.terms <- selected$terms
    } else {
        fit <- ordinal_stream_native_fit(fetch,source$n_chunks,p,q+1L,c(alpha,numeric(p)),firth,
            ctl$score.tol,ctl$step.tol,as.integer(maxit),as.integer(block.size),ctl$max.step,
            as.integer(ctl$max.halves),ctl$condition.tol,ctl$trace,link.code)
        object <- .ordinal_stream_result(fit,center,scale,response,frequency,positive.rows,
            firth,link,ctl,tol,maxit,block.size,state$columns)
    }
    object$source <- source;object$fetch <- read;object$state <- state;object$call <- call
    object$deviance <- 2*(saturated-fit$loglik)
    object$terms <- source$schema$terms;object$xlevels <- source$schema$xlevels
    object$contrasts <- source$schema$contrasts
    object$design.assign <- if(!is.null(source$schema$assign)) source$schema$assign[source$schema$assign!=0L]
    object$prediction.offset.required <- length(source$schema$offset)>0L
    object$source.passes <- state$passes;object$source.reads <- state$reads
    object$maximum.block.rows <- state$maximum.block.rows
    object
}

# The fit keeps this reader. A separate frame keeps fitting variables, including
# the last source block, out of the fitted object.
.ordinal_stream_reader <- function(source,state) {
    force(source);force(state)
    integer.frequency <- function(x) all(abs(x-round(x))<=pmin(1e-7,16*.Machine$double.eps*pmax(1,abs(x))))
    read <- function(k,summaries=FALSE) {
        if(k==1L) state$passes <- state$passes+1L
        state$reads <- state$reads+1L
        z <- source$callback(k)
        if(!is.list(z)) stop("an ordinal source block must be a list",call.=FALSE)
        x <- z$X;y <- z$y;grouped <- is.matrix(y)
        if(!is.matrix(x) || !is.numeric(x) || is.complex(x) || any(!is.finite(x)))
            stop("each ordinal block$X must be a finite dense numeric matrix",call.=FALSE)
        if(!is.null(source$schema$assign)) x <- x[,source$schema$assign!=0L,drop=FALSE]
        n <- nrow(x)
        if(n<1L || NROW(y)!=n) stop("ordinal source blocks require matching, nonempty design and response rows",call.=FALSE)
        if(ncol(x)>0L && is.null(colnames(x))) colnames(x) <- paste0("x",seq_len(ncol(x)))
        if(anyNA(colnames(x)) || any(!nzchar(colnames(x))) || anyDuplicated(colnames(x)))
            stop("ordinal predictor column names must be distinct and nonmissing",call.=FALSE)
        if(is.null(state$p)) {state$p <- ncol(x);state$columns <- colnames(x);state$grouped <- grouped}
        if(ncol(x)!=state$p || !identical(colnames(x),state$columns))
            stop("ordinal source predictor columns changed from the schema",call.=FALSE)
        if(!identical(grouped,state$grouped)) stop("ordinal source response representation changed",call.=FALSE)
        if(state$rows[k]!=0L && state$rows[k]!=n) stop("ordinal source block size changed between passes",call.=FALSE)
        state$rows[k] <- n;state$maximum.block.rows <- max(state$maximum.block.rows,n)
        if(is.null(state$levels)) {
            state$levels <- if(grouped) colnames(y) else if(is.ordered(y)) base::levels(y)
            if(is.null(state$levels)) stop("supply 'levels' or retain ordered-factor levels or count column names in every block",call.=FALSE)
        }
        labels <- as.character(state$levels);K <- length(labels)
        if(K<2L || anyNA(labels) || any(!nzchar(labels)) || anyDuplicated(labels))
            stop("at least two distinct, nonmissing ordinal category labels are required",call.=FALSE)
        state$levels <- labels
        w <- if(is.null(z$weights)) rep(1,n) else z$weights
        off <- if(is.null(z$offset)) numeric(n) else z$offset
        if(!is.numeric(w) || is.complex(w) || !is.null(dim(w)) || length(w)!=n || any(!is.finite(w)|w<0))
            stop("ordinal frequency weights must be finite and nonnegative",call.=FALSE)
        if(!is.numeric(off) || is.complex(off) || !is.null(dim(off)) || length(off)!=n || any(!is.finite(off)))
            stop("ordinal source offsets must be finite and match block rows",call.=FALSE)
        counts <- matrix(numeric(),0L,0L);codes <- integer()
        if(grouped) {
            if(!is.numeric(y) || is.complex(y) || ncol(y)!=K || any(!is.finite(y)|y<0))
                stop("ordinal count blocks require finite nonnegative counts in the declared columns",call.=FALSE)
            if(is.null(state$count.columns)) state$count.columns <- list(colnames(y))
            if(!identical(colnames(y),state$count.columns[[1L]])) stop("ordinal count columns changed from the category schema",call.=FALSE)
            if(!is.null(colnames(y)) && !identical(colnames(y),labels))
                stop("ordinal count column names must match 'levels' in order",call.=FALSE)
            effective <- y*w
            if(any(!is.finite(effective)) || !integer.frequency(effective))
                stop("ordinal fitting requires finite integer effective category counts",call.=FALSE)
            counts <- y;storage.mode(counts) <- "double"
            totals <- rowSums(effective);category.totals <- if(summaries) colSums(effective)
            saturated <- 0
            if(summaries) {
                take <- effective>0
                saturated <- sum(effective[take]*log((effective/totals)[take]))
            }
        } else {
            if(!is.null(dim(y))) stop("invalid ordinal source response dimensions",call.=FALSE)
            if(is.factor(y)) {
                if(is.null(state$factor.levels)) state$factor.levels <- base::levels(y)
                if(!identical(base::levels(y),state$factor.levels)) stop("ordinal factor levels changed from the category schema",call.=FALSE)
            }
            if(is.numeric(y)) {
                if(any(!is.finite(y)|y!=floor(y)|y<1|y>K)) stop("numeric ordinal responses must be category codes",call.=FALSE)
                codes <- as.integer(y)
            } else codes <- match(as.character(y),labels)
            if(anyNA(codes) || !integer.frequency(w)) stop("ordinal source responses and integer frequency weights must match the category schema",call.=FALSE)
            totals <- w;category.totals <- if(summaries) vapply(seq_len(K),function(j)sum(w[codes==j]),numeric(1));saturated <- 0
        }
        if(any(!is.finite(totals)) || any(!is.finite(category.totals))) stop("ordinal source frequencies overflow",call.=FALSE)
        storage.mode(x) <- "double"
        list(X=x,codes=as.integer(codes),counts=counts,weights=as.numeric(w),offset=as.numeric(off),
            totals=as.numeric(totals),category.totals=category.totals,saturated=saturated)
    }
    read
}

.ordinal_stream_result <- function(fit,center,scale,response,frequency,positive.rows,
    firth,link,ctl,tol,maxit,block.size,columns) {
    q <- length(response$levels)-1L;p <- length(columns);d <- q+p
    T <- Ti <- diag(d)
    if(p) {
        T[seq_len(q),q+seq_len(p)] <- matrix(center/scale,q,p,byrow=TRUE)
        T[q+seq_len(p),q+seq_len(p)] <- diag(1/scale,nrow=p)
        Ti[seq_len(q),q+seq_len(p)] <- matrix(-center,q,p,byrow=TRUE)
        Ti[q+seq_len(p),q+seq_len(p)] <- diag(scale,nrow=p)
    }
    nms <- c(paste(response$levels[-(q+1L)],response$levels[-1L],sep="|"),columns)
    effective <- list(coefficients=drop(T%*%fit$coefficients),covariance=T%*%fit$cov.unscaled%*%t(T),
        information=crossprod(Ti,fit$information%*%Ti),score=drop(crossprod(Ti,fit$score)),
        ordinary.score=drop(crossprod(Ti,fit$ordinary.score)),adjustment=drop(crossprod(Ti,fit$adjustment)),
        step=drop(T%*%fit$step),levels=response$levels)
    for(nm in c("coefficients","score","ordinary.score","adjustment","step")) names(effective[[nm]]) <- nms
    for(nm in c("covariance","information")) dimnames(effective[[nm]]) <- list(nms,nms)
    effective$alpha <- effective$coefficients[seq_len(q)];effective$beta <- effective$coefficients[q+seq_len(p)]
    working <- list(coefficients=as.numeric(fit$coefficients),center=center,scale=scale,
        covariance=fit$cov.unscaled,information=fit$information,score=as.numeric(fit$score),
        ordinary.score=as.numeric(fit$ordinary.score),adjustment=as.numeric(fit$adjustment),
        step=as.numeric(fit$step),scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,
        information.rcond=fit$information.rcond)
    names(working$coefficients) <- nms
    working$alpha <- working$coefficients[seq_len(q)];working$beta <- working$coefficients[q+seq_len(p)]
    K <- length(response$original.levels);expand <- c(response$threshold.map,q+seq_len(p))
    original.names <- c(paste0("threshold:",paste(response$original.levels[-K],response$original.levels[-1L],sep="|")),columns)
    b <- effective$coefficients[expand];names(b) <- original.names
    V <- effective$covariance[expand,expand,drop=FALSE];dimnames(V) <- list(original.names,original.names)
    colnames(fit$history) <- c("loglik","scaled.score","predictor.step","minimum.gap")
    if(response$boundary) warning("empty interior ordinal categories were collapsed for fitting; their labels are retained with zero probability and coincident thresholds",call.=FALSE)
    if(!fit$converged) warning("ordinal fitting did not converge: ",fit$convergence.reason,call.=FALSE)
    object <- list(coefficients=b,alpha=b[seq_len(K-1L)],beta=b[K-1L+seq_len(p)],thresholds=b[seq_len(K-1L)],
        levels=response$original.levels,effective.levels=response$levels,threshold.map=response$threshold.map,
        retained.categories=response$retained.categories,collapsed.categories=response$collapsed.categories,
        boundary=response$boundary,empty.end.categories=response$empty.end.categories,
        tied.thresholds=duplicated(response$threshold.map)|duplicated(response$threshold.map,fromLast=TRUE),
        effective=effective,working=working,mapped.covariance=V,cov.unscaled=V,information=effective$information,
        score=effective$score,ordinary.score=effective$ordinary.score,adjustment=effective$adjustment,
        scaled.score=fit$scaled.score,predictor.step=fit$predictor.step,information.rcond=fit$information.rcond,
        minimum.log.probability=fit$minimum.log.probability,converged=fit$converged,convergence.reason=fit$convergence.reason,
        iterations=fit$iterations,evaluations=fit$evaluations,fallbacks=fit$fallbacks,
        last.rejected.evaluation=fit$last.rejected.evaluation,history=fit$history,loglik=fit$loglik,
        rank=d,df.residual=frequency-d,nobs=frequency,nobs.rows=positive.rows,frequency=frequency,
        category=response,response=response,preparation.reused=FALSE,firth=firth,link=link,
        method=if(firth) "meanBR" else "ML",control=ctl,tol=tol,maxit=as.integer(maxit),
        block.size=as.integer(block.size),design.columns=columns)
    class(object) <- c("fastglm_ordinal_stream","fastglm_ordinal")
    names(object$alpha) <- names(object$thresholds) <- sub("^threshold:","",names(object$alpha))
    object
}

.ordinal_stream_collect <- function(object,fun,sink=NULL) {
    if(!is.null(sink) && !is.function(sink)) stop("'sink' must be a function",call.=FALSE)
    out <- if(is.null(sink)) vector("list",object$source$n_chunks)
    for(k in seq_len(object$source$n_chunks)) {
        value <- fun(object$fetch(k))
        if(is.null(sink)) out[[k]] <- value else sink(value,k)
    }
    if(!is.null(sink)) return(invisible(object$source$n_chunks))
    bind <- function(x) {
        first <- x[[1L]]
        if(is.matrix(first)) return(do.call(rbind,x))
        # c() keeps factor attributes only from R 4.1.0 onward.
        if(is.factor(first)) return(structure(unlist(lapply(x,unclass)),levels=levels(first),class=class(first)))
        do.call(c,x)
    }
    if(inherits(out[[1L]],"fastglm_prediction")) {
        value <- out[[1L]];value$fit <- bind(lapply(out,`[[`,"fit"));value$se.fit <- bind(lapply(out,`[[`,"se.fit"));value
    } else bind(out)
}

#' Predictions and residuals from an ordinal source fit
#' @param object a `fastglm_ordinal_stream` fit.
#' @param newdata,offset,type,se.fit as in [predict.fastglm_ordinal()].
#' @param sink optional function of `(value, chunk_index)` that receives each
#'   result block, for example to write it to a file. Its return value is ignored.
#'   Without a function, all blocks are collected into a single R object.
#' @param ... further method arguments.
#' @returns Collected predictions or residuals, or invisibly the number of
#'   blocks supplied to `sink`.
#' @export
predict.fastglm_ordinal_stream <- function(object,newdata=NULL,type=c("probability","response","cumulative","class","link"),
    se.fit=FALSE,offset=NULL,sink=NULL,...) {
    type <- match.arg(type)
    predictor <- if(inherits(object,"fastglm_ordinal_partial"))
        predict.fastglm_ordinal_partial else predict.fastglm_ordinal
    if(!is.null(newdata)) {
        if(!is.null(sink)) stop("'sink' is used when predicting from the fitted source",call.=FALSE)
        return(predictor(object,newdata,type,se.fit,offset,...))
    }
    if(!is.null(offset)) stop("source prediction uses its recorded block offsets",call.=FALSE)
    .ordinal_stream_collect(object,function(z) {
        model <- object;model$terms <- NULL;model$prediction.offset.required <- FALSE
        predictor(model,z$X,type,se.fit,z$offset,...)
    },sink)
}

#' @rdname predict.fastglm_ordinal_stream
#' @export
residuals.fastglm_ordinal_stream <- function(object,type=c("probability","cumulative","pearson"),sink=NULL,...) {
    type <- match.arg(type)
    .ordinal_stream_collect(object,function(z) {
        model <- object;model$x <- z$X;model$offset <- z$offset;model$weights <- z$weights
        model$category$totals <- z$totals;model$category$original.codes <- z$codes
        if(model$category$grouped) model$category$counts <- z$counts[,model$retained.categories,drop=FALSE]
        residuals.fastglm_ordinal(model,type,...)
    },sink)
}

#' @export
weights.fastglm_ordinal_stream <- function(object,sink=NULL,...) {
    if(length(list(...))) stop("unused ordinal source weights arguments",call.=FALSE)
    .ordinal_stream_collect(object,function(z) z$weights,sink)
}

#' @export
deviance.fastglm_ordinal_stream <- function(object,...) object$deviance

#' @export
model.matrix.fastglm_ordinal_stream <- function(object,...) {
    stop("ordinal source fits do not retain a model matrix; read the source blocks explicitly",call.=FALSE)
}

#' @export
simulate.fastglm_ordinal_stream <- function(object,nsim=1L,seed=NULL,...) {
    stop("simulation from ordinal source fits is not implemented",call.=FALSE)
}
