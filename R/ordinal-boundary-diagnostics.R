# Explicitly requested row diagnostics: one data pass, with only two bounded
# lists of the most extreme row/category combinations retained.
.ordinal_boundary_diagnostics <- function(object,max.rows=5L) {
    if(!inherits(object,"fastglm_ordinal"))
        stop("boundary row diagnostics require an ordinal fit",call.=FALSE)
    if(!is.numeric(max.rows) || is.complex(max.rows) || length(max.rows)!=1L ||
       !is.finite(max.rows) || max.rows<1 || max.rows!=floor(max.rows) ||
       max.rows>.Machine$integer.max)
        stop("'max.rows' must be a positive integer",call.=FALSE)
    max.rows <- as.integer(max.rows)
    raw <- object$effective$score
    maximum <- function(x) if(length(x) && all(is.finite(x))) max(abs(x)) else NA_real_
    out <- list(available=FALSE,reason=NULL,
        scope="positive-frequency input rows in the effective category model",
        max.rows=max.rows,rows.examined=0,frequency.examined=0,
        source.passes=0L,source.reads=0L,
        collapsed.categories=object$levels[object$collapsed.categories],
        minimum.probability=NA_real_,minimum.log.probability=NA_real_,minimum.gap=NA_real_,
        raw.score.maximum=maximum(raw),
        raw.score.coordinates="effective threshold and slope parameters in reported predictor units",
        standardized.score=object$scaled.score,
        standardized.score.coordinates="centered/scaled fitting parameters; divided by square roots of expected-information diagonal entries",
        score.type=if(isTRUE(object$firth)) "mean-bias-reduced" else "ordinary",
        information.rcond=object$information.rcond,
        smallest.probabilities=NULL,smallest.gaps=NULL)
    if(is.null(object$working$coefficients) || is.null(object$category)) {
        out$reason <- "the fit does not retain its working coefficients and category metadata"
        return(out)
    }
    streamed <- inherits(object,"fastglm_ordinal_stream")
    if(streamed && (!is.function(object$fetch) || is.null(object$source$n_chunks))) {
        out$reason <- "the source fit does not retain its recorded block reader"
        return(out)
    }
    if(!streamed && (is.null(object$x) || is.null(object$y))) {
        out$reason <- "row diagnostics require the retained design and response"
        return(out)
    }
    # The general cumulative-predictor kernel also describes a common-slope
    # model with zero varying columns. Its compensated differences and log
    # probabilities retain gaps and tails lost by subtracting probabilities.
    kernel.object <- object
    if(!inherits(object,"fastglm_ordinal_partial")) {
        kernel.object$alpha <- object$working$alpha
        kernel.object$beta <- object$working$beta
        kernel.object$common <- seq_along(object$working$beta)
        kernel.object$nonparallel <- integer()
        kernel.object$working$gamma <- matrix(numeric(),0L,length(object$working$alpha))
    }
    retained <- object$retained.categories
    if(is.null(retained)) retained <- seq_along(object$effective$levels)
    K <- length(retained)
    if(K<2L) {
        out$reason <- "at least two effective categories are required"
        return(out)
    }
    empty <- data.frame(row=numeric(),row.name=character(),block=integer(),
        row.in.block=integer(),category=character(),category.index=integer(),
        probability=numeric(),log.probability=numeric(),gap=numeric(),
        observed.frequency=numeric(),row.frequency=numeric(),offset=numeric(),
        stringsAsFactors=FALSE)
    empty$predictors <- I(list())
    probabilities <- gaps <- empty
    keep <- function(old,new,field) {
        combined <- rbind(old,new)
        order <- order(combined[[field]],combined$row,combined$category.index)
        combined[utils::head(order,max.rows),,drop=FALSE]
    }
    add <- function(x,codes,counts,weights,offset,totals,rows,block,row.in.block) {
        positive <- which(totals>0)
        if(!length(positive)) return(invisible(NULL))
        xx <- x[positive,,drop=FALSE]
        pieces <- .ordinal_partial_probability_block(kernel.object,xx,offset[positive],
            derivatives=FALSE,rows=rows[positive])
        if(anyNA(pieces$log.probability) || any(!is.finite(pieces$gap)))
            stop("category probabilities or predictor gaps could not be evaluated",call.=FALSE)
        out$rows.examined <<- out$rows.examined+length(positive)
        out$frequency.examined <<- out$frequency.examined+sum(totals[positive])
        for(j in seq_len(K)) {
            lp <- pieces$log.probability[,j]
            gap <- if(j>1L && j<K) pieces$gap[,j-1L] else rep(NA_real_,length(positive))
            by.probability <- utils::head(order(lp,rows[positive]),max.rows)
            by.gap <- if(j>1L && j<K) utils::head(order(gap,rows[positive]),max.rows) else integer()
            selected <- union(by.probability,by.gap)
            index <- positive[selected]
            original.category <- retained[j]
            observed <- if(length(counts)) counts[index,original.category]*weights[index] else
                weights[index]*(codes[index]==original.category)
            row.names <- rownames(x)
            records <- data.frame(row=rows[index],
                row.name=if(is.null(row.names)) rep(NA_character_,length(index)) else row.names[index],
                block=rep(block,length(index)),row.in.block=row.in.block[index],
                category=rep(object$levels[original.category],length(index)),
                category.index=rep(original.category,length(index)),
                probability=exp(lp[selected]),log.probability=lp[selected],gap=gap[selected],
                observed.frequency=observed,row.frequency=totals[index],offset=offset[index],
                stringsAsFactors=FALSE)
            records$predictors <- I(lapply(index,function(i)
                setNames(as.numeric(x[i,,drop=FALSE]),object$design.columns)))
            probabilities <<- keep(probabilities,records[match(by.probability,selected),,drop=FALSE],"log.probability")
            if(length(by.gap)) gaps <<- keep(gaps,records[match(by.gap,selected),,drop=FALSE],"gap")
        }
        invisible(NULL)
    }
    block.size <- object$block.size
    if(is.null(block.size)) block.size <- 2048L
    failure <- tryCatch({
        if(streamed) {
            first <- 0
            for(k in seq_len(object$source$n_chunks)) {
                out$source.reads <- out$source.reads+1L
                if(k==1L) out$source.passes <- 1L
                z <- object$fetch(k)
                .design_blocks(z$X,function(x,i) add(x,
                    if(length(z$codes)) z$codes[i] else integer(),
                    if(length(z$counts)) z$counts[i,,drop=FALSE] else matrix(numeric(),0L,0L),
                    z$weights[i],z$offset[i],z$totals[i],first+i,k,i),block.size)
                first <- first+nrow(z$X)
            }
        } else {
            category <- object$category
            .design_blocks(object$x,function(x,i) add(x,
                if(category$grouped) integer() else category$original.codes[i],
                if(category$grouped) object$y[i,,drop=FALSE] else matrix(numeric(),0L,0L),
                object$weights[i],object$offset[i],category$totals[i],i,NA_integer_,i),block.size)
        }
        NULL
    },error=function(e) conditionMessage(e))
    if(!is.null(failure)) {
        out$reason <- paste0("row diagnostics could not complete: ",failure)
        return(out)
    }
    if(!nrow(probabilities)) {
        out$reason <- "no positive-frequency rows were available"
        return(out)
    }
    rownames(probabilities) <- rownames(gaps) <- NULL
    out$available <- TRUE
    out$minimum.probability <- probabilities$probability[1L]
    out$minimum.log.probability <- probabilities$log.probability[1L]
    if(nrow(gaps)) out$minimum.gap <- gaps$gap[1L]
    out$smallest.probabilities <- probabilities
    out$smallest.gaps <- gaps
    out$interpretation <- paste0("These are the smallest probabilities and gaps, not a test for root existence. ",
        "A displayed zero probability can be floating-point underflow; its log probability distinguishes that case. ",
        "Scores and conditioning are stored at the fit; row details use the retained data or current source.")
    out
}
