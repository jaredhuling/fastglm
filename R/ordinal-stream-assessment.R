# Reuse dense assessment on each original block. The source's recorded reader
# validates its schema; the temporary dense view retains no previous blocks.
.ordinal_stream_assessment <- function(object,bins,po.test,po.terms,po.control,
                                       estimation,keep.rows,sink,call) {
    if(!is.function(object$fetch) || is.null(object$source$n_chunks))
        stop("ordinal source assessment requires its recorded block reader",call.=FALSE)
    retained <- if(keep.rows) vector("list",object$source$n_chunks)
    calibration <- NULL;exclusions <- numeric()
    frequency <- rows <- 0;loss <- rep(NA_real_,3L)
    for(k in seq_len(object$source$n_chunks)) {
        block.error <- function(e) stop("ordinal source assessment block ",k,": ",
            conditionMessage(e),call.=FALSE)
        block <- tryCatch(object$fetch(k),error=block.error)
        model <- object
        class(model) <- setdiff(class(model),"fastglm_ordinal_stream")
        model$x <- block$X;model$offset <- block$offset;model$weights <- block$weights
        model$y <- if(object$category$grouped) block$counts else block$codes
        model$na.action <- NULL
        value <- tryCatch(fastglm_ordinal_assess(model,bins=bins),error=block.error)
        total <- value$scores$frequency
        next.frequency <- frequency+total
        if(!is.finite(next.frequency)) stop("total assessment frequency overflows",call.=FALSE)
        if(total>0) {
            current <- unlist(value$scores[1L,-1L],use.names=FALSE)
            if(frequency==0) loss <- current else {
                # Normalize before combining: frequency times loss may overflow.
                # Infinite losses stay infinite even if a relative frequency
                # underflows to zero in floating-point arithmetic.
                infinite <- is.infinite(loss)|is.infinite(current)
                loss <- loss*(frequency/next.frequency)+current*(total/next.frequency)
                loss[infinite] <- Inf
            }
        }
        frequency <- next.frequency
        counts <- value$calibration
        counts$predicted <- ifelse(counts$frequency>0,counts$frequency*counts$predicted,0)
        counts$observed <- ifelse(counts$frequency>0,counts$frequency*counts$observed,0)
        if(is.null(calibration)) calibration <- counts else
            calibration[,c("frequency","predicted","observed")] <-
                calibration[,c("frequency","predicted","observed")]+counts[,c("frequency","predicted","observed")]
        for(j in seq_len(nrow(value$exclusions))) {
            reason <- value$exclusions$reason[j]
            if(!reason%in%names(exclusions)) exclusions[reason] <- 0
            exclusions[reason] <- exclusions[reason]+value$exclusions$rows[j]
        }
        value$rows$row <- value$rows$row+rows
        rows <- rows+nrow(block$X)
        if(keep.rows) retained[[k]] <- value$rows
        if(!is.null(sink)) sink(value$rows,k)
    }
    calibration$predicted <- calibration$predicted/calibration$frequency
    calibration$observed <- calibration$observed/calibration$frequency
    calibration$predicted[calibration$frequency==0] <- NA_real_
    calibration$observed[calibration$frequency==0] <- NA_real_
    exclusions <- exclusions[order(as.character(names(exclusions)))]
    scores <- data.frame(frequency=frequency,log_loss=loss[1L],rps=loss[2L],rps_normalized=loss[3L])
    test <- if(po.test=="score") .ordinal_stream_po_test(object,terms=po.terms,
        control=po.control,estimation=estimation) else NULL
    if(keep.rows) {
        retained <- do.call(rbind,retained)
        rownames(retained) <- NULL
    }
    structure(list(scores=scores,rows=retained,calibration=calibration,
        proportional.odds=NULL,proportional.odds.test=test,evaluation="training source",
        levels=object$levels,link=.ordinal_link(object),
        exclusions=data.frame(reason=as.character(names(exclusions)),rows=unname(exclusions),stringsAsFactors=FALSE),
        bins=bins,source=list(blocks=object$source$n_chunks,rows=rows,
            rows.retained=keep.rows,rows.delivered=!is.null(sink)),call=call),
        class="fastglm_ordinal_assessment")
}
