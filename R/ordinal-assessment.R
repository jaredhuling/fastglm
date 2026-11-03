# Assessment keeps the fitted category scale; it must not collapse categories
# according to the validation outcomes or refit prediction preprocessing.
.ordinal_assessment_response <- function(y,weights,levels,n) {
    grouped <- is.matrix(y); K <- length(levels)
    if((if(grouped) nrow(y) else length(y))!=n)
        stop("assessment outcomes must match the prediction rows",call.=FALSE)
    if(is.null(weights)) weights <- rep(1,n)
    if(!is.numeric(weights) || is.complex(weights) || !is.null(dim(weights)) ||
       length(weights)!=n || any(!is.na(weights) & (!is.finite(weights) | weights<0)))
        stop("assessment frequency weights must be nonnegative and finite or missing",call.=FALSE)
    integer.frequency <- function(z)
        all(abs(z-round(z))<=pmin(1e-7,16*.Machine$double.eps*pmax(1,abs(z))))
    codes <- NULL
    if(grouped) {
        if(!is.numeric(y) || is.complex(y) || ncol(y)!=K ||
           any(!is.na(y) & (!is.finite(y) | y<0)))
            stop("assessment counts must have one nonnegative numeric column per fitted category",call.=FALSE)
        if(!is.null(colnames(y))) {
            if(anyDuplicated(colnames(y)) || !setequal(colnames(y),levels))
                stop("assessment count columns must match the fitted category labels",call.=FALSE)
            y <- y[,levels,drop=FALSE]
        }
        missing.response <- rowSums(is.na(y))>0L
        totals <- rep(NA_real_,n)
        # Check effective counts in blocks, avoiding another full count matrix.
        for(first in seq.int(1L,max(1L,n),by=8192L)) {
            if(!n) break
            index <- seq.int(first,min(n,first+8191L))
            good <- !missing.response[index] & !is.na(weights[index])
            index <- index[good]
            z <- y[index,,drop=FALSE]*weights[index]
            if(any(!is.finite(z)) || !integer.frequency(z))
                stop("assessment requires finite integer effective category counts",call.=FALSE)
            totals[index] <- rowSums(z)
        }
    } else {
        if(!is.null(dim(y))) stop("invalid assessment response dimensions",call.=FALSE)
        missing.response <- is.na(y)
        if(is.numeric(y)) {
            if(any(!missing.response & (!is.finite(y) | y!=floor(y) | y<1 | y>K)))
                stop("numeric assessment responses must be codes in the fitted category order",call.=FALSE)
            codes <- as.integer(y)
        } else {
            codes <- match(as.character(y),levels)
            if(any(is.na(codes) & !missing.response))
                stop("unknown assessment response category",call.=FALSE)
        }
        if(!integer.frequency(weights[!is.na(weights)]))
            stop("assessment requires integer individual frequency weights",call.=FALSE)
        totals <- weights
    }
    if(any(!is.na(totals) & !is.finite(totals)) || !is.finite(sum(totals,na.rm=TRUE)))
        stop("total assessment frequency overflows",call.=FALSE)
    list(y=y,weights=weights,codes=codes,totals=totals,grouped=grouped,
         missing.response=missing.response)
}

.ordinal_assessment_counts <- function(response,index,K) {
    if(response$grouped) return(response$y[index,,drop=FALSE]*response$weights[index])
    z <- matrix(0,length(index),K)
    if(length(index)) z[cbind(seq_along(index),response$codes[index])] <- response$weights[index]
    z
}

.ordinal_assessment_po <- function(object,design,response,included,control) {
    if(!is.list(control) || (length(control) && (is.null(names(control)) ||
       anyDuplicated(names(control)) || any(!names(control)%in%c("tol","maxit")))))
        stop("'po.control' permits only named 'tol' and 'maxit' entries",call.=FALSE)
    ctl <- utils::modifyList(list(tol=1e-8,maxit=200L),control)
    if(!is.numeric(ctl$tol) || is.complex(ctl$tol) || length(ctl$tol)!=1L || !is.finite(ctl$tol) || ctl$tol<=0 ||
       !is.numeric(ctl$maxit) || is.complex(ctl$maxit) || length(ctl$maxit)!=1L || !is.finite(ctl$maxit) ||
       ctl$maxit<1 || ctl$maxit!=floor(ctl$maxit) || ctl$maxit>.Machine$integer.max)
        stop("'po.control' requires positive 'tol' and positive integer 'maxit'",call.=FALSE)
    p <- length(object$beta); K <- length(object$levels)
    out <- data.frame(threshold=integer(),category=character(),term=character(),
        estimate=numeric(),std.error=numeric(),common.estimate=numeric(),
        converged=logical(),reason=character(),event.frequency=numeric(),
        nonevent.frequency=numeric(),warning=character(),stringsAsFactors=FALSE)
    if(!p) return(out)
    index <- which(included)
    xx <- design$x[index,,drop=FALSE]
    xx <- cbind(`(Intercept)`=rep(1,length(index)),xx)
    off <- design$offset[index]
    rows <- vector("list",K-1L)
    low <- numeric(length(index))
    if(response$grouped) {
        upper <- response$y[index,,drop=FALSE]*response$weights[index]
        for(j in seq.int(K-1L,1L)) upper[,j] <- upper[,j]+upper[,j+1L]
    }
    for(j in seq_len(K-1L)) {
        if(response$grouped) {
            # Sum both tails directly: total-minus-lower loses small upper counts.
            low <- low+response$y[index,j]*response$weights[index]
            high <- upper[,j+1L]
        } else {
            high <- response$weights[index]*(response$codes[index]>j)
            low <- response$weights[index]*(response$codes[index]<=j)
        }
        warning <- character(); fit <- NULL; reason <- "no_outcome_variation"
        # The binary fitting API converts counts to a proportion. Do not let a
        # rounded endpoint erase a real outcome from this optional diagnostic.
        fraction <- high/(high+low)
        lost.tail <- any(high>0 & low>0 & (fraction==0 | fraction==1))
        if(lost.tail) reason <- "unrepresentable_binary_counts"
        if(sum(high)>0 && sum(low)>0 && !lost.tail) {
            fit <- tryCatch(withCallingHandlers(fastglm(xx,cbind(high,low),
                family=binomial(),offset=off,firth=TRUE,method="auto",
                tol=ctl$tol,maxit=ctl$maxit,retain.data=FALSE),warning=function(w) {
                    warning <<- c(warning,conditionMessage(w)); invokeRestart("muffleWarning")
                }),error=function(e) { reason <<- conditionMessage(e); NULL })
        }
        ok <- !is.null(fit) && isTRUE(fit$converged) &&
            all(is.finite(fit$coefficients)) && all(is.finite(diag(fit$cov.unscaled))) &&
            all(diag(fit$cov.unscaled)>=0)
        estimate <- se <- rep(NA_real_,p)
        if(!is.null(fit)) {
            reason <- if(is.null(fit$convergence.reason))
                if(isTRUE(fit$converged)) "converged" else "not_converged" else fit$convergence.reason
            if(ok) {
                estimate <- unname(fit$coefficients[-1L])
                se <- sqrt(diag(fit$cov.unscaled)[-1L])
            } else if(isTRUE(fit$converged)) reason <- "unavailable_coefficients_or_covariance"
        }
        rows[[j]] <- data.frame(threshold=j,category=object$levels[j],term=names(object$beta),
            estimate=estimate,std.error=se,common.estimate=unname(object$beta),
            converged=ok,reason=reason,event.frequency=sum(high),nonevent.frequency=sum(low),
            warning=paste(unique(warning),collapse="; "),stringsAsFactors=FALSE)
    }
    out <- do.call(rbind,rows)
    rownames(out) <- NULL
    out
}

#' Assess ordinal prediction and the common-slope assumption
#'
#' Compares predicted cumulative probabilities with observed frequencies,
#' scores the full ordered prediction distribution, and optionally compares
#' predictor slopes across binary divisions of the outcome scale.
#' @param object a converged \code{fastglm_ordinal} model.
#' @param newdata prediction data frame or matrix, as in \code{predict}.
#' @param y ordered outcomes, category labels, numeric codes in the fitted
#'   category order, or a matrix of category counts. Required with
#'   \code{newdata}; otherwise defaults to the retained fitting response.
#'   Named count columns are aligned with the fitted category labels.
#' @param weights frequency weights. Defaults to the fitting weights only when
#'   both \code{newdata} and \code{y} are omitted; otherwise defaults to one.
#'   Counts times weights, or individual weights, must be integer frequencies.
#' @param offset optional replacement prediction offset.
#' @param bins positive integer number of equal-width calibration bins on
#'   \eqn{[0,1]}. Bin boundaries do not depend on observed outcomes.
#' @param proportional.odds also fit separate mean bias-reduced binary logistic
#'   models for the events \eqn{Y>j}, using the assessment rows and offsets.
#' @param po.control named list containing \code{tol} and/or \code{maxit} for
#'   these auxiliary binary fits; defaults are \code{1e-8} and \code{200}.
#' @details The ranked probability score (\code{rps}) is
#'   \eqn{\sum_{j=1}^{K-1}\{F_j-I(Y\leq j)\}^2};
#'   \code{rps_normalized} divides by \eqn{K-1}, and lies between zero and one.
#'   \code{log_loss} is \eqn{-\log p_Y}. All are losses: smaller values are
#'   better. Grouped rows average the individual-outcome losses, rather than
#'   squaring differences from grouped outcome proportions. Overall scores
#'   average over the included individual frequency. Log probabilities are
#'   evaluated directly without clipping. A positive count in a category with
#'   structural probability zero has infinite log loss; a zero count contributes
#'   zero. Extremely small positive probabilities can have a finite log loss
#'   even when the displayed probability underflows to zero.
#'
#'   Missing outcomes, weights, predictors, or offsets, and zero-frequency rows,
#'   are retained in \code{rows} with missing scores and an exclusion reason.
#'   Infinite predictors or offsets are errors. Empty calibration bins remain
#'   in the table with zero frequency and missing means. With no usable outcomes,
#'   overall scores are missing. Bins include their lower boundary; the last bin
#'   also includes one. Fitting rows omitted under \code{na.exclude}
#'   are restored when assessing the original response. Explicit responses with
#'   no \code{newdata} must match the retained design rows.
#'
#'   Scores on the fitting outcomes describe apparent performance. Supplying
#'   held-out outcomes evaluates the frozen model; the function does not establish
#'   that those outcomes were excluded during fitting. The optional binary fits
#'   use the supplied assessment outcomes and design when supplied. They affect
#'   only the proportional-odds display, never predictions or scores. The dashed
#'   reference slopes always come from the original ordinal model.
#'   Positive slopes correspond to higher outcomes, matching the ordinal model.
#'   The binary estimates are correlated, and bias reduction can produce different
#'   finite-sample changes at each threshold. Their separate standard errors are
#'   descriptive; neither interval overlap nor this plot is a formal test of
#'   proportional odds. Thresholds with no outcome variation or unsuccessful
#'   fits retain their status and have unavailable estimates. A threshold is also
#'   unavailable when binary count conversion would round a positive minority
#'   frequency away (\code{unrepresentable_binary_counts}). Calibration alone
#'   is also not a test of proportional odds.
#'
#'   Scoring and calibration use blocks of rows and require work proportional to
#'   the number of rows times the number of categories. The optional binary
#'   comparison fits one model per threshold without expanding frequency rows.
#' @returns An object of class \code{fastglm_ordinal_assessment}, containing
#'   \code{scores}, the overall frequency-weighted losses; \code{rows}, individual
#'   row averages and exclusions; \code{calibration}, predicted and observed
#'   cumulative probabilities by threshold and bin; and \code{proportional.odds},
#'   the optional slope comparison table. It also records the evaluation source,
#'   category order, and exclusion counts. Plot methods use these saved tables.
#' @references Epstein, E. S. (1969). A scoring system for probability forecasts
#'   of ranked categories. Journal of Applied Meteorology, 8, 985--987.
#'
#'   Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules,
#'   prediction, and estimation. Journal of the American Statistical Association,
#'   102, 359--378. \doi{10.1198/016214506000001437}.
#'
#'   Brant, R. (1990). Assessing proportionality in the proportional odds model
#'   for ordinal logistic regression. Biometrics, 46, 1171--1178.
#'   \doi{10.2307/2532457}.
#' @export
fastglm_ordinal_assess <- function(object,newdata=NULL,y=NULL,weights=NULL,
        offset=NULL,bins=10L,proportional.odds=FALSE,po.control=list()) {
    if(!inherits(object,"fastglm_ordinal") || !isTRUE(object$converged))
        stop("ordinal assessment requires a converged fastglm_ordinal fit",call.=FALSE)
    if(!is.numeric(bins) || is.complex(bins) || length(bins)!=1L || !is.finite(bins) ||
       bins<1 || bins!=floor(bins) || bins>.Machine$integer.max)
        stop("'bins' must be a positive integer",call.=FALSE)
    if(!is.logical(proportional.odds) || length(proportional.odds)!=1L || is.na(proportional.odds))
        stop("'proportional.odds' must be TRUE or FALSE",call.=FALSE)
    training <- is.null(newdata) && is.null(y)
    if(!is.null(newdata) && is.null(y))
        stop("supply assessment outcomes 'y' with 'newdata'",call.=FALSE)
    if(is.null(y)) y <- object$y
    if(training && is.null(weights)) weights <- object$weights
    design <- .ordinal_prediction_design(object,newdata,offset)
    n <- nrow(design$x); K <- length(object$levels); q <- K-1L
    response <- .ordinal_assessment_response(y,weights,object$levels,n)
    exclusion <- rep("",n)
    exclusion[!is.na(response$totals) & response$totals==0] <- "zero_frequency"
    exclusion[response$missing.response] <- "missing_outcome"
    exclusion[is.na(response$weights)] <- "missing_weight"
    # Separate unshared output vectors permit in-place block assignment. A data
    # frame here copies whole columns at every replacement inside the callback.
    row.logloss <- rep(NA_real_,n); row.rps <- rep(NA_real_,n)
    included <- rep(FALSE,n)
    calibration <- data.frame(threshold=rep(seq_len(q),each=bins),
        category=rep(object$levels[-K],each=bins),bin=rep(seq_len(bins),q),
        lower=rep((seq_len(bins)-1)/bins,q),upper=rep(seq_len(bins)/bins,q),
        frequency=0,predicted=0,observed=0)
    .design_blocks(design$x,function(z,index) {
        off <- design$offset[index]
        if(any(is.infinite(z)) || any(is.infinite(off)))
            stop("ordinal assessment predictors and offsets must be finite or missing",call.=FALSE)
        missing <- rowSums(is.na(z))>0L | is.na(off)
        exclusion[index[missing]] <<- "missing_predictor_or_offset"
        good <- !missing & exclusion[index]==""
        if(!any(good)) return(invisible(NULL))
        ii <- index[good]; z <- z[good,,drop=FALSE]; off <- off[good]
        xx <- .ordinal_prediction_scale(z,object$working$center,object$working$scale)
        location <- streaming_qr_eta(xx,object$working$beta,off)
        if(any(!is.finite(location))) stop("ordinal assessment linear predictor is nonfinite",call.=FALSE)
        counts <- .ordinal_assessment_counts(response,ii,K)
        frequency <- response$totals[ii]
        suffix <- counts
        for(j in seq.int(K-1L,1L)) suffix[,j] <- suffix[,j]+suffix[,j+1L]
        low <- rps <- logloss <- numeric(length(ii))
        alpha <- object$working$alpha
        for(j in seq_len(q)) {
            low <- low+counts[,j]
            eta <- alpha[object$threshold.map[j]]-location
            F <- plogis(eta); S <- plogis(-eta)
            rps <- rps+(low/frequency)*S^2+(suffix[,j+1L]/frequency)*F^2
            bin <- pmin(bins,floor(F*bins)+1L)
            sums <- rowsum(cbind(frequency,frequency*F,low),bin,reorder=FALSE)
            at <- (j-1L)*bins+as.integer(rownames(sums))
            calibration$frequency[at] <<- calibration$frequency[at]+sums[,1L]
            calibration$predicted[at] <<- calibration$predicted[at]+sums[,2L]
            calibration$observed[at] <<- calibration$observed[at]+sums[,3L]
        }
        # Direct log tails avoid clipping and preserve finite losses below the
        # floating-point probability range. Removed categories remain true zeros.
        retained <- object$retained.categories
        for(k in seq_len(K)) {
            active <- counts[,k]>0
            if(!any(active)) next
            r <- match(k,retained)
            lp <- if(is.na(r)) rep(-Inf,sum(active)) else if(r==1L)
                plogis(alpha[1L]-location[active],log.p=TRUE) else if(r==length(retained))
                plogis(location[active]-alpha[length(alpha)],log.p=TRUE) else
                plogis(alpha[r]-location[active],log.p=TRUE)+
                log(-expm1(alpha[r-1L]-alpha[r]))+
                plogis(location[active]-alpha[r-1L],log.p=TRUE)
            logloss[active] <- logloss[active]-(counts[active,k]/frequency[active])*lp
        }
        row.logloss[ii] <<- logloss; row.rps[ii] <<- rps; included[ii] <<- TRUE
    },object$block.size)
    rows <- data.frame(row=seq_len(n),frequency=response$totals,log_loss=row.logloss,
        rps=row.rps,rps_normalized=row.rps/q,included=included,exclusion=exclusion)
    calibration$predicted <- calibration$predicted/calibration$frequency
    calibration$observed <- calibration$observed/calibration$frequency
    calibration$predicted[calibration$frequency==0] <- NA_real_
    calibration$observed[calibration$frequency==0] <- NA_real_
    take <- rows$included; frequency <- sum(rows$frequency[take])
    score <- function(z) if(frequency>0) sum((rows$frequency[take]/frequency)*z[take]) else NA_real_
    scores <- data.frame(frequency=frequency,log_loss=score(rows$log_loss),
                         rps=score(rows$rps),rps_normalized=score(rows$rps_normalized))
    po <- if(proportional.odds) .ordinal_assessment_po(object,design,response,take,po.control) else NULL
    if(training && !is.null(object$na.action)) {
        index <- napredict(object$na.action,seq_len(n))
        rows <- rows[index,,drop=FALSE]
        omitted <- is.na(index); rows$row <- seq_len(nrow(rows))
        rows$included[omitted] <- FALSE; rows$exclusion[omitted] <- "omitted_during_fitting"
        rownames(rows) <- NULL
    }
    excluded <- table(rows$exclusion[!rows$included])
    exclusions <- data.frame(reason=as.character(names(excluded)),rows=as.integer(excluded),stringsAsFactors=FALSE)
    structure(list(scores=scores,rows=rows,calibration=calibration,proportional.odds=po,
        evaluation=if(training) "training" else "supplied outcomes",levels=object$levels,
        exclusions=exclusions,bins=bins,call=match.call()),class="fastglm_ordinal_assessment")
}

#' @rdname fastglm_ordinal_assess
#' @param x an ordinal assessment object.
#' @param ... additional arguments passed to the plotting function; unused by
#'   the print method.
#' @method print fastglm_ordinal_assessment
#' @export
print.fastglm_ordinal_assessment <- function(x,...) {
    cat("Ordinal assessment (",x$evaluation,")\n",sep="")
    print(x$scores,row.names=FALSE)
    if(nrow(x$exclusions)) { cat("Excluded rows:\n"); print(x$exclusions,row.names=FALSE) }
    if(!is.null(x$proportional.odds)) {
        failed <- unique(x$proportional.odds$threshold[!x$proportional.odds$converged])
        cat("Descriptive proportional-odds comparison; ",length(failed),
            " unavailable threshold fit(s).\n",sep="")
    }
    invisible(x)
}

#' @rdname fastglm_ordinal_assess
#' @param type cumulative calibration or the descriptive proportional-odds
#'   slope comparison.
#' @param level level for separate Wald intervals in the slope display. These
#'   are not simultaneous intervals or tests of differences between thresholds.
#' @method plot fastglm_ordinal_assessment
#' @export
plot.fastglm_ordinal_assessment <- function(x,type=c("calibration","proportional.odds"),
                                           level=.95,...) {
    type <- match.arg(type)
    dots <- list(...)
    draw <- function(defaults) do.call(graphics::plot,utils::modifyList(defaults,dots))
    if(type=="calibration") {
        groups <- seq_len(length(x$levels)-1L)
    } else {
        if(is.null(x$proportional.odds) || !nrow(x$proportional.odds))
            stop("request 'proportional.odds = TRUE' for a model with predictors first",call.=FALSE)
        if(!is.numeric(level) || is.complex(level) || length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
            stop("'level' must lie strictly between zero and one",call.=FALSE)
        groups <- unique(x$proportional.odds$term)
    }
    layout <- if(length(groups)==2L) c(1L,2L) else grDevices::n2mfrow(length(groups))
    old <- graphics::par(mfrow=layout,mar=c(4.1,4.1,2.5,1))
    on.exit(graphics::par(old))
    for(group in groups) {
        if(type=="calibration") {
            z <- x$calibration[x$calibration$threshold==group,,drop=FALSE]
            draw(list(x=z$predicted,y=z$observed,xlim=c(0,1),ylim=c(0,1),
                xlab="Mean predicted cumulative probability",ylab="Observed cumulative proportion",
                main=paste0("Y <= ",x$levels[group]),pch=19))
            graphics::abline(0,1,lty=2,col="gray50")
        } else {
            z <- x$proportional.odds[x$proportional.odds$term==group,,drop=FALSE]
            delta <- qnorm((1+level)/2)*z$std.error
            limits <- range(c(z$common.estimate,z$estimate-delta,z$estimate+delta),finite=TRUE)
            draw(list(x=z$threshold,y=z$estimate,xlim=c(.5,length(x$levels)-.5),ylim=limits,
                xaxt="n",xlab="Event: outcome above category",ylab="Slope (positive = higher outcomes)",
                main=group,pch=19))
            graphics::axis(1,at=z$threshold,labels=z$category)
            graphics::arrows(z$threshold,z$estimate-delta,z$threshold,z$estimate+delta,
                             angle=90,code=3,length=.04)
            graphics::abline(h=z$common.estimate[1L],lty=2,col="gray50")
            if(any(!z$converged)) graphics::mtext(paste(sum(!z$converged),"unavailable threshold fit(s)"),cex=.75)
        }
    }
    invisible(x)
}
