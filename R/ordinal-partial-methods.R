#' Methods for ordinal models with threshold-specific slopes
#'
#' Partial proportional-odds models allow selected predictor effects to differ
#' across cumulative thresholds. The same parameter layout is available for
#' probit and complementary log-log links, with interpretation on their link
#' scales. Common slopes and threshold-specific slopes are displayed separately.
#' @param object,x a fitted `fastglm_ordinal_partial` model, or its summary.
#' @param component coefficient block: `"all"`, `"location"` (all slopes),
#'   `"threshold"`, `"common"`, or `"nonparallel"`.
#' @param vcov. optional full coefficient covariance for a converged fit.
#' @param digits number of significant digits used for display.
#' @param ... further method arguments.
#' @details The coefficient vector contains thresholds, common slopes, and
#'   threshold-specific slopes, in that order. Within the last block, all
#'   selected predictors for the first threshold precede those for the second
#'   threshold. `object$gamma` also stores these slopes as a predictor-by-threshold
#'   matrix. Selecting `component="location"` includes both kinds of slope.
#'
#'   Thresholds describe the cumulative predictors when the covariates and
#'   offset equal zero. That covariate combination can lie outside the valid
#'   prediction domain, so the reported thresholds need not be ordered in
#'   isolation. The cumulative predictors are constrained to be ordered at
#'   positive-frequency training rows and checked again at prediction rows.
#'   Invalid predictions report the affected rows. Probability clipping or
#'   threshold reordering is not used to repair an invalid prediction.
#'
#'   Model-based covariance is inverse expected Fisher information, including
#'   covariance between all thresholds and slopes. Wald intervals and
#'   delta-method prediction standard errors are first-order approximations
#'   under an interior correctly specified model with independent observations
#'   or integer frequency counts and nonsingular limiting information.
#'   Numerical convergence alone does not establish those assumptions.
#'   With `firth=TRUE`, the full-model cumulative-link adjustment targets the
#'   leading mean bias of the reported thresholds, common slopes, and
#'   threshold-specific slopes. These adjusted equations do not generally
#'   maximize a scalar penalized likelihood. Empty-category reductions require
#'   additional parameter constraints and are not supported for these fits.
#' @returns Coefficients, covariance, predictions, or a summary as appropriate.
#'   Printing invisibly returns the supplied object.
#' @name ordinal-partial-methods
#' @md
NULL

#' @rdname ordinal-partial-methods
#' @md
#' @export
coef.fastglm_ordinal_partial <- function(object,component=c("all","location","threshold","common","nonparallel"),...) {
    if(length(list(...))) stop("unused nonparallel coefficient arguments",call.=FALSE)
    object$coefficients[.ordinal_partial_indices(object,match.arg(component))]
}

#' @rdname ordinal-partial-methods
#' @md
#' @param type for `vcov()`, `"model"` for inverse expected information, or
#'   `"HC0"` or `"HC1"` for ordinary-score sandwich covariance. All thresholds
#'   and both slope blocks enter the joint observed-information calculation.
#'   For `predict()`, the scale: `"probability"` (or `"response"`),
#'   `"cumulative"`, `"class"`, or `"link"`. Nonparallel link predictions
#'   are an observation-by-threshold matrix of cumulative predictors
#'   `alpha[j] - x_common beta - x_nonparallel gamma[,j] - offset`.
#'   Applying the inverse link gives the cumulative probabilities.
#' @param cluster,cadjust,block.size cluster identifiers, cluster adjustment,
#'   and processing block size, as in [fastglm_vcov()]. Supplying `cluster`
#'   requires `type="HC0"` or `type="HC1"`. For mean-bias-reduced fits,
#'   sandwich covariance uses the same first-order ordinary-score convention
#'   and regularity assumptions described there.
#' @export
vcov.fastglm_ordinal_partial <- function(object,component=c("all","location","threshold","common","nonparallel"),
    type=c("model","HC0","HC1"),cluster=NULL,cadjust=TRUE,block.size=8192L,...) {
    if(length(list(...))) stop("unused nonparallel covariance arguments",call.=FALSE)
    component <- match.arg(component);type <- match.arg(type)
    if(type=="model" && !is.null(cluster))
        stop("cluster identifiers require type = 'HC0' or 'HC1'",call.=FALSE)
    .ordinal_check_inference(object)
    i <- .ordinal_partial_indices(object,component)
    if(type!="model") {
        full <- fastglm_vcov(object,type,cluster,cadjust,block.size)
        out <- full[i,i,drop=FALSE]
        for(nm in setdiff(names(attributes(full)),c("dim","dimnames"))) attr(out,nm) <- attr(full,nm)
        return(out)
    }
    out <- object$mapped.covariance[i,i,drop=FALSE]
    attr(out,"covariance.scope") <- "expected Fisher information"
    out
}

#' @rdname ordinal-partial-methods
#' @md
#' @export
summary.fastglm_ordinal_partial <- function(object,vcov.=NULL,...) {
    if(length(list(...))) stop("unused nonparallel summary arguments",call.=FALSE)
    b <- coef(object);se <- rep(NA_real_,length(b))
    if(!is.null(vcov.) || isTRUE(object$converged)) se <- sqrt(diag(.ordinal_summary_covariance(object,vcov.)))
    z <- b/se;tab <- cbind(Estimate=b,"Std. Error"=se,"z value"=z,"Pr(>|z|)"=2*pnorm(-abs(z)))
    selected <- .ordinal_partial_indices(object,"nonparallel")
    varying <- data.frame(Predictor=rep(rownames(object$gamma),ncol(object$gamma)),
        Threshold=rep(colnames(object$gamma),each=nrow(object$gamma)),tab[selected,,drop=FALSE],
        row.names=NULL,check.names=FALSE)
    structure(list(call=object$call,coefficients=tab,
        location=tab[.ordinal_partial_indices(object,"location"),,drop=FALSE],
        common=tab[.ordinal_partial_indices(object,"common"),,drop=FALSE],nonparallel=varying,
        thresholds=tab[.ordinal_partial_indices(object,"threshold"),1:2,drop=FALSE],
        converged=object$converged,convergence.reason=object$convergence.reason,
        iterations=object$iterations,firth=object$firth,link=.ordinal_link(object),
        frequency=object$frequency,nobs.rows=object$nobs.rows,loglik=object$loglik,boundary=object$boundary,
        covariance.type=if(is.null(vcov.)) "expected Fisher information" else "supplied covariance",
        predictor.step=object$predictor.step,scaled.score=object$scaled.score,minimum.gap=object$minimum.gap),
        class="summary.fastglm_ordinal_partial")
}

.ordinal_partial_title <- function(object) if(.ordinal_link(object)=="logit")
    "Partial proportional-odds regression" else paste0("Nonparallel cumulative-link regression (",.ordinal_link(object),")")

#' @rdname ordinal-partial-methods
#' @md
#' @export
print.summary.fastglm_ordinal_partial <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(.ordinal_partial_title(x),x$call)
    .fastglm_print_field("Estimation",if(x$firth) "mean bias reduction" else "maximum likelihood")
    .fastglm_print_coefficients(x$common,"Common slopes",digits,...)
    cat("\nThreshold-specific slopes:\n")
    .fastglm_print_table(x$nonparallel,digits,p.columns="Pr(>|z|)")
    .fastglm_print_coefficients(x$thresholds,"Thresholds at zero covariates and offset",digits,
        P.values=FALSE,has.Pvalue=FALSE,...)
    cat("\n")
    if(isTRUE(x$converged)) .fastglm_print_field("Uncertainty",x$covariance.type)
    .fastglm_print_field("Sample",paste0(x$frequency," observations represented by ",x$nobs.rows," positive-frequency rows"))
    .fastglm_print_convergence(x$converged,x$iterations,x$convergence.reason)
    invisible(x)
}

#' @rdname ordinal-partial-methods
#' @md
#' @export
print.fastglm_ordinal_partial <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(.ordinal_partial_title(x),x$call)
    .fastglm_print_field("Estimation",if(x$firth) "mean bias reduction" else "maximum likelihood")
    .fastglm_print_coefficients(x$beta,"Common slopes",digits,...)
    cat("\nThreshold-specific slopes:\n")
    .fastglm_print_table(data.frame(Predictor=rep(rownames(x$gamma),ncol(x$gamma)),
        Threshold=rep(colnames(x$gamma),each=nrow(x$gamma)),Estimate=as.numeric(x$gamma)),digits)
    .fastglm_print_coefficients(x$alpha,"Thresholds at zero covariates and offset",digits,...)
    cat("\n");.fastglm_print_convergence(x$converged,x$iterations,x$convergence.reason)
    invisible(x)
}

#' @rdname ordinal-partial-methods
#' @md
#' @param parm parameters by name or position; defaults to all parameters.
#' @param level confidence level.
#' @param method interval method: `"wald"` or, for maximum-likelihood fits
#'   with retained data, `"profile"`.
#' @param control numerical controls for profile fitting, as described in
#'   [fastglm_score_test()].
#' @export
confint.fastglm_ordinal_partial <- function(object,parm,level=.95,method=c("wald","profile"),control=list(),...) {
    if(length(list(...))) stop("unused nonparallel confidence-interval arguments",call.=FALSE)
    method <- match.arg(method)
    if(!is.numeric(level) || is.complex(level) || length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie between zero and one",call.=FALSE)
    b <- coef(object);if(missing(parm)) parm <- seq_along(b)
    if(method=="profile") return(.ordinal_profile_confint(object,parm,level,control))
    if(length(control)) stop("nonparallel Wald intervals do not use optimizer controls",call.=FALSE)
    if(is.character(parm)) parm <- match(parm,names(b))
    if(!is.numeric(parm) || is.complex(parm) || anyNA(parm) ||
       any(parm<1 | parm>length(b) | parm!=floor(parm)) || anyDuplicated(parm))
        stop("invalid or duplicated nonparallel parameter selection",call.=FALSE)
    v <- vcov(object);half <- qnorm((1+level)/2)*sqrt(diag(v))[parm]
    out <- cbind(b[parm]-half,b[parm]+half)
    dimnames(out) <- list(names(b)[parm],paste0(format(100*c((1-level)/2,(1+level)/2),trim=TRUE),"%"))
    attr(out,"method") <- "Wald with expected Fisher information"
    out
}

#' @rdname ordinal-partial-methods
#' @md
#' @param newdata prediction data, as in [predict.fastglm_ordinal()].
#' @param se.fit include full-model delta-method standard errors.
#' @param offset optional replacement location offset.
#' @export
predict.fastglm_ordinal_partial <- function(object,newdata=NULL,
    type=c("probability","cumulative","class","link","response"),se.fit=FALSE,offset=NULL,...) {
    if(length(list(...))) stop("unused nonparallel prediction arguments",call.=FALSE)
    type <- match.arg(type);if(type=="response") type <- "probability"
    if(!is.logical(se.fit) || length(se.fit)!=1L || is.na(se.fit)) stop("'se.fit' must be TRUE or FALSE",call.=FALSE)
    if(se.fit && type=="class") stop("class predictions do not have delta-method standard errors",call.=FALSE)
    if(se.fit) .ordinal_check_inference(object)
    design <- .ordinal_prediction_design(object,newdata,offset)
    n <- nrow(design$x);K <- length(object$levels);q <- K-1L
    labels <- if(type%in%c("cumulative","link")) object$levels[-K] else object$levels
    fit <- if(type=="class") rep(NA_integer_,n) else matrix(NA_real_,n,length(labels),dimnames=list(rownames(design$x),labels))
    se <- if(se.fit) fit else NULL
    .design_blocks(design$x,function(z,i) {
        off <- design$offset[i]
        if(any(is.infinite(z)) || any(is.infinite(off))) stop("ordinal prediction inputs must be finite or missing",call.=FALSE)
        good <- rowSums(!is.finite(z))==0L & is.finite(off)
        if(!any(good)) return(invisible(NULL))
        rows <- i[good];z <- z[good,,drop=FALSE];off <- off[good]
        parts <- .ordinal_partial_probability_block(object,z,off,rows=rows)
        if(type=="class") {fit[rows] <<- max.col(parts$probability,ties.method="first");return(invisible(NULL))}
        fit[rows,] <<- if(type=="link") parts$eta else parts[[type]]
        if(se.fit) for(j in seq_along(labels)) {
            if(type=="link") {
                pc <- length(object$beta);pn <- length(object$nonparallel)
                g <- matrix(0,nrow(z),length(object$coefficients));g[,j] <- 1
                if(pc) g[,q+seq_len(pc)] <- -parts$x[,object$common,drop=FALSE]
                g[,q+pc+(j-1L)*pn+seq_len(pn)] <- -parts$x[,object$nonparallel,drop=FALSE]
            } else g <- .ordinal_partial_gradient(parts,j,type)
            se[rows,j] <<- sqrt(pmax(0,rowSums((g%*%object$working$covariance)*g)))
        }
    },object$block.size)
    if(type=="class") {fit <- ordered(object$levels[fit],levels=object$levels);names(fit) <- rownames(design$x)}
    if(is.null(newdata) && !is.null(object$na.action)) {
        fit <- napredict(object$na.action,fit)
        if(se.fit) se <- napredict(object$na.action,se)
    }
    if(se.fit) .fastglm_prediction(list(fit=fit,se.fit=se,covariance.scope="expected Fisher information"),
        if(type=="link") "cumulative.link" else type) else fit
}

#' @rdname ordinal-partial-methods
#' @md
#' @param conf.int include Wald confidence limits in `broom::tidy()` output.
#' @param conf.level confidence level for `broom::tidy()`.
#' @param exponentiate exponentiate slopes and their confidence limits.
#'   Odds-ratio interpretation applies to the logit link, with separate ratios
#'   for each threshold when a predictor has a nonparallel effect.
#' @exportS3Method broom::tidy fastglm_ordinal_partial
tidy.fastglm_ordinal_partial <- function(x,conf.int=FALSE,conf.level=.95,exponentiate=FALSE,
    component=c("all","location","threshold","common","nonparallel"),vcov.=NULL,...) {
    if(length(list(...))) stop("unused nonparallel tidy arguments",call.=FALSE)
    for(flag in list(conf.int,exponentiate)) if(!is.logical(flag) || length(flag)!=1L || is.na(flag))
        stop("tidy flags must be TRUE or FALSE",call.=FALSE)
    if(!is.numeric(conf.level) || is.complex(conf.level) || length(conf.level)!=1L || !is.finite(conf.level) || conf.level<=0 || conf.level>=1)
        stop("'conf.level' must lie between zero and one",call.=FALSE)
    component <- match.arg(component);tab <- summary(x,vcov.=vcov.)$coefficients
    q <- length(x$alpha);pc <- length(x$beta);pn <- nrow(x$gamma)
    out <- data.frame(term=rownames(tab),component=c(rep("threshold",q),rep("common",pc),rep("nonparallel",q*pn)),
        threshold=c(names(x$alpha),rep(NA_character_,pc),rep(colnames(x$gamma),each=pn)),
        estimate=unname(tab[,1L]),std.error=unname(tab[,2L]),statistic=unname(tab[,3L]),
        p.value=unname(tab[,4L]),row.names=NULL)
    out$statistic[seq_len(q)] <- out$p.value[seq_len(q)] <- NA_real_
    if(conf.int) {
        half <- qnorm((1+conf.level)/2)*out$std.error
        out$conf.low <- out$estimate-half;out$conf.high <- out$estimate+half
    }
    if(exponentiate) for(nm in intersect(c("estimate","conf.low","conf.high"),names(out)))
        out[out$component!="threshold",nm] <- exp(out[out$component!="threshold",nm])
    out[.ordinal_partial_indices(x,component),,drop=FALSE]
}
