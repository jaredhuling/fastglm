#' Fit a cumulative-link model for ordered outcomes
#'
#' Fits a cumulative-link model with common slopes and freely estimated
#' thresholds, by maximum likelihood or mean bias reduction. Selected terms
#' can have threshold-specific slopes in a dense fit with either estimation method.
#' Positive slopes shift probability toward higher outcome categories.
#' @param x a formula, an object from [fastglm_ordinal_prepare()], or a finite
#'   dense numeric matrix of location predictors.
#'   A matrix must omit the intercept; formula fits remove the model-matrix
#'   intercept because the thresholds already specify location.
#' @param y an ordered factor, a factor or character response with an explicit
#'   category order in `levels`, integer category codes with `levels`, or a
#'   matrix whose columns contain category counts in increasing outcome order.
#' @param data,subset,na.action,contrasts formula and model-frame arguments.
#' @param weights nonnegative integer frequency weights for individual
#'   responses. For count matrices, counts times weights must be integers.
#'   Frequencies are used without normalization.
#' @param offset a location offset: the model is
#'   `link{Pr(Y <= j)} = alpha[j] - x beta - offset`.
#' @param levels category labels in increasing order. Numeric responses are
#'   interpreted as codes from one to the number of labels.
#' @param firth use mean bias reduction through adjusted score equations.
#'   `NULL` in a prepared-design fit uses that design's default.
#' @param link cumulative link: `"logit"` (proportional odds), `"probit"`,
#'   or `"cloglog"` (complementary log-log). Prepared fits use their saved
#'   link when `NULL` is supplied.
#' @param nonparallel terms whose slopes may differ across cumulative
#'   thresholds. For a formula fit, supply a one-sided formula selecting
#'   complete terms from the fitted formula, such as `~ exposure` or
#'   `~ exposure:group`. `~ .` selects every predictor term. For a matrix,
#'   supply distinct column names or indices. `NULL` or an empty selection
#'   retains the common-slope model and its existing fitting path.
#' @param tol tolerance for the remaining change in cumulative linear
#'   predictors and the standardized adjusted score.
#' @param maxit maximum fitting iterations.
#' @param block.size maximum observations processed together by the numerical
#'   kernel. The input design is retained in memory.
#' @param control named optimizer controls: `score.tol`, `step.tol`,
#'   `max.step`, `max.halves`, `condition.tol`, `standardize`, and `trace`.
#' @param model retain the formula model frame.
#' @param ... additional arguments to the default or formula method.
#' @details By default, thresholds are estimated in increasing order. The same location
#'   slopes apply to all cumulative probabilities. Full-rank location designs
#'   are required. Sparse matrices and
#'   repeatable sources are not accepted by this interface.
#'
#'   With `nonparallel`, the model is
#'   `link{Pr(Y <= j)} = alpha[j] - x_common beta - x_nonparallel gamma[,j] - offset`.
#'   A selected predictor has a separate slope at each threshold and no
#'   additional common slope. Fitting requires every declared category to
#'   have positive frequency. The cumulative predictors must remain ordered
#'   at the observed predictor values, and predictions check this condition
#'   at each supplied row. Ordering at training values does not establish
#'   validity under extrapolation. See [ordinal-partial-methods].
#'
#'   In common-slope models, an empty interior category is removed from the numerical fit and then
#'   restored to the declared scale with zero fitted probability and coincident
#'   thresholds. Convergence and information refer to the reduced model.
#'   Standard errors for the tied thresholds are unavailable. Bias-reduced fits
#'   retain empty endpoint categories; maximum likelihood with empty endpoints
#'   requires infinite thresholds and is rejected.
#'
#'   The covariance uses inverse expected Fisher information at the fitted
#'   estimate. It gives a first-order model-based approximation; this interface
#'   does not provide ordinal penalized-profile inference. Robust covariance
#'   is available through `fastglm_vcov()` or `vcov(type="HC0")` for converged
#'   common-slope fits with every declared category observed. Mean bias
#'   reduction solves the adjusted equations for the original thresholds and slopes.
#'   It does not maximize a log-determinant penalized ordinal likelihood.
#'
#'   `logLik()` evaluates the ordinary categorical frequency likelihood,
#'   omitting multinomial grouping constants. Consequently, grouping identical
#'   covariate patterns preserves this likelihood convention. For bias-reduced
#'   fits it is the ordinary likelihood evaluated at the bias-reduced estimate.
#'   `deviance()` compares this kernel with a saturated fit to the supplied
#'   response rows. It therefore uses a grouped multinomial saturated model
#'   for a count-matrix response and depends on the chosen grouping.
#' @returns An object of class `fastglm_ordinal`, with thresholds `alpha`,
#'   location coefficients `beta`, the full `coefficients` vector, category
#'   mapping, covariance, convergence diagnostics, and retained design.
#'   With selected nonparallel effects, the additional class
#'   `fastglm_ordinal_partial` provides threshold-specific slope methods;
#'   `gamma` stores these slopes as a predictor-by-threshold matrix, and `beta`
#'   contains only the common slopes.
#'   `nobs()` gives the total frequency, while `nobs.rows` counts rows with
#'   positive total frequency.
#' @references Kosmidis, I. (2014). Improved estimation in cumulative link
#'   models. Journal of the Royal Statistical Society, Series B, 76, 169--196.
#'   \doi{10.1111/rssb.12025}.
#' @examples
#' set.seed(91)
#' x <- rnorm(150)
#' y <- ordered(cut(.5*x + rlogis(150), c(-Inf,-.5,.7,Inf),
#'                  labels=c("low","middle","high")))
#' fit <- fastglm_ordinal(y ~ x, firth=TRUE)
#' coef(fit, component="location")
#' head(predict(fit))
#' @md
#' @export
fastglm_ordinal <- function(x, ...) UseMethod("fastglm_ordinal")

#' @rdname fastglm_ordinal
#' @export
fastglm_ordinal.default <- function(x,y,weights=NULL,offset=NULL,levels=NULL,
                                    firth=FALSE,tol=1e-8,maxit=200L,
                                    block.size=2048L,control=list(),link="logit",nonparallel=NULL,...) {
    if(length(list(...))) stop("unused ordinal fitting arguments",call.=FALSE)
    if(missing(y)) stop("an ordinal response is required",call.=FALSE)
    fit <- .ordinal_fit_matrix(x,y,weights,offset,levels,firth,tol,maxit,block.size,control,
        link=link,nonparallel=nonparallel)
    fit$call <- match.call()
    fit$call[[1L]] <- quote(fastglm::fastglm_ordinal)
    fit
}

.ordinal_fit_matrix <- function(x,y,weights,offset,levels,firth,tol,maxit,block.size,control,cache=NULL,link="logit",nonparallel=NULL) {
    if(!is.matrix(x) || !is.numeric(x))
        stop("ordinal fitting currently requires a dense numeric matrix or formula",call.=FALSE)
    if(is.factor(y) && !is.ordered(y) && is.null(levels))
        stop("use an ordered response factor or supply its category order in 'levels'",call.=FALSE)
    if(ncol(x)>0L && is.null(colnames(x))) colnames(x) <- paste0("x",seq_len(ncol(x)))
    if(anyNA(colnames(x)) || any(!nzchar(colnames(x))) || anyDuplicated(colnames(x)))
        stop("ordinal predictor column names must be distinct and nonmissing",call.=FALSE)
    selected <- .ordinal_nonparallel_columns(nonparallel,colnames(x))
    if(length(selected))
        return(.ordinal_partial_fit_matrix(x,y,weights,offset,levels,firth,tol,maxit,
            block.size,control,link,selected))
    fit <- .ordinal_fit_core(x,y,weights,offset,levels,firth,tol,maxit,block.size,control,cache,link)
    q <- length(fit$alpha)
    names(fit$coefficients) <- c(paste0("threshold:",names(fit$alpha)),names(fit$beta))
    dimnames(fit$mapped.covariance) <- rep(list(names(fit$coefficients)),2L)
    fit$cov.unscaled <- fit$mapped.covariance
    fit$thresholds <- fit$alpha
    fit$tied.thresholds <- duplicated(fit$threshold.map) |
        duplicated(fit$threshold.map,fromLast=TRUE)
    fit$x <- x; fit$y <- y; fit$design.columns <- colnames(x)
    fit$nobs.rows <- fit$nobs; fit$nobs <- fit$frequency
    class(fit) <- "fastglm_ordinal"
    fit
}

#' @rdname fastglm_ordinal
#' @export
fastglm_ordinal.formula <- function(x,data,weights,offset,subset,na.action,
                                    contrasts=NULL,levels=NULL,firth=FALSE,
                                    tol=1e-8,maxit=200L,block.size=2048L,
                                    control=list(),model=TRUE,link="logit",nonparallel=NULL,...) {
    if(length(list(...))) stop("unused ordinal fitting arguments",call.=FALSE)
    if(!is.logical(model) || length(model)!=1L || is.na(model))
        stop("'model' must be TRUE or FALSE",call.=FALSE)
    call <- match.call()
    mf <- call[c(1L,match(c("x","data","weights","offset","subset","na.action"),names(call),0L))]
    names(mf)[names(mf)=="x"] <- "formula"
    mf[[1L]] <- quote(stats::model.frame)
    mf$drop.unused.levels <- FALSE
    mf <- eval(mf,parent.frame())
    terms <- attr(mf,"terms")
    design <- model.matrix(terms,mf,contrasts.arg=contrasts)
    coding <- attr(design,"contrasts")
    assignment <- attr(design,"assign")
    design <- design[,attr(design,"assign")!=0L,drop=FALSE]
    selected <- .ordinal_nonparallel_terms(nonparallel,terms,assignment[assignment!=0L])
    fit <- fastglm_ordinal.default(design,model.response(mf),
        weights=model.weights(mf),offset=model.offset(mf),levels=levels,
        firth=firth,tol=tol,maxit=maxit,block.size=block.size,control=control,link=link,
        nonparallel=selected$columns)
    fit$terms <- terms
    fit$design.assign <- assignment[assignment!=0L]
    fit$xlevels <- stats::.getXlevels(terms,mf)
    fit$contrasts <- coding
    fit$na.action <- attr(mf,"na.action")
    fit$offset.expression <- call$offset
    if(inherits(fit,"fastglm_ordinal_partial")) fit$nonparallel.terms <- selected$terms
    if(model) fit$model <- mf
    fit$call <- call
    fit$call[[1L]] <- quote(fastglm::fastglm_ordinal)
    fit
}

#' @rdname ordinal-methods
#' @param formula. optional change to the formula, as in `update()`.
#' @param evaluate evaluate the updated call or return it for inspection.
#' @exportS3Method update fastglm_ordinal
update.fastglm_ordinal <- function(object,formula.,...,evaluate=TRUE) {
    if(inherits(object,"fastglm_ordinal_stream") && !missing(formula.))
        stop("formula changes require rebuilding the ordinal source schema with fastglm_source() and refitting",call.=FALSE)
    if(!is.null(object$batch.response))
        stop("update is unavailable for ordinal batch fits; refit the response with fastglm_ordinal(prepared, y, ...)",call.=FALSE)
    if(isTRUE(object$prepared) && !missing(formula.))
        stop("formula changes require preparing the revised ordinal design again",call.=FALSE)
    call <- object$call
    if(!missing(formula.)) {
        if(is.null(object$terms)) stop("formula updates require an ordinal formula fit",call.=FALSE)
        call$x <- update.formula(formula(object$terms),formula.)
    }
    extras <- match.call(expand.dots=FALSE)$...
    if(length(extras)) {
        if(is.null(names(extras)) || any(!nzchar(names(extras))))
            stop("updated ordinal fitting arguments must be named",call.=FALSE)
        for(name in names(extras)) call[name] <- extras[name]
    }
    if(evaluate) eval(call,parent.frame()) else call
}

.ordinal_indices <- function(object,component) {
    q <- length(object$alpha); p <- length(object$beta)
    switch(component,all=seq_len(q+p),location=q+seq_len(p),threshold=seq_len(q))
}

#' Methods for cumulative-link ordinal fits
#' @param object a fitted \code{fastglm_ordinal} model.
#' @param component return all parameters, location slopes, or thresholds.
#' @param ... further method arguments.
#' @returns Coefficients, covariance, or model summaries, as appropriate.
#' @rdname ordinal-methods
#' @export
coef.fastglm_ordinal <- function(object,component=c("all","location","threshold"),...) {
    object$coefficients[.ordinal_indices(object,match.arg(component))]
}

.ordinal_check_inference <- function(object) {
    if(!isTRUE(object$converged))
        stop("ordinal uncertainty requires a converged fit; inspect the convergence diagnostics",call.=FALSE)
    invisible(NULL)
}

#' @rdname ordinal-methods
#' @param type covariance convention: \code{"model"} uses expected Fisher
#'   information; \code{"HC0"} and \code{"HC1"} use the ordinary-score sandwich
#'   described in \code{fastglm_vcov()}.
#' @param cluster optional cluster identifiers for robust covariance.
#' @param cadjust apply the cluster-count correction for robust covariance.
#' @param block.size number of rows processed together for robust covariance.
#' @export
vcov.fastglm_ordinal <- function(object,component=c("all","location","threshold"),
                                type=c("model","HC0","HC1"),cluster=NULL,
                                cadjust=TRUE,block.size=8192L,...) {
    if(length(list(...))) stop("unused ordinal covariance arguments",call.=FALSE)
    .ordinal_check_inference(object)
    component <- match.arg(component); type <- match.arg(type)
    if(type!="model") {
        v <- fastglm_vcov(object,type=type,cluster=cluster,cadjust=cadjust,block.size=block.size)
        i <- .ordinal_indices(object,component)
        if(component=="all") return(v)
        out <- v[i,i,drop=FALSE]
        for(nm in setdiff(names(attributes(v)),c("dim","dimnames"))) attr(out,nm) <- attr(v,nm)
        return(out)
    }
    if(!is.null(cluster)) stop("cluster covariance requires type = 'HC0' or 'HC1'",call.=FALSE)
    v <- object$mapped.covariance
    tied <- which(object$tied.thresholds)
    if(length(tied)) { v[tied,] <- NA_real_; v[,tied] <- NA_real_ }
    i <- .ordinal_indices(object,component)
    v <- v[i,i,drop=FALSE]
    attr(v,"covariance.scope") <- if(object$boundary) "reduced category model" else "expected Fisher information"
    v
}

.ordinal_summary_covariance <- function(object,vcov.) {
    if(is.null(vcov.)) return(vcov(object))
    .ordinal_check_inference(object)
    if(object$boundary) stop("supplied summary covariance is unsupported with collapsed ordinal categories",call.=FALSE)
    b <- coef(object); d <- length(b)
    if(!is.matrix(vcov.) || !is.numeric(vcov.) || is.complex(vcov.) ||
       !identical(dim(vcov.),c(d,d)) || any(!is.finite(vcov.)) ||
       !isSymmetric(vcov.,tol=1e-8) || any(diag(vcov.)<0))
        stop("supply a finite symmetric full ordinal covariance with nonnegative variances",call.=FALSE)
    if(!is.null(colnames(vcov.)) || !is.null(rownames(vcov.))) {
        if(anyDuplicated(colnames(vcov.)) || anyDuplicated(rownames(vcov.)) ||
           !setequal(colnames(vcov.),names(b)) || !setequal(rownames(vcov.),names(b)))
            stop("supplied covariance names do not match the ordinal parameters",call.=FALSE)
        vcov. <- vcov.[names(b),names(b),drop=FALSE]
    }
    vcov.
}

#' @rdname ordinal-methods
#' @param vcov. optional full coefficient covariance for the summary. Defaults
#'   to model-based expected Fisher information.
#' @method summary fastglm_ordinal
#' @exportS3Method summary fastglm_ordinal
summary.fastglm_ordinal <- function(object,vcov.=NULL,...) {
    b <- coef(object); se <- rep(NA_real_,length(b))
    if(!is.null(vcov.) || isTRUE(object$converged)) se <- sqrt(diag(.ordinal_summary_covariance(object,vcov.)))
    z <- b/se
    tab <- cbind(Estimate=b,"Std. Error"=se,"z value"=z,"Pr(>|z|)"=2*pnorm(-abs(z)))
    q <- length(object$alpha)
    structure(list(call=object$call,location=tab[q+seq_along(object$beta),,drop=FALSE],
        thresholds=tab[seq_len(q),1:2,drop=FALSE],converged=object$converged,
        convergence.reason=object$convergence.reason,iterations=object$iterations,
        firth=object$firth,link=.ordinal_link(object),frequency=object$frequency,nobs.rows=object$nobs.rows,
        loglik=object$loglik,boundary=object$boundary,
        covariance.type=if(is.null(vcov.)) "expected Fisher information" else "supplied covariance",
        collapsed.categories=object$levels[object$collapsed.categories],
        predictor.step=object$predictor.step,scaled.score=object$scaled.score),
        class="summary.fastglm_ordinal")
}

#' @exportS3Method print summary.fastglm_ordinal
print.summary.fastglm_ordinal <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(if(.ordinal_link(x)=="logit") "Proportional-odds regression" else
        paste0("Cumulative-link regression (",.ordinal_link(x),")"),x$call)
    .fastglm_print_field("Estimation",if(x$firth) "mean bias reduction" else "maximum likelihood")
    .fastglm_print_coefficients(x$location,"Location coefficients (positive means higher outcomes)",digits,...)
    .fastglm_print_coefficients(x$thresholds,"Thresholds",digits,P.values=FALSE,has.Pvalue=FALSE,...)
    cat("\n")
    if(isTRUE(x$converged)) .fastglm_print_field("Uncertainty",x$covariance.type)
    .fastglm_print_field("Sample",paste0(x$frequency," observations represented by ",x$nobs.rows," positive-frequency rows"))
    .fastglm_print_convergence(x$converged,x$iterations,x$convergence.reason)
    if(x$boundary) {
        .fastglm_print_field("Empty interior categories",x$collapsed.categories)
        cat("Tied-threshold standard errors are unavailable; inference uses the reduced category model.\n")
    }
    invisible(x)
}

#' @exportS3Method print fastglm_ordinal
print.fastglm_ordinal <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(if(.ordinal_link(x)=="logit") "Proportional-odds regression" else
        paste0("Cumulative-link regression (",.ordinal_link(x),")"),x$call)
    .fastglm_print_field("Estimation",if(x$firth) "mean bias reduction" else "maximum likelihood")
    .fastglm_print_coefficients(x$beta,"Location coefficients",digits,...)
    .fastglm_print_coefficients(x$alpha,"Thresholds",digits,...)
    cat("\n"); .fastglm_print_convergence(x$converged,x$iterations,x$convergence.reason)
    invisible(x)
}

#' @export
model.matrix.fastglm_ordinal <- function(object,...) object$x

#' @export
model.frame.fastglm_ordinal <- function(formula,...) {
    if(is.null(formula$model)) stop("the ordinal formula model frame was not retained",call.=FALSE)
    formula$model
}

#' @export
nobs.fastglm_ordinal <- function(object,...) object$frequency

#' @export
weights.fastglm_ordinal <- function(object,...) object$weights

#' @export
logLik.fastglm_ordinal <- function(object,...) {
    structure(object$loglik,df=object$rank,nobs=object$frequency,
              converged=object$converged,estimation=object$method,class="logLik")
}

#' @export
deviance.fastglm_ordinal <- function(object,...) {
    saturated <- 0
    if(object$category$grouped) {
        counts <- object$category$counts
        for(first in seq.int(1L,nrow(counts),by=object$block.size)) {
            i <- first:min(nrow(counts),first+object$block.size-1L)
            y <- counts[i,,drop=FALSE]*object$weights[i]
            positive <- y>0
            proportions <- y/object$category$totals[i]
            saturated <- saturated+sum(y[positive]*log(proportions[positive]))
        }
    }
    2*(saturated-object$loglik)
}

#' @rdname ordinal-methods
#' @param parm parameters by name or position; defaults to all parameters.
#' @param level confidence level.
#' @param method \code{"wald"} for the default expected-information normal
#'   intervals, or \code{"profile"} for maximum-likelihood profile intervals.
#'   Profiles re-estimate all remaining coefficients at each selected value
#'   and invert an asymptotic likelihood-ratio test. They require an ML fit,
#'   retained dense data, and every declared category observed.
#' @param control numerical controls for profile inference: \code{maxit=300},
#'   \code{score.tol=1e-7}, \code{step.tol=1e-7}, \code{root.tol=1e-6},
#'   \code{max.expand=25}, \code{max.root=100}, \code{max.halves=35},
#'   \code{max.step=4}, and \code{condition.tol=1e-14}. Scores are standardized
#'   by expected information in the nuisance coordinates; \code{step.tol}
#'   controls the remaining cumulative-predictor change. Root tolerances use
#'   units of the coefficient's fitted standard error. Failed endpoints are
#'   missing with attached \code{diagnostics}; exhausting the search is not
#'   interpreted as an infinite endpoint. These are first-order model-based
#'   intervals, not exact small-sample or cluster-robust intervals.
#' @export
confint.fastglm_ordinal <- function(object,parm,level=.95,method=c("wald","profile"),control=list(),...) {
    if(length(list(...))) stop("unused ordinal confidence-interval arguments",call.=FALSE)
    method <- match.arg(method)
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie between zero and one",call.=FALSE)
    b <- coef(object)
    if(missing(parm)) parm <- seq_along(b)
    if(method=="profile") return(.ordinal_profile_confint(object,parm,level,control))
    if(length(control)) stop("'control' is only used for profile intervals",call.=FALSE)
    v <- vcov(object)
    if(is.character(parm)) parm <- match(parm,names(b))
    if(!is.numeric(parm) || anyNA(parm) || any(parm<1 | parm>length(b) | parm!=floor(parm)))
        stop("invalid ordinal parameter selection",call.=FALSE)
    q <- qnorm((1+level)/2)
    out <- b[parm]+outer(sqrt(diag(v))[parm],c(-q,q))
    dimnames(out) <- list(names(b)[parm],paste0(format(100*c((1-level)/2,(1+level)/2),trim=TRUE),"%"))
    attr(out,"method") <- "Wald with expected Fisher information"
    out
}

.ordinal_prediction_design <- function(object,newdata=NULL,offset=NULL) {
    if(inherits(object,"fastglm_ordinal_stream") && is.null(newdata))
        stop("supply explicit evaluation data for this ordinal source operation",call.=FALSE)
    if(!is.null(newdata) && isTRUE(object$prediction.offset.required) && is.null(offset))
        stop("supply an explicit prediction offset after changing prepared ordinal offsets",call.=FALSE)
    if(is.null(newdata)) { x <- object$x; off <- object$offset }
    else if(!is.null(object$terms)) {
        tt <- delete.response(object$terms)
        mf <- model.frame(tt,newdata,na.action=na.pass,xlev=object$xlevels)
        x <- model.matrix(tt,mf,contrasts.arg=object$contrasts)
        x <- x[,attr(x,"assign")!=0L,drop=FALSE]
        off <- model.offset(mf)
        if(is.null(offset) && !is.null(object$offset.expression)) {
            ext <- eval(object$offset.expression,newdata,environment(object$terms))
            off <- if(is.null(off)) ext else off+ext
        }
    } else { x <- newdata; off <- NULL }
    if(!is.matrix(x) && !inherits(x,"Matrix")) stop("ordinal prediction requires a numeric design matrix",call.=FALSE)
    if((is.matrix(x) && (!is.numeric(x) || is.complex(x))) ||
       (inherits(x,"Matrix") && !inherits(x,"dMatrix")))
        stop("ordinal prediction requires a real numeric design matrix",call.=FALSE)
    expected.columns <- if(inherits(object,"fastglm_ordinal_partial")) length(object$design.columns) else length(object$beta)
    if(ncol(x)!=expected.columns) stop("ordinal prediction design has the wrong number of columns",call.=FALSE)
    if(!is.null(colnames(x)) && length(object$design.columns)) {
        if(anyDuplicated(colnames(x)) || !setequal(colnames(x),object$design.columns))
            stop("ordinal prediction columns do not match the fitted design",call.=FALSE)
        if(!identical(colnames(x),object$design.columns))
            x <- x[,object$design.columns,drop=FALSE]
    }
    if(!is.null(offset)) off <- offset
    if(is.null(off)) off <- numeric(nrow(x))
    if(length(off)==1L) off <- rep(off,nrow(x))
    if(!is.numeric(off) || is.complex(off) || length(off)!=nrow(x))
        stop("invalid ordinal prediction offset",call.=FALSE)
    list(x=x,offset=off)
}

.ordinal_prediction_scale <- function(x,center,scale) {
    if(!ncol(x)) return(x)
    out <- matrix(NA_real_,nrow(x),ncol(x),dimnames=dimnames(x))
    for(j in seq_len(ncol(x))) out[,j] <- (x[,j]-center[j])/scale[j]
    out
}

.ordinal_probability_block <- function(object,x,offset,what="all",derivatives=TRUE,rows=NULL) {
    if(inherits(object,"fastglm_ordinal_partial"))
        return(.ordinal_partial_probability_block(object,x,offset,what,derivatives,rows))
    p <- length(object$beta); a <- object$working$alpha; q <- length(a)
    xx <- .ordinal_prediction_scale(x,object$working$center,object$working$scale)
    location <- streaming_qr_eta(xx,object$working$beta,offset)
    if(.ordinal_link(object)!="logit") {
        out <- .ordinal_link_kernel(a,location,.ordinal_link(object))
        out$x <- xx
        return(out)
    }
    n <- nrow(x); full <- what=="all"
    want.probability <- full || what=="probability"
    threshold.names <- if(!is.null(names(a))) list(NULL,names(a)) else NULL
    cumulative <- if(full || what=="cumulative") matrix(NA_real_,n,q,dimnames=threshold.names) else NULL
    survival <- if(full) matrix(NA_real_,n,q,dimnames=threshold.names) else NULL
    derivative <- if(derivatives) matrix(NA_real_,n,q,dimnames=threshold.names) else NULL
    probability <- if(want.probability) matrix(NA_real_,n,q+1L) else NULL
    softplus <- function(z) pmax(z,0)+log1p(exp(-abs(z)))
    previous.survival <- previous.eta <- NULL
    for(j in seq_len(q)) {
        eta <- a[j]-location
        lower <- plogis(eta)
        upper <- if(want.probability || full || derivatives) plogis(-eta) else NULL
        if(!is.null(cumulative)) cumulative[,j] <- lower
        if(full) survival[,j] <- upper
        if(derivatives) derivative[,j] <- lower*upper
        if(want.probability) {
            if(j==1L) probability[,1L] <- lower
            else {
                gap <- -expm1(a[j-1L]-a[j])
                value <- lower*previous.survival*gap
                # This exact product avoids subtracting cumulative probabilities.
                # R's logistic tails can become zero before exp() underflows;
                # preserve the log calculation for all subnormal probabilities.
                tiny <- which(value < .Machine$double.xmin)
                if(length(tiny)) value[tiny] <- exp(-softplus(-eta[tiny])+log(gap)-softplus(previous.eta[tiny]))
                probability[,j] <- value
            }
            if(j==q) probability[,q+1L] <- upper
            previous.survival <- upper; previous.eta <- eta
        }
    }
    list(x=if(full || derivatives) xx else NULL,probability=probability,
         cumulative=cumulative,survival=survival,derivative=derivative)
}

.ordinal_gradient <- function(parts,j,type) {
    if(!is.null(parts$partial)) return(.ordinal_partial_gradient(parts,j,type))
    q <- ncol(parts$derivative); p <- ncol(parts$x); n <- nrow(parts$x)
    g <- matrix(0,n,q+p)
    if(type=="cumulative") {
        d <- parts$derivative[,j]; g[,j] <- d
    } else {
        upper <- if(j<=q) parts$derivative[,j] else numeric(n)
        lower <- if(j>1L) parts$derivative[,j-1L] else numeric(n)
        d <- upper-lower
        if(j<=q) g[,j] <- upper
        if(j>1L) g[,j-1L] <- -lower
    }
    if(p) g[,q+seq_len(p)] <- -parts$x*d
    g
}

#' Predict ordinal probabilities or outcome categories
#' @param object a fitted \code{fastglm_ordinal} model.
#' @param newdata prediction data frame for a formula fit, or a matrix with the
#'   location columns used during matrix fitting. Defaults to the fitted design.
#' @param type \code{"probability"} (or \code{"response"}) returns category
#'   probabilities; \code{"cumulative"} returns probabilities through each
#'   category except the last; \code{"class"} returns the most probable ordered
#'   category; \code{"link"} returns the location predictor \code{x beta + offset}.
#' @param se.fit include model-based delta-method standard errors. They are
#'   unavailable for class predictions or unconverged fits.
#' @param offset optional replacement location offset.
#' @param ... further arguments; currently unused.
#' @details With collapsed empty interior categories, probability and cumulative
#'   uncertainty refer to the reduced category model. The removed category has
#'   fitted probability zero and unavailable probability standard error.
#' @returns A vector or matrix, or a \code{fastglm_prediction} list with
#'   \code{fit}, \code{se.fit}, and \code{covariance.scope}. See
#'   \link{fastglm-prediction-methods} for printing and table extraction.
#' @export
predict.fastglm_ordinal <- function(object,newdata=NULL,
        type=c("probability","cumulative","class","link","response"),
        se.fit=FALSE,offset=NULL,...) {
    if(length(list(...))) stop("unused ordinal prediction arguments",call.=FALSE)
    type <- match.arg(type)
    if(type=="response") type <- "probability"
    if(!is.logical(se.fit) || length(se.fit)!=1L || is.na(se.fit))
        stop("'se.fit' must be TRUE or FALSE",call.=FALSE)
    if(se.fit && type=="class") stop("class predictions do not have delta-method standard errors",call.=FALSE)
    if(se.fit) .ordinal_check_inference(object)
    design <- .ordinal_prediction_design(object,newdata,offset)
    n <- nrow(design$x); K <- length(object$levels); q <- length(object$effective$alpha)
    calc.type <- if(type=="class") "probability" else type
    labels <- if(calc.type=="cumulative") object$levels[-K] else object$levels
    fit <- if(type=="link") rep(NA_real_,n) else if(type=="class") rep(NA_integer_,n) else
        matrix(NA_real_,n,length(labels),dimnames=list(rownames(design$x),labels))
    se <- if(se.fit) fit else NULL
    probability.map <- match(seq_len(K),object$retained.categories)
    complete.categories <- length(object$retained.categories)==K
    .design_blocks(design$x,function(z,i) {
        off <- design$offset[i]
        if(!is.numeric(z) || is.complex(z)) stop("ordinal prediction requires a real numeric design",call.=FALSE)
        if(any(is.infinite(z)) || any(is.infinite(off))) stop("ordinal prediction inputs must be finite or missing",call.=FALSE)
        ok <- rowSums(!is.finite(z))==0L & is.finite(off)
        if(!any(ok)) return(invisible(NULL))
        rows <- i
        if(!all(ok)) {rows <- i[ok]; z <- z[ok,,drop=FALSE]; off <- off[ok]}
        if(type=="link") {
            fit[rows] <<- streaming_qr_eta(z,object$beta,off)
            if(se.fit) {
                g <- if(ncol(z)) sweep(z,2,object$working$scale,"/") else z
                v <- object$working$covariance[q+seq_len(ncol(z)),q+seq_len(ncol(z)),drop=FALSE]
                se[rows] <<- sqrt(pmax(0,.quad_rows(g,v)))
            }
            return(invisible(NULL))
        }
        parts <- .ordinal_probability_block(object,z,off,what=calc.type,derivatives=se.fit)
        if(type=="class") {
            fit[rows] <<- object$retained.categories[max.col(parts$probability,ties.method="first")]
            return(invisible(NULL))
        }
        if(calc.type=="probability") {
            if(complete.categories) value <- parts$probability else {
                value <- matrix(0,nrow(z),K)
                value[,object$retained.categories] <- parts$probability
            }
            map <- probability.map
        } else {
            map <- object$threshold.map
            value <- if(complete.categories) parts$cumulative else parts$cumulative[,map,drop=FALSE]
        }
        fit[rows,] <<- value
        if(se.fit) for(j in seq_along(map)) if(!is.na(map[j])) {
            g <- .ordinal_gradient(parts,map[j],calc.type)
            variance <- if(nrow(g)<=8192L) rowSums((g%*%object$working$covariance)*g) else
                .quad_rows(g,object$working$covariance)
            se[rows,j] <<- sqrt(pmax(0,variance))
        }
    },object$block.size)
    if(type=="class") {
        fit <- ordered(object$levels[fit],levels=object$levels)
    }
    if(type=="link" || type=="class") names(fit) <- rownames(design$x)
    if(is.null(newdata) && !is.null(object$na.action)) {
        fit <- napredict(object$na.action,fit)
        if(se.fit) se <- napredict(object$na.action,se)
    }
    if(se.fit) .fastglm_prediction(list(fit=fit,se.fit=se,
        covariance.scope=if(object$boundary) "reduced category model" else "expected Fisher information"),type) else fit
}

#' @export
fitted.fastglm_ordinal <- function(object,...) predict(object,...)

.ordinal_diagnostics <- function(object) {
    out <- list(converged=object$converged,convergence.reason=object$convergence.reason,
        iterations=object$iterations,evaluations=object$evaluations,fallbacks=object$fallbacks,
        rank=object$rank,boundary=object$boundary,
        collapsed.categories=object$levels[object$collapsed.categories],
        score=object$effective$score,ordinary.score=object$effective$ordinary.score,
        score.type=if(object$firth) "mean-bias-reduced" else "ordinary",
        score.coordinates="effective threshold and location parameters",
        standardized.score=object$scaled.score,remaining.predictor.step=object$predictor.step,
        information.rcond=object$information.rcond,
        minimum.log.probability=object$minimum.log.probability,
        last.rejected.evaluation=object$last.rejected.evaluation)
    if(inherits(object,"fastglm_ordinal_partial")) out$minimum.gap <- object$minimum.gap
    out
}

.ordinal_average <- function(object,newdata,contrast,weights,vcov.,level,type) {
    type <- match.arg(type,c("response","probability","cumulative"))
    if(type=="response") type <- "probability"
    .ordinal_check_inference(object)
    if(!is.null(contrast) && !is.function(contrast))
        stop("an ordinal contrast must be a function of the location design",call.=FALSE)
    design <- .ordinal_prediction_design(object,newdata)
    n <- nrow(design$x)
    if(!is.null(weights) && (!is.numeric(weights) || is.complex(weights) ||
       length(weights)!=n || any(!is.finite(weights)|weights<0)))
        stop("invalid averaging weights",call.=FALSE)
    total <- if(is.null(weights)) n else sum(weights)
    if(!is.finite(total) || total<=0) stop("averaging weights must have positive finite total",call.=FALSE)
    altered <- if(!is.null(contrast)) contrast(design$x)
    if(!is.null(contrast) && !identical(dim(altered),dim(design$x)))
        stop("contrast must preserve design dimensions",call.=FALSE)
    q <- length(object$effective$alpha); p <- length(object$beta)
    m <- if(type=="probability") q+1L else q
    estimate <- numeric(m); gradient <- matrix(0,nrow(object$working$covariance),m)
    .design_blocks(design$x,function(z,i) {
        w <- if(is.null(weights)) rep(1,length(i)) else weights[i]
        positive <- w>0
        if(!any(positive)) return(invisible(NULL))
        z <- z[positive,,drop=FALSE]; off <- design$offset[i[positive]]; w <- w[positive]
        if(!is.numeric(z) || is.complex(z) || any(!is.finite(z)) || any(!is.finite(off)))
            stop("ordinal averages require complete finite prediction rows with positive averaging weight",call.=FALSE)
        parts <- .ordinal_probability_block(object,z,off,rows=i[positive])
        value <- parts[[type]]
        if(!is.null(contrast)) {
            zz <- as.matrix(altered[i[positive],,drop=FALSE])
            if(!is.numeric(zz) || is.complex(zz) || any(!is.finite(zz)))
                stop("ordinal contrast must return finite numeric predictor values",call.=FALSE)
            alternative <- .ordinal_probability_block(object,zz,off,rows=i[positive])
            value <- alternative[[type]]-value
        }
        estimate <<- estimate+colSums(w*value)
        for(j in seq_len(m)) {
            g <- .ordinal_gradient(parts,j,type)
            if(!is.null(contrast)) g <- .ordinal_gradient(alternative,j,type)-g
            gradient[,j] <<- gradient[,j]+colSums(w*g)
        }
    },object$block.size)
    estimate <- estimate/total; gradient <- gradient/total
    .ordinal_average_summary(object,estimate,gradient,vcov.,level,type)
}

.ordinal_average_summary <- function(object,estimate,gradient,vcov.,level,type) {
    q <- length(object$effective$alpha); p <- length(object$beta)
    d <- nrow(object$working$covariance)
    if(is.null(vcov.)) covariance <- object$working$covariance
    else {
        if(object$boundary)
            stop("a supplied covariance is not supported for ordinal averages with collapsed categories",call.=FALSE)
        if(!is.matrix(vcov.) || !is.numeric(vcov.) || is.complex(vcov.) ||
           !identical(dim(vcov.),c(d,d)) || any(!is.finite(vcov.)) ||
           !isSymmetric(vcov.,tol=1e-8))
            stop("ordinal averages require a finite symmetric full coefficient covariance",call.=FALSE)
        cn <- names(coef(object))
        if(!is.null(colnames(vcov.))) {
            if(!setequal(colnames(vcov.),cn) || anyDuplicated(colnames(vcov.)) ||
               !setequal(rownames(vcov.),cn))
                stop("ordinal covariance names do not match the parameters",call.=FALSE)
            vcov. <- vcov.[cn,cn,drop=FALSE]
        }
        covariance <- vcov.
        if(inherits(object,"fastglm_ordinal_partial"))
            gradient <- crossprod(object$working$Ti,gradient)
        else if(p) gradient[q+seq_len(p),] <-
            gradient[q+seq_len(p),,drop=FALSE]*object$working$scale -
            outer(object$working$center,colSums(gradient[seq_len(q),,drop=FALSE]))
    }
    variances <- colSums(gradient*(covariance%*%gradient))
    if(any(variances < -1e-12*(1+max(abs(variances)))))
        stop("the supplied ordinal covariance gives negative prediction variance",call.=FALSE)
    se <- sqrt(pmax(0,variances)); K <- length(object$levels)
    if(type=="probability") {
        result <- data.frame(category=object$levels,estimate=0,std.error=NA_real_)
        result$estimate[object$retained.categories] <- estimate
        result$std.error[object$retained.categories] <- se
    } else result <- data.frame(category=object$levels[-K],
        estimate=estimate[object$threshold.map],std.error=se[object$threshold.map])
    quantile <- qnorm((1+level)/2)
    result$conf.low <- result$estimate-quantile*result$std.error
    result$conf.high <- result$estimate+quantile*result$std.error
    attr(result,"covariance.scope") <- if(object$boundary) "reduced category model" else "full model"
    result
}
