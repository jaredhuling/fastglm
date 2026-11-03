#' Fit a proportional-odds model for ordered outcomes
#'
#' Fits a cumulative-logit model with common slopes and freely estimated
#' thresholds, by maximum likelihood or mean bias reduction. Positive slopes
#' shift probability toward higher outcome categories.
#' @param x a formula or a finite dense numeric matrix of location predictors.
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
#'   `logit Pr(Y <= j) = alpha[j] - x beta - offset`.
#' @param levels category labels in increasing order. Numeric responses are
#'   interpreted as codes from one to the number of labels.
#' @param firth use mean bias reduction through adjusted score equations.
#' @param tol tolerance for the remaining change in cumulative linear
#'   predictors and the standardized adjusted score.
#' @param maxit maximum fitting iterations.
#' @param block.size maximum observations processed together by the numerical
#'   kernel. The input design is retained in memory.
#' @param control named optimizer controls: `score.tol`, `step.tol`,
#'   `max.step`, `max.halves`, `condition.tol`, `standardize`, and `trace`.
#' @param model retain the formula model frame.
#' @param ... additional arguments to the default or formula method.
#' @details Thresholds are estimated in increasing order. The same location
#'   slopes apply to all cumulative probabilities. Only the logit link and
#'   full-rank location designs are currently supported. Sparse matrices and
#'   repeatable sources are not accepted by this interface.
#'
#'   An empty interior category is removed from the numerical fit and then
#'   restored to the declared scale with zero fitted probability and coincident
#'   thresholds. Convergence and information refer to the reduced model.
#'   Standard errors for the tied thresholds are unavailable. Bias-reduced fits
#'   retain empty endpoint categories; maximum likelihood with empty endpoints
#'   requires infinite thresholds and is rejected.
#'
#'   The covariance uses inverse expected Fisher information at the fitted
#'   estimate. It gives a first-order model-based approximation; this interface
#'   does not provide ordinal penalized-profile or robust inference. Mean bias
#'   reduction solves the original threshold/common-slope adjusted equations.
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
                                    block.size=2048L,control=list(),...) {
    if(length(list(...))) stop("unused ordinal fitting arguments",call.=FALSE)
    if(missing(y)) stop("an ordinal response is required",call.=FALSE)
    if(!is.matrix(x) || !is.numeric(x))
        stop("ordinal fitting currently requires a dense numeric matrix or formula",call.=FALSE)
    if(is.factor(y) && !is.ordered(y) && is.null(levels))
        stop("use an ordered response factor or supply its category order in 'levels'",call.=FALSE)
    if(ncol(x)>0L && is.null(colnames(x))) colnames(x) <- paste0("x",seq_len(ncol(x)))
    if(anyNA(colnames(x)) || any(!nzchar(colnames(x))) || anyDuplicated(colnames(x)))
        stop("ordinal predictor column names must be distinct and nonmissing",call.=FALSE)
    fit <- .ordinal_fit_core(x,y,weights,offset,levels,firth,tol,maxit,block.size,control)
    q <- length(fit$alpha)
    names(fit$coefficients) <- c(paste0("threshold:",names(fit$alpha)),names(fit$beta))
    dimnames(fit$mapped.covariance) <- rep(list(names(fit$coefficients)),2L)
    fit$cov.unscaled <- fit$mapped.covariance
    fit$thresholds <- fit$alpha
    fit$tied.thresholds <- duplicated(fit$threshold.map) |
        duplicated(fit$threshold.map,fromLast=TRUE)
    fit$x <- x; fit$y <- y; fit$design.columns <- colnames(x)
    fit$nobs.rows <- fit$nobs; fit$nobs <- fit$frequency
    fit$call <- match.call()
    fit$call[[1L]] <- quote(fastglm::fastglm_ordinal)
    class(fit) <- "fastglm_ordinal"
    fit
}

#' @rdname fastglm_ordinal
#' @export
fastglm_ordinal.formula <- function(x,data,weights,offset,subset,na.action,
                                    contrasts=NULL,levels=NULL,firth=FALSE,
                                    tol=1e-8,maxit=200L,block.size=2048L,
                                    control=list(),model=TRUE,...) {
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
    design <- design[,attr(design,"assign")!=0L,drop=FALSE]
    fit <- fastglm_ordinal.default(design,model.response(mf),
        weights=model.weights(mf),offset=model.offset(mf),levels=levels,
        firth=firth,tol=tol,maxit=maxit,block.size=block.size,control=control)
    fit$terms <- terms
    fit$xlevels <- stats::.getXlevels(terms,mf)
    fit$contrasts <- coding
    fit$na.action <- attr(mf,"na.action")
    fit$offset.expression <- call$offset
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

#' Methods for proportional-odds fits
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
#' @export
vcov.fastglm_ordinal <- function(object,component=c("all","location","threshold"),...) {
    .ordinal_check_inference(object)
    v <- object$mapped.covariance
    tied <- which(object$tied.thresholds)
    if(length(tied)) { v[tied,] <- NA_real_; v[,tied] <- NA_real_ }
    i <- .ordinal_indices(object,match.arg(component))
    v <- v[i,i,drop=FALSE]
    attr(v,"covariance.scope") <- if(object$boundary) "reduced category model" else "expected Fisher information"
    v
}

#' @exportS3Method summary fastglm_ordinal
summary.fastglm_ordinal <- function(object,...) {
    b <- coef(object); se <- rep(NA_real_,length(b))
    if(isTRUE(object$converged)) se <- sqrt(diag(vcov(object)))
    z <- b/se
    tab <- cbind(Estimate=b,"Std. Error"=se,"z value"=z,"Pr(>|z|)"=2*pnorm(-abs(z)))
    q <- length(object$alpha)
    structure(list(call=object$call,location=tab[q+seq_along(object$beta),,drop=FALSE],
        thresholds=tab[seq_len(q),1:2,drop=FALSE],converged=object$converged,
        convergence.reason=object$convergence.reason,iterations=object$iterations,
        firth=object$firth,frequency=object$frequency,nobs.rows=object$nobs.rows,
        loglik=object$loglik,boundary=object$boundary,
        collapsed.categories=object$levels[object$collapsed.categories],
        predictor.step=object$predictor.step,scaled.score=object$scaled.score),
        class="summary.fastglm_ordinal")
}

#' @exportS3Method print summary.fastglm_ordinal
print.summary.fastglm_ordinal <- function(x,...) {
    print(x$call)
    cat("\nCumulative logit with common slopes; ",
        if(x$firth) "mean bias reduction" else "maximum likelihood","\n",sep="")
    if(nrow(x$location)) { cat("\nLocation coefficients (positive means higher outcomes):\n"); printCoefmat(x$location,...) }
    cat("\nThresholds:\n"); printCoefmat(x$thresholds,P.values=FALSE,has.Pvalue=FALSE,...)
    cat("\nFrequency total:",x$frequency,"  Positive-frequency rows:",x$nobs.rows,
        "\nConverged:",x$converged," (",x$convergence.reason,"); iterations:",x$iterations,"\n")
    if(x$boundary) cat("Empty interior categories:",paste(x$collapsed.categories,collapse=", "),
                       "\nTied-threshold standard errors are unavailable; inference uses the reduced category model.\n")
    invisible(x)
}

#' @exportS3Method print fastglm_ordinal
print.fastglm_ordinal <- function(x,...) {
    print(x$call)
    cat("\nLocation coefficients:\n"); print(x$beta,...)
    cat("\nThresholds:\n"); print(x$alpha,...)
    cat("\nConverged:",x$converged," (",x$convergence.reason,")\n")
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
#' @param level confidence level for asymptotic normal intervals.
#' @export
confint.fastglm_ordinal <- function(object,parm,level=.95,...) {
    if(length(list(...))) stop("only model-based Wald intervals are available for ordinal fits",call.=FALSE)
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie between zero and one",call.=FALSE)
    b <- coef(object); v <- vcov(object)
    if(missing(parm)) parm <- seq_along(b)
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
    if(ncol(x)!=length(object$beta)) stop("ordinal prediction design has the wrong number of columns",call.=FALSE)
    if(!is.null(colnames(x)) && length(object$design.columns)) {
        if(anyDuplicated(colnames(x)) || !setequal(colnames(x),object$design.columns))
            stop("ordinal prediction columns do not match the fitted design",call.=FALSE)
        x <- x[,object$design.columns,drop=FALSE]
    }
    if(!is.null(offset)) off <- offset
    if(is.null(off)) off <- numeric(nrow(x))
    if(length(off)==1L) off <- rep(off,nrow(x))
    if(!is.numeric(off) || is.complex(off) || length(off)!=nrow(x))
        stop("invalid ordinal prediction offset",call.=FALSE)
    list(x=x,offset=off)
}

.ordinal_probability_block <- function(object,x,offset) {
    p <- length(object$beta); a <- object$working$alpha; q <- length(a)
    xx <- if(p) sweep(sweep(x,2,object$working$center,"-"),2,object$working$scale,"/") else x
    location <- streaming_qr_eta(xx,object$working$beta,offset)
    eta <- outer(-location,a,"+")
    cumulative <- plogis(eta)
    derivative <- cumulative*plogis(-eta)
    probability <- matrix(0,nrow(x),q+1L)
    probability[,1L] <- cumulative[,1L]
    probability[,q+1L] <- plogis(-eta[,q])
    softplus <- function(z) pmax(z,0)+log1p(exp(-abs(z)))
    if(q>1L) for(j in 2:q) {
        loggap <- log(-expm1(a[j-1L]-a[j]))
        probability[,j] <- exp(-softplus(-eta[,j])+loggap-softplus(eta[,j-1L]))
    }
    list(x=xx,probability=probability,cumulative=cumulative,derivative=derivative)
}

.ordinal_gradient <- function(parts,j,type) {
    q <- ncol(parts$cumulative); p <- ncol(parts$x); n <- nrow(parts$x)
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
#' @returns A vector or matrix, or a list with \code{fit} and \code{se.fit}.
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
    fit <- se <- if(type=="link") rep(NA_real_,n) else matrix(NA_real_,n,length(labels),
        dimnames=list(rownames(design$x),labels))
    .design_blocks(design$x,function(z,i) {
        off <- design$offset[i]
        if(!is.numeric(z) || is.complex(z)) stop("ordinal prediction requires a real numeric design",call.=FALSE)
        if(any(is.infinite(z)) || any(is.infinite(off))) stop("ordinal prediction inputs must be finite or missing",call.=FALSE)
        ok <- rowSums(!is.finite(z))==0L & is.finite(off)
        if(!any(ok)) return(invisible(NULL))
        rows <- i[ok]; z <- z[ok,,drop=FALSE]; off <- off[ok]
        if(type=="link") {
            fit[rows] <<- streaming_qr_eta(z,object$beta,off)
            if(se.fit) {
                g <- if(ncol(z)) sweep(z,2,object$working$scale,"/") else z
                v <- object$working$covariance[q+seq_len(ncol(z)),q+seq_len(ncol(z)),drop=FALSE]
                se[rows] <<- sqrt(pmax(0,.quad_rows(g,v)))
            }
            return(invisible(NULL))
        }
        parts <- .ordinal_probability_block(object,z,off)
        if(calc.type=="probability") {
            value <- matrix(0,nrow(z),K)
            value[,object$retained.categories] <- parts$probability
            map <- match(seq_len(K),object$retained.categories)
        } else {
            map <- object$threshold.map
            value <- parts$cumulative[,map,drop=FALSE]
        }
        fit[rows,] <<- value
        if(se.fit) for(j in seq_along(map)) if(!is.na(map[j])) {
            g <- .ordinal_gradient(parts,map[j],calc.type)
            se[rows,j] <<- sqrt(pmax(0,.quad_rows(g,object$working$covariance)))
        }
    },object$block.size)
    if(type=="class") {
        code <- rep(NA_integer_,n)
        ok <- complete.cases(fit)
        if(any(ok)) code[ok] <- max.col(fit[ok,,drop=FALSE],ties.method="first")
        fit <- ordered(object$levels[code],levels=object$levels)
    }
    if(type=="link" || type=="class") names(fit) <- rownames(design$x)
    if(is.null(newdata) && !is.null(object$na.action)) {
        fit <- napredict(object$na.action,fit)
        if(se.fit) se <- napredict(object$na.action,se)
    }
    if(se.fit) list(fit=fit,se.fit=se,
        covariance.scope=if(object$boundary) "reduced category model" else "expected Fisher information") else fit
}

#' @export
fitted.fastglm_ordinal <- function(object,...) predict(object,...)

.ordinal_diagnostics <- function(object) {
    list(converged=object$converged,convergence.reason=object$convergence.reason,
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
    estimate <- numeric(m); gradient <- matrix(0,q+p,m)
    .design_blocks(design$x,function(z,i) {
        w <- if(is.null(weights)) rep(1,length(i)) else weights[i]
        positive <- w>0
        if(!any(positive)) return(invisible(NULL))
        z <- z[positive,,drop=FALSE]; off <- design$offset[i[positive]]; w <- w[positive]
        if(!is.numeric(z) || is.complex(z) || any(!is.finite(z)) || any(!is.finite(off)))
            stop("ordinal averages require complete finite prediction rows with positive averaging weight",call.=FALSE)
        parts <- .ordinal_probability_block(object,z,off)
        value <- parts[[type]]
        if(!is.null(contrast)) {
            zz <- as.matrix(altered[i[positive],,drop=FALSE])
            if(!is.numeric(zz) || is.complex(zz) || any(!is.finite(zz)))
                stop("ordinal contrast must return finite numeric predictor values",call.=FALSE)
            alternative <- .ordinal_probability_block(object,zz,off)
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
    if(is.null(vcov.)) covariance <- object$working$covariance
    else {
        if(object$boundary)
            stop("a supplied covariance is not supported for ordinal averages with collapsed categories",call.=FALSE)
        if(!is.matrix(vcov.) || !is.numeric(vcov.) || is.complex(vcov.) ||
           !identical(dim(vcov.),c(q+p,q+p)) || any(!is.finite(vcov.)) ||
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
        if(p) gradient[q+seq_len(p),] <-
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
