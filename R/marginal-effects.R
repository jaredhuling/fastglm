#' Average marginal effects
#'
#' Estimates the average rate of change of a prediction as a continuous
#' predictor changes: the fitted mean of a generalized linear model, the mean,
#' count-component mean, or zero probability of a count or two-part model, or
#' each category probability of an ordinal or multinomial model. Uncertainty uses the delta method, conditional
#' on the supplied predictor values and averaging weights.
#' @param object a converged GLM (\code{fastglm()}), negative binomial,
#'   zero-truncated, hurdle, zero-inflated, two-part, ordinal, or multinomial fit.
#' @param variable name of one continuous numeric variable. For a formula fit,
#'   this names the original variable in the data; interactions and smooth
#'   transformations are differentiated together. For a matrix fit, this names
#'   one design column, with the other columns held fixed; for two-component
#'   count models the column is changed in both component designs where present.
#' @param newdata data frame for a formula fit, or location-design matrix for a
#'   matrix fit. Defaults to retained fitting data. Supply the original variables
#'   explicitly when they cannot be recovered from the retained model frame.
#' @param weights nonnegative averaging weights, defaulting to equal weights.
#'   Fitting frequency weights are not reused automatically.
#' @param vcov. optional full coefficient covariance matrix, as in
#'   \code{fastglm_average()}.
#' @param level confidence level for separate normal intervals.
#' @param type prediction target. For GLMs, \code{"response"} (the mean).
#'   For count and two-part models, \code{"response"}, \code{"count"}, or
#'   \code{"zero"}, as in \code{fastglm_average()}. For ordinal models, category
#'   probabilities (\code{"response"} or \code{"probability"}) or
#'   \code{"cumulative"} probabilities. For multinomial models,
#'   \code{"response"} or \code{"probability"}.
#' @param eps positive absolute step for numerical differentiation of formula
#'   terms. The default is \code{.Machine$double.eps^(1/3) * pmax(1, abs(x))}
#'   for the selected variable. For ordinal matrix fits the derivative is exact
#'   and does not use this argument; GLM and count models use centered
#'   differences for matrix columns as well.
#' @param offset optional replacement prediction offset (GLM, ordinal, and multinomial fits).
#'   All offsets are held fixed when differentiating, including formula offsets
#'   involving \code{variable}.
#' @details A marginal effect is a derivative, measured in probability units
#'   per unit of the continuous predictor. It describes a local rate of change;
#'   a finite change can give a different probability difference. Use
#'   \code{fastglm_average(contrast = ...)} for discrete comparisons, including
#'   factors. A numeric binary indicator is treated as continuous if requested.
#'
#'   Formula design derivatives use centered differences, with the fitted
#'   prediction basis and factor coding retained. Terms must be smooth functions
#'   of each row's predictors, or use a prediction basis frozen during fitting
#'   (for example, \code{poly()} or \code{splines::ns()}). Expressions that
#'   recompute sample summaries such as \code{I(x - mean(x))} are unsupported.
#'   Rows and perturbed values with positive averaging weight must be finite
#'   and within the domain of the formula transformations. A smaller \code{eps}
#'   may be needed near a transformation's domain boundary.
#'
#'   Derivatives of probabilities and their parameter gradients are analytic
#'   conditional on the computed design derivative. The gradient includes
#'   uncertainty in both thresholds and slopes. Computation uses blocks of rows
#'   and retains the base prediction design, without storing a full array of
#'   observation-by-parameter-by-category derivatives. Normal intervals are
#'   first-order approximations; the covariance and reduced-category conventions
#'   match \code{fastglm_average()}.
#'   For nonparallel models, each cumulative probability uses its own slope.
#'   Evaluation rows must lie within the fitted model's valid predictor domain.
#'   Supplied covariance, including cluster covariance, retains covariance among
#'   thresholds, common slopes, and threshold-specific slopes.
#'
#'   For GLM and count models, each row's derivative is the centered difference
#'   of its prediction at the variable plus and minus the step, with every other
#'   predictor and every offset held fixed, and the standard error uses the
#'   parameter gradient of that same difference with the model covariance
#'   (including theta for count models when it is estimated). Streaming fits
#'   require \code{newdata}.
#'
#'   For multinomial fits, each row's derivative of every category probability
#'   is the centered difference of its probabilities at the variable plus and
#'   minus the step, with other predictors and all offsets held fixed. The
#'   effects sum to zero across categories, and the standard error of each
#'   category uses the gradient of that difference with the covariance of all
#'   coefficients. A formula fit without \code{newdata} recovers its fitting data
#'   from the call; a source fit requires \code{newdata}.
#' @returns A data frame with \code{variable}, \code{estimate}, \code{std.error},
#'   \code{conf.low}, and \code{conf.high}, and for ordinal models also
#'   \code{category}, with one row per category (also for multinomial models, whose
#'   category effects sum to zero). Ordinal category effects
#'   sum to zero. Removed categories have zero effects and unavailable standard
#'   errors; inference otherwise refers to the reduced category model.
#' @examples
#' set.seed(91)
#' d <- data.frame(x=rnorm(150))
#' d$y <- ordered(cut(.5*d$x + rlogis(150), c(-Inf,-.5,.7,Inf)))
#' fit <- fastglm_ordinal(y ~ x, data=d, firth=TRUE)
#' fastglm_marginal_effects(fit, "x")
#'
#' # Logistic regression: average change in probability per unit of x
#' d$z <- rbinom(150, 1, plogis(-0.5 + 0.8 * d$x))
#' fastglm_marginal_effects(fastglm(z ~ x, data = d, family = binomial()), "x")
#' @export
fastglm_marginal_effects <- function(object,variable,newdata=NULL,weights=NULL,
                                     vcov.=NULL,level=.95,type="response",
                                     eps=NULL,offset=NULL) {
    if(inherits(object,"fastglm_vglm"))
        return(.vglm_marginal_effects(object,variable,newdata,weights,vcov.,level,type,eps,offset))
    if(!inherits(object,"fastglm_ordinal"))
        return(.response_marginal_effects(object,variable,newdata,weights,vcov.,level,type,eps,offset))
    .ordinal_check_inference(object)
    if(!is.character(variable) || length(variable)!=1L || is.na(variable) || !nzchar(variable))
        stop("'variable' must name one continuous numeric predictor",call.=FALSE)
    if(!is.numeric(level) || is.complex(level) || length(level)!=1L ||
       !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie strictly between zero and one",call.=FALSE)
    if(!is.null(eps) && (!is.numeric(eps) || is.complex(eps) || length(eps)!=1L ||
       !is.finite(eps) || eps<=0)) stop("'eps' must be a positive finite number",call.=FALSE)
    type <- match.arg(type,c("response","probability","cumulative"))
    if(type=="response") type <- "probability"
    formula.fit <- !is.null(object$terms)
    data <- if(is.null(newdata)) if(formula.fit) object$model else object$x else newdata
    if(formula.fit) {
        needed <- all.vars(delete.response(object$terms))
        if(!is.data.frame(data) || !all(needed%in%names(data)))
            stop("supply 'newdata' containing the original formula predictors",call.=FALSE)
        x <- data[[variable]]
        if(!variable%in%needed || !is.numeric(x) || is.complex(x) ||
           !is.null(dim(x)) || is.object(x))
            stop("'variable' must name an unclassed continuous numeric formula predictor; use contrasts for factors",call.=FALSE)
        # Reject common non-rowwise terms before block evaluation. Arbitrary
        # user functions still require the documented smooth rowwise contract.
        reducers <- c("mean","median","sd","var","cov","sum","prod","min","max",
                      "range","quantile","rank","order","sort","cumsum","cumprod",
                      "cummin","cummax")
        check <- function(expr) {
            if(!is.call(expr)) return(invisible(NULL))
            head <- expr[[1L]]
            name <- if(is.symbol(head)) as.character(head) else if(is.call(head) &&
                as.character(head[[1L]])%in%c("::",":::")) as.character(head[[3L]]) else ""
            if(name%in%reducers && any(all.vars(expr)%in%needed))
                stop("marginal effects require rowwise formula terms; compute sample summaries before fitting",call.=FALSE)
            for(arg in as.list(expr)[-1L]) check(arg)
            invisible(NULL)
        }
        check(attr(delete.response(object$terms),"predvars"))
    }
    design <- .ordinal_prediction_design(object,newdata,offset)
    if(!formula.fit && !variable%in%object$design.columns)
        stop("'variable' must name a fitted location-design column",call.=FALSE)
    n <- nrow(design$x)
    if(!is.null(weights) && (!is.numeric(weights) || is.complex(weights) ||
       !is.null(dim(weights)) || length(weights)!=n || any(!is.finite(weights)|weights<0)))
        stop("invalid averaging weights",call.=FALSE)
    total <- if(is.null(weights)) n else sum(weights)
    if(!is.finite(total) || total<=0) stop("averaging weights must have positive finite total",call.=FALSE)
    partial <- inherits(object,"fastglm_ordinal_partial")
    q <- length(object$working$alpha);p <- length(object$design.columns)
    d <- length(object$working$coefficients)
    estimate <- numeric(q);gradient <- matrix(0,d,q)
    .design_blocks(design$x,function(z,i) {
        w <- if(is.null(weights)) rep(1,length(i)) else weights[i]
        keep <- w>0; i <- i[keep]; w <- w[keep]/total; z <- z[keep,,drop=FALSE]
        if(!length(i)) return(invisible(NULL))
        off <- design$offset[i]
        if(any(!is.finite(z)) || any(!is.finite(off)))
            stop("marginal effects require complete finite rows with positive averaging weight",call.=FALSE)
        if(formula.fit) {
            upper <- lower <- data[i,,drop=FALSE]
            h <- if(is.null(eps)) .Machine$double.eps^(1/3)*pmax(1,abs(x[i])) else eps
            upper[[variable]] <- x[i]+h; lower[[variable]] <- x[i]-h
            width <- upper[[variable]]-lower[[variable]]
            if(any(!is.finite(width) | width<=0))
                stop("'eps' cannot produce finite distinct predictor values",call.=FALSE)
            high <- .ordinal_prediction_design(object,upper,off)$x
            low <- .ordinal_prediction_design(object,lower,off)$x
            dx <- (high-low)/width
            if(any(!is.finite(dx)))
                stop("formula derivatives must be finite; check transformation domains and 'eps'",call.=FALSE)
        } else {
            dx <- matrix(0,length(i),p)
            dx[,match(variable,object$design.columns)] <- 1
        }
        dx <- sweep(dx,2L,object$working$scale,"/")
        parts <- .ordinal_probability_block(object,z,off,what="all",rows=i)
        if(!partial) slope <- drop(dx%*%object$working$beta)
        for(j in seq_len(q)) {
            if(partial) {
                slopes <- numeric(p);slopes[object$common] <- object$working$beta
                slopes[object$nonparallel] <- object$working$gamma[,j]
                slope <- streaming_qr_eta(dx,slopes,numeric(length(i)))
            }
            density <- parts$derivative[,j]
            curvature <- if(is.null(parts$curvature)) density*(parts$survival[,j]-parts$cumulative[,j]) else parts$curvature[,j]
            value <- -density*slope
            g <- matrix(0,length(i),d)
            g[,j] <- -curvature*slope
            if(partial) {
                pc <- length(object$common);pn <- length(object$nonparallel)
                if(pc) g[,q+seq_len(pc)] <- parts$x[,object$common,drop=FALSE]*(curvature*slope)-
                    dx[,object$common,drop=FALSE]*density
                at <- q+pc+(j-1L)*pn+seq_len(pn)
                g[,at] <- parts$x[,object$nonparallel,drop=FALSE]*(curvature*slope)-
                    dx[,object$nonparallel,drop=FALSE]*density
            } else if(p) g[,q+seq_len(p)] <- parts$x*(curvature*slope)-dx*density
            if(any(!is.finite(value)) || any(!is.finite(g)))
                stop("marginal effect or parameter derivative is nonfinite",call.=FALSE)
            estimate[j] <<- estimate[j]+sum(w*value)
            gradient[,j] <<- gradient[,j]+colSums(w*g)
        }
    },object$block.size)
    if(type=="probability") {
        estimate <- diff(c(0,estimate,0))
        gradient <- cbind(gradient,0)-cbind(0,gradient)
    }
    out <- .ordinal_average_summary(object,estimate,gradient,vcov.,level,type)
    out <- cbind(variable=variable,out)
    attr(out,"covariance.scope") <- if(object$boundary) "reduced category model" else "full model"
    out
}

# Formula predictors that a marginal effect can differentiate: an unclassed
# numeric variable entering only through rowwise terms.
.marginal_formula_variable <- function(terms_list,data,variable) {
    needed <- unique(unlist(lapply(terms_list,function(tt) all.vars(delete.response(tt)))))
    if(!variable%in%needed)
        stop("'variable' must name an unclassed continuous numeric formula predictor; use contrasts for factors",call.=FALSE)
    if(!is.data.frame(data) || !all(needed%in%names(data)))
        stop("supply 'newdata' containing the original formula predictors",call.=FALSE)
    x <- data[[variable]]
    if(!variable%in%needed || !is.numeric(x) || is.complex(x) || !is.null(dim(x)) || is.object(x))
        stop("'variable' must name an unclassed continuous numeric formula predictor; use contrasts for factors",call.=FALSE)
    reducers <- c("mean","median","sd","var","cov","sum","prod","min","max","range","quantile",
                  "rank","order","sort","cumsum","cumprod","cummin","cummax")
    check <- function(expr) {
        if(!is.call(expr)) return(invisible(NULL))
        head <- expr[[1L]]
        name <- if(is.symbol(head)) as.character(head) else if(is.call(head) &&
            as.character(head[[1L]])%in%c("::",":::")) as.character(head[[3L]]) else ""
        if(name%in%reducers && any(all.vars(expr)%in%needed))
            stop("marginal effects require rowwise formula terms; compute sample summaries before fitting",call.=FALSE)
        for(arg in as.list(expr)[-1L]) check(arg)
        invisible(NULL)
    }
    for(tt in terms_list) check(attr(delete.response(tt),"predvars"))
    x
}

# Average marginal effects for GLM, count, and two-part fits by centered
# differences of each row's prediction; see fastglm_marginal_effects().
.response_marginal_effects <- function(object,variable,newdata,weights,vcov.,level,type,eps,offset,
                                       block.size=8192L) {
    if(!is.character(variable) || length(variable)!=1L || is.na(variable) || !nzchar(variable))
        stop("'variable' must name one continuous numeric predictor",call.=FALSE)
    if(!is.numeric(level) || length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie strictly between zero and one",call.=FALSE)
    if(!is.null(eps) && (!is.numeric(eps) || length(eps)!=1L || !is.finite(eps) || eps<=0))
        stop("'eps' must be a positive finite number",call.=FALSE)
    if(!isTRUE(object$converged)) stop("marginal effects require a converged model",call.=FALSE)
    count <- .count_inference_model(object)
    two <- count && .count_two_component(object)
    if(!count && !inherits(object,c("fastglm","fastglmFit","fastglm_stream")))
        stop("marginal effects support GLM, count, two-part, and ordinal fits",call.=FALSE)
    if(count) {
        type <- match.arg(type,c("response","count","zero"))
        if(!two && type=="zero") stop("zero probabilities are a target only for two-component models",call.=FALSE)
        if(!is.null(offset)) stop("replacement offsets are not supported for count models; include them in newdata",call.=FALSE)
        pars <- .count_parameters(object,FALSE)
        if(is.null(vcov.)) vcov. <- .count_default_vcov(object)
    } else {
        if(!identical(type,"response")) stop("GLM marginal effects use type = 'response'",call.=FALSE)
        if(anyNA(coef(object))) stop("marginal effects require a full-rank fit",call.=FALSE)
        pars <- coef(object)
        if(is.null(vcov.)) vcov. <- vcov(object)
    }
    if(!identical(dim(vcov.),c(length(pars),length(pars))))
        stop("'vcov.' must be the full covariance of the model parameters, including theta when estimated",call.=FALSE)
    streaming <- !is.null(object$source) && is.function(object$fetch)
    if(streaming && is.null(newdata)) stop("marginal effects for a streaming fit require 'newdata'",call.=FALSE)
    terms_list <- if(is.null(object$terms)) NULL else if(inherits(object$terms,"terms")) list(object$terms) else object$terms
    formula.fit <- !is.null(terms_list) && (is.null(newdata) || is.data.frame(newdata))
    # Design, offsets, and the variable for a set of data rows.
    design_of <- function(dat) {
        if(two) {
            d <- .count_design(object,dat)
            return(list(X=d$x$count,Z=d$x$zero,offset=d$offset))
        }
        d <- if(count) .prediction_design(object,dat) else .prediction_design(object,dat,offset)
        list(X=d$x,offset=d$offset)
    }
    if(formula.fit) {
        data <- if(is.null(newdata)) object$model else newdata
        x <- .marginal_formula_variable(terms_list,data,variable)
        n <- nrow(data)
    } else {
        base <- if(is.null(newdata)) {
            if(two) { d <- .count_design(object); list(X=d$x$count,Z=d$x$zero,offset=d$offset) }
            else { d <- .prediction_design(object); list(X=d$x,offset=d$offset) }
        } else design_of(newdata)
        cols <- c(colnames(base$X),if(two) colnames(base$Z))
        if(!variable%in%cols) stop("'variable' must name a fitted design column",call.=FALSE)
        x <- if(variable%in%colnames(base$X)) base$X[,variable] else base$Z[,variable]
        n <- nrow(base$X)
    }
    if(!is.null(weights) && (!is.numeric(weights) || length(weights)!=n || any(!is.finite(weights)|weights<0)))
        stop("invalid averaging weights",call.=FALSE)
    total <- if(is.null(weights)) n else sum(weights)
    if(!is.finite(total) || total<=0) stop("averaging weights must have positive finite total",call.=FALSE)
    predict_at <- function(z) {
        if(count) {
            zz <- .count_normalize_block(list(X=z$X,Z=z$Z,offset=z$offset),two)
            list(value=.count_prediction_parameters(object,zz,pars,type),
                 gradient=.count_prediction_gradient(object,zz,pars,type))
        } else {
            eta <- .model_eta(z$X,object,z$offset)
            list(value=object$family$linkinv(eta),gradient=as.matrix(z$X)*.glm_mu_eta(object,eta))
        }
    }
    estimate <- 0; gradient <- numeric(length(pars))
    for(start in seq(1L,n,by=block.size)) {
        i <- start:min(n,start+block.size-1L)
        w <- if(is.null(weights)) rep(1,length(i)) else weights[i]
        keep <- w>0; i <- i[keep]; w <- w[keep]
        if(!length(i)) next
        h <- if(is.null(eps)) .Machine$double.eps^(1/3)*pmax(1,abs(x[i])) else rep(eps,length(i))
        if(formula.fit) {
            rows <- data[i,,drop=FALSE]
            fixed <- design_of(rows)$offset
            upper <- lower <- rows
            upper[[variable]] <- x[i]+h; lower[[variable]] <- x[i]-h
            width <- upper[[variable]]-lower[[variable]]
            hi <- design_of(upper); lo <- design_of(lower)
            hi$offset <- lo$offset <- fixed
        } else {
            sub <- function(m) if(is.null(m)) NULL else m[i,,drop=FALSE]
            rows <- list(X=sub(base$X),Z=sub(base$Z),
                offset=if(is.list(base$offset)) lapply(base$offset,`[`,i) else base$offset[i])
            shift <- function(z,delta) {
                if(variable%in%colnames(z$X)) z$X[,variable] <- z$X[,variable]+delta
                if(!is.null(z$Z) && variable%in%colnames(z$Z)) z$Z[,variable] <- z$Z[,variable]+delta
                z
            }
            hi <- shift(rows,h); lo <- shift(rows,-h)
            width <- (x[i]+h)-(x[i]-h)
        }
        if(any(!is.finite(width) | width<=0)) stop("'eps' cannot produce finite distinct predictor values",call.=FALSE)
        a <- predict_at(hi); b <- predict_at(lo)
        value <- (a$value-b$value)/width
        g <- (a$gradient-b$gradient)/width
        if(any(!is.finite(value)) || any(!is.finite(g)))
            stop("marginal effect or parameter derivative is nonfinite; check transformation domains and 'eps'",call.=FALSE)
        estimate <- estimate+sum(w*value); gradient <- gradient+colSums(w*g)
    }
    estimate <- estimate/total; gradient <- gradient/total
    active <- which(gradient!=0)
    variance <- if(length(active)) drop(crossprod(gradient[active],vcov.[active,active,drop=FALSE]%*%gradient[active])) else 0
    if(!is.finite(variance) || variance<0) stop("invalid covariance for the requested marginal effect",call.=FALSE)
    se <- sqrt(variance); q <- stats::qnorm((1+level)/2)
    data.frame(variable=variable,estimate=estimate,std.error=se,conf.low=estimate-q*se,conf.high=estimate+q*se)
}
