#' Average marginal effects for ordinal probabilities
#'
#' Estimates the average rate of change in each category probability as a
#' continuous predictor changes. Uncertainty uses the delta method, conditional
#' on the supplied predictor values and averaging weights.
#' @param object a converged \code{fastglm_ordinal} fit, with common or
#'   threshold-specific slopes.
#' @param variable name of one continuous numeric variable. For a formula fit,
#'   this names the original variable in the data; interactions and smooth
#'   transformations are differentiated together. For a matrix fit, this names
#'   one location-design column, with the other columns held fixed.
#' @param newdata data frame for a formula fit, or location-design matrix for a
#'   matrix fit. Defaults to retained fitting data. Supply the original variables
#'   explicitly when they cannot be recovered from the retained model frame.
#' @param weights nonnegative averaging weights, defaulting to equal weights.
#'   Fitting frequency weights are not reused automatically.
#' @param vcov. optional full coefficient covariance matrix, as in
#'   \code{fastglm_average()}.
#' @param level confidence level for separate normal intervals.
#' @param type category probabilities (\code{"response"} or
#'   \code{"probability"}) or \code{"cumulative"} probabilities.
#' @param eps positive absolute step for numerical differentiation of formula
#'   terms. The default is \code{.Machine$double.eps^(1/3) * pmax(1, abs(x))}
#'   for the selected variable. Matrix-column derivatives are exact and do not
#'   use this argument.
#' @param offset optional replacement prediction offset. All offsets are held
#'   fixed when differentiating, including formula offsets involving
#'   \code{variable}.
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
#' @returns A data frame with \code{variable}, \code{category}, \code{estimate},
#'   \code{std.error}, \code{conf.low}, and \code{conf.high}. Category effects
#'   sum to zero. Removed categories have zero effects and unavailable standard
#'   errors; inference otherwise refers to the reduced category model.
#' @examples
#' set.seed(91)
#' d <- data.frame(x=rnorm(150))
#' d$y <- ordered(cut(.5*d$x + rlogis(150), c(-Inf,-.5,.7,Inf)))
#' fit <- fastglm_ordinal(y ~ x, data=d, firth=TRUE)
#' fastglm_marginal_effects(fit, "x")
#' @export
fastglm_marginal_effects <- function(object,variable,newdata=NULL,weights=NULL,
                                     vcov.=NULL,level=.95,type="response",
                                     eps=NULL,offset=NULL) {
    if(!inherits(object,"fastglm_ordinal"))
        stop("marginal effects currently require a fastglm_ordinal fit",call.=FALSE)
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
