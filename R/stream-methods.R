#' Stream predictions or diagnostics to a callback
#'
#' Prediction for a large dataset can create a vector too large to retain in
#' memory. A sink is a user-supplied function that receives one block of results
#' at a time, for example to write predictions to a file or accumulate a summary.
#' Without a sink, the result blocks are collected into a single R object, whose
#' memory requirement grows with the number of observations.
#' @param object a streaming GLM fit.
#' @param newdata optional prediction data; otherwise rereads the fitted source.
#' @param type,dispersion,se.fit prediction arguments as in [predict.fastglm()].
#' @param sink optional function of `(value, chunk_index)` that processes or
#'   writes each result block. Its return value is ignored.
#' @param ... additional prediction arguments.
#' @returns Predictions (a `fastglm_prediction` list when `se.fit=TRUE`), or
#'   invisibly the number of chunks delivered to a sink. Each prediction block
#'   supplied to a sink uses the same structure. See \link{fastglm-prediction-methods}.
#' @export
predict.fastglm_stream <- function(object,newdata=NULL,type=c("link","response"),
                                   se.fit=FALSE,dispersion=NULL,sink=NULL,...) {
    type <- match.arg(type)
    if(!is.null(newdata)) return(predict.fastglm(object,newdata,type,se.fit,dispersion,...))
    if(is.null(dispersion)) dispersion <- object$dispersion
    if(!is.null(sink) && !is.function(sink)) stop("'sink' must be a function",call.=FALSE)
    n <- object$source$n_chunks; out <- if(is.null(sink)) vector("list",n)
    for(k in seq_len(n)) {
        z <- object$fetch(k); eta <- .model_eta(z$X,object,z$offset)
        value <- if(type=="response") object$family$linkinv(eta) else eta
        if(se.fit) {
            se <- sqrt(pmax(0,dispersion*.model_quad_rows(z$X,object)))
            if(type=="response") se <- se*abs(.glm_mu_eta(object,eta))
            value <- .fastglm_prediction(list(fit=value,se.fit=se,residual.scale=sqrt(dispersion)),type)
        }
        if(is.null(sink)) out[[k]] <- value else sink(value,k)
    }
    if(!is.null(sink)) return(invisible(n))
    if(se.fit) .fastglm_prediction(list(fit=unlist(lapply(out,`[[`,"fit"),use.names=FALSE),
                    se.fit=unlist(lapply(out,`[[`,"se.fit"),use.names=FALSE),residual.scale=sqrt(dispersion)),type)
    else unlist(out,use.names=FALSE)
}

#' @export
residuals.fastglm_stream <- function(object,type=c("deviance","pearson","working","response"),sink=NULL,...) {
    type <- match.arg(type); n <- object$source$n_chunks
    out <- if(is.null(sink)) vector("list",n)
    for(k in seq_len(n)) {
        z <- object$fetch(k); eta <- .model_eta(z$X,object,z$offset)
        mu <- object$family$linkinv(eta); r <- z$y-mu
        r <- switch(type,response=r,working=r/object$family$mu.eta(eta),
            pearson=r*sqrt(z$weights/object$family$variance(mu)),
            deviance=sign(r)*sqrt(pmax(0,object$family$dev.resids(z$y,mu,z$weights))))
        code <- .stable_binomial_code(object)
        if(code>=0L) {
            moments <- binomial_firth_moments(as.numeric(eta),as.numeric(z$y),z$weights,code)
            if(type=="working") {
                r <- .binomial_working_residual(moments,z$y,eta,code)
                r[z$weights==0] <- 0
            } else r <- .binomial_tail_residual(moments,z$y,z$weights,type)
        }
        if(is.null(sink)) out[[k]] <- r else sink(r,k)
    }
    if(is.null(sink)) unlist(out,use.names=FALSE) else invisible(n)
}

#' @export
logLik.fastglm_stream <- function(object,...) {
    df <- object$rank+as.integer(is.na(.fixed_dispersion(object$family)))
    structure((2*df-object$aic)/2,df=df,nobs=object$nobs,class="logLik")
}

#' @export
hatvalues.fastglm_stream <- function(model,sink=NULL,...) {
    out <- if(is.null(sink)) vector("list",model$source$n_chunks)
    for(k in seq_len(model$source$n_chunks)) {
        z <- model$fetch(k); eta <- .model_eta(z$X,model,z$offset); mu <- model$family$linkinv(eta)
        w <- z$weights*model$family$mu.eta(eta)^2/model$family$variance(mu)
        code <- .stable_binomial_code(model)
        if(code>=0L) w <- binomial_firth_moments(as.numeric(eta),as.numeric(z$y),z$weights,code)$working
        h <- w*.model_quad_rows(z$X,model)
        if(is.null(sink)) out[[k]] <- h else sink(h,k)
    }
    if(is.null(sink)) unlist(out,use.names=FALSE) else invisible(model$source$n_chunks)
}

#' Robust covariance with bounded design memory
#'
#' Computes the sandwich meat in row blocks, optionally accumulating scores
#' by cluster. Cluster storage is proportional to the number of distinct
#' clusters times the number of coefficients. Streaming fits reread the source.
#' @param object a fitted fastglm GLM, count, two-part, or ordinal model.
#' @param type HC0, HC1, HC2, or HC3 correction. Cluster covariance, count,
#'   two-part, and ordinal models support HC0 and HC1. The default is HC3 for
#'   GLMs and HC0 for count, two-part, and ordinal models.
#' @param cluster optional cluster identifiers; for streaming fits, a function
#'   of chunk index returning identifiers in the source row order. For ordinal
#'   fits, supply one identifier per retained row, or per row before missing-value
#'   omission after any formula subset. Each represented individual in a grouped
#'   frequency row belongs to that row's cluster; split the row if its members
#'   belong to different clusters.
#' @param cadjust multiply clustered covariance by `G/(G-1)`.
#' @param block.size maximum number of design rows processed at once.
#' @param full include the estimated negative-binomial size parameter when
#'   applicable. Its covariance with the regression coefficients is included.
#' @details For count and two-part models, the sandwich covariance uses joint
#'   estimating scores and the observed derivative of their sum. Cluster scores
#'   include both model components, so within-cluster dependence can produce
#'   nonzero covariance between them. The Gamma mean equation does not depend
#'   on dispersion; removing its common dispersion multiplier preserves the
#'   fitted mean parameters and their sandwich covariance. These approximations
#'   require independent observations, or independent clusters when supplied,
#'   and identifiable interior parameters. For GLM, count, and two-part models,
#'   sandwich covariance for bias-adjusted scores is not implemented.
#'
#'   Ordinal robust covariance currently requires a retained common-slope fit.
#'   For ordinal fits, the covariance uses the inverse observed negative
#'   Jacobian of the ordinary likelihood score, rather than inverse expected
#'   Fisher information. Independent-observation meat sums individual score
#'   outer products linearly in the effective category frequencies. Cluster meat
#'   instead sums individual scores within each cluster before taking outer
#'   products. Thus a frequency row represents repeated independent individuals
#'   unless a cluster identifier groups them together. With total frequency
#'   \eqn{N}, parameter count \eqn{d}, and positive-frequency cluster count
#'   \eqn{G}, HC1 multiplies independent covariance by \eqn{N/(N-d)} and
#'   cluster covariance by \eqn{(N-1)/(N-d)}; \code{cadjust = TRUE} additionally
#'   applies \eqn{G/(G-1)} to cluster covariance. Every declared category must
#'   have positive frequency, and the fit must have converged.
#'
#'   For mean-bias-reduced ordinal fits, this is explicitly a first-order
#'   ordinary-score sandwich evaluated at the bias-reduced estimate, not the
#'   covariance of the exact adjusted equations. Its asymptotic interpretation
#'   requires an interior regular limit, fixed parameter dimension, nonsingular
#'   limiting score derivative, an appropriate independent-observation or
#'   independent-cluster central limit theorem, and a score adjustment negligible
#'   relative to the square root of the sample size. Finite estimates under
#'   separation do not establish those conditions. This convention does not
#'   promise finite-sample calibration or bias reduction under misspecification.
#'   See [fastglm_bootstrap()] for refit-based uncertainty.
#' @returns A coefficient covariance matrix.
#' @export
fastglm_vcov <- function(object,type=c("HC3","HC2","HC1","HC0"),cluster=NULL,
                          cadjust=TRUE,block.size=8192L,full=FALSE) {
    .ordinal_partial_require_common(object,"Robust covariance")
    if(inherits(object,"fastglm_ordinal_stream"))
        stop("robust covariance is not yet available for ordinal source fits; vcov() returns model-based covariance",call.=FALSE)
    if(inherits(object,"fastglm_ordinal")) {
        if(missing(type)) type <- "HC0"
        return(.ordinal_robust_vcov(object,type,cluster,cadjust,block.size))
    }
    if(.count_inference_model(object)) {
        if(missing(type)) type <- "HC0"
        type <- match.arg(type,c("HC0","HC1","HC2","HC3"))
        return(.count_robust_vcov(object,type,cluster,cadjust,block.size,full))
    }
    type <- match.arg(type)
    if(isTRUE(object$firth)) stop("sandwich inference for adjusted scores is not implemented",call.=FALSE)
    if(!is.null(cluster) && !type %in% c("HC0","HC1")) stop("cluster covariance supports HC0 or HC1",call.=FALSE)
    good <- !is.na(coef(object)); b <- coef(object)[good]
    if(is.null(object$cov.unscaled)) stop("covariance information was not retained",call.=FALSE)
    bread <- object$cov.unscaled[good,good,drop=FALSE]
    use.qr <- .has_stream_qr(object)
    p <- length(b); meat <- matrix(0,p,p); clusters <- new.env(hash=TRUE,parent=emptyenv())
    n <- 0L
    add <- function(x,y,w,off,groups=NULL) {
        if(!all(good)) x <- x[,good,drop=FALSE]
        eta <- .model_eta(x,object,off); mu <- object$family$linkinv(eta)
        dm <- object$family$mu.eta(eta); vv <- object$family$variance(mu)
        score <- w*dm*(y-mu)/vv; score[w==0] <- 0
        n <<- n+sum(w>0)
        if(use.qr) x <- .stream_qr_transform(x,object$qr)
        if(is.null(groups)) {
            omega <- score^2
            if(type %in% c("HC2","HC3")) {
                h <- w*dm^2/vv*(if(use.qr) rowSums(x^2) else .quad_rows(x,bread))
                if(any(h>=1 & w>0)) stop("HC2/HC3 undefined for unit leverage",call.=FALSE)
                omega <- omega/(1-h)^(if(type=="HC2") 1 else 2)
            }
            meat <<- meat+crossprod(x,omega*x)
        } else {
            if(length(groups)!=nrow(x) || anyNA(groups)) stop("invalid cluster identifiers",call.=FALSE)
            positive <- w>0
            if(!any(positive)) return(invisible(NULL))
            sums <- rowsum(x[positive,,drop=FALSE]*score[positive],
                           as.character(groups[positive]),reorder=FALSE)
            for(j in seq_len(nrow(sums))) {
                key <- rownames(sums)[j]
                if(exists(key,clusters,inherits=FALSE)) sums[j,] <- sums[j,]+get(key,clusters)
                assign(key,sums[j,],clusters)
            }
        }
    }
    if(inherits(object,"fastglm_stream")) {
        if(!is.null(cluster) && !is.function(cluster)) stop("streaming clusters require a function of chunk index",call.=FALSE)
        for(k in seq_len(object$source$n_chunks)) {
            z <- object$fetch(k)
            add(z$X,z$y,z$weights,z$offset,if(!is.null(cluster)) cluster(k))
        }
    } else {
        x <- model.matrix(object)
        off <- object$offset; if(is.null(off)) off <- rep(0,nrow(x))
        .design_blocks(x,function(z,i) add(z,object$y[i],object$prior.weights[i],off[i],if(!is.null(cluster)) cluster[i]),block.size)
    }
    if(!is.null(cluster)) {
        keys <- ls(clusters,all.names=TRUE); g <- length(keys)
        if(g<2L) stop("cluster covariance requires at least two clusters",call.=FALSE)
        for(key in keys) { z <- get(key,clusters); meat <- meat+tcrossprod(z) }
        if(cadjust) meat <- meat*g/(g-1)
        if(type=="HC1") meat <- meat*(n-1)/(n-p)
    } else if(type=="HC1") meat <- meat*n/(n-p)
    if(use.qr) {
        transform <- .stream_qr_transform(diag(p),object$qr)
        v <- transform%*%meat%*%t(transform)
    } else v <- bread%*%meat%*%bread
    dimnames(v) <- list(names(b),names(b))
    if(!all(good)) { full <- matrix(NA_real_,length(good),length(good),dimnames=rep(list(names(coef(object))),2L)); full[good,good] <- v; v <- full }
    v
}

#' @exportS3Method sandwich::vcovHC fastglm_stream
vcovHC.fastglm_stream <- function(object,type=c("HC3","HC2","HC1","HC0"),...) fastglm_vcov(object,match.arg(type),...)
#' Average predictions and contrasts with delta-method uncertainty
#'
#' Computes averages without retaining individual predictions. For a streaming
#' fit, a contrast function can transform each design chunk before prediction.
#' Uncertainty conditions on the supplied covariate distribution.
#' For ordinal fits, returns an average for each category or cumulative
#' probability. With collapsed categories, inference refers to the reduced
#' category model and removed-category standard errors are unavailable.
#' Nonparallel ordinal fits include the joint uncertainty of thresholds,
#' common slopes, and threshold-specific slopes and check prediction validity
#' at all rows with positive averaging weight.
#' @param object a fitted fastglm GLM, count, two-part, or ordinal model.
#' @param newdata optional prediction data.
#' @param contrast optional function transforming a design matrix. For models
#'   with two components, the function receives and returns a list with design
#'   matrices named `count` and `zero`. The result is the average transformed
#'   prediction minus the original prediction; offsets remain fixed.
#' @param weights optional averaging weights; for a streaming fit, a function
#'   of chunk index. Defaults to equal weights.
#' @param vcov. optional covariance matrix. Count models require the full
#'   joint covariance, including theta when estimated; for robust inference use
#'   `fastglm_vcov(object, full=TRUE)`. Ordinal models require the full
#'   threshold--location covariance; supplied covariance matrices are supported
#'   only when no categories were collapsed.
#' @param level confidence level for a normal interval.
#' @param type prediction target: `"response"` for the marginal mean;
#'   `"count"` and `"zero"` are also available for two-component models.
#'   For ordinal fits, `"response"` or `"probability"` gives category
#'   probabilities and `"cumulative"` gives cumulative probabilities.
#' @returns A data frame with estimate, standard error, and confidence limits.
#' @export
fastglm_average <- function(object,newdata=NULL,contrast=NULL,weights=NULL,
                             vcov.=NULL,level=.95,type="response") {
    if(anyNA(coef(object))) stop("average prediction inference requires a full-rank fit",call.=FALSE)
    if(length(level)!=1L || !is.finite(level) || level<=0 || level>=1)
        stop("'level' must lie between zero and one",call.=FALSE)
    if(inherits(object,"fastglm_ordinal"))
        return(.ordinal_average(object,newdata,contrast,weights,vcov.,level,type))
    if(.count_inference_model(object))
        return(.count_average(object,newdata,contrast,weights,vcov.,level,type))
    if(!identical(type,"response")) stop("GLM averages use type = 'response'",call.=FALSE)
    use.qr <- is.null(vcov.) && .has_stream_qr(object)
    if(is.null(vcov.) && !use.qr) vcov. <- vcov(object)
    total <- estimate <- 0; gradient <- numeric(length(coef(object)))
    add <- function(x,off,w,contrast.x=NULL) {
        if(is.null(w)) w <- rep(1,nrow(x))
        if(length(w)!=nrow(x) || any(!is.finite(w)|w<0)) stop("invalid averaging weights",call.=FALSE)
        eta <- .model_eta(x,object,off); mu <- object$family$linkinv(eta)
        g <- (if(use.qr) .stream_qr_transform(x,object$qr) else x)*.glm_mu_eta(object,eta)
        if(!is.null(contrast)) {
            xx <- if(is.null(contrast.x)) contrast(x) else contrast.x
            if(!identical(dim(xx),dim(x))) stop("contrast must preserve design dimensions",call.=FALSE)
            ee <- .model_eta(xx,object,off)
            mu <- object$family$linkinv(ee)-mu
            g <- (if(use.qr) .stream_qr_transform(xx,object$qr) else xx)*.glm_mu_eta(object,ee)-g
        }
        total <<- total+sum(w); estimate <<- estimate+sum(w*mu)
        gradient <<- gradient+colSums(w*g)
    }
    if(inherits(object,"fastglm_stream") && is.null(newdata)) {
        for(k in seq_len(object$source$n_chunks)) { z <- object$fetch(k); add(z$X,z$offset,if(!is.null(weights)) weights(k)) }
    } else {
        d <- .prediction_design(object,newdata)
        if(use.qr) {
            if(!is.null(weights) && (length(weights)!=nrow(d$x) || any(!is.finite(weights)|weights<0)))
                stop("invalid averaging weights",call.=FALSE)
            # Apply a user transformation once to the supplied design, preserving
            # transformations that depend on all rows. Densify only each block.
            xx <- if(!is.null(contrast)) contrast(d$x)
            if(!is.null(contrast) && !identical(dim(xx),dim(d$x)))
                stop("contrast must preserve design dimensions",call.=FALSE)
            .design_blocks(d$x,function(z,i) add(z,d$offset[i],if(!is.null(weights)) weights[i],
                if(!is.null(contrast)) xx[i,,drop=FALSE]))
        } else add(d$x,d$offset,weights)
    }
    if(total<=0) stop("averaging weights must have positive total",call.=FALSE)
    gradient <- gradient/total
    se <- if(use.qr) sqrt(object$dispersion*sum(gradient^2)) else
        sqrt(drop(crossprod(gradient,vcov.%*%gradient)))
    estimate <- estimate/total; q <- qnorm((1+level)/2)
    data.frame(estimate=estimate,std.error=se,conf.low=estimate-q*se,conf.high=estimate+q*se)
}

#' @export
fitted.fastglm_stream <- function(object,...) predict(object,type="response",...)

#' @exportS3Method summary fastglm_stream
summary.fastglm_stream <- function(object,dispersion=NULL,...) {
    estimated <- is.null(dispersion) && is.na(.fixed_dispersion(object$family))
    if(is.null(dispersion)) dispersion <- object$dispersion
    b <- coef(object); se <- sqrt(diag(object$cov.unscaled)*dispersion)
    stat <- b/se
    pv <- if(estimated) 2*pt(-abs(stat),object$df.residual) else 2*pnorm(-abs(stat))
    tab <- cbind(b,se,stat,pv)
    colnames(tab) <- c("Estimate","Std. Error",if(estimated) "t value" else "z value",
                       if(estimated) "Pr(>|t|)" else "Pr(>|z|)")
    structure(list(call=object$call,coefficients=tab,dispersion=dispersion,
        deviance=object$deviance,df.residual=object$df.residual,nobs=object$nobs,
        aic=object$aic,converged=object$converged),class="summary.fastglm_stream")
}

#' @exportS3Method print summary.fastglm_stream
print.summary.fastglm_stream <- function(x,digits=max(3L,getOption("digits")-3L),...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header("Streaming generalized linear model",x$call)
    .fastglm_print_coefficients(x$coefficients,digits=digits,...)
    cat("\n")
    .fastglm_print_table(data.frame(Dispersion=x$dispersion,Deviance=x$deviance,
        `Residual df`=x$df.residual,AIC=x$aic,check.names=FALSE),digits)
    .fastglm_print_field("Observations",x$nobs)
    .fastglm_print_convergence(x$converged)
    invisible(x)
}
