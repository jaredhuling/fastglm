#' Prepare a reusable design for repeated GLM fits
#'
#' Records the predictors and weights shared by several outcome models. Formula
#' inputs also retain the factor coding and transformations needed for prediction.
#' Gaussian identity fits can reuse their matrix decomposition, the expensive
#' calculation that depends only on predictors and weights. Dense Firth fits
#' with Gaussian identity, Gamma log, or Poisson square-root links also reuse
#' their common decomposition and bias-adjustment setup with the automatic or
#' LLT solver. Saved calculations are rebuilt when predictors or weights change.
#' Mean-bias-reduced Gaussian identity, Gamma log, and Poisson square-root fits
#' also reuse the observation leverages needed for the adjusted score. A
#' leverage measures how strongly a row contributes to the fitted regression.
#' The shared setup is used by `method = "auto"` and `method = "llt"` (or `2`).
#' @param x model matrix or a one-sided formula.
#' @param data data frame for a formula.
#' @param weights,offset optional common weights and offsets.
#' @param firth default for mean bias reduction in subsequent batch fits.
#' @returns A `fastglm_design` object for [fastglm_batch()].
#' @export
fastglm_prepare <- function(x, data = NULL, weights = NULL, offset = NULL, firth = FALSE) {
    if(!is.logical(firth) || length(firth)!=1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE",call.=FALSE)
    tt <- mf <- NULL
    if (inherits(x, "formula")) {
        mf <- model.frame(x, data, na.action=na.fail)
        tt <- attr(mf,"terms"); x <- model.matrix(tt,mf)
        if (is.null(offset)) offset <- model.offset(mf)
    }
    if(is.matrix(x) && is.integer(x)) storage.mode(x)<-"double"
    if (is.null(weights)) weights <- rep(1,nrow(x))
    if (is.null(offset)) offset <- rep(0,nrow(x))
    .validate_glm_data(x, rep(0,nrow(x)), weights, offset)
    cache <- new.env(parent=emptyenv())
    structure(list(x=x, weights=weights, offset=offset, terms=tt,
        xlevels=if (!is.null(tt)) .getXlevels(tt,mf), contrasts=attr(x,"contrasts"),
        cache=cache,firth=firth), class="fastglm_design")
}

.prepared_factor <- function(design,weights,code,method,firth,inference) {
    if(!is.matrix(design$x) || ncol(design$x)==0L ||
       !(identical(method,"auto") || identical(.solver_method(method),2L))) return(NULL)
    key<-list(x=design$x,weights=weights,code=code,method=method,firth=firth)
    cache<-design$cache
    if(!identical(cache$key,key)) {
        cache$key<-key
        scale<-if(identical(method,"auto"))
            as.numeric(pmax(abs(colMax_dense(design$x)),abs(colMin_dense(design$x)))) else rep(1,ncol(design$x))
        scale[scale==0]<-1
        xx<-if(all(scale==1)) design$x else sweep(design$x,2L,scale,"/")
        if(is.integer(xx)) storage.mode(xx)<-"double"
        factor<-prepare_glm_constant(xx,weights*if(code==9L) 4 else 1,firth,code)
        if(!is.null(factor)) {
            factor$x<-xx;factor$scale<-scale
            factor$intercept<-.intercept_columns(design$x)
            factor$covariance<-matrix(numeric(),0L,0L)
        }
        cache$factor<-factor
        cache$builds<-if(is.null(cache$builds)) 1L else cache$builds+1L
    }
    factor<-cache$factor
    if(!is.null(factor) && inference!="none" && !length(factor$covariance)) {
        factor$covariance<-chol2inv(t(factor$lower));cache$factor<-factor
    }
    factor
}

.prepared_firth_fit <- function(design,response,family,weights,offset,start,
                                factor,tol,maxit,inference) {
    y<-response;nobs<-n<-length(y);mustart<-etastart<-NULL
    eval(family$initialize)
    if(!is.null(start)) start<-start*factor$scale
    eta<-if(is.null(start)) family$linkfun(mustart) else drop(factor$x%*%start)+offset
    mu<-family$linkinv(eta)
    # Batch warm starts are optional. Use the family's ordinary initializer
    # when the preceding response supplied a start outside this link's domain.
    if(!is.null(start) && (!family$validmu(mu) || !family$valideta(eta))) {
        start<-NULL;eta<-family$linkfun(mustart);mu<-family$linkinv(eta)
    }
    if(!family$validmu(mu) || !family$valideta(eta))
        stop("cannot find valid starting values: please specify some",call.=FALSE)
    if(is.null(start)) start<-rep(0,ncol(design$x))
    fit<-fit_glm_firth(factor$x,as.numeric(y),weights,offset,start,mu,eta,
        family$variance,family$mu.eta,family$linkinv,family$dev.resids,
        family$valideta,family$validmu,2L,tol,as.integer(maxit),family_code(family),
        family_params(family),inference=inference!="none",prepared=factor)
    if(!fit$converged) warning("fit_glm: algorithm did not converge",call.=FALSE)
    cn<-colnames(design$x);is.int<-factor$intercept
    if(is.null(cn)) {
        cn<-paste0("X",seq_len(ncol(design$x)))
        cn[is.int]<-if(sum(is.int)==1L) "(Intercept)" else paste0("(Intercept)",seq_len(sum(is.int)))
    }
    fit$coefficients<-setNames(fit$coefficients/factor$scale,cn)
    if(length(fit$cov.unscaled)) fit$cov.unscaled<-fit$cov.unscaled/outer(factor$scale,factor$scale)
    fit$log.det.XtWX<-fit$log.det.XtWX+2*sum(log(factor$scale))
    fit$penalized.deviance<-fit$penalized.deviance-2*sum(log(factor$scale))
    fit$x<-design$x;fit$y<-as.numeric(y);fit$n<-n;fit$family<-family
    fit$intercept<-any(is.int);fit$inference<-inference;fit$solver<-"cached_llt"
    fit<-.finish_glm(fit,offset)
    class(fit)<-c("fastglm_firth","fastglm")
    fit
}

#' Fit multiple responses using a prepared design
#'
#' The same design can be used for many outcomes. Gaussian identity fits share
#' the weighted crossproduct factor. With `firth = TRUE`, Gaussian identity,
#' Gamma log, and Poisson square-root fits share their factor and observation
#' leverages when using automatic or LLT fitting. Changes to predictors or
#' weights rebuild this setup; offsets and responses can vary independently.
#' Other families and decompositions use the ordinary fitting routine.
#' Sequential fits may use the preceding converged fit as a starting value.
#' Independent fits can run in separate R processes.
#' @param design a model matrix or object from [fastglm_prepare()].
#' @param y matrix whose columns are responses, or a list of responses.
#' @param family common GLM family.
#' @param weights,offset common vectors or matrices with one column per fit.
#' @param method,tol,maxit solver and convergence controls.
#' @param warm.start reuse coefficients from the preceding converged fit.
#' @param workers number of independent worker processes; defaults to one.
#' @param inference `"full"` retains covariance, `"diagonal"` retains only
#'   standard errors, and `"none"` omits inference from the result.
#' @param retain.data retain the design and response on each model.
#' @param firth apply mean bias reduction; `NULL` uses the prepared design's
#'   default, or `FALSE` for a matrix supplied directly.
#' @returns A named list of fitted models.
#' @export
fastglm_batch <- function(design, y, family = gaussian(), weights = NULL,
                          offset = NULL, method = "auto", tol = 1e-8,
                          maxit = 100L, warm.start = TRUE, workers = 1L,
                          inference = c("full", "diagonal", "none"), retain.data = TRUE,
                          firth = NULL) {
    if (!inherits(design,"fastglm_design")) design <- fastglm_prepare(design)
    if(is.matrix(design$x) && is.integer(design$x))
        storage.mode(design$x)<-"double"
    if(is.null(firth)) firth<-isTRUE(design$firth)
    if(!is.logical(firth) || length(firth)!=1L || is.na(firth))
        stop("'firth' must be TRUE or FALSE",call.=FALSE)
    if(!is.environment(design$cache)) design$cache<-new.env(parent=emptyenv())
    family <- .resolve_family(family); inference <- match.arg(inference)
    .check_iterations(tol,maxit)
    if (!is.list(y)) {
        if (is.null(dim(y))) y <- matrix(y,ncol=1L)
        yy <- lapply(seq_len(ncol(y)),function(j) y[,j]); names(yy) <- colnames(y); y <- yy
    }
    if (!length(y)) return(list())
    if (length(workers) != 1L || !is.finite(workers) || workers < 1 || workers != as.integer(workers)) stop("'workers' must be a positive integer",call.=FALSE)
    if (is.null(weights)) weights <- design$weights
    if (is.null(offset)) offset <- design$offset
    if(is.matrix(weights) && ncol(weights)!=length(y)) stop("weights must have one column per response",call.=FALSE)
    if(is.matrix(offset) && ncol(offset)!=length(y)) stop("offset must have one column per response",call.=FALSE)
    take <- function(z,j) if (is.matrix(z)) z[,j] else z
    .validate_glm_data(design$x,y[[1L]],take(weights,1L),take(offset,1L))
    code<-family_code(family)
    canonical.initialize<-tryCatch(identical(family$initialize,
        get(family$family,asNamespace("stats"))(link=family$link)$initialize),error=function(e) FALSE)
    eligible<-canonical.initialize && code %in% (if(firth) c(0L,9L,10L) else 0L)
    if(firth && (grepl("^quasi",family$family) || code<0L || code>=20L))
        stop("mean bias reduction currently requires an unmodified supported family",call.=FALSE)
    if(firth && code %in% c(0:2,10:16) && !is.null(family$dispersion) && is.finite(family$dispersion))
        stop("mean bias reduction with an externally fixed dispersion is not implemented for this family",call.=FALSE)
    # The cache contains only R arrays and can be shared by forked workers or
    # serialized to independent processes. Build common setup once per batch.
    if(eligible && !is.matrix(weights)) .prepared_factor(design,weights,code,method,firth,inference)
    one <- function(j, start=NULL) {
        w <- take(weights,j); off <- take(offset,j); response <- y[[j]]
        if(NROW(response)!=nrow(design$x) || anyNA(response) ||
           (!is.numeric(response) && !is.factor(response)) || (is.numeric(response) && any(!is.finite(response))))
            stop("responses must contain one finite value per design row",call.=FALSE)
        for(v in list(w,off)) if(!is.numeric(v) || length(v)!=nrow(design$x) || any(!is.finite(v)))
            stop("weights and offsets must contain one finite value per design row",call.=FALSE)
        if(any(w<0) || sum(w>0)<ncol(design$x) || !any(w>0))
            stop("weights must be nonnegative with enough positive-weight observations",call.=FALSE)
        f<-if(eligible && is.numeric(response) && is.null(dim(response)))
            .prepared_factor(design,w,code,method,firth,inference) else NULL
        if(!is.null(f) && firth) {
            fit<-.prepared_firth_fit(design,response,family,w,off,start,f,tol,maxit,inference)
        } else if(!is.null(f)) {
            b <- backsolve(t(f$lower),forwardsolve(f$lower,crossprod(f$x,w*(response-off))))/f$scale
            eta <- drop(design$x%*%b)+off
            cn <- colnames(design$x); if(is.null(cn)) cn <- paste0("X",seq_len(ncol(design$x)))
            b <- setNames(drop(b),cn)
            fit <- list(coefficients=b, cov.unscaled=if(length(f$covariance)) f$covariance/outer(f$scale,f$scale), fitted.values=eta,
                linear.predictors=eta, weights=w, prior.weights=w, family=family,
                y=response,n=rep(1,length(response)),x=design$x,rank=ncol(design$x),
                intercept=any(f$intercept),deviance=sum(w*(response-eta)^2),
                iter=1L,converged=TRUE,solver="cached_llt",inference=inference)
            fit <- .finish_glm(fit,off); class(fit) <- "fastglm"
        } else {
            if(!is.null(start) && (is.matrix(design$x) || inherits(design$x,"dgCMatrix"))) {
                eta<-drop(design$x%*%start)+off;mu<-family$linkinv(eta)
                valideta<-if(is.null(family$valideta)) TRUE else family$valideta(eta)
                validmu<-if(is.null(family$validmu)) TRUE else family$validmu(mu)
                if(!valideta || !validmu) start<-NULL
            }
            fit <- fastglm(design$x,response,family=family,weights=w,offset=off,
                start=start,method=method,tol=tol,maxit=maxit,firth=firth,inference=inference,retain.data=retain.data)
        }
        fit$terms <- design$terms; fit$xlevels <- design$xlevels; fit$contrasts <- design$contrasts
        if (inference != "full") fit$cov.unscaled <- NULL
        if (inference == "none") fit$se <- NULL
        fit$inference <- inference
        if (!retain.data) { fit$x <- NULL; fit$y <- NULL }
        fit
    }
    workers <- min(as.integer(workers),length(y))
    if (workers > 1L) {
        # Each process owns its R callbacks and numerical workspace.
        if (.Platform$OS.type == "windows") {
            cl <- parallel::makeCluster(workers); on.exit(parallel::stopCluster(cl))
            initialize <- function(paths) {.libPaths(paths);loadNamespace("fastglm");NULL}
            environment(initialize) <- baseenv()
            parallel::clusterCall(cl,initialize,.libPaths())
            out <- parallel::parLapply(cl,seq_along(y),one)
        } else out <- parallel::mclapply(seq_along(y),one,mc.cores=workers)
        bad <- vapply(out,inherits,logical(1),"try-error")
        if (any(bad)) stop("a batch worker failed: ",out[[which(bad)[1L]]],call.=FALSE)
    } else {
        out <- vector("list",length(y)); b <- NULL
        for(j in seq_along(y)) {
            out[[j]] <- one(j,b)
            b<-if(warm.start && out[[j]]$converged && all(is.finite(coef(out[[j]])))) coef(out[[j]]) else NULL
        }
    }
    names(out) <- names(y); out
}
