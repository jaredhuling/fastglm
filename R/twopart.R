#' Two-part regression for nonnegative continuous outcomes
#'
#' Fits a binomial model for a positive outcome and a Gamma model with a log
#' link among positive outcomes. The marginal mean is the product of the
#' probability of a positive outcome and the conditional positive mean.
#' @param formula numeric matrix, `dgCMatrix`, [fastglm_source()], or a formula
#'   with one or two right-hand sides, as in [fastglm_hurdle()].
#' @param data,subset,na.action,weights,offset model-frame arguments. A vector
#'   offset applies to the positive component; a two-column matrix supplies
#'   positive and binary offsets.
#' @param link link for the probability of a positive outcome.
#' @param method,tol,maxit solver and convergence controls for [fastglm()].
#' @param firth apply mean bias reduction to both components.
#' @param response nonnegative continuous response for a matrix design.
#' @param z optional binary-component design for matrix inputs; defaults to
#'   the positive-component matrix. Source blocks can supply `Z` and `X`.
#' @param block.size maximum number of matrix rows processed together for
#'   model validation. A source determines its own block sizes.
#' @returns A `fastglm_twopart` model supporting prediction, delta-method
#'   standard errors, simulation, and component summaries.
#' @export
fastglm_twopart <- function(formula, data, subset, na.action, weights, offset,
                            link = "logit", method = "auto", tol = 1e-8,
                            maxit = 100L, firth = FALSE,
                            response = NULL, z = NULL, block.size = 8192L) {
    if(is.matrix(formula) || inherits(formula,c("dgCMatrix","fastglm_source"))) {
        if(!missing(subset) || !missing(na.action))
            stop("subset and na.action apply only to formula fits; prepare matrix or source rows first",call.=FALSE)
        out <- .count_backend_twopart(formula,response,z,
            if(missing(weights)) NULL else weights,if(missing(offset)) NULL else offset,
            link,method,tol,maxit,firth,block.size)
        out$call <- match.call()
        return(out)
    }
    cl <- match.call(); f <- Formula::Formula(formula)
    if (missing(data)) data <- environment(formula)
    mc <- match.call(expand.dots = FALSE)
    mc <- mc[c(1L, match(c("formula", "data", "subset", "na.action", "weights", "offset"), names(mc), 0L))]
    mc[[1L]] <- quote(stats::model.frame); mc$formula <- f
    mf <- eval(mc, parent.frame())
    tt <- list(count = terms(f, data=data, rhs=1L), zero = terms(f, data=data, rhs=if (length(f)[2L] > 1L) 2L else 1L))
    x <- lapply(tt, model.matrix, data=mf); y <- model.response(mf)
    w <- model.weights(mf); if (is.null(w)) w <- rep(1, length(y))
    off <- .component_offsets(mf, tt)
    .validate_glm_data(x$count, y, w, off$count)
    if (any(y < 0)) stop("two-part responses must be nonnegative", call. = FALSE)
    positive <- y > 0 & w > 0
    if (!any(positive) || all(y[w > 0] > 0)) stop("two-part models require zeros and positive observations", call. = FALSE)
    binary <- fastglm(x$zero, as.numeric(y > 0), family=binomial(link), weights=w, offset=off$zero,
                      method=method, tol=tol, maxit=maxit, firth=firth)
    fit <- fastglm(x$count[positive,,drop=FALSE], y[positive], family=Gamma("log"),
                   weights=w[positive], offset=off$count[positive], method=method, tol=tol, maxit=maxit, firth=firth)
    pc <- length(coef(fit)); pz <- length(coef(binary))
    v <- matrix(0, pc+pz, pc+pz); v[seq_len(pc),seq_len(pc)] <- vcov(fit)
    v[pc+seq_len(pz),pc+seq_len(pz)] <- vcov(binary)
    cn <- c(paste0("count_", names(coef(fit))),paste0("zero_", names(coef(binary))))
    dimnames(v) <- list(cn,cn)
    out <- list(coefficients=list(count=coef(fit),zero=coef(binary)),
        vcov=v, vcov.full=v, positive=fit, binary=binary,
        loglik=as.numeric(logLik(fit))+as.numeric(logLik(binary)),
        converged=fit$converged && binary$converged, dist="Gamma", link=link,
        model=mf, terms=tt, formula=f, x=x, y=y, weights=w, offset=off,
        levels=lapply(mf[vapply(mf,is.factor,logical(1))],levels),
        contrasts=lapply(x,attr,"contrasts"), na.action=attr(mf,"na.action"),
        call=cl, df.residual=sum(w>0)-pc-pz-1, firth=firth,
        refit.controls=list(link=link,method=method,tol=tol,maxit=maxit,firth=firth,block.size=block.size))
    class(out) <- c("fastglm_twopart","fastglm_hurdle")
    out
}
#' @export
logLik.fastglm_twopart <- function(object, ...) {
    structure(object$loglik, df=length(coef(object))+1L, nobs=nobs(object), class="logLik")
}
#' @exportS3Method print fastglm_twopart
print.fastglm_twopart <- function(x, ...) {
    cat("\nTwo-part binomial / Gamma regression\n")
    print(summary(x), ...); invisible(x)
}
