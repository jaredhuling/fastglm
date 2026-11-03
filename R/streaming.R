#' Fit a GLM by reading blocks of observations as needed
#'
#' `fastglm_streaming()` fits a GLM while keeping only a block of design rows
#' and coefficient-sized summary matrices in memory. Reading successive
#' blocks this way is called streaming. It permits fitting data that cannot
#' be loaded into memory together, at the cost of rereading the observations
#' as the coefficient estimates are updated.
#'
#' @param chunk_callback a function. Called as `chunk_callback(k)` for
#'   `k = 1, ..., n_chunks`. Must return a list with elements
#'   \describe{
#'     \item{`X`}{an `n_k x p` numeric matrix (chunk of the design matrix).}
#'     \item{`y`}{numeric vector of length `n_k` (response).}
#'     \item{`weights`}{optional; numeric vector of length `n_k` of prior weights.}
#'     \item{`offset`}{optional; numeric vector of length `n_k` of offsets.}
#'   }
#'   Every chunk must have the same number of columns, in the same order. The
#'   supplied function, called a callback, is called repeatedly during fitting.
#'   It must return the same rows and values each time a block is requested.
#' @param n_chunks integer; the number of chunks to iterate over.
#' @param family a `family` object describing the error distribution and link.
#'   See [stats::family()].
#' @param start optional length-`p` numeric vector of starting coefficients.
#' @param method integer; `2` for LLT Cholesky (default) or `3` for LDLT
#'   Cholesky. QR / SVD methods are not supported in streaming mode.
#' @param tol convergence tolerance on the relative change in deviance
#'   (or sup-norm of coefficient update when `firth = TRUE`). Gamma-log
#'   bias-reduced fits also check the remaining adjusted-score update and
#'   relative change in dispersion.
#' @param maxit maximum number of IRLS iterations.
#' @param firth logical; if `TRUE` apply Firth bias reduction using
#'   lagged leverages from the previous IRLS iteration.  At convergence
#'   the result matches the exact-leverage dense path.
#'
#' @returns An object with classes `"fastglm_stream"` and `"fastglm"` containing
#'   coefficient estimates and model summaries as in
#'   [fastglm()], including `coefficients`, `cov.unscaled`, `deviance`,
#'   `iter`, and `converged`. The object retains the function used to read
#'   the observations. Prediction and [fastglm_vcov()] call that function
#'   again, so the data must remain accessible and unchanged.
#'
#' @details
#' Iteratively reweighted least squares (IRLS) updates the coefficients by
#' solving a sequence of weighted regressions. The updates and safeguards
#' that shorten unsuccessful steps run in C++. For supported standard
#' families, the distribution calculations also run in C++; the R function
#' is called to supply each block of observations.
#'
#' Standard errors and `cov.unscaled` come from the final `(X' W X)`
#' Cholesky factor, exactly as in the in-memory path.
#'
#' @section Files and databases:
#' [fastglm_source_arrow()] reads one Parquet file per block, and
#' [fastglm_source_dbi()] reads the rows returned by one database query per
#' block. [fastglm_source()] records the factor levels and transformations
#' needed to give the design columns the same meaning in each block.
#'
#' @examples
#' # Simulate a "data source" that yields the design matrix in 4 row-blocks.
#' set.seed(1)
#' n <- 1000; p <- 5
#' X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
#' y <- rbinom(n, 1, plogis(X %*% c(0.2, 0.5, -0.3, 0.4, -0.2)))
#' chunk_size <- 250
#' chunks <- function(k) {
#'   idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
#'   list(X = X[idx, , drop = FALSE], y = y[idx])
#' }
#'
#' fit_stream <- fastglm_streaming(chunks, n_chunks = 4, family = binomial())
#' fit_full   <- fastglm(X, y, family = binomial(), method = 2)
#' max(abs(coef(fit_stream) - coef(fit_full)))
#'
#' @export
fastglm_streaming <- function(chunk_callback, n_chunks = NULL,
                              family = gaussian(), start = NULL, method = 2L,
                              tol = 1e-8, maxit = 100L, firth = FALSE) {
    source <- if (inherits(chunk_callback, "fastglm_source")) chunk_callback else
        fastglm_source(chunk_callback, n_chunks)
    if(!is.null(source$schema$terms) && !inherits(source$schema$terms,"terms"))
        stop("a two-component source requires fastglm_hurdle() or fastglm_twopart()",call.=FALSE)
    n_chunks <- source$n_chunks
    family <- .resolve_family(family)
    .check_iterations(tol,maxit)
    method <- .solver_method(method)
    if (!(method %in% 2:3)) stop("streaming requires method 'llt' or 'ldlt'",call.=FALSE)
    fc <- family_code(family)
    if (firth && fc %in% c(0:2,10:16) && !is.null(family$dispersion) && is.finite(family$dispersion)) stop("mean bias reduction with externally fixed dispersion is not implemented for this family",call.=FALSE)
    if (firth && (fc < 0 || fc >= 20 || grepl("^quasi",family$family)))
        stop("mean bias reduction requires a supported likelihood family",call.=FALSE)
    if (!is.logical(firth) || length(firth) != 1L || is.na(firth)) stop("'firth' must be TRUE or FALSE",call.=FALSE)
    state <- new.env(parent=emptyenv()); state$passes <- 0L; state$reads <- 0L
    state$columns <- NULL; state$p <- NULL; state$rows <- integer(n_chunks)
    state$initialization.warnings <- character()
    initialize <- function(z) {
        x <- z$X; y <- z$y; nobs <- n <- NROW(y)
        if(inherits(x,"sparseMatrix")) x <- as.matrix(x)
        weights <- if(is.null(z$weights)) rep(1,nobs) else z$weights
        offset <- if(is.null(z$offset)) rep(0,nobs) else z$offset
        if (!is.matrix(x) || !is.numeric(x)) stop("each chunk$X must be a numeric matrix",call.=FALSE)
        if(is.integer(x)) storage.mode(x)<-"double"
        if (nobs == 0L) stop("empty chunks are not supported; omit them from the source",call.=FALSE)
        if (nrow(x) != nobs || any(!is.finite(x)) || anyNA(y) ||
            any(!is.finite(weights)) || length(weights)!=nobs || any(weights<0) ||
            length(offset)!=nobs || any(!is.finite(offset))) stop("invalid chunk rows, weights, offsets, or values",call.=FALSE)
        mustart <- etastart <- NULL
        withCallingHandlers(eval(family$initialize),warning=function(w) {
            message <- conditionMessage(w)
            if(message %in% state$initialization.warnings) invokeRestart("muffleWarning")
            state$initialization.warnings <- c(state$initialization.warnings,message)
        })
        list(X=x,y=as.numeric(y),weights=as.numeric(weights),offset=as.numeric(offset),
             trials=n,mustart=mustart)
    }
    fetch <- function(k) {
        if(k == 1L) state$passes <- state$passes+1L
        state$reads <- state$reads+1L
        z <- initialize(source$callback(k))
        if(is.null(state$p)) { state$p <- ncol(z$X); state$columns <- colnames(z$X) }
        if(ncol(z$X)!=state$p || !identical(colnames(z$X),state$columns)) stop("chunk columns changed from the source schema",call.=FALSE)
        if(state$rows[k] != 0L && state$rows[k] != nrow(z$X)) stop("chunk size changed between passes",call.=FALSE)
        state$rows[k] <- nrow(z$X)
        z
    }
    # One validation pass also constructs the usual family-initialized WLS start.
    a <- rhs <- low <- high <- NULL; nobs <- sumw <- sumwy <- 0; has.offset <- FALSE
    gamma.null.total <- -Inf
    for(k in seq_len(n_chunks)) {
        z <- fetch(k); x <- z$X; w <- z$weights
        has.offset <- has.offset || any(z$offset != 0)
        if(is.null(a)) {
            a <- matrix(0,ncol(x),ncol(x)); rhs <- numeric(ncol(x))
            low <- rep(Inf,ncol(x)); high <- rep(-Inf,ncol(x))
        }
        low <- pmin(low,apply(x,2,min)); high <- pmax(high,apply(x,2,max))
        eta <- family$linkfun(z$mustart); mu <- family$linkinv(eta)
        dm <- family$mu.eta(eta)
        ww <- if(fc==10L) w else w*dm^2/family$variance(mu)
        ww[w==0] <- 0
        zz <- eta-z$offset+(z$y-mu)/dm; zz[w==0] <- 0
        a <- a+crossprod(x,ww*x); rhs <- rhs+drop(crossprod(x,ww*zz))
        nobs <- nobs+sum(w>0); sumw <- sumw+sum(w); sumwy <- sumwy+sum(w*z$y)
        if(fc==10L) gamma.null.total <- .gamma_log_null_total(z$y,w,z$offset,gamma.null.total)
    }
    p <- state$p
    if(nobs <= p) stop("streaming requires more positive-weight observations than coefficients",call.=FALSE)
    is.int <- low == high & low != 0
    if(is.null(start)) start <- tryCatch(drop(solve(a,rhs)),error=function(e) stop("streaming design is rank deficient",call.=FALSE))
    if(length(start)!=p || any(!is.finite(start))) stop("invalid starting coefficients",call.=FALSE)
    res <- fit_streaming_glm(fetch,n_chunks,as.integer(p),method,tol,as.integer(maxit),fc,
        family$variance,family$mu.eta,family$linkinv,family$dev.resids,
        if(is.null(family$valideta)) function(x) TRUE else family$valideta,
        if(is.null(family$validmu)) function(x) TRUE else family$validmu,
        as.numeric(start),family_params(family),firth)
    cn <- state$columns
    if(is.null(cn)) { cn <- paste0("X",seq_len(p)); cn[is.int] <- paste0("(Intercept)",if(sum(is.int)>1) seq_len(sum(is.int)) else "") }
    names(res$coefficients) <- cn; dimnames(res$cov.unscaled) <- list(cn,cn)
    res$family <- family; res$nobs <- nobs; res$df.residual <- nobs-p
    res$intercept <- any(is.int); res$df.null <- nobs-as.integer(res$intercept)
    disp <- .fixed_dispersion(family)
    if(is.na(disp)) disp <- res$pearson/res$df.residual
    if(firth) disp <- res$bias.dispersion
    res$dispersion <- disp; res$se <- sqrt(diag(res$cov.unscaled)*disp)
    res$source <- source; res$fetch <- fetch; res$state <- state
    res$terms <- source$schema$terms; res$xlevels <- source$schema$xlevels
    res$contrasts <- source$schema$contrasts
    res$call <- match.call(); res$firth <- firth; res$solver <- c("llt","ldlt")[method-1L]
    # Evaluate the likelihood using original binomial trial totals. Gaussian
    # and Gamma likelihood normalizers depend on global dispersion, so they
    # are accumulated explicitly rather than summing per-chunk AIC estimates.
    ll <- 0; null.dev <- 0
    for(k in seq_len(n_chunks)) {
        z <- fetch(k); mu <- family$linkinv(drop(z$X%*%res$coefficients)+z$offset)
        null.mu <- if(res$intercept && fc==10L) exp(z$offset+gamma.null.total-log(sumw)) else
            if(res$intercept) rep(sumwy/sumw,length(mu)) else family$linkinv(z$offset)
        null.dev <- null.dev+sum(family$dev.resids(z$y,null.mu,z$weights))
        w <- z$weights; y <- z$y; keep <- w>0; ff <- family$family
        if(ff == "gaussian") ll <- ll+sum(dnorm(y[keep],mu[keep],sqrt(res$deviance/nobs/w[keep]),log=TRUE))
        else if(ff == "Gamma") ll <- ll+sum(w[keep]*dgamma(y[keep],shape=sumw/res$deviance,scale=mu[keep]*res$deviance/sumw,log=TRUE))
        else if(ff == "inverse.gaussian") ll <- ll-.5*sum(w[keep]*(1+log(res$deviance/sumw*2*pi)+3*log(y[keep])))
        else ll <- ll - .family_aic(family,y,z$trials,mu,w,sum(family$dev.resids(y,mu,w)))/2
    }
    if(family$family %in% c("gaussian","Gamma","inverse.gaussian")) res$aic <- -2*ll+2*(p+1L)
    else res$aic <- -2*ll+2*p
    # An intercept with offsets is a one-parameter GLM, not a weighted mean.
    res$null.deviance <- null.dev
    if(res$intercept && has.offset && fc!=10L) {
        nullfetch <- function(k) { z <- fetch(k); z$X <- matrix(1,length(z$y),1L); z }
        eta0 <- family$linkfun(sumwy/sumw)
        nf <- fit_streaming_glm(nullfetch,n_chunks,1L,2L,tol,as.integer(maxit),fc,
            family$variance,family$mu.eta,family$linkinv,family$dev.resids,
            if(is.null(family$valideta)) function(x) TRUE else family$valideta,
            if(is.null(family$validmu)) function(x) TRUE else family$validmu,
            as.numeric(eta0),family_params(family),FALSE)
        res$null.deviance <- nf$deviance
    }
    res$passes <- state$passes; res$chunk.reads <- state$reads
    if(!res$converged) warning("streaming GLM did not converge",call.=FALSE)
    class(res) <- c("fastglm_stream",if(firth) "fastglm_firth","fastglm")
    res
}
