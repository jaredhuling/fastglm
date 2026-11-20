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
#'   Alternatively, a [fastglm_source()], or a design matrix (a numeric
#'   matrix, `dgCMatrix`, or file-backed `big.matrix`) with `y` and optional
#'   `weights` and `offset`; the matrix is then read in blocks of `block.size`
#'   rows. For a `big.matrix` this keeps memory use bounded by the block size,
#'   whereas `fastglm()` on a `big.matrix` is faster but holds several vectors
#'   of length n during the fit.
#' @param n_chunks integer; the number of chunks to iterate over.
#' @param family a `family` object describing the error distribution and link.
#'   See [stats::family()].
#' @param start optional length-`p` numeric vector of starting coefficients.
#' @param method `"llt"` or `2` for LLT Cholesky (default), `"ldlt"` or
#'   `3` for LDLT Cholesky, or `"qr"` or `0` for incremental QR. All
#'   require full column rank across the complete weighted design. The
#'   `"auto"` choice retains the default Cholesky method for streaming fits.
#' @param tol convergence tolerance on relative deviance change for ordinary
#'   fits. Cholesky bias-reduced fits use the largest coefficient update;
#'   Gamma-log fits also check the remaining adjusted-score update and relative
#'   dispersion change. QR bias-reduced fits require small remaining changes in
#'   fitted linear predictors, a small joint information-scaled adjusted score
#'   per positive-weight row, and dispersion stationarity where applicable.
#' @param maxit maximum number of IRLS iterations.
#' @param covariance where the information matrix behind the standard errors
#'   is evaluated. `"estimate"` (default) evaluates it at the reported
#'   coefficients. `"previous"` uses the factorization from the final
#'   iteration, the convention of [stats::glm.fit()], and changes standard
#'   errors by a relative amount of order 1e-6 at the default tolerance.
#'   For streamed fits both read the data the same number of times, because
#'   the information at the reported coefficients is accumulated in the pass
#'   that evaluates them. Coefficients agree to within the convergence
#'   tolerance. Bias-reduced fits (`firth = TRUE`) are not affected.
#' @param y,weights,offset response, prior weights, and offset when
#'   `chunk_callback` is a design matrix.
#' @param block.size number of rows read at a time from a design matrix.
#' @param firth logical; if `TRUE` apply Firth bias reduction using
#'   lagged leverages from the previous IRLS iteration. At convergence,
#'   the fit solves the same adjusted-score equations as the dense path.
#' @param threads the number of threads for the compiled computations within
#'   each block of observations, or `NULL` (default) for the session setting
#'   of [fastglm_threads()]. Blocks are still read one at a time. Results are
#'   identical for every number of threads.
#' @param trace `TRUE` to print one line per coefficient update (the
#'   iteration, the passes over the source so far including the validation
#'   pass, the deviance, the step halvings of the iteration, and the elapsed
#'   seconds), `FALSE` for none, or `NULL` (default) for
#'   `getOption("fastglm.trace")`, which defaults to `TRUE` in an interactive
#'   session and `FALSE` otherwise (in scripts, knitr, and tests).
#'
#' @returns An object with classes `"fastglm_stream"` and `"fastglm"` containing
#'   coefficient estimates and model summaries as in
#'   [fastglm()], including `coefficients`, `cov.unscaled`, `deviance`,
#'   `iter`, `converged`, and `convergence.reason`. Convergence and starting
#'   values follow [fastglm()] (see its Convergence section); `iter` counts
#'   coefficient updates as there, and `passes` counts reads of the data,
#'   which include validation and initialization passes. The object retains
#'   the function used to read
#'   the observations. Prediction and [fastglm_vcov()] call that function
#'   again, so the data must remain accessible and unchanged. QR fits also
#'   retain a `qr` list containing a triangular matrix `R`, column permutation
#'   `pivot`, and column scales `scale` for the final weighted design.
#'   QR bias-reduced fits additionally save `remaining.predictor.step`,
#'   `adjusted.score.rms`, `remaining.dispersion.step`, and
#'   `adjusted.dispersion.score.rms`. The dispersion diagnostics are missing
#'   for fixed-dispersion families. `refined.convergence.verification` records
#'   whether verification required additional precision for the design.
#'
#' @details
#' Iteratively reweighted least squares (IRLS) updates the coefficients by
#' solving a sequence of weighted regressions. The updates and safeguards
#' that shorten unsuccessful steps run in C++. For supported standard
#' families, the distribution calculations also run in C++; the R function
#' is called to supply each block of observations.
#'
#' Cholesky methods accumulate the weighted crossproduct `X' W X`. The QR
#' method instead updates a triangular factor as blocks arrive, avoiding
#' crossproducts in initialization, coefficient updates, and covariance
#' calculation. It can retain precision for nearly redundant predictors at
#' the cost of more arithmetic. Rank is assessed after combining all blocks;
#' individual blocks may contain fewer rows than columns or lack factor
#' levels. Globally redundant predictors cause an error.
#'
#' Standard errors and `cov.unscaled` use the final weighted design. For QR
#' fits, prediction uncertainty, leverage, robust covariance, and default
#' average-prediction inference use the triangular factor directly. Supplying
#' a covariance matrix explicitly to [fastglm_average()] uses that matrix.
#' Storage includes the current block and matrices with dimensions determined
#' by the number of coefficients, including a quadratic storage requirement
#' in that number. Neither method retains all design rows.
#'
#' QR bias-reduced convergence checks evaluate the actual adjusted equations
#' using current working weights and leverage values. Verification requires
#' additional source passes. For poorly conditioned designs, the calculation
#' forms and refactors a better-scaled basis before checking these equations.
#' A small score alone cannot certify convergence: the remaining predictor
#' change and dispersion checks must also pass. Failed step searches retain a
#' failure flag.
#'
#' @references Zietkiewicz, P. and Kosmidis, I. (2024). Bounded-memory adjusted
#' scores estimation in generalized linear models with large data sets.
#' \url{https://arxiv.org/abs/2307.07342}.
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
#' @md
#' @export
fastglm_streaming <- function(chunk_callback, n_chunks = NULL,
                              family = gaussian(), start = NULL, method = 2L,
                              tol = 1e-8, maxit = 100L, firth = FALSE,
                              covariance = c("estimate", "previous"),
                              y = NULL, weights = NULL, offset = NULL, block.size = 100000L,
                              threads = NULL, trace = NULL) {
    method.supplied <- !missing(method)
    threads <- .fastglm_resolve_threads(threads)
    restore.threads <- .fastglm_with_threads(threads)
    on.exit(restore.threads(), add = TRUE)
    tr <- .fastglm_scope_trace(trace, streamed = TRUE)
    on.exit(tr$restore(), add = TRUE)
    if (is.matrix(chunk_callback) || inherits(chunk_callback, c("dgCMatrix", "big.matrix"))) {
        chunk_callback <- .matrix_source(chunk_callback, y, weights, offset, block.size)
    } else if (!is.null(y) || !is.null(weights) || !is.null(offset))
        stop("'y', 'weights', and 'offset' apply to a design matrix; a source or callback supplies its own",call.=FALSE)
    old.rule <- set_conv_rule(.covariance_rule(covariance))
    on.exit(set_conv_rule(old.rule), add = TRUE)
    source <- if (inherits(chunk_callback, "fastglm_source")) chunk_callback else
        fastglm_source(chunk_callback, n_chunks)
    if(!is.null(source$schema$terms) && !inherits(source$schema$terms,"terms"))
        stop("a two-component source requires fastglm_hurdle() or fastglm_twopart()",call.=FALSE)
    n_chunks <- source$n_chunks
    family <- .resolve_family(family)
    .check_iterations(tol,maxit)
    method <- .solver_method(method)
    if (!(method %in% c(0L,2L,3L))) stop("streaming requires method 'qr', 'llt', or 'ldlt'",call.=FALSE)
    fc <- family_code(family)
    if (firth && fc %in% c(0:2,10:16) && !is.null(family$dispersion) && is.finite(family$dispersion)) stop("mean bias reduction with externally fixed dispersion is not implemented for this family",call.=FALSE)
    if (firth && (fc < 0 || fc >= 20 || startsWith(family$family, "quasi")))
        stop("mean bias reduction requires a supported likelihood family",call.=FALSE)
    if (!is.logical(firth) || length(firth) != 1L || is.na(firth)) stop("'firth' must be TRUE or FALSE",call.=FALSE)
    state <- new.env(parent=emptyenv()); state$passes <- 0L; state$reads <- 0L
    state$columns <- NULL; state$p <- NULL; state$rows <- integer(n_chunks)
    state$initialization.warnings <- character()
    fetch <- .stream_reader(source,state,family,start)
    # One validation pass also constructs the usual family-initialized WLS start.
    a <- rhs <- low <- high <- qr.state <- NULL; nobs <- sumw <- sumwy <- 0; has.offset <- FALSE
    off.lo <- Inf; off.hi <- -Inf; sumwo <- 0
    gamma.null.total <- -Inf
    # Data-only terms of the log likelihood, so that the final likelihood
    # follows from the final deviance without another pass (see below).
    closed.loglik <- .stream_closed_loglik(family, firth)
    saturated <- 0; sum.log.w <- 0; sum.w.log.y <- 0
    # Response fingerprint for comparisons of fits (see .fastglm_fingerprint).
    fingerprint <- NULL
    for(k in seq_len(n_chunks)) {
        z <- fetch(k); x <- z$X; w <- z$weights
        fingerprint <- .fastglm_fingerprint_add(fingerprint, z$y, w)
        has.offset <- has.offset || any(z$offset != 0)
        if(any(w > 0)) { off.lo <- min(off.lo, z$offset[w > 0]); off.hi <- max(off.hi, z$offset[w > 0]) }
        sumwo <- sumwo + sum(w * z$offset)
        if(is.null(low)) {
            if(method!=0L) { a <- matrix(0,ncol(x),ncol(x)); rhs <- numeric(ncol(x)) }
            low <- rep(Inf,ncol(x)); high <- rep(-Inf,ncol(x))
        }
        low <- pmin(low,drop(colMin_dense(x))); high <- pmax(high,drop(colMax_dense(x)))
        eta <- family$linkfun(z$mustart); mu <- family$linkinv(eta)
        dm <- family$mu.eta(eta)
        ww <- if(fc==10L) w else w*dm^2/family$variance(mu)
        ww[w==0] <- 0
        zz <- eta-z$offset+(z$y-mu)/dm; zz[w==0] <- 0
        if(method==0L) qr.state <- streaming_qr_update(x,zz,ww,qr.state)
        else { a <- a+weighted_crossprod(x,ww); rhs <- rhs+drop(crossprod(x,ww*zz)) }
        nobs <- nobs+sum(w>0); sumw <- sumw+sum(w); sumwy <- sumwy+sum(w*z$y)
        if(fc==10L) gamma.null.total <- .gamma_log_null_total(z$y,w,z$offset,gamma.null.total)
        if(closed.loglik) {
            keep <- w>0
            if(family$family %in% c("binomial","poisson"))
                saturated <- saturated-.family_aic(family,z$y,z$trials,z$y,w,0)/2
            sum.log.w <- sum.log.w+sum(log(w[keep]))
            if(family$family %in% c("Gamma","inverse.gaussian"))
                sum.w.log.y <- sum.w.log.y+sum(w[keep]*log(z$y[keep]))
        }
    }
    p <- state$p
    if(nobs <= p) stop("streaming requires more positive-weight observations than coefficients",call.=FALSE)
    is.int <- low == high & low != 0
    # When the start is the weighted least squares fit from the usual starting
    # means, that solve is the first coefficient update (as in fastglm()).
    start.from.means <- is.null(start)
    if(method==0L) {
        initial.qr <- streaming_qr_finish(qr.state)
        if(is.null(start)) start <- initial.qr$coefficients
    } else if(is.null(start)) start <- tryCatch(drop(solve(a,rhs)),error=function(e) stop("streaming design is rank deficient",call.=FALSE))
    if(length(start)!=p || any(!is.finite(start))) stop("invalid starting coefficients",call.=FALSE)
    # Intercept-only fit as the fallback for an invalid starting step (as in
    # fastglm()), and the totals from this validation pass, which spare the
    # C++ fitter a pass of its own.
    anchor <- if (!has.offset) .feasible_start(matrix(low, 1L), sumwy / sumw, 1, 0, family,
        if(is.null(family$valideta)) function(x) TRUE else family$valideta,
        if(is.null(family$validmu)) function(x) TRUE else family$validmu, is.int) else
        .feasible_start_offset(low, is.int, sumwy / sumw, off.lo, off.hi, sumwo / sumw, family)
    if (!is.null(anchor) && all(anchor == 0)) anchor <- NULL
    # A Cholesky failure (working weights near the boundary of the valid
    # region can make X'WX ill-conditioned) is refit with the streaming QR
    # decomposition, as dense fits fall back to pivoted QR.
    qr.fallback <- FALSE
    res <- tryCatch(fit_streaming_glm(fetch,n_chunks,as.integer(p),method,tol,as.integer(maxit),fc,
        family$variance,family$mu.eta,family$linkinv,family$dev.resids,
        if(is.null(family$valideta)) function(x) TRUE else family$valideta,
        if(is.null(family$validmu)) function(x) TRUE else family$validmu,
        as.numeric(start),family_params(family),firth,pmax(abs(low),abs(high)),
        anchor, c(sumwy, sumw, nobs), as.integer(!firth && start.from.means)),
        error = function(e) {
            # Only for the default method: an explicitly requested Cholesky
            # method still reports the failure (e.g. a near-collinear design).
            if (!method.supplied && !firth && method %in% 2:3 &&
                grepl("Cholesky factorization", conditionMessage(e), fixed = TRUE)) {
                qr.fallback <<- TRUE; return(NULL)
            }
            stop(e)
        })
    if (qr.fallback) {
        return(fastglm_streaming(source, family = family, start = if (start.from.means) NULL else start,
                                 method = 0L, tol = tol, maxit = maxit, firth = firth, covariance = covariance,
                                 threads = threads, trace = tr$trace))
    }
    if(!firth && start.from.means) res$iter <- res$iter + 1L
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
    res$assign <- source$schema$assign
    res$call <- match.call(); res$firth <- firth; res$solver <- if(method==0L) "qr" else c("llt","ldlt")[method-1L]
    # Evaluate the likelihood using original binomial trial totals. Gaussian
    # and Gamma likelihood normalizers depend on global dispersion, so they
    # are accumulated explicitly rather than summing per-chunk AIC estimates.
    ll <- 0; null.dev <- 0
    # For the standard families the log likelihood is a function of the
    # deviance and of data-only totals from the validation pass, and the C++
    # fitter returns the null deviance and Pearson statistic of its last pass,
    # so no further pass over the data is needed:
    #   binomial, Poisson:  l = -D/2 + l(saturated)
    #   Gaussian:           l = -(n/2) (log(2 pi D/n) + 1) + (1/2) sum log w
    #   Gamma (a = sum w/D): l = sum w (a log a - lgamma a) + (a - 1) sum w log y
    #                            - a (D/2 + sum w log y + sum w)
    #   inverse Gaussian:   l = -(1/2) (sum w (1 + log(2 pi D/sum w)) + 3 sum w log y)
    # These equal the expressions of the final pass below; the Gamma form uses
    # D = 2 sum w (log mu - log y + y/mu - 1).
    closed <- closed.loglik && is.finite(res$deviance) && is.finite(res$null.deviance) &&
        is.finite(res$pearson) && is.finite(saturated)
    if(closed) {
        dev <- res$deviance; ff <- family$family
        ll <- switch(ff,
            binomial=, poisson= -dev/2+saturated,
            gaussian= -nobs/2*(log(2*pi*dev/nobs)+1)+sum.log.w/2,
            Gamma={a <- sumw/dev; sumw*(a*log(a)-lgamma(a))+(a-1)*sum.w.log.y-a*(dev/2+sum.w.log.y+sumw)},
            inverse.gaussian= -.5*(sumw*(1+log(2*pi*dev/sumw))+3*sum.w.log.y))
        null.dev <- res$null.deviance
        # A Gamma-log null model with offsets is not the weighted mean the C++
        # fitter uses; that case keeps the final pass.
        if(res$intercept && fc==10L && has.offset) { closed <- FALSE; ll <- 0; null.dev <- 0 }
    }
    if(!closed && .stable_binomial_code(res)>=0L) res$pearson <- 0
    if(!closed) for(k in seq_len(n_chunks)) {
        z <- fetch(k)
        eta <- if(method==0L) streaming_qr_eta(z$X,res$coefficients,z$offset) else
            drop(z$X%*%res$coefficients)+z$offset
        mu <- family$linkinv(eta)
        null.mu <- if(res$intercept && fc==10L) exp(z$offset+gamma.null.total-log(sumw)) else
            if(res$intercept) rep(sumwy/sumw,length(mu)) else family$linkinv(z$offset)
        null.dev <- null.dev+sum(family$dev.resids(z$y,null.mu,z$weights))
        w <- z$weights; y <- z$y; keep <- w>0; ff <- family$family
        if(ff == "gaussian") ll <- ll+sum(dnorm(y[keep],mu[keep],sqrt(res$deviance/nobs/w[keep]),log=TRUE))
        else if(ff == "Gamma") ll <- ll+sum(w[keep]*dgamma(y[keep],shape=sumw/res$deviance,scale=mu[keep]*res$deviance/sumw,log=TRUE))
        else if(ff == "inverse.gaussian") ll <- ll-.5*sum(w[keep]*(1+log(res$deviance/sumw*2*pi)+3*log(y[keep])))
        else if(.stable_binomial_code(res)>=0L) {
            moments <- binomial_firth_moments(as.numeric(eta),as.numeric(y),w,.stable_binomial_code(res))
            ll <- ll-.binomial_tail_aic(moments,y,z$trials,w)/2
            res$pearson <- res$pearson+sum(.binomial_tail_residual(moments,y,w,"pearson")^2)
        } else ll <- ll - .family_aic(family,y,z$trials,mu,w,sum(family$dev.resids(y,mu,w)))/2
    }
    if(family$family %in% c("gaussian","Gamma","inverse.gaussian")) res$aic <- -2*ll+2*(p+1L)
    else res$aic <- -2*ll+2*p
    # An intercept with offsets is a one-parameter GLM, not a weighted mean.
    res$null.deviance <- null.dev
    # At an all-zero/all-one binomial response the intercept-only ML supremum
    # is at infinity, with zero deviance even when finite offsets are present.
    boundary.null <- family$family=="binomial" && (sumwy==0 || sumwy==sumw)
    if(res$intercept && boundary.null) res$null.deviance <- 0
    if(res$intercept && has.offset && fc!=10L && !boundary.null) {
        nullfetch <- function(k) { z <- fetch(k); z$X <- matrix(1,length(z$y),1L); z }
        eta0 <- family$linkfun(sumwy/sumw)
        nf <- .fastglm_untraced(fit_streaming_glm(nullfetch,n_chunks,1L,if(method==0L) 0L else 2L,tol,as.integer(maxit),fc,
            family$variance,family$mu.eta,family$linkinv,family$dev.resids,
            if(is.null(family$valideta)) function(x) TRUE else family$valideta,
            if(is.null(family$validmu)) function(x) TRUE else family$validmu,
            as.numeric(eta0),family_params(family),FALSE))
        res$null.deviance <- nf$deviance
    }
    res$passes <- state$passes; res$chunk.reads <- state$reads
    res$response.fingerprint <- fingerprint
    res$compare.controls <- list(tol = tol, maxit = maxit, covariance = covariance[1L])
    res$threads <- threads
    if(!res$converged) warning("streaming GLM did not converge",call.=FALSE)
    if(identical(res$convergence.reason,"converged_boundary"))
        warning("streaming GLM: ",BOUNDARY_WARNING,call.=FALSE)
    class(res) <- c("fastglm_stream",if(firth) "fastglm_firth","fastglm")
    res
}

# The fit keeps this reader. A separate frame keeps fitting variables, including
# the last source block, out of the fitted object. Family initialization
# expressions can read `family` and `start` from this enclosing frame.
.stream_reader <- function(source,state,family,start) {
    force(source);force(state);force(family);force(start)
    initialize <- function(z) {
        x <- z$X; y <- z$y; nobs <- n <- NROW(y)
        if(inherits(x,"sparseMatrix")) x <- as.matrix(x)
        weights <- if(is.null(z$weights)) rep(1,nobs) else z$weights
        offset <- if(is.null(z$offset)) rep(0,nobs) else z$offset
        if (!is.matrix(x) || !is.numeric(x)) stop("each chunk$X must be a numeric matrix",call.=FALSE)
        if(is.integer(x)) storage.mode(x)<-"double"
        if (nobs == 0L) stop("empty chunks are not supported; omit them from the source",call.=FALSE)
        if (nrow(x) != nobs || !all_finite_dense(x) || anyNA(y) ||
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
    fetch
}

# TRUE when the log likelihood of a streamed fit follows from its deviance:
# the standard families with their unmodified likelihood (aic) functions,
# fitted by maximum likelihood.
.stream_closed_loglik <- function(family, firth) {
    if(firth || !family$family %in% c("binomial","poisson","gaussian","Gamma","inverse.gaussian"))
        return(FALSE)
    reference <- tryCatch(get(family$family,asNamespace("stats"))(link=family$link),error=function(e) NULL)
    !is.null(reference) && identical(deparse(family$aic),deparse(reference$aic)) &&
        identical(deparse(family$dev.resids),deparse(reference$dev.resids))
}

# A design matrix (dense, dgCMatrix, or file-backed big.matrix) as a source
# of row blocks.
.matrix_source <- function(x,y,weights,offset,block.size) {
    .check_iterations(1e-8,block.size)
    n <- nrow(x)
    if(is.null(y) || NROW(y)!=n) stop("supply 'y' with one entry per row of the design matrix",call.=FALSE)
    for(a in list(weights,offset)) if(!is.null(a) && length(a)!=n)
        stop("weights and offsets must have one entry per row of the design matrix",call.=FALSE)
    columns <- colnames(x)
    block.size <- as.integer(min(block.size,n))
    fastglm_source(function(k) {
        i <- seq.int((k-1L)*block.size+1L,min(n,k*block.size))
        X <- x[i,,drop=FALSE]
        if(!is.null(columns)) colnames(X) <- columns
        out <- list(X=X,y=if(is.matrix(y)) y[i,,drop=FALSE] else y[i])
        if(!is.null(weights)) out$weights <- weights[i]
        if(!is.null(offset)) out$offset <- offset[i]
        out
    },ceiling(n/block.size))
}
