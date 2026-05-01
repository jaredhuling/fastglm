#' Fit a GLM by streaming row-blocks of the design matrix
#'
#' `fastglm_streaming()` runs an IRLS GLM fit where the design matrix is
#' produced one chunk at a time by a user-supplied closure. It never holds
#' more than a single chunk in memory at once, plus a `p x p` accumulator.
#' This is the front-end for fitting on data sources too large to load
#' completely into RAM (Arrow datasets, Parquet files, DuckDB query results,
#' on-disk CSV streams, etc.).
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
#'   closure is called multiple times per IRLS iteration, so it should be
#'   reasonably cheap (e.g. an Arrow scanner that reads from columnar files).
#' @param n_chunks integer; the number of chunks to iterate over.
#' @param family a `family` object describing the error distribution and link.
#'   See [stats::family()].
#' @param start optional length-`p` numeric vector of starting coefficients.
#' @param method integer; `2` for LLT Cholesky (default) or `3` for LDLT
#'   Cholesky. QR / SVD methods are not supported in streaming mode.
#' @param tol convergence tolerance on the relative change in deviance.
#' @param maxit maximum number of IRLS iterations.
#'
#' @returns A list with class `"fastglm"` containing the same elements as
#'   [fastglm()], including `coefficients`, `cov.unscaled`, `deviance`,
#'   `iter`, `converged`, etc. The design matrix is *not* attached, so
#'   `sandwich::vcovHC()` / `sandwich::vcovCL()` will require re-streaming.
#'
#' @details
#' The IRLS loop and step-halving (Marschner 2011) run entirely in C++; the
#' R closure is called only to *deliver* one chunk at a time. For tier-1
#' families (gaussian / binomial / poisson / Gamma / inverse.gaussian on
#' their common links) the family functions are evaluated inline in C++, so
#' the only R round-trip per iteration is the chunk fetch.
#'
#' Standard errors and `cov.unscaled` come from the final `(X' W X)`
#' Cholesky factor, exactly as in the in-memory path.
#'
#' @section Arrow / Parquet recipe:
#'
#' Wrap an Arrow scanner in a closure -- pull each `RecordBatch` as a chunk,
#' build the model matrix, and return it together with the response:
#' `chunks(k)` opens the dataset, scans the `k`-th batch, calls
#' `as.data.frame()`, then returns
#' `list(X = model.matrix(~ x1 + x2, data = tbl), y = tbl$y)`. Pass that
#' closure plus `n_chunks = length(batches)` to `fastglm_streaming()`.
#'
#' The same recipe works for DuckDB (one `dbReadTable` per chunk), CSV
#' streamers, and custom binary formats.
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
fastglm_streaming <- function(chunk_callback,
                              n_chunks,
                              family   = gaussian(),
                              start    = NULL,
                              method   = 2L,
                              tol      = 1e-7,
                              maxit    = 100L)
{
    if (!is.function(chunk_callback))
        stop("'chunk_callback' must be a function", call. = FALSE)
    if (!is.numeric(n_chunks) || length(n_chunks) != 1L || n_chunks < 1)
        stop("'n_chunks' must be a positive integer", call. = FALSE)
    n_chunks <- as.integer(n_chunks)
    method <- as.integer(method)
    if (!(method %in% c(2L, 3L)))
        stop("for streaming fits, 'method' must be 2 (LLT) or 3 (LDLT).", call. = FALSE)

    if (is.character(family))   family <- get(family, mode = "function", envir = parent.frame())
    if (is.function(family))    family <- family()
    if (is.null(family$family)) stop("'family' not recognized", call. = FALSE)

    fam_code <- family_code(family)
    fam_par  <- family_params(family)

    # Pull one chunk to discover p and column names, and to do up-front
    # shape validation.  The C++ solver re-pulls all chunks (including this
    # one) on each pass.
    c1 <- chunk_callback(1L)
    if (!is.list(c1) || is.null(c1$X) || is.null(c1$y))
        stop("chunk_callback(1) did not return a list with 'X' and 'y'.",
             call. = FALSE)
    if (!is.matrix(c1$X) && !inherits(c1$X, "Matrix"))
        stop("chunk$X must be a matrix", call. = FALSE)
    if (nrow(c1$X) != length(c1$y))
        stop(sprintf("chunk$X has %d rows but length(y) = %d",
                     nrow(c1$X), length(c1$y)), call. = FALSE)
    p <- ncol(c1$X)
    cnames <- colnames(c1$X)
    rm(c1)

    # Default start = 0_p.
    if (is.null(start)) {
        start_vec <- rep(0, p)
    } else {
        if (length(start) != p)
            stop(sprintf("length(start) = %d does not match number of columns (%d)",
                         length(start), p), call. = FALSE)
        start_vec <- as.numeric(start)
    }

    # R callbacks are only used when fam_code == -1.  For tier-1 families the
    # C++ path never invokes them, but Rcpp still requires Function values.
    valideta <- if (is.null(family$valideta)) function(eta) TRUE else family$valideta
    validmu  <- if (is.null(family$validmu))  function(mu)  TRUE else family$validmu

    res <- fit_streaming_glm(
        chunk_callback = chunk_callback,
        n_chunks       = n_chunks,
        p              = as.integer(p),
        type           = method,
        tol            = tol,
        maxit          = as.integer(maxit),
        fam_code       = as.integer(fam_code),
        var            = family$variance,
        mu_eta         = family$mu.eta,
        linkinv        = family$linkinv,
        dev_resids     = family$dev.resids,
        valideta       = valideta,
        validmu        = validmu,
        start          = start_vec,
        fam_params     = fam_par
    )

    # C++ always reports Pearson-based dispersion; override to 1 for
    # poisson / binomial.  Quasi-binomial / quasi-poisson share C++ family
    # codes with binomial / poisson but keep the estimated dispersion.
    # NB families also use the Pearson estimate (same convention as
    # glm() + MASS::negative.binomial in summary()).
    dispersion <- res$dispersion
    is_fixed_disp <- family$family %in% c("poisson", "binomial")
    if (is_fixed_disp) dispersion <- 1
    if (!identical(dispersion, res$dispersion)) {
        if (is.finite(res$dispersion) && res$dispersion > 0)
            res$se <- res$se / sqrt(res$dispersion)
        if (is.finite(dispersion) && dispersion > 0)
            res$se <- res$se * sqrt(dispersion)
    }

    has_intercept <- isTRUE(res$has_intercept)
    intercept_mask <- as.logical(res$intercept_mask)

    # Names
    if (is.null(cnames)) {
        if (has_intercept) {
            ix <- which(intercept_mask)
            cnames <- character(p)
            cnames[ix]  <- "(Intercept)"
            cnames[-ix] <- paste0("X", seq_len(p - length(ix)))
        } else {
            cnames <- paste0("X", seq_len(p))
        }
    }
    beta <- as.numeric(res$coefficients); names(beta) <- cnames
    se   <- as.numeric(res$se);           names(se)   <- cnames
    cov_unscaled <- res$cov.unscaled
    rownames(cov_unscaled) <- colnames(cov_unscaled) <- cnames

    out <- list(
        coefficients      = beta,
        se                = se,
        cov.unscaled      = cov_unscaled,
        fitted.values     = NULL,
        linear.predictors = NULL,
        deviance          = res$deviance,
        null.deviance     = res$null.deviance,
        df.null           = res$df.null,
        df.residual       = res$df.residual,
        rank              = res$rank,
        iter              = res$iter,
        converged         = res$converged,
        family            = family,
        dispersion        = dispersion,
        n                 = res$n,
        intercept         = has_intercept,
        prior.weights     = NULL,
        y                 = NULL,
        x                 = NULL,
        call              = match.call()
    )
    class(out) <- "fastglm"
    out
}
