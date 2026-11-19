#' Methods for vector GLM fits
#'
#' Standard methods for objects returned by [fastglm_multinom()]: printing and
#' summaries, extraction of coefficients, covariance, fitted values and
#' residuals, Wald confidence intervals, and prediction.
#' @param object,x a fitted `fastglm_vglm` object (or, for `print`, a summary).
#' @param matrix if `TRUE`, `coef()` returns a `K - 1` by `p` matrix whose rows
#'   are the non-reference categories and whose columns are the design columns,
#'   the layout of `coef()` for `nnet::multinom()`.
#' @param parm coefficients, by name or position; defaults to all.
#' @param level confidence level.
#' @param type for `residuals()`, `"response"` (observed minus fitted
#'   proportions) or `"pearson"` (the response residual divided by
#'   `sqrt(p (1 - p) / m)`, with `m` the row's number of trials: the row total
#'   times its weight for grouped data, and for individual data one, with the
#'   result multiplied by the square root of the prior weight, as in
#'   [stats::glm()]; a row with no trials or zero weight has residual zero). For `predict()`, `"probability"`
#'   (an `n` by `K` matrix), `"class"` (the most probable category, a factor
#'   with the levels of the fit), or `"link"` (the `n` by `K - 1` matrix of
#'   log odds against the reference category, including offsets).
#' @param newdata for formula fits a data frame (the response is not needed and
#'   factors may have a subset of the training levels); for matrix fits a
#'   numeric design matrix or a `dgCMatrix` with the training columns. If
#'   `NULL`, the training design is used.
#' @param offset an optional offset for `newdata`, a vector or an `n` by
#'   `K - 1` matrix. For formula fits without this argument, `offset()` terms
#'   in the formula and the `offset` argument of the fit are evaluated from
#'   `newdata`; an explicit `offset` replaces them (it is not added to them),
#'   as elsewhere in the package. For matrix fits with a nonzero offset the
#'   offset must be supplied again here.
#' @param sink for a fit over a [fastglm_source()] with `newdata = NULL`, an
#'   optional function of `(value, chunk_index)` that receives the prediction
#'   for each source block, as for [predict.fastglm_stream()]; `predict()` then
#'   returns the number of blocks invisibly and collects nothing. Without a
#'   sink the blocks are combined in source order (without row names).
#' @param se.fit if `TRUE`, also return delta-method standard errors, from the
#'   expected-information covariance, for probabilities or log odds. Standard
#'   errors are not available for `type = "class"`.
#' @param digits number of significant digits to display.
#' @param ... further arguments (not used, except that `print` and `summary`
#'   pass them to the coefficient printer).
#' @returns `predict()` returns a matrix or factor, or with `se.fit = TRUE` a
#'   list with components `fit` and `se.fit`. Other methods return the
#'   conventional objects. The `summary()` coefficient table has one row per
#'   coefficient, with Wald z statistics; its Akaike and Bayesian information
#'   criteria use the rank of the fit as the number of parameters.
#' @name fastglm-vglm-methods
#' @md
NULL

# Wald standard errors; NA for aliased coefficients.
.vglm_se <- function(object) {
    v <- object$cov.unscaled
    se <- sqrt(diag(v))
    names(se) <- names(object$coefficients)
    se
}

.vglm_title <- function(object) {
    if (identical(object$family, "multinomial")) "Multinomial logit regression"
    else "Vector GLM regression"
}

#' @rdname fastglm-vglm-methods
#' @export
coef.fastglm_vglm <- function(object, matrix = FALSE, ...) {
    b <- object$coefficients
    if (!matrix) return(b)
    M <- length(object$levels) - 1L
    out <- base::matrix(b, M, dimnames = list(setdiff(object$levels, object$reference),
                                              .vglm_design_names(object)))
    out
}

#' @rdname fastglm-vglm-methods
#' @export
vcov.fastglm_vglm <- function(object, ...) object$cov.unscaled

#' @rdname fastglm-vglm-methods
#' @export
confint.fastglm_vglm <- function(object, parm, level = 0.95, ...) {
    if (length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1)
        stop("'level' must lie between zero and one", call. = FALSE)
    b <- object$coefficients
    if (missing(parm)) parm <- seq_along(b)
    else if (is.character(parm)) {
        i <- match(parm, names(b))
        if (anyNA(i)) stop("unknown coefficient name in 'parm'", call. = FALSE)
        parm <- i
    }
    if (!is.numeric(parm) || anyNA(parm) || any(parm < 1 | parm > length(b)))
        stop("invalid 'parm'", call. = FALSE)
    a <- (1 - level) / 2
    q <- qnorm(1 - a)
    se <- .vglm_se(object)
    out <- cbind(b[parm] - q * se[parm], b[parm] + q * se[parm])
    dimnames(out) <- list(names(b)[parm],
                          paste(format(100 * c(a, 1 - a), trim = TRUE, digits = 3), "%"))
    out
}

#' @rdname fastglm-vglm-methods
#' @export
logLik.fastglm_vglm <- function(object, ...)
    structure(object$loglik, df = object$rank, nobs = object$nobs,
              class = "logLik")

#' @rdname fastglm-vglm-methods
#' @export
nobs.fastglm_vglm <- function(object, ...) object$nobs

#' @rdname fastglm-vglm-methods
#' @export
deviance.fastglm_vglm <- function(object, ...) object$deviance

#' @rdname fastglm-vglm-methods
#' @export
fitted.fastglm_vglm <- function(object, ...) object$fitted.values

#' @rdname fastglm-vglm-methods
#' @export
model.matrix.fastglm_vglm <- function(object, ...) object$x

#' @rdname fastglm-vglm-methods
#' @export
residuals.fastglm_vglm <- function(object, type = c("response", "pearson"), ...) {
    type <- match.arg(type)
    if (.vglm_is_source(object))
        stop("residuals are not available for a source fit, which does not retain its rows",
             call. = FALSE)
    p <- object$fitted.values
    y <- object$y
    if (is.matrix(y)) {
        tot <- rowSums(y)
        prop <- y / ifelse(tot > 0, tot, 1)
        prop[tot == 0, ] <- p[tot == 0, ]
        m <- tot * object$prior.weights
    } else {
        prop <- outer(as.integer(y), seq_along(object$levels), "==") * 1
        m <- rep(1, nrow(p))
    }
    w <- object$prior.weights
    dimnames(prop) <- dimnames(p)
    r <- prop - p
    if (type == "pearson") {
        r <- r / sqrt(p * (1 - p) / ifelse(m > 0, m, 1))
        # Individual data: scale by sqrt(prior weight), as glm does (grouped
        # data carry the weight in m already).
        if (!is.matrix(y)) r <- r * sqrt(w)
        r[!is.finite(r)] <- 0
        r[m == 0, ] <- 0
    }
    r
}

#' @rdname fastglm-vglm-methods
#' @export
print.fastglm_vglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(.vglm_title(x), x$call)
    .fastglm_print_field("Estimation", if (isTRUE(x$firth)) "mean bias reduction" else "maximum likelihood")
    .fastglm_print_field("Reference category", x$reference)
    cat("\nCoefficients (rows: non-reference categories):\n")
    print(format(coef(x, matrix = TRUE), digits = digits), quote = FALSE, right = TRUE)
    cat("\n")
    .vglm_print_fit_line(x, digits)
    .fastglm_print_convergence(x$converged, x$iter, x$convergence.reason)
    invisible(x)
}

.vglm_print_fit_line <- function(x, digits) {
    cat("Residual deviance: ", format(x$deviance, digits = digits),
        " on ", x$df.residual, " degrees of freedom\n", sep = "")
}

#' @rdname fastglm-vglm-methods
#' @export
summary.fastglm_vglm <- function(object, ...) {
    b <- object$coefficients
    se <- .vglm_se(object)
    z <- b / se
    tab <- cbind(Estimate = b, "Std. Error" = se, "z value" = z,
                 "Pr(>|z|)" = 2 * pnorm(-abs(z)))
    structure(list(call = object$call, coefficients = tab, family = object$family,
                   levels = object$levels, reference = object$reference,
                   loglik = object$loglik, deviance = object$deviance,
                   null.deviance = object$null.deviance,
                   df.residual = object$df.residual, nobs = object$nobs,
                   rank = object$rank, AIC = AIC(object), BIC = BIC(object),
                   iter = object$iter, converged = object$converged,
                   convergence.reason = object$convergence.reason,
                   firth = object$firth, aliased = object$aliased,
                   separation = object$separation, singular = isTRUE(object$singular)),
              class = c(paste0("summary.", class(object)[1L]), "summary.fastglm_vglm"))
}

#' @rdname fastglm-vglm-methods
#' @export
print.summary.fastglm_vglm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
    digits <- .fastglm_print_digits(digits)
    .fastglm_print_header(.vglm_title(x), x$call)
    .fastglm_print_field("Estimation", if (isTRUE(x$firth)) "mean bias reduction" else "maximum likelihood")
    .fastglm_print_field("Reference category", x$reference)
    .fastglm_print_coefficients(x$coefficients, "Coefficients", digits, ...)
    if (any(x$aliased)) cat("\n(", sum(x$aliased), " coefficient(s) not defined because of singularities)\n", sep = "")
    cat("\n")
    cat("Null deviance:     ", format(x$null.deviance, digits = digits), "\n", sep = "")
    .vglm_print_fit_line(x, digits)
    cat("Log likelihood: ", format(x$loglik, digits = digits), " (", x$rank, " parameters, ",
        x$nobs, " observations)\n", sep = "")
    cat("AIC: ", format(x$AIC, digits = digits), "  BIC: ", format(x$BIC, digits = digits), "\n", sep = "")
    .fastglm_print_convergence(x$converged, x$iter, x$convergence.reason)
    if (isTRUE(x$separation) && !isTRUE(x$firth))
        cat("Note: fitted probabilities are numerically 0 or 1, so the Wald standard errors are\n",
            "unreliable; firth = TRUE gives finite estimates.\n", sep = "")
    if (isTRUE(x$singular))
        cat("Note: the information matrix is numerically singular, so the covariance is a\n",
            "pseudoinverse and standard errors in the singular direction are not reliable.\n", sep = "")
    invisible(x)
}

# Design matrix and offsets for prediction.
.vglm_prediction_design <- function(object, newdata, offset) {
    M <- length(object$levels) - 1L
    if (is.null(newdata)) {
        x <- object$x
        .check_big_alive(x)
        off <- object$offset
    } else if (!is.null(object$terms)) {
        if (!is.data.frame(newdata) && !is.list(newdata))
            stop("'newdata' must be a data frame for a formula fit", call. = FALSE)
        tt <- stats::delete.response(object$terms)
        mf <- stats::model.frame(tt, newdata, na.action = stats::na.pass,
                                 xlev = object$xlevels)
        x <- stats::model.matrix(tt, mf, contrasts.arg = object$contrasts)
        off <- stats::model.offset(mf)
        if (is.null(offset) && length(object$source.offset)) {
            # A source fit's offset columns are read from newdata, as for
            # fastglm_streaming() fits.
            miss <- setdiff(object$source.offset, names(newdata))
            if (length(miss))
                stop("newdata lacks the source offset column(s) ", paste(miss, collapse = ", "),
                     call. = FALSE)
            ext <- matrix(vapply(object$source.offset, function(nm) as.numeric(newdata[[nm]]),
                                 numeric(nrow(x))), nrow(x))
            # One offset column applies to every linear predictor.
            if (ncol(ext) == 1L) ext <- ext[, 1L]
            off <- if (is.null(off)) ext else ext + off
        }
        if (is.null(offset) && !is.null(object$call$offset)) {
            ext <- eval(object$call$offset, newdata, environment(object$terms))
            # An offset written against the fitting data (for example data$e)
            # evaluates to the fitting rows; recycling it would be silently wrong.
            if (NROW(ext) != 1L && NROW(ext) != nrow(x))
                stop(sprintf(paste("the 'offset' argument of the fit evaluates to %d rows in newdata,",
                                   "which has %d; write the offset in terms of newdata columns",
                                   "(for example log(exposure))"), NROW(ext), nrow(x)), call. = FALSE)
            if (is.matrix(ext)) {
                if (ncol(ext) != M)
                    stop("the 'offset' argument of the fit must be a vector or an n by K - 1 matrix",
                         call. = FALSE)
                off <- if (is.null(off)) ext else ext + off
            } else off <- if (is.null(off)) ext else off + ext
        }
    } else {
        x <- newdata
        if (is.data.frame(x)) x <- as.matrix(x)
        off <- NULL
        if (is.null(offset) && (any(object$offset != 0) || isTRUE(object$source.offset.used)))
            stop("the fit used a nonzero offset, which cannot be replayed for a design matrix; supply 'offset' for newdata",
                 call. = FALSE)
    }
    sparse <- inherits(x, "dgCMatrix")
    big <- inherits(x, "big.matrix")
    .check_big_alive(x)
    if (!sparse && !big && (!is.matrix(x) || !is.numeric(x)))
        stop("'newdata' must be a numeric design matrix, a dgCMatrix, or a big.matrix for a matrix fit",
             call. = FALSE)
    cols <- NULL
    cn <- .vglm_design_names(object)
    if (ncol(x) != length(cn))
        stop("the prediction design has the wrong number of columns (expected ",
             length(cn), ")", call. = FALSE)
    if (!is.null(colnames(x)) && !identical(colnames(x), cn)) {
        if (!setequal(colnames(x), cn) || anyDuplicated(colnames(x)))
            stop("the prediction design columns do not match the fitted design columns",
                 call. = FALSE)
        # A big.matrix is not reordered (that would copy it); the columns are
        # selected by name when each block is read.
        if (big) cols <- match(cn, colnames(x)) else x <- x[, cn, drop = FALSE]
    }
    if (!sparse && !big) storage.mode(x) <- "double"
    n <- nrow(x)
    if (!is.null(offset)) {
        if (!is.numeric(offset)) stop("'offset' must be numeric", call. = FALSE)
        off <- offset
    }
    if (is.null(off)) off <- 0
    if (is.matrix(off)) {
        if (nrow(off) != n || ncol(off) != M)
            stop("'offset' must be a vector of length n or an n by K - 1 matrix", call. = FALSE)
    } else {
        if (length(off) == 1L) off <- rep(off, n)
        if (length(off) != n)
            stop("'offset' must be a vector of length n or an n by K - 1 matrix", call. = FALSE)
        off <- matrix(as.numeric(off), n, M)
    }
    list(x = x, offset = off, cols = cols)
}

#' @rdname fastglm-vglm-methods
#' @export
predict.fastglm_vglm <- function(object, newdata = NULL,
                                 type = c("probability", "class", "link"),
                                 se.fit = FALSE, offset = NULL, sink = NULL, ...) {
    type <- match.arg(type)
    if (!is.logical(se.fit) || length(se.fit) != 1L || is.na(se.fit))
        stop("'se.fit' must be TRUE or FALSE", call. = FALSE)
    if (se.fit && type == "class")
        stop("'se.fit' is not available for type = \"class\"", call. = FALSE)
    if (!is.null(sink) && !is.function(sink)) stop("'sink' must be a function", call. = FALSE)
    if (is.null(newdata) && .vglm_is_source(object))
        return(.vglm_source_predict(object, type, se.fit, offset, sink))
    if (!is.null(sink))
        stop("'sink' is used when predicting the rows of a source fit (newdata = NULL)", call. = FALSE)
    K <- length(object$levels); M <- K - 1L
    ref <- match(object$reference, object$levels)
    nonref <- object$levels[-ref]
    d <- .vglm_prediction_design(object, newdata, offset)
    x <- d$x; n <- nrow(x)
    B <- object$coefficients
    B[is.na(B)] <- 0
    eta <- .vglm_linear(x, B, M, d$cols) + d$offset
    rn <- if (inherits(x, "big.matrix")) NULL else rownames(x)
    if (is.null(rn)) rn <- as.character(seq_len(n))
    dimnames(eta) <- list(rn, nonref)
    full <- matrix(0, n, K)
    full[, -ref] <- eta
    pr <- exp(full - apply(full, 1L, max))
    pr <- pr / rowSums(pr)
    dimnames(pr) <- list(rn, object$levels)
    if (type == "class") {
        cls <- factor(object$levels[max.col(pr, ties.method = "first")],
                      levels = object$levels)
        names(cls) <- rn
        return(cls)
    }
    fit <- if (type == "link") eta else pr
    if (!se.fit) return(fit)
    if (n == 0L) {
        se <- matrix(NA_real_, 0L, ncol(fit), dimnames = dimnames(fit))
        return(list(fit = fit, se.fit = se))
    }

    # Delta method. Coefficient index is term-major: (term c, predictor j) is
    # at (c - 1) * M + j, so the gradient of a function g(eta_1, ..., eta_M)
    # is kronecker(x, dg/deta).
    ok <- !is.na(object$coefficients)
    V <- object$cov.unscaled[ok, ok, drop = FALSE]
    keep <- which(ok)
    se <- matrix(NA_real_, n, if (type == "link") M else K)
    dimnames(se) <- dimnames(fit)
    block <- max(1L, floor(2e6 / max(1L, length(keep))))
    # A sparse design is transposed once and each block of rows is read as a
    # dense block from a contiguous range of its columns.
    xt <- if (inherits(x, "dgCMatrix")) Matrix::t(x)
    for (first in seq.int(1L, n, by = block)) {
        i <- first:min(n, first + block - 1L)
        xi <- if (is.null(xt)) x[i, if (is.null(d$cols)) seq_len(ncol(x)) else d$cols, drop = FALSE] else
            .vglm_dense_rows(xt, first, i[length(i)], colnames(x))
        if (type == "link") {
            for (j in seq_len(M)) {
                G <- xi[, rep(seq_len(ncol(xi)), each = M) , drop = FALSE]
                G[, rep(seq_len(M), ncol(xi)) != j] <- 0
                G <- G[, keep, drop = FALSE]
                se[i, j] <- sqrt(rowSums((G %*% V) * G))
            }
        } else {
            for (k in seq_len(K)) {
                # d p_k / d eta_j = p_k (1[k = j] - p_j) over non-reference j.
                pk <- pr[i, k]
                D <- -pk * pr[i, -ref, drop = FALSE]
                if (k != ref) D[, match(k, seq_len(K)[-ref])] <- D[, match(k, seq_len(K)[-ref])] + pk
                G <- xi[, rep(seq_len(ncol(xi)), each = M), drop = FALSE] *
                    D[, rep(seq_len(M), ncol(xi)), drop = FALSE]
                G <- G[, keep, drop = FALSE]
                se[i, k] <- sqrt(pmax(rowSums((G %*% V) * G), 0))
            }
        }
    }
    list(fit = fit, se.fit = se)
}

# Prediction for the training rows of a source fit: the source is reread
# block by block and each block is predicted from its design and recorded
# offsets, as predict() does for fastglm_streaming() fits. With a sink, each
# block's result is passed to sink(value, k) and nothing is collected;
# otherwise the blocks are combined in source order, without row names.
.vglm_source_predict <- function(object, type, se.fit, offset, sink) {
    if (!is.null(offset))
        stop("prediction for the source rows uses the offsets recorded in the source; supply newdata to change them",
             call. = FALSE)
    model <- object
    model$source <- NULL
    model$terms <- NULL
    n <- object$source$n_chunks
    out <- if (is.null(sink)) vector("list", n)
    for (k in seq_len(n)) {
        z <- object$fetch(k)
        value <- predict.fastglm_vglm(model, z$X, type = type, se.fit = se.fit, offset = z$offset)
        if (type == "class") names(value) <- NULL
        else if (se.fit) { rownames(value$fit) <- NULL; rownames(value$se.fit) <- NULL }
        else rownames(value) <- NULL
        if (is.null(sink)) out[[k]] <- value else sink(value, k)
    }
    if (!is.null(sink)) return(invisible(n))
    if (type == "class")
        return(factor(unlist(lapply(out, as.character), use.names = FALSE), levels = object$levels))
    if (!se.fit) return(do.call(rbind, out))
    list(fit = do.call(rbind, lapply(out, `[[`, "fit")),
         se.fit = do.call(rbind, lapply(out, `[[`, "se.fit")))
}
