# Per-row evaluation and covariance inference for vector GLM fits
# (class "fastglm_vglm"). Everything works from one per-row evaluation,
# .vglm_rows(), so later model types inherit it.

# Response pieces for the compiled evaluator: 1-based codes (individual) or a
# count matrix (grouped).
.vglm_response_parts <- function(y, levels) {
    if (is.matrix(y)) {
        counts <- unname(y)
        storage.mode(counts) <- "double"
        list(codes = integer(0), counts = counts)
    } else {
        codes <- match(as.character(y), levels)
        if (anyNA(codes)) stop("the response has values outside the levels of the fit", call. = FALSE)
        list(codes = as.integer(codes), counts = matrix(0, 0L, 0L))
    }
}

# The data a vglm evaluator works on: design x, response y, prior weights w,
# and offsets off. For the training data these are the stored ones; for newdata
# (formula fits only; newdata must contain the response) they are rebuilt, with
# weights from the fit's `weights` argument evaluated in newdata (default 1).
.vglm_eval_data <- function(object, newdata = NULL) {
    if (is.null(newdata) && .vglm_is_source(object))
        stop("a source fit does not retain its rows; its training data are evaluated block by block",
             call. = FALSE)
    if (is.null(newdata)) {
        x <- object$x
        .check_big_alive(x)
        # A sparse design is transposed once here, so that every block of rows
        # is a contiguous range of columns of xt.
        xt <- if (inherits(x, "dgCMatrix")) Matrix::t(x)
        return(list(x = x, y = object$y, w = object$prior.weights, off = object$offset,
                    xt = xt))
    }
    if (is.null(object$terms))
        stop("evaluation at newdata requires a formula fit", call. = FALSE)
    mf <- stats::model.frame(object$terms, newdata, na.action = stats::na.fail,
                             xlev = object$xlevels)
    y <- stats::model.response(mf)
    if (is.null(y)) stop("newdata must contain the response", call. = FALSE)
    d <- .vglm_prediction_design(object, newdata, NULL)
    n <- nrow(d$x)
    w <- rep(1, n)
    if (!is.null(object$call$weights)) {
        ext <- eval(object$call$weights, newdata, environment(object$terms))
        if (length(ext) != 1L && length(ext) != n)
            stop(sprintf(paste("the 'weights' argument of the fit evaluates to %d values in newdata,",
                               "which has %d rows; write the weights in terms of newdata columns"),
                         length(ext), n), call. = FALSE)
        w <- rep_len(as.numeric(ext), n)
    }
    if (is.logical(y) || is.character(y)) y <- factor(y, levels = object$levels)
    list(x = d$x, y = y, w = w, off = d$offset)
}

# Evaluate the model at its coefficients on the rows `i` of the data `dat`
# (a .vglm_eval_data list). Returns eta, u, W, P, loglik_rows, and estfun, the
# matrix whose row is X_i*' u_i with X_i* the row's design on the coefficient
# scale (every coefficient is a column; aliased ones are zero), plus weights,
# used (rows that enter the likelihood), and x (the block's design, dense). A
# dense design uses the dense compiled evaluator and a dgCMatrix design the
# sparse one; both return the same list for a block of rows, so the
# accumulators below work unchanged. For a sparse design i must be a
# contiguous range of rows: the evaluator maps that range of columns of the
# transposed design without subsetting it, so a pass over all blocks costs
# O(nnz) in all. For a streamed source, dat is list(block = z) with z a block
# from the fit's reader (.vglm_stream_reader), evaluated whole by the source
# evaluator; i is not used.
.vglm_eval_block <- function(object, dat, i = seq_len(nrow(dat$x))) {
    K <- length(object$levels)
    ref <- match(object$reference, object$levels)
    coefs <- unname(object$coefficients)
    if (!is.null(dat$block)) {
        # One normalized block of a streamed source, evaluated whole.
        z <- dat$block
        x <- z$X
        w <- z$weights
        r <- vglm_evaluate_source(z, object$constraints, object$family, as.integer(K),
                                  as.integer(ref), coefs, max(1L, nrow(x)))
        used <- w > 0
        if (length(z$counts)) used <- used & rowSums(z$counts) > 0
        return(.vglm_block_rows(object, r, x, w, used))
    }
    y <- if (is.matrix(dat$y)) dat$y[i, , drop = FALSE] else dat$y[i]
    w <- dat$w[i]
    parts <- .vglm_response_parts(y, object$levels)
    off <- dat$off[i, , drop = FALSE]
    if (inherits(dat$x, "big.matrix")) {
        .check_big_alive(dat$x)
        if (!length(i) || any(diff(i) != 1L))
            stop("big.matrix evaluation needs a contiguous, nonempty range of rows", call. = FALSE)
        # Only the block's rows are read into R; the compiled evaluator maps
        # the same rows of the backing memory without copying.
        x <- dat$x[i, , drop = FALSE]
        r <- vglm_evaluate_big(dat$x@address, as.integer(i[1L]), as.integer(i[length(i)]),
                               parts$codes, parts$counts, as.numeric(w), off,
                               object$constraints, object$family, as.integer(K),
                               as.integer(ref), coefs, max(1L, length(i)))
    } else if (inherits(dat$x, "dgCMatrix")) {
        if (!length(i) || any(diff(i) != 1L))
            stop("sparse evaluation needs a contiguous, nonempty range of rows", call. = FALSE)
        xt <- if (is.null(dat$xt)) Matrix::t(dat$x) else dat$xt
        first <- i[1L]; last <- i[length(i)]
        x <- .vglm_dense_rows(xt, first, last, colnames(dat$x))
        r <- vglm_evaluate_sparse(xt, as.integer(first), as.integer(last), parts$codes,
                                  parts$counts, as.numeric(w), off, object$constraints,
                                  object$family, as.integer(K), as.integer(ref), coefs,
                                  max(1L, length(i)))
    } else {
        x <- dat$x[i, , drop = FALSE]
        r <- vglm_evaluate_dense(x, parts$codes, parts$counts, as.numeric(w), off,
                                 object$constraints, object$family, as.integer(K),
                                 as.integer(ref), coefs, max(1L, length(i)))
    }
    used <- w > 0
    if (is.matrix(y)) used <- used & rowSums(y) > 0
    .vglm_block_rows(object, r, x, w, used)
}

# The evaluator's list for a block: r from a compiled evaluator, x the block's
# dense design, w its prior weights, and used its rows that enter the
# likelihood; adds estfun (see .vglm_eval_block).
.vglm_block_rows <- function(object, r, x, w, used) {
    coefs <- object$coefficients
    estfun <- matrix(0, nrow(x), length(coefs))
    pos <- 0L
    for (k in seq_len(ncol(x))) {
        H <- object$constraints[[k]]
        idx <- pos + seq_len(ncol(H))
        estfun[, idx] <- (r$u %*% H) * x[, k]
        pos <- pos + ncol(H)
    }
    estfun[, is.na(object$coefficients)] <- 0
    colnames(estfun) <- names(object$coefficients)
    c(r[c("eta", "u", "W", "P", "loglik_rows")],
      list(estfun = estfun, weights = w, used = used, x = x))
}

# Rows first, ..., last of a sparse design as a dense matrix, read directly
# from the slots of its transpose xt (a p by n dgCMatrix): the cost is the
# nonzeros of those rows plus the size of the dense block, independent of n.
.vglm_dense_rows <- function(xt, first, last, names = NULL) {
    b <- last - first + 1L
    out <- matrix(0, b, nrow(xt), dimnames = list(NULL, names))
    from <- xt@p[first] + 1L
    to <- xt@p[last + 1L]
    if (to >= from) {
        idx <- from:to
        row <- rep.int(seq_len(b), diff(xt@p[first:(last + 1L)]))
        out[cbind(row, xt@i[idx] + 1L)] <- xt@x[idx]
    }
    out
}

# Per-row evaluation of the whole data (training data when newdata is NULL).
.vglm_rows <- function(object, newdata = NULL, block.size = 8192L) {
    if (!inherits(object, "fastglm_vglm"))
        stop("a fastglm vector GLM fit is required", call. = FALSE)
    .vglm_eval_block(object, .vglm_eval_data(object, newdata))
}

# Visit the training rows in blocks of at most block.size rows, calling
# fun(rows, index) with the evaluator's output for each block. Accumulators
# need only this iterator, so evaluators for other representations supply their
# own and reuse them.
.vglm_for_blocks <- function(object, fun, block.size = 8192L) {
    if (.vglm_is_source(object)) {
        # A streamed source: each of its blocks is read once and evaluated
        # whole, with index the rows' positions in source order.
        last <- 0L
        for (k in seq_len(object$source$n_chunks)) {
            z <- object$fetch(k)
            i <- last + seq_len(nrow(z$X))
            fun(.vglm_eval_block(object, list(block = z)), i)
            last <- last + nrow(z$X)
        }
        return(invisible(NULL))
    }
    dat <- .vglm_eval_data(object)
    n <- nrow(dat$x)
    for (first in seq.int(1L, n, by = block.size)) {
        i <- first:min(n, first + block.size - 1L)
        fun(.vglm_eval_block(object, dat, i), i)
    }
    invisible(NULL)
}

# Whether a fit is over a streamed source (its rows are not retained).
.vglm_is_source <- function(object) !is.null(object$source)

# The number of rows of the training data (for a source, the rows read).
.vglm_nrows <- function(object)
    if (.vglm_is_source(object)) object$n.rows else nrow(object$x)

# Cluster identifiers: a vector over the retained rows, or over the rows
# before NA omission.
.vglm_cluster_vector <- function(cluster, object, n) {
    if (is.null(cluster)) return(NULL)
    if (!is.atomic(cluster) || !is.null(dim(cluster)) || is.complex(cluster))
        stop("'cluster' must be a vector of identifiers", call. = FALSE)
    omitted <- as.integer(object$na.action)
    if (length(cluster) != n) {
        if (!length(omitted) || length(cluster) != n + length(omitted) ||
            anyNA(omitted) || any(omitted < 1L | omitted > length(cluster)) ||
            anyDuplicated(omitted))
            stop("cluster identifiers must match the retained rows, or the rows before NA omission after any subset",
                 call. = FALSE)
        cluster <- cluster[-omitted]
    }
    # Integer keys keep numeric labels that differ beyond printed precision distinct.
    match(cluster, unique(cluster[!is.na(cluster)]))
}

# Sandwich covariance, accumulated block by block: only the P by P cross
# product (or the per-cluster score sums) is kept, never the n by P matrix.
.vglm_robust_vcov <- function(object, type, cluster, cadjust, block.size) {
    type <- match.arg(type, c("HC0", "HC1", "HC2", "HC3"))
    if (type %in% c("HC2", "HC3"))
        stop(type, " covariance is not available for vector GLM fits; use HC0 or HC1", call. = FALSE)
    if (!is.logical(cadjust) || length(cadjust) != 1L || is.na(cadjust))
        stop("'cadjust' must be TRUE or FALSE", call. = FALSE)
    if (!is.numeric(block.size) || length(block.size) != 1L || !is.finite(block.size) || block.size < 1)
        stop("'block.size' must be a positive integer", call. = FALSE)
    block.size <- as.integer(block.size)
    .check_big_alive(object$x)
    if (isTRUE(object$firth))
        stop("sandwich inference for adjusted scores is not implemented", call. = FALSE)
    if (is.null(object$cov.unscaled)) stop("covariance information was not retained", call. = FALSE)
    good <- !is.na(object$coefficients)
    n <- object$nobs; P <- sum(good)
    if (type == "HC1" && n <= P)
        stop("HC1 requires more positive-weight observations than estimated parameters", call. = FALSE)
    cluster <- .vglm_cluster_vector(cluster, object, .vglm_nrows(object))
    meat <- matrix(0, P, P)
    sums <- if (!is.null(cluster)) new.env(hash = TRUE, parent = emptyenv())
    .vglm_for_blocks(object, function(rows, i) {
        est <- rows$estfun[rows$used, good, drop = FALSE]
        if (is.null(cluster)) {
            meat <<- meat + crossprod(est)
        } else {
            g <- cluster[i][rows$used]
            if (anyNA(g))
                stop("cluster identifiers are missing for rows that enter the likelihood", call. = FALSE)
            s <- rowsum(est, as.character(g), reorder = FALSE)
            for (j in seq_len(nrow(s))) {
                key <- rownames(s)[j]
                v <- s[j, ]
                if (!is.null(old <- sums[[key]])) v <- v + old
                sums[[key]] <- v
            }
        }
    }, block.size)
    if (is.null(cluster)) {
        if (type == "HC1") meat <- meat * n / (n - P)
    } else {
        keys <- ls(sums, all.names = TRUE)
        G <- length(keys)
        if (G < 2L) stop("cluster covariance requires at least two clusters", call. = FALSE)
        for (key in keys) meat <- meat + tcrossprod(sums[[key]])
        if (cadjust) meat <- meat * G / (G - 1)
        if (type == "HC1") meat <- meat * (n - 1) / (n - P)
    }
    bread <- object$cov.unscaled[good, good, drop = FALSE]
    v <- bread %*% meat %*% bread
    nm <- names(object$coefficients)
    v <- (v + t(v)) / 2
    out <- matrix(NA_real_, length(good), length(good), dimnames = list(nm, nm))
    out[good, good] <- v
    attr(out, "covariance.type") <- type
    out
}

# Diagnostics from the ordinary score at the estimate (before any bias
# adjustment) and the covariance.
.vglm_diagnostics <- function(object) {
    good <- !is.na(object$coefficients)
    nm <- names(object$coefficients)[good]
    score <- numeric(sum(good))
    .vglm_for_blocks(object, function(rows, i)
        score <<- score + colSums(rows$estfun[, good, drop = FALSE]))
    info <- diag(object$information)[good]
    v <- object$cov.unscaled[good, good, drop = FALSE]
    reason <- object$convergence.reason
    if (!length(reason) || !nzchar(reason)) reason <- if (isTRUE(object$converged)) "converged" else "not_converged"
    names(score) <- nm
    list(converged = isTRUE(object$converged), iterations = object$iter,
         convergence.reason = reason, rank = object$rank,
         aliased = names(object$coefficients)[!good],
         score = score,
         score.type = if (isTRUE(object$firth)) "ordinary (before bias adjustment)" else "ordinary",
         ordinary.score.available = TRUE,
         standardized.score = if (length(score)) max(abs(score / sqrt(info))) else 0,
         covariance.condition = if (length(v) && all(is.finite(v))) kappa(v, exact = TRUE) else NA_real_,
         boundary = isTRUE(object$separation))
}

# ---- Averages, marginal effects, and bootstrap helpers ---------------------

# The training data of a formula fit, recovered from the call when the caller
# supplies no newdata: the data argument (or, without one, the formula's
# variables in its environment, as model.frame() finds them), then subset and
# NA omission. The result must have one row per row of the stored design.
.vglm_training_data <- function(object) {
    env <- environment(object$terms)
    dat <- if (is.null(object$call$data)) {
        tryCatch(stats::get_all_vars(object$terms, data = NULL), error = function(e) NULL)
    } else tryCatch(eval(object$call$data, env), error = function(e) NULL)
    if (is.data.frame(dat)) {
        if (!is.null(object$call$subset))
            dat <- dat[eval(object$call$subset, dat, env), , drop = FALSE]
        if (length(object$na.action))
            dat <- dat[-as.integer(object$na.action), , drop = FALSE]
    }
    if (!is.data.frame(dat) || nrow(dat) != nrow(object$x))
        stop("the fitting data cannot be recovered from the call; supply 'newdata'", call. = FALSE)
    dat
}

# Dense rows i of a prediction design d (from .vglm_prediction_design), with
# columns in the fitted order; xt is the transpose of a sparse design.
.vglm_design_rows <- function(d, i, xt = NULL) {
    x <- d$x
    if (!is.null(xt)) return(.vglm_dense_rows(xt, i[1L], i[length(i)], colnames(x)))
    as.matrix(x[i, if (is.null(d$cols)) seq_len(ncol(x)) else d$cols, drop = FALSE])
}

# Probabilities (b by K) for a dense design block x, term-major coefficients
# with aliased entries zero, and the b by M offset block off.
.vglm_block_probs <- function(object, x, off) {
    M <- length(object$levels) - 1L
    ref <- match(object$reference, object$levels)
    B <- object$coefficients
    B[is.na(B)] <- 0
    full <- matrix(0, nrow(x), M + 1L)
    full[, -ref] <- x %*% t(matrix(B, M)) + off
    pr <- exp(full - apply(full, 1L, max))
    pr / rowSums(pr)
}

# Gradient of sum_i w_i pi_ik with respect to the coefficients (term-major),
# as a (P M) by K matrix: d pi_k / d eta_j = pi_k (1[k = j] - pi_j).
.vglm_prob_gradient <- function(object, x, pr, w) {
    K <- length(object$levels); M <- K - 1L
    ref <- match(object$reference, object$levels)
    out <- matrix(0, ncol(x) * M, K)
    for (k in seq_len(K)) {
        D <- -pr[, k] * pr[, -ref, drop = FALSE]
        if (k != ref) D[, match(k, seq_len(K)[-ref])] <- D[, match(k, seq_len(K)[-ref])] + pr[, k]
        out[, k] <- as.vector(crossprod(w * D, x))
    }
    out
}

# Estimates, standard errors, and intervals from an estimate vector and its
# gradient matrix (one column per category).
.vglm_effect_summary <- function(object, estimate, gradient, vcov., level) {
    if (is.null(vcov.)) vcov. <- object$cov.unscaled
    if (!is.matrix(vcov.) || !identical(dim(vcov.), rep(length(object$coefficients), 2L)))
        stop("'vcov.' must be the full coefficient covariance matrix of the fit", call. = FALSE)
    v <- colSums(gradient * (vcov. %*% gradient))
    if (any(!is.finite(v)) || any(v < -1e-10 * (1 + max(abs(v)))))
        stop("invalid covariance for the requested estimate", call. = FALSE)
    se <- sqrt(pmax(0, v))
    q <- stats::qnorm((1 + level) / 2)
    data.frame(category = object$levels, estimate = estimate, std.error = se,
               conf.low = estimate - q * se, conf.high = estimate + q * se)
}

.vglm_check_weights <- function(w, n) {
    if (!is.numeric(w) || length(w) != n || any(!is.finite(w) | w < 0))
        stop("invalid averaging weights", call. = FALSE)
    w
}

# Average predicted probabilities (or, with a contrast, the average change)
# for a multinomial fit; see fastglm_average().
.vglm_average <- function(object, newdata, contrast, weights, vcov., level, type) {
    type <- match.arg(type, c("response", "probability"))
    if (!is.null(contrast) && !is.function(contrast)) stop("'contrast' must be a function", call. = FALSE)
    P <- length(object$coefficients)
    K <- length(object$levels)
    total <- 0; estimate <- numeric(K); gradient <- matrix(0, P, K)
    add <- function(x, off, w, xc) {
        if (!length(w)) return(invisible(NULL))
        pr <- .vglm_block_probs(object, x, off)
        g <- .vglm_prob_gradient(object, x, pr, w)
        value <- pr
        if (!is.null(xc)) {
            if (!identical(dim(xc), dim(x))) stop("contrast must preserve design dimensions", call. = FALSE)
            pc <- .vglm_block_probs(object, xc, off)
            g <- .vglm_prob_gradient(object, xc, pc, w) - g
            value <- pc - pr
        }
        if (any(!is.finite(value))) stop("average prediction requires finite rows", call. = FALSE)
        total <<- total + sum(w)
        estimate <<- estimate + colSums(w * value)
        gradient <<- gradient + g
    }
    if (.vglm_is_source(object) && is.null(newdata)) {
        model <- object; model$source <- NULL; model$terms <- NULL
        for (k in seq_len(object$source$n_chunks)) {
            z <- object$fetch(k)
            d <- .vglm_prediction_design(model, z$X, z$offset)
            x <- as.matrix(d$x)
            w <- if (is.function(weights)) weights(k) else rep(1, nrow(x))
            if (length(w) != nrow(x) || any(!is.finite(w) | w < 0))
                stop("invalid averaging weights", call. = FALSE)
            add(x, d$offset, w, if (!is.null(contrast)) contrast(x))
        }
    } else {
        if (is.function(weights)) stop("'weights' must be a vector for in-memory fits", call. = FALSE)
        d <- .vglm_prediction_design(object, newdata, NULL)
        n <- nrow(d$x)
        w <- if (is.null(weights)) rep(1, n) else .vglm_check_weights(weights, n)
        dense <- is.matrix(d$x) && is.null(d$cols)
        # A dense design is contrasted whole, so contrasts that depend on all
        # rows are preserved; sparse and big.matrix designs are contrasted block by block.
        xx <- if (dense && !is.null(contrast)) contrast(d$x)
        if (!is.null(xx) && !identical(dim(xx), dim(d$x)))
            stop("contrast must preserve design dimensions", call. = FALSE)
        xt <- if (inherits(d$x, "dgCMatrix")) Matrix::t(d$x)
        for (first in seq.int(1L, max(n, 1L), by = 8192L)) {
            if (n == 0L) break
            i <- first:min(n, first + 8192L - 1L)
            x <- .vglm_design_rows(d, i, xt)
            xc <- if (is.null(contrast)) NULL else if (dense) xx[i, , drop = FALSE] else contrast(x)
            keep <- w[i] > 0
            add(x[keep, , drop = FALSE], d$offset[i[keep], , drop = FALSE], w[i][keep],
                if (!is.null(xc)) xc[keep, , drop = FALSE])
        }
    }
    if (total <= 0) stop("averaging weights must have positive total", call. = FALSE)
    .vglm_effect_summary(object, estimate / total, gradient / total, vcov., level)
}

# Average marginal effects of one continuous predictor on every category
# probability, by centered differences of each row's probabilities; see
# fastglm_marginal_effects().
.vglm_marginal_effects <- function(object, variable, newdata, weights, vcov., level, type,
                                   eps, offset) {
    if (!is.character(variable) || length(variable) != 1L || is.na(variable) || !nzchar(variable))
        stop("'variable' must name one continuous numeric predictor", call. = FALSE)
    if (!is.numeric(level) || length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1)
        stop("'level' must lie strictly between zero and one", call. = FALSE)
    if (!is.null(eps) && (!is.numeric(eps) || length(eps) != 1L || !is.finite(eps) || eps <= 0))
        stop("'eps' must be a positive finite number", call. = FALSE)
    type <- match.arg(type, c("response", "probability"))
    if (!isTRUE(object$converged)) stop("marginal effects require a converged model", call. = FALSE)
    if (anyNA(object$coefficients)) stop("marginal effects require a full-rank fit", call. = FALSE)
    if (.vglm_is_source(object) && is.null(newdata))
        stop("marginal effects for a source fit require 'newdata'", call. = FALSE)
    formula.fit <- !is.null(object$terms) && (is.null(newdata) || is.data.frame(newdata))
    if (formula.fit) {
        data <- if (is.null(newdata)) .vglm_training_data(object) else newdata
        x <- .marginal_formula_variable(list(object$terms), data, variable)
    }
    base <- .vglm_prediction_design(object, if (is.null(newdata)) NULL else newdata, offset)
    n <- nrow(base$x)
    if (formula.fit && nrow(data) != n) stop("the data and design have different numbers of rows", call. = FALSE)
    cn <- .vglm_design_names(object)
    if (!formula.fit) {
        col <- match(variable, cn)
        if (is.na(col)) stop("'variable' must name a fitted design column", call. = FALSE)
    }
    w <- if (is.null(weights)) rep(1, n) else .vglm_check_weights(weights, n)
    total <- sum(w)
    if (!is.finite(total) || total <= 0) stop("averaging weights must have positive finite total", call. = FALSE)
    K <- length(object$levels)
    estimate <- numeric(K); gradient <- matrix(0, length(object$coefficients), K)
    xt <- if (inherits(base$x, "dgCMatrix")) Matrix::t(base$x)
    for (first in seq.int(1L, max(n, 1L), by = 8192L)) {
        if (n == 0L) break
        i <- first:min(n, first + 8192L - 1L)
        keep <- w[i] > 0
        if (!any(keep)) next
        xi <- .vglm_design_rows(base, i, xt)[keep, , drop = FALSE]
        off <- base$offset[i[keep], , drop = FALSE]
        wi <- w[i][keep] / total
        v <- if (formula.fit) x[i[keep]] else xi[, col]
        h <- if (is.null(eps)) .Machine$double.eps^(1/3) * pmax(1, abs(v)) else rep(eps, length(v))
        if (formula.fit) {
            rows <- data[i[keep], , drop = FALSE]
            up <- lo <- rows
            up[[variable]] <- v + h; lo[[variable]] <- v - h
            width <- up[[variable]] - lo[[variable]]
            xh <- as.matrix(.vglm_formula_design(object, up))
            xl <- as.matrix(.vglm_formula_design(object, lo))
        } else {
            xh <- xl <- xi
            xh[, col] <- v + h; xl[, col] <- v - h
            width <- (v + h) - (v - h)
        }
        if (any(!is.finite(width) | width <= 0))
            stop("'eps' cannot produce finite distinct predictor values", call. = FALSE)
        ph <- .vglm_block_probs(object, xh, off)
        pl <- .vglm_block_probs(object, xl, off)
        value <- (ph - pl) / width
        g <- .vglm_prob_gradient(object, xh, ph, wi / width) - .vglm_prob_gradient(object, xl, pl, wi / width)
        if (any(!is.finite(value)) || any(!is.finite(g)))
            stop("marginal effect or parameter derivative is nonfinite; check transformation domains and 'eps'",
                 call. = FALSE)
        estimate <- estimate + colSums(wi * value)
        gradient <- gradient + g
    }
    out <- .vglm_effect_summary(object, estimate, gradient, vcov., level)
    cbind(variable = variable, out)
}

# The design matrix of a formula fit for the rows of a data frame, without
# evaluating any offset (the marginal-effect offsets come from the base design).
.vglm_formula_design <- function(object, rows) {
    tt <- stats::delete.response(object$terms)
    mf <- stats::model.frame(tt, rows, na.action = stats::na.pass, xlev = object$xlevels)
    stats::model.matrix(tt, mf, contrasts.arg = object$contrasts)
}

# Bootstrap input and refit for multinomial fits. The refit uses the
# tolerances and controls stored on the fit. Dense and sparse designs are
# subset to the drawn rows; memory-mapped designs and sources keep every row
# and multiply its prior weight by the number of times it was drawn.
.vglm_bootstrap_input <- function(object) {
    if (!is.null(object$source)) {
        rows <- environment(object$fetch)$state$rows
        return(list(rows = rows, n = sum(rows)))
    }
    if (is.null(object$y)) stop("bootstrap requires a retained response", call. = FALSE)
    x <- object$x
    n <- nrow(x)
    off <- object$offset
    if (is.null(off)) off <- matrix(0, n, length(object$levels) - 1L)
    list(data = list(X = x, y = object$y, weights = object$prior.weights, offset = off,
                     vglm = TRUE), n = n)
}

.vglm_bootstrap_refit <- function(object, z, control, source = NULL) {
    rc <- object$refit.controls
    if (is.null(rc)) rc <- list(tol = 1e-8, maxit = 100L, block.size = 8192L, control = list())
    args <- list(reference = object$reference, firth = isTRUE(object$firth),
                 tol = rc$tol, maxit = rc$maxit, block.size = rc$block.size,
                 control = rc$control)
    args <- if (is.null(source))
        c(list(x = z$X, y = z$y, weights = z$weights, offset = z$offset), args)
    else c(list(x = source, levels = object$levels), args)
    fit <- do.call(fastglm_multinom, utils::modifyList(args, control))
    for (key in c("terms", "xlevels", "contrasts")) fit[[key]] <- object[[key]]
    fit$call <- object$call
    fit$na.action <- NULL
    fit
}

# A source refit reads the fit's own blocks in order on every pass (so
# sequential sources stay valid), with each row's weight multiplied by the
# number of times it was drawn.
.vglm_bootstrap_stream_refit <- function(object, input, index, control) {
    frequencies <- tabulate(index, nbins = input$n)
    starts <- c(0L, cumsum(input$rows))
    lev <- object$levels
    callback <- function(k) {
        z <- object$fetch(k)
        b <- nrow(z$X)
        y <- if (length(z$codes)) factor(lev[z$codes], levels = lev)
        else `colnames<-`(z$counts, lev)
        list(X = z$X, y = y, weights = z$weights * frequencies[starts[k] + seq_len(b)],
             offset = z$offset)
    }
    .vglm_bootstrap_refit(object, NULL, control,
                          source = fastglm_source(callback, length(input$rows)))
}
