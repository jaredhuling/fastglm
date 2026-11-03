fastglm_po_test <-
function (object, terms = NULL, null.estimation = c("ML", "meanBR"), 
    newdata = NULL, y = NULL, weights = NULL, offset = NULL, 
    control = list(), by.term = FALSE) 
{
    if (!inherits(object, "fastglm_ordinal")) 
        stop("supply a fastglm_ordinal fit", call. = FALSE)
    null.estimation <- match.arg(null.estimation)
    if (!is.logical(by.term) || length(by.term) != 1L || is.na(by.term)) 
        stop("'by.term' must be TRUE or FALSE", call. = FALSE)
    training <- is.null(newdata) && is.null(y)
    if (!is.null(newdata) && is.null(y)) 
        stop("supply assessment outcomes 'y' with 'newdata'", 
            call. = FALSE)
    if (is.null(y)) 
        y <- object$y
    if (training && is.null(weights)) 
        weights <- object$weights
    design <- .ordinal_prediction_design(object, newdata, offset)
    response <- .ordinal_assessment_response(y, weights, object$levels, 
        nrow(design$x))
    included <- !is.na(response$totals) & response$totals > 0 & 
        !response$missing.response & !is.na(response$weights)
    .design_blocks(design$x, function(z, index) {
        off <- design$offset[index]
        if (any(is.infinite(z)) || any(is.infinite(off))) 
            stop("ordinal test predictors and offsets must be finite or missing", 
                call. = FALSE)
        included[index] <<- included[index] & rowSums(is.na(z)) == 
            0L & !is.na(off)
    }, object$block.size)
    result <- .ordinal_assessment_po_test(object, design, response, 
        included, terms, control, null.estimation, by.term)
    data.name <- paste(deparse(substitute(object)), collapse = " ")
    if (by.term) 
        result$tests <- lapply(result$tests, function(z) {
            z$data.name <- data.name
            z
        })
    else result$data.name <- data.name
    result
}
.ordinal_assessment_po_test <-
function (object, design, response, included, terms, control, 
    estimation = "ML", by.term = FALSE) 
{
    selected <- .ordinal_po_test_columns(object, terms)
    ctl <- .ordinal_po_test_control(control)
    K <- length(object$levels)
    p <- length(object$beta)
    out <- structure(list(method = if (.ordinal_link(object) == 
        "logit") "Expected-information proportional-odds score test" else paste0("Expected-information common-slope score test (", 
        .ordinal_link(object), ")"), available = FALSE, statistic = NA_real_, 
        df = length(selected$index) * (K - 2L), p.value = NA_real_, 
        null.estimation = estimation, terms = selected$terms, 
        columns = selected$columns, observations = sum(included), 
        frequency = sum(response$totals[included]), reason = "", 
        detail = "", null.converged = FALSE, null.iterations = NA_integer_, 
        information.rcond = NA_real_, empty.categories = character(), 
        warning = character(), separation = NULL, parameter = c(df = length(selected$index) * 
            (K - 2L)), data.name = "ordinal assessment outcomes", 
        note = if (estimation == "ML") "First-order chi-square approximation." else "Ordinary efficient score at a mean-bias-reduced null; first-order chi-square approximation."), 
        class = c("fastglm_po_test", "fastglm_test", "htest"))
    selections <- list(Joint = selected)
    if (by.term) 
        for (term in selected$terms) selections[[length(selections) + 
            1L]] <- .ordinal_po_test_columns(object, term)
    if (by.term) 
        names(selections) <- make.unique(c("Joint", selected$terms))
    template <- function(selection) {
        z <- out
        z$terms <- selection$terms
        z$columns <- selection$columns
        z$df <- length(selection$index) * (K - 2L)
        z$parameter <- c(df = z$df)
        z
    }
    collect <- function(results) if (!by.term) 
        results[[1L]]
    else structure(list(tests = results, method = out$method, 
        null.estimation = estimation, note = paste(out$note, 
            "Term-specific p-values are unadjusted for multiple testing.")), 
        class = "fastglm_test_list")
    unavailable <- function(reason, detail) {
        collect(lapply(selections, function(selection) {
            z <- template(selection)
            z$reason <- reason
            z$detail <- detail
            z
        }))
    }
    if (K < 3L || !p) 
        return(unavailable("no_slope_contrasts", "At least three categories and one predictor are required."))
    if (!any(included)) 
        return(unavailable("no_usable_outcomes", "No positive-frequency complete assessment rows are available."))
    if (!is.matrix(design$x) || !is.numeric(design$x)) 
        return(unavailable("unsupported_design", "The score test currently requires a dense numeric design."))
    index <- which(included)
    xx <- design$x[index, , drop = FALSE]
    off <- design$offset[index]
    yy <- if (response$grouped) 
        response$y[index, , drop = FALSE]
    else response$codes[index]
    rr <- .ordinal_assessment_response(yy, response$weights[index], 
        object$levels, length(index))
    totals <- numeric(K)
    for (first in seq.int(1L, length(index), by = object$block.size)) {
        at <- seq.int(first, min(length(index), first + object$block.size - 
            1L))
        totals <- totals + colSums(.ordinal_assessment_counts(rr, 
            at, K))
    }
    out$empty.categories <- object$levels[totals == 0]
    if (length(out$empty.categories)) 
        return(unavailable("empty_categories", paste0("Every declared category must have positive frequency; empty: ", 
            paste(out$empty.categories, collapse = ", "), ".")))
    constant <- vapply(seq_len(p), function(j) all(xx[, j] == 
        xx[1L, j]), logical(1))
    if (any(constant)) 
        return(unavailable("rank_deficient_design", paste0("No predictor variation in the assessment rows: ", 
            paste(colnames(xx)[constant], collapse = ", "), ". The requested model is not identifiable.")))
    out$separation <- .ordinal_po_separation(xx, rr, K)
    if (estimation == "ML" && !is.null(out$separation)) {
        sep <- out$separation
        out$null.iterations <- 0L
        return(unavailable("null_separation", paste0("Predictor '", 
            sep$column, "' at ", format(sep$value, trim = TRUE), 
            " occurs only in endpoint category '", object$levels[sep$category], 
            "'. The ML null has no finite solution. A separately requested meanBR null may be used; its test remains an approximation.")))
    }
    warning <- character()
    error <- ""
    fit <- tryCatch(withCallingHandlers(.ordinal_fit_core(xx, 
        yy, rr$weights, off, object$levels, firth = estimation == 
            "meanBR", tol = ctl$tol, maxit = ctl$maxit, block.size = object$block.size, 
        link = .ordinal_link(object)), warning = function(w) {
        warning <<- c(warning, conditionMessage(w))
        invokeRestart("muffleWarning")
    }), error = function(e) {
        error <<- conditionMessage(e)
        NULL
    })
    out$warning <- unique(warning)
    if (is.null(fit)) 
        return(unavailable(if (grepl("rank|constant column", 
            error)) "rank_deficient_design" else "null_fit_error", 
            error))
    out$null.converged <- isTRUE(fit$converged)
    out$null.iterations <- fit$iterations
    if (!out$null.converged) 
        return(unavailable("null_fit_not_converged", paste0("The ", 
            estimation, " common-slope null did not converge: ", 
            gsub("_", " ", fit$convergence.reason, fixed = TRUE), 
            ".")))
    moments <- tryCatch(.ordinal_po_score_moments(xx, rr, off, 
        fit, selected$index, object$block.size), error = function(e) {
        error <<- conditionMessage(e)
        NULL
    })
    if (is.null(moments)) 
        return(unavailable("information_unavailable", error))
    collect(lapply(selections, function(selection) {
        z <- template(selection)
        positions <- match(selection$index, selected$index)
        tested <- moments$nuisance + as.vector(outer(positions, 
            seq.int(0L, K - 3L) * length(selected$index), "+"))
        result <- tryCatch(.ordinal_po_score_statistic(moments, 
            tested), error = function(e) {
            error <<- conditionMessage(e)
            NULL
        })
        if (is.null(result)) {
            z$reason <- "information_unavailable"
            z$detail <- error
            return(z)
        }
        z$available <- TRUE
        z$statistic <- result$statistic
        z$p.value <- pchisq(z$statistic, z$df, lower.tail = FALSE)
        z$information.rcond <- result$condition
        z$reason <- "available"
        z
    }))
}
.ordinal_po_expected_score <-
function (x, response, offset, fit, selected, block.size) 
{
    .ordinal_po_score_statistic(.ordinal_po_score_moments(x, 
        response, offset, fit, selected, block.size))
}
.ordinal_po_score_moments <-
function (x, response, offset, fit, selected, block.size) 
{
    n <- nrow(x)
    p <- ncol(x)
    q <- length(fit$alpha)
    s <- length(selected)
    d0 <- q + p
    d <- d0 + s * (q - 1L)
    frequency <- fit$frequency
    alpha <- fit$working$alpha
    beta <- fit$working$beta
    gap <- diff(alpha)
    if (any(!is.finite(gap) | gap <= 0)) 
        stop("the null fit requires finite, strictly ordered thresholds", 
            call. = FALSE)
    tgap <- exp(-gap)/(-expm1(-gap))
    U <- numeric(d)
    I <- matrix(0, d, d)
    ia <- seq_len(q)
    ib <- q + seq_len(p)
    ig <- function(j) d0 + (j - 2L) * s + seq_len(s)
    for (first in seq.int(1L, n, by = block.size)) {
        at <- seq.int(first, min(n, first + block.size - 1L))
        xx <- .ordinal_prediction_scale(x[at, , drop = FALSE], 
            fit$working$center, fit$working$scale)
        zz <- xx[, selected, drop = FALSE]
        eta <- outer(-streaming_qr_eta(xx, beta, offset[at]), 
            alpha, "+")
        logistic <- .ordinal_link(fit) == "logit"
        if (logistic) {
            F <- plogis(eta)
            S <- plogis(-eta)
            g <- F * S
        }
        else {
            kernel <- .ordinal_link_kernel(alpha, streaming_qr_eta(xx, 
                beta, offset[at]), .ordinal_link(fit), scores = TRUE)
            F <- kernel$cumulative
            S <- kernel$survival
            g <- kernel$derivative
        }
        counts <- .ordinal_assessment_counts(response, at, q + 
            1L)/frequency
        mass <- response$totals[at]/frequency
        threshold <- diagonal <- rowsum <- matrix(0, length(at), 
            q)
        adjacent <- matrix(0, length(at), q - 1L)
        location <- if (logistic) 
            numeric(length(at))
        else -rowSums(counts * kernel$translation)
        probability <- if (logistic) 
            cbind(F[, 1L], F[, -1L, drop = FALSE] * S[, -q, drop = FALSE] * 
                rep(-expm1(-gap), each = length(at)), S[, q])
        else kernel$probability
        if (any(!is.finite(probability) | probability <= 0)) 
            stop("the fitted null has a category probability outside the representable positive range", 
                call. = FALSE)
        for (j in seq_len(q)) {
            left <- if (logistic) 
                S[, j] + if (j > 1L) 
                  tgap[j - 1L]
                else 0
            else kernel$left[, j]
            right <- if (logistic) 
                F[, j] + if (j < q) 
                  tgap[j]
                else 0
            else kernel$right[, j]
            threshold[, j] <- counts[, j] * left - counts[, j + 
                1L] * right
            diagonal[, j] <- mass * g[, j] * (left + right)
            rowsum[, j] <- mass * g[, j] * (if (logistic) 
                probability[, j] + probability[, j + 1L]
            else kernel$translation.difference[, j])
            if (j < q) 
                adjacent[, j] <- -mass * g[, j + 1L] * right
            if (logistic) 
                location <- location + counts[, j + 1L] * F[, 
                  j] - counts[, j] * S[, j]
        }
        U[ia] <- U[ia] + colSums(threshold)
        U[ib] <- U[ib] + drop(crossprod(xx, location))
        I[cbind(ia, ia)] <- I[cbind(ia, ia)] + colSums(diagonal)
        I[cbind(ia[-q], ia[-1L])] <- I[cbind(ia[-q], ia[-1L])] + 
            colSums(adjacent)
        I[ia, ib] <- I[ia, ib] - crossprod(rowsum, xx)
        I[ib, ib] <- I[ib, ib] + crossprod(xx, xx * rowSums(rowsum))
        dg <- crossprod(diagonal, zz)
        ad <- crossprod(adjacent, zz)
        for (j in 2:q) {
            index <- ig(j)
            U[index] <- U[index] - drop(crossprod(zz, threshold[, 
                j]))
            I[j, index] <- I[j, index] - dg[j, ]
            I[j - 1L, index] <- I[j - 1L, index] - ad[j - 1L, 
                ]
            if (j < q) 
                I[j + 1L, index] <- I[j + 1L, index] - ad[j, 
                  ]
            I[ib, index] <- I[ib, index] + crossprod(xx, zz * 
                rowsum[, j])
            I[index, index] <- I[index, index] + crossprod(zz, 
                zz * diagonal[, j])
            if (j < q) 
                I[index, ig(j + 1L)] <- I[index, ig(j + 1L)] + 
                  crossprod(zz, zz * adjacent[, j])
        }
    }
    I[lower.tri(I)] <- t(I)[lower.tri(I)]
    if (any(!is.finite(U)) || any(!is.finite(I))) 
        stop("score or expected information is nonfinite", call. = FALSE)
    list(score = U, information = I, nuisance = d0, frequency = frequency)
}
.ordinal_po_score_statistic <-
function (moments, tested = NULL) 
{
    d0 <- moments$nuisance
    if (is.null(tested)) 
        tested <- seq.int(d0 + 1L, length(moments$score))
    keep <- c(seq_len(d0), tested)
    U <- moments$score[keep]
    I <- moments$information[keep, keep, drop = FALSE]
    d <- length(U)
    frequency <- moments$frequency
    scale <- sqrt(diag(I))
    if (any(!is.finite(scale) | scale <= 0)) 
        stop("expected information has a nonpositive diagonal", 
            call. = FALSE)
    I <- I/outer(scale, scale)
    U <- U/scale
    nuisance <- seq_len(d0)
    tested <- seq.int(d0 + 1L, d)
    A <- I[nuisance, nuisance, drop = FALSE]
    C <- I[tested, nuisance, drop = FALSE]
    root <- chol(A)
    projection <- t(backsolve(root, forwardsolve(t(root), t(C))))
    e <- drop(U[tested] - projection %*% U[nuisance])
    J <- I[tested, tested, drop = FALSE] - projection %*% t(C)
    J <- (J + t(J))/2
    js <- sqrt(diag(J))
    if (any(!is.finite(js) | js <= 0)) 
        stop("efficient expected information is not positive definite", 
            call. = FALSE)
    J <- J/outer(js, js)
    e <- e/js
    root <- chol(J)
    condition <- min(rcond(A), rcond(J))
    if (!is.finite(condition) || condition < 1e-14) 
        stop("expected information is numerically singular", 
            call. = FALSE)
    statistic <- frequency * sum(forwardsolve(t(root), e)^2)
    if (!is.finite(statistic)) 
        stop("the score statistic is nonfinite", call. = FALSE)
    list(statistic = statistic, condition = condition)
}
