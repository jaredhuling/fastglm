# Pure-R reference implementation of multinomial logit Fisher scoring, with
# optional mean bias reduction. It is the independent oracle that the compiled
# vector GLM engine is tested against, so it favors clarity over speed. It
# loops only over the K - 1 linear predictors, never over rows.
#
# Notation (design document, Sections 5 and 6). There are M = K - 1 linear
# predictors, one per non-reference category, and the coefficients are ordered
# term-major: all M coefficients of design column 1, then those of column 2,
# and so on. With H_k = I_M for every column, X_i* = I_M (x) x_i' and
#   U = sum_i X_i*' u_i,   I = sum_i X_i*' W_i X_i*,
# where u_i and W_i are the (prior-weighted) row score and expected
# information with respect to eta_i.

.vglm_reference_fit <- function(x, counts, weights = NULL, offset = NULL,
                                reference = 1L, firth = FALSE,
                                tol = 1e-12, maxit = 200L) {
    x <- as.matrix(x)
    counts <- as.matrix(counts)
    n <- nrow(x)
    p <- ncol(x)
    K <- ncol(counts)
    M <- K - 1L
    if (nrow(counts) != n) stop("'x' and 'counts' must have the same number of rows")
    if (K < 2L) stop("'counts' must have at least two columns")
    if (is.null(weights)) weights <- rep(1, n)
    if (is.null(offset)) offset <- matrix(0, n, M)
    offset <- matrix(offset, n, M)
    reference <- as.integer(reference)
    lev <- colnames(counts)
    if (is.null(lev)) lev <- as.character(seq_len(K))
    nonref <- setdiff(seq_len(K), reference)
    cnr <- counts[, nonref, drop = FALSE]     # counts of non-reference categories
    tot <- rowSums(counts)
    mw <- weights * tot                       # weighted number of trials per row
    P <- p * M
    idx <- function(a) a + M * (seq_len(p) - 1L)   # positions of predictor a

    # Fitted non-reference probabilities for a coefficient vector.
    probs <- function(beta) {
        eta <- x %*% t(matrix(beta, M, p)) + offset
        mx <- pmax(0, apply(eta, 1, max))
        e <- exp(eta - mx)
        den <- exp(-mx) + rowSums(e)
        list(pi = e / den, eta = eta, lden = log(den) + mx)
    }
    loglik_of <- function(beta) {
        pr <- probs(beta)
        # log pi_nonref = eta - log(den); log pi_ref = -log(den)
        ll <- weights * (rowSums(cnr * pr$eta) - tot * pr$lden)
        sum(ll)
    }
    score_info <- function(beta) {
        pi <- probs(beta)$pi
        u <- weights * cnr - mw * pi                       # n by M
        U <- as.vector(t(crossprod(x, u)))                  # term-major
        Imat <- matrix(0, P, P)
        Wab <- vector("list", M * M)
        for (a in seq_len(M)) for (b in seq_len(M)) {
            W <- mw * ((a == b) * pi[, a] - pi[, a] * pi[, b])
            Wab[[(b - 1L) * M + a]] <- W
            Imat[idx(a), idx(b)] <- crossprod(x, W * x)
        }
        list(pi = pi, U = U, I = Imat, Wab = Wab)
    }
    penalized <- function(beta) {
        ll <- loglik_of(beta)
        if (!firth) return(ll)
        ld <- determinant(score_info(beta)$I, logarithm = TRUE)
        ll + 0.5 * as.numeric(ld$modulus)
    }
    # Adjusted score U + sum_i X_i*' t_i, with
    # t_i[m] = 1/2 sum_{a,b} A_i[a,b] dW_i[a,b]/d eta_m, A_i = X_i* I^-1 X_i*'.
    bias_term <- function(s) {
        Iinv <- solve(s$I)
        A <- vector("list", M * M)
        for (a in seq_len(M)) for (b in seq_len(M))
            A[[(b - 1L) * M + a]] <- rowSums((x %*% Iinv[idx(a), idx(b), drop = FALSE]) * x)
        pi <- s$pi
        tm <- matrix(0, n, M)
        for (m in seq_len(M)) for (a in seq_len(M)) for (b in seq_len(M)) {
            # d W_ab / d eta_m with W_ab = mw (delta_ab pi_a - pi_a pi_b)
            dpa <- pi[, a] * ((a == m) - pi[, m])
            dpb <- pi[, b] * ((b == m) - pi[, m])
            dW <- mw * ((a == b) * dpa - dpa * pi[, b] - pi[, a] * dpb)
            tm[, m] <- tm[, m] + 0.5 * A[[(b - 1L) * M + a]] * dW
        }
        as.vector(t(crossprod(x, tm)))
    }

    beta <- numeric(P)
    obj <- penalized(beta)
    converged <- FALSE
    iter <- 0L
    for (iter in seq_len(maxit)) {
        s <- score_info(beta)
        Ustar <- if (firth) s$U + bias_term(s) else s$U
        delta <- as.vector(solve(s$I, Ustar))
        decrement <- sum(Ustar * delta)    # Fisher decrement
        if (is.finite(decrement) && decrement < tol) {
            converged <- TRUE
            break
        }
        step <- 1
        repeat {
            cand <- beta + step * delta
            obj_new <- penalized(cand)
            if (is.finite(obj_new) && obj_new >= obj - 1e-14 * abs(obj)) break
            step <- step / 2
            if (step < 1e-12) break
        }
        if (step < 1e-12) { converged <- FALSE; break }
        beta <- cand
        obj <- obj_new
    }
    s <- score_info(beta)
    coefficients <- matrix(beta, M, p)
    rownames(coefficients) <- lev[nonref]
    colnames(coefficients) <- colnames(x)
    cov.unscaled <- solve(s$I)
    nm <- paste(rep(lev[nonref], p), rep(colnames(x), each = M), sep = ":")
    if (!is.null(colnames(x))) dimnames(cov.unscaled) <- list(nm, nm)
    list(coefficients = coefficients, loglik = loglik_of(beta),
         cov.unscaled = cov.unscaled, iter = as.integer(iter),
         converged = converged)
}
