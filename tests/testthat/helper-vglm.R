# Helpers shared by the vector GLM tests.

# Simulate a multinomial logit data set. Returns the design matrix with an
# "(Intercept)" column and columns x1..xp, a factor response with levels
# A, B, C, ... (C1, ..., CK when K > 26), and the n by K 0/1 indicator matrix of the response.
vglm_sim_data <- function(n = 200L, p = 3L, K = 3L, seed = 1L) {
    set.seed(seed)
    lev <- if (K <= 26L) LETTERS[seq_len(K)] else paste0("C", seq_len(K))
    x <- cbind(1, matrix(rnorm(n * p), n, p))
    colnames(x) <- c("(Intercept)", paste0("x", seq_len(p)))
    beta <- matrix(rnorm((K - 1L) * (p + 1L), sd = 0.5), K - 1L, p + 1L)
    eta <- cbind(0, x %*% t(beta))
    pr <- exp(eta - apply(eta, 1, max))
    pr <- pr / rowSums(pr)
    cls <- apply(pr, 1, function(q) sample.int(K, 1L, prob = q))
    y <- factor(lev[cls], levels = lev)
    counts <- vglm_indicator(y)
    list(x = x, y = y, counts = counts)
}

vglm_indicator <- function(y) {
    counts <- outer(as.integer(y), seq_len(nlevels(y)), "==") * 1
    colnames(counts) <- levels(y)
    counts
}

# A small data set (n = 60, K = 3) in which x1 perfectly separates level C
# (every observation with x1 > 1 is in C and no other is), so the maximum
# likelihood estimate does not exist, while the bias-reduced estimate does.
vglm_separated_data <- function() {
    set.seed(11L)
    n <- 60L
    x1 <- c(seq(1.2, 3, length.out = 10L), rnorm(n - 10L, sd = 0.5))
    x1[11:n] <- pmin(x1[11:n], 0.9)
    x2 <- rnorm(n)
    cls <- integer(n)
    cls[1:10] <- 3L
    pB <- plogis(0.8 * x2[11:n])
    cls[11:n] <- ifelse(runif(n - 10L) < pB, 2L, 1L)
    y <- factor(c("A", "B", "C")[cls], levels = c("A", "B", "C"))
    x <- cbind("(Intercept)" = 1, x1 = x1, x2 = x2)
    list(x = x, y = y, counts = vglm_indicator(y))
}

# Elementwise agreement: passes only if
# max(abs(object - expected)) <= tol * max(1, max(abs(expected))).
# testthat's `tolerance` is a mean relative difference, which is weaker.
expect_max_diff <- function(object, expected, tol) {
    obj <- as.numeric(object)
    ref <- as.numeric(expected)
    if (length(obj) != length(ref)) {
        testthat::expect(FALSE, sprintf(
            "lengths differ: object has %d elements, expected has %d",
            length(obj), length(ref)))
        return(invisible(object))
    }
    if (!length(obj)) {
        testthat::expect(TRUE, "")
        return(invisible(object))
    }
    observed <- max(abs(obj - ref))
    bound <- tol * max(1, max(abs(ref)))
    testthat::expect(
        !is.na(observed) && observed <= bound,
        sprintf("max abs difference %.6g exceeds bound %.6g (tol = %g)",
                observed, bound, tol))
    invisible(object)
}
