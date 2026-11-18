# Per-row evaluation and covariance inference for multinomial fits.

# Estimating functions, information, and bread computed directly from the
# multinomial formulas at the fit's coefficients (identity constraints).
direct_inference <- function(fit, x, counts, w = rep(1, nrow(x))) {
    K <- ncol(counts); M <- K - 1L; p <- ncol(x)
    B <- matrix(fit$coefficients, M, p)
    eta <- x %*% t(B)
    e <- exp(eta - pmax(0, apply(eta, 1, max)))
    mx <- pmax(0, apply(eta, 1, max))
    pi <- e / (exp(-mx) + rowSums(e))
    cnr <- counts[, -1L, drop = FALSE]
    mw <- w * rowSums(counts)
    u <- w * cnr - mw * pi
    est <- matrix(0, nrow(x), p * M)
    for (k in seq_len(p)) for (a in seq_len(M)) est[, (k - 1L) * M + a] <- x[, k] * u[, a]
    I <- matrix(0, p * M, p * M)
    idx <- function(a) a + M * (seq_len(p) - 1L)
    for (a in seq_len(M)) for (b in seq_len(M))
        I[idx(a), idx(b)] <- crossprod(x, mw * ((a == b) * pi[, a] - pi[, a] * pi[, b]) * x)
    list(est = est, bread = solve(I))
}

vglm_rows_for_test <- function(fit) fastglm:::.vglm_rows(fit)

inf_fit <- function(n = 300L, K = 3L, weighted = FALSE) {
    d <- vglm_sim_data(n, 2L, K, seed = 21L)
    w <- if (weighted) {set.seed(5L); runif(n, 0.5, 2)} else rep(1, n)
    fit <- fastglm_multinom(d$x, d$y, weights = w, tol = 1e-12)
    c(d, list(fit = fit, w = w))
}

test_that("vglm_evaluate_dense agrees with the multinomial formulas", {
    d <- inf_fit(weighted = TRUE)
    n <- nrow(d$x); M <- 2L
    r <- vglm_evaluate_dense(d$x, integer(0), d$counts, d$w, matrix(0, n, M),
                             rep(list(diag(M)), ncol(d$x)), "multinomial", 3L, 1L,
                             unname(d$fit$coefficients), 64L)
    di <- direct_inference(d$fit, d$x, d$counts, d$w)
    est <- matrix(0, n, ncol(d$x) * M)
    for (k in 1:ncol(d$x)) for (a in 1:M) est[, (k - 1L) * M + a] <- d$x[, k] * r$u[, a]
    expect_max_diff(est, di$est, 1e-12)
    expect_max_diff(rowSums(r$P), rep(1, n), 1e-12)
    expect_equal(sum(r$loglik_rows), d$fit$loglik, tolerance = 1e-10)
    expect_identical(dim(r$W), c(n, 3L))
})

test_that("HC0 equals bread %*% crossprod(estfun) %*% bread, HC1 scales it", {
    d <- inf_fit(weighted = TRUE)
    di <- direct_inference(d$fit, d$x, d$counts, d$w)
    P <- ncol(di$est)
    meat <- crossprod(di$est)
    hc0 <- di$bread %*% meat %*% di$bread
    v <- fastglm_vcov(d$fit)
    expect_max_diff(v, hc0, 1e-10)
    expect_max_diff(fastglm_vcov(d$fit, "HC0"), hc0, 1e-10)
    expect_identical(dimnames(v), list(names(d$fit$coefficients), names(d$fit$coefficients)))
    n <- d$fit$nobs
    expect_max_diff(fastglm_vcov(d$fit, "HC1"), hc0 * n / (n - P), 1e-10)
    expect_error(fastglm_vcov(d$fit, "HC2"), "HC2")
    expect_error(fastglm_vcov(d$fit, "HC3"), "HC3")
    ev <- vglm_rows_for_test(d$fit)
    expect_max_diff(ev$estfun, di$est, 1e-10)
})

test_that("cluster covariance equals the direct cluster sum", {
    d <- inf_fit()
    di <- direct_inference(d$fit, d$x, d$counts)
    n <- nrow(d$x); P <- ncol(di$est)
    set.seed(3L)
    cl <- sample(1:40, n, TRUE)
    S <- rowsum(di$est, cl)
    G <- nrow(S)
    meat <- crossprod(S)
    v0 <- di$bread %*% meat %*% di$bread * G / (G - 1)
    expect_max_diff(fastglm_vcov(d$fit, cluster = cl), v0, 1e-10)
    expect_max_diff(fastglm_vcov(d$fit, cluster = cl, cadjust = FALSE),
                    di$bread %*% meat %*% di$bread, 1e-10)
    expect_max_diff(fastglm_vcov(d$fit, "HC1", cluster = cl),
                    v0 * (n - 1) / (n - P), 1e-10)
    expect_error(fastglm_vcov(d$fit, cluster = rep(1, n)), "two clusters")
    expect_error(fastglm_vcov(d$fit, cluster = 1:3), "cluster")
})

test_that("cluster vectors align with NA-omitted rows; formulas are not accepted", {
    sim <- vglm_sim_data(250L, 2L, 3L, seed = 4L)
    dd <- data.frame(sim$x[, -1L])
    dd$y <- sim$y
    set.seed(9L)
    dd$g <- sample(letters[1:25], 250L, TRUE)
    dd$x1[7] <- NA
    fit <- fastglm_multinom(y ~ x1 + x2, data = dd, tol = 1e-12)
    expect_identical(fit$nobs, 249L)
    v2 <- fastglm_vcov(fit, cluster = dd$g)           # length before omission
    v3 <- fastglm_vcov(fit, cluster = dd$g[-7])       # retained rows
    expect_max_diff(v3, v2, 1e-12)
    expect_error(fastglm_vcov(fit, cluster = ~ g), "vector")
})

test_that("grouped responses: estfun and HC0/HC1 match the direct formula", {
    set.seed(12L)
    n <- 120L
    x <- cbind(1, matrix(rnorm(n * 2L), n, 2L))
    colnames(x) <- c("(Intercept)", "x1", "x2")
    pr <- exp(cbind(0, x %*% matrix(c(.3, .5, -.4, -.2, .4, .1), 3L)))
    pr <- pr / rowSums(pr)
    tot <- sample(2:9, n, TRUE)
    counts <- t(vapply(seq_len(n), function(i) as.numeric(rmultinom(1, tot[i], pr[i, ])),
                       numeric(3L)))
    colnames(counts) <- LETTERS[1:3]
    w <- runif(n, 0.5, 2)
    fit <- fastglm_multinom(x, counts, weights = w, tol = 1e-12)
    expect_gt(max(counts), 1)
    di <- direct_inference(fit, x, counts, w)
    expect_max_diff(fastglm:::.vglm_rows(fit)$estfun, di$est, 1e-12)
    P <- ncol(di$est)
    hc0 <- di$bread %*% crossprod(di$est) %*% di$bread
    expect_max_diff(fastglm_vcov(fit), hc0, 1e-10)
    expect_max_diff(fastglm_vcov(fit, "HC1"), hc0 * n / (n - P), 1e-10)
    # Block size does not change the result.
    expect_max_diff(fastglm_vcov(fit, block.size = 7L), hc0, 1e-10)
    cl <- rep(1:30, length.out = n)
    S <- rowsum(di$est, cl)
    expect_max_diff(fastglm_vcov(fit, cluster = cl, block.size = 11L),
                    di$bread %*% crossprod(S) %*% di$bread * 30 / 29, 1e-10)
    dg <- fastglm_diagnostics(fit)
    expect_max_diff(dg$score, colSums(di$est), 1e-12)
})

test_that(".vglm_rows(newdata) evaluates the fit's weights in newdata", {
    sim <- vglm_sim_data(150L, 1L, 3L, seed = 6L)
    set.seed(6L)
    dd <- data.frame(y = sim$y, x1 = sim$x[, "x1"], wt = runif(150L, 0.5, 3))
    fit <- fastglm_multinom(y ~ x1, data = dd, weights = wt, tol = 1e-12)
    r <- fastglm:::.vglm_rows(fit, newdata = dd)
    expect_max_diff(r$weights, dd$wt, 1e-15)
    expect_max_diff(sum(r$loglik_rows), fit$loglik, 1e-10)
    expect_max_diff(r$estfun, fastglm:::.vglm_rows(fit)$estfun, 1e-12)
    half <- dd[1:50, ]
    expect_max_diff(fastglm:::.vglm_rows(fit, half)$weights, half$wt, 1e-15)
    fit0 <- fastglm_multinom(y ~ x1, data = dd, tol = 1e-12)
    expect_max_diff(fastglm:::.vglm_rows(fit0, dd)$weights, rep(1, 150L), 1e-15)
})

test_that("firth fits stop for sandwich inference; aliased columns are dropped", {
    d <- inf_fit()
    fb <- fastglm_multinom(d$x, d$y, firth = TRUE)
    expect_error(fastglm_vcov(fb), "sandwich inference for adjusted scores is not implemented")
    x2 <- cbind(d$x, dup = d$x[, "x1"])
    fa <- fastglm_multinom(x2, d$y, tol = 1e-12)
    expect_true(anyNA(coef(fa)))
    v <- fastglm_vcov(fa)
    bad <- is.na(coef(fa))
    expect_true(all(is.na(v[bad, bad])))
    di <- direct_inference(d$fit, d$x, d$counts)
    expect_max_diff(v[!bad, !bad], fastglm_vcov(d$fit), 1e-8)
})

test_that("fastglm_wald_test gives the chi-square on K - 1 df", {
    d <- inf_fit()
    fit <- d$fit
    wt <- fastglm_wald_test(fit, terms = "x1")
    expect_true(wt$available)
    expect_identical(as.numeric(wt$df), 2L * 1)
    i <- grep(":x1$", names(fit$coefficients))
    b <- fit$coefficients[i]; V <- vcov(fit)[i, i]
    expect_max_diff(wt$statistic, drop(t(b) %*% solve(V, b)), 1e-10)
    wr <- fastglm_wald_test(fit, terms = "x1", vcov. = function(o) fastglm_vcov(o))
    V2 <- fastglm_vcov(fit)[i, i]
    expect_max_diff(wr$statistic, drop(t(b) %*% solve(V2, b)), 1e-10)
    # Formula fit, a three-level factor: 2 * 2 df.
    set.seed(2L)
    df <- data.frame(y = d$y, x1 = d$x[, "x1"], f = factor(sample(c("a", "b", "c"), nrow(d$x), TRUE)))
    ff <- fastglm_multinom(y ~ x1 + f, data = df, tol = 1e-12)
    wf <- fastglm_wald_test(ff, terms = "f")
    expect_identical(as.numeric(wf$df), 4)
    j <- grep(":f", names(ff$coefficients))
    bf <- ff$coefficients[j]
    expect_max_diff(wf$statistic, drop(t(bf) %*% solve(vcov(ff)[j, j], bf)), 1e-10)
    expect_s3_class(fastglm_wald_test(ff), "fastglm_wald_test")
})

test_that("fastglm_diagnostics works for multinomial fits", {
    d <- inf_fit()
    dg <- fastglm_diagnostics(d$fit)
    expect_s3_class(dg, "fastglm_diagnostics")
    expect_true(dg$converged)
    expect_identical(dg$rank, d$fit$rank)
    expect_lt(dg$standardized.score, 1e-5)
    expect_true(is.finite(dg$covariance.condition))
    expect_output(print(dg), "Convergence: converged")
    expect_error(fastglm_diagnostics(d$fit, boundary = TRUE), "boundary")
})

test_that("existing models keep their routes", {
    set.seed(1L)
    x <- cbind(1, rnorm(80)); y <- rbinom(80, 1, plogis(x[, 2]))
    g <- fastglm(x, y, family = binomial())
    expect_true(is.matrix(fastglm_vcov(g, "HC0")))
    expect_s3_class(fastglm_diagnostics(g), "fastglm_diagnostics")
})
