# Averages, marginal effects, and bootstrap for multinomial fits.

eff_frame <- function(n = 400L, seed = 71L) {
    set.seed(seed)
    d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), f = factor(sample(c("u", "v"), n, TRUE)))
    eta <- cbind(0, 0.3 + 0.8 * d$x1 - 0.4 * d$x2, -0.2 - 0.5 * d$x2 + 0.5 * (d$f == "v"),
                 0.1 + 0.3 * d$x1)
    pr <- exp(eta) / rowSums(exp(eta))
    d$y <- factor(LETTERS[1:4][apply(pr, 1L, function(q) sample.int(4L, 1L, prob = q))],
                  levels = LETTERS[1:4])
    d
}

# Probabilities from a term-major coefficient vector b (M = K - 1) and a design x.
eff_probs <- function(b, x, K) {
    M <- K - 1L
    eta <- cbind(0, x %*% t(matrix(b, M)))
    e <- exp(eta - apply(eta, 1L, max))
    e / rowSums(e)
}

test_that("averages equal mean predicted probabilities with delta-method errors", {
    skip_if_not_installed("numDeriv")
    d <- eff_frame()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = d, tol = 1e-12)
    nd <- d[1:150, ]
    a <- fastglm_average(fit, newdata = nd)
    expect_identical(a$category, fit$levels)
    expect_identical(names(a), c("category", "estimate", "std.error", "conf.low", "conf.high"))
    expect_max_diff(a$estimate, colMeans(predict(fit, nd)), 1e-12)
    xm <- model.matrix(~ x1 + x2 + f, nd)
    b <- fit$coefficients
    J <- numDeriv::jacobian(function(b) colMeans(eff_probs(b, xm, 4L)), b)
    se <- sqrt(rowSums((J %*% vcov(fit)) * J))
    expect_max_diff(a$std.error, se, 1e-6)
    q <- qnorm(0.975)
    expect_max_diff(a$conf.low, a$estimate - q * a$std.error, 1e-12)
    # training rows, with averaging weights and a different level
    w <- runif(nrow(d), 0.5, 2)
    a2 <- fastglm_average(fit, weights = w, level = 0.9)
    expect_max_diff(a2$estimate, colSums(w * predict(fit)) / sum(w), 1e-12)
    expect_max_diff(a2$conf.high - a2$conf.low, 2 * qnorm(0.95) * a2$std.error, 1e-12)
})

test_that("average contrasts give the mean change in probabilities", {
    skip_if_not_installed("numDeriv")
    d <- eff_frame()
    x <- model.matrix(~ x1 + x2 + f, d)
    fit <- fastglm_multinom(x, d$y, tol = 1e-12)
    ct <- function(z) { z[, "x1"] <- z[, "x1"] + 1; z }
    a <- fastglm_average(fit, contrast = ct)
    b <- fit$coefficients
    f <- function(b) colMeans(eff_probs(b, ct(x), 4L) - eff_probs(b, x, 4L))
    expect_max_diff(a$estimate, f(b), 1e-12)
    J <- numDeriv::jacobian(f, b)
    expect_max_diff(a$std.error, sqrt(rowSums((J %*% vcov(fit)) * J)), 1e-6)
    expect_error(fastglm_average(fit, contrast = function(z) z[, 1:2]), "preserve design dimensions")
    # a supplied covariance is used
    a3 <- fastglm_average(fit, vcov. = 4 * vcov(fit))
    a0 <- fastglm_average(fit)
    expect_max_diff(a3$std.error, 2 * a0$std.error, 1e-10)
})

test_that("average marginal effects match the analytic average derivative", {
    skip_if_not_installed("numDeriv")
    d <- eff_frame()
    x <- model.matrix(~ x1 + x2, d)
    fit <- fastglm_multinom(x, d$y, tol = 1e-12)
    me <- fastglm_marginal_effects(fit, "x1")
    expect_identical(me$category, fit$levels)
    expect_identical(names(me), c("variable", "category", "estimate", "std.error",
                                  "conf.low", "conf.high"))
    ame <- function(b) {
        pr <- eff_probs(b, x, 4L)
        bk <- c(0, matrix(b, 3L)[, 2L])
        colMeans(pr * (rep(bk, each = nrow(pr)) - drop(pr %*% bk)))
    }
    expect_max_diff(me$estimate, ame(fit$coefficients), 1e-6)
    expect_max_diff(sum(me$estimate), 0, 1e-10)
    J <- numDeriv::jacobian(ame, fit$coefficients)
    expect_max_diff(me$std.error, sqrt(rowSums((J %*% vcov(fit)) * J)), 1e-5)
    # formula fit with a factor and an interaction, evaluated on supplied data
    fit2 <- fastglm_multinom(y ~ x1 * f + x2, data = d, tol = 1e-12)
    m2 <- fastglm_marginal_effects(fit2, "x1", newdata = d[1:100, ])
    m3 <- fastglm_marginal_effects(fit2, "x1")
    expect_max_diff(sum(m2$estimate), 0, 1e-8)
    expect_max_diff(sum(m3$estimate), 0, 1e-8)
    h <- 1e-5
    up <- lo <- d; up$x1 <- up$x1 + h; lo$x1 <- lo$x1 - h
    expect_max_diff(m3$estimate, colMeans((predict(fit2, up) - predict(fit2, lo)) / (2 * h)), 1e-6)
    expect_error(fastglm_marginal_effects(fit2, "f", newdata = d), "continuous")
})

test_that("the bootstrap is reproducible and replicates equal refits", {
    d <- eff_frame(n = 250L)
    x <- model.matrix(~ x1 + x2, d)
    fit <- fastglm_multinom(x, d$y)
    b1 <- fastglm_bootstrap(fit, R = 20L, seed = 5L)
    b2 <- fastglm_bootstrap(fit, R = 20L, seed = 5L)
    expect_identical(b1$replicates, b2$replicates)
    expect_equal(b1$successful, 20L)
    expect_identical(colnames(b1$replicates), names(fit$coefficients))
    for (j in c(1L, 7L, 20L)) {
        set.seed(b1$seeds[j])
        idx <- sample.int(nrow(x), nrow(x), replace = TRUE)
        r <- fastglm_multinom(x[idx, ], d$y[idx])
        expect_max_diff(b1$replicates[j, ], r$coefficients, 1e-10)
    }
    g <- sample.int(25L, nrow(x), TRUE)
    bc <- fastglm_bootstrap(fit, R = 5L, type = "cluster", cluster = g, seed = 1L)
    expect_equal(bc$successful, 5L)
    # formula fit: a statistic can predict from the refit
    fit2 <- fastglm_multinom(y ~ x1 + x2, data = d)
    nd <- d[1:30, ]
    bs <- fastglm_bootstrap(fit2, R = 4L, seed = 2L,
                            statistic = function(f) colMeans(predict(f, nd)))
    expect_identical(dim(bs$replicates), c(4L, 4L))
    expect_error(fastglm_bootstrap(fit, R = 4L, type = "parametric"), "multinomial")
})

test_that("source fits stop for the bootstrap", {
    d <- eff_frame(n = 120L)
    d$blk <- rep(1:3, length.out = nrow(d))
    idx <- split(seq_len(nrow(d)), d$blk)
    src <- fastglm_source(function(k) d[idx[[k]], ], 3L, formula = y ~ x1 + x2,
                          prototype = d[1:20, ])
    sf <- fastglm_multinom(src)
    expect_error(fastglm_bootstrap(sf, R = 3L),
                 "bootstrap refitting is not yet available for multinomial source fits")
})

test_that("sparse and big.matrix fits give the same averages and effects", {
    skip_if_not_installed("Matrix")
    d <- eff_frame(n = 300L)
    x <- model.matrix(~ x1 + x2, d)
    de <- fastglm_multinom(x, d$y, tol = 1e-12)
    sp <- fastglm_multinom(Matrix::Matrix(x, sparse = TRUE), d$y, tol = 1e-12)
    expect_max_diff(fastglm_average(sp)$estimate, fastglm_average(de)$estimate, 1e-10)
    expect_max_diff(fastglm_average(sp)$std.error, fastglm_average(de)$std.error, 1e-8)
    expect_max_diff(fastglm_marginal_effects(sp, "x1")$estimate,
                    fastglm_marginal_effects(de, "x1")$estimate, 1e-8)
    skip_if_not_installed("bigmemory")
    bg <- fastglm_multinom(bigmemory::as.big.matrix(x), d$y, tol = 1e-12)
    expect_max_diff(fastglm_average(bg)$estimate, fastglm_average(de)$estimate, 1e-10)
    expect_max_diff(fastglm_marginal_effects(bg, "x1")$std.error,
                    fastglm_marginal_effects(de, "x1")$std.error, 1e-8)
})

test_that("bootstrap refits use the stored settings, weights, offsets, reference, and firth", {
    d <- eff_frame(n = 200L)
    x <- model.matrix(~ x1 + x2, d)
    set.seed(3L)
    w <- runif(nrow(x), 0.5, 2)
    off <- matrix(rnorm(nrow(x) * 3L, sd = 0.2), nrow(x), 3L)
    tl <- 1e-12
    fit <- fastglm_multinom(x, d$y, weights = w, offset = off, reference = "C",
                            firth = TRUE, tol = tl, maxit = 7L)
    expect_equal(fit$refit.controls$maxit, 7L)
    expect_equal(fit$refit.controls$tol, 1e-12)
    b <- fastglm_bootstrap(fit, R = 4L, seed = 9L)
    for (j in 1:4) {
        set.seed(b$seeds[j])
        idx <- sample.int(nrow(x), nrow(x), replace = TRUE)
        r <- fastglm_multinom(x[idx, ], d$y[idx], weights = w[idx], offset = off[idx, ],
                              reference = "C", firth = TRUE, tol = tl, maxit = 7L)
        expect_max_diff(b$replicates[j, ], r$coefficients, 1e-10)
    }
    # a maxit too small to converge is reported, not replaced by defaults
    low <- suppressWarnings(fastglm_multinom(x, d$y, maxit = 2L, tol = 1e-12))
    expect_false(isTRUE(low$converged))
})

test_that("bootstrap stops for big.matrix fits", {
    skip_if_not_installed("bigmemory")
    d <- eff_frame(n = 100L)
    bg <- fastglm_multinom(bigmemory::as.big.matrix(model.matrix(~ x1, d)), d$y)
    expect_error(fastglm_bootstrap(bg, R = 3L), "memory-mapped \\(big.matrix\\) multinomial")
})

test_that("formula source fits work with newdata and agree with the in-memory fit", {
    d <- eff_frame(n = 210L)
    d$blk <- rep(1:3, length.out = nrow(d))
    idx <- split(seq_len(nrow(d)), d$blk)
    src <- fastglm_source(function(k) d[idx[[k]], ], 3L, formula = y ~ x1 + x2 + f,
                          prototype = d[1:20, ])
    sf <- fastglm_multinom(src, tol = 1e-12)
    de <- fastglm_multinom(y ~ x1 + x2 + f, data = d, tol = 1e-12)
    expect_identical(sf$source.offset, character(0))
    nd <- d[1:40, ]
    expect_max_diff(predict(sf, nd), predict(de, nd), 1e-10)
    expect_max_diff(fastglm_average(sf, newdata = nd)$estimate,
                    fastglm_average(de, newdata = nd)$estimate, 1e-10)
    expect_max_diff(fastglm_average(sf, newdata = nd)$std.error,
                    fastglm_average(de, newdata = nd)$std.error, 1e-10)
    expect_max_diff(fastglm_average(sf)$estimate, fastglm_average(de)$estimate, 1e-10)
    expect_max_diff(fastglm_average(sf)$std.error, fastglm_average(de)$std.error, 1e-10)
    m1 <- fastglm_marginal_effects(sf, "x1", newdata = d)
    m2 <- fastglm_marginal_effects(de, "x1", newdata = d)
    expect_max_diff(m1$estimate, m2$estimate, 1e-10)
    expect_max_diff(m1$std.error, m2$std.error, 1e-10)
    expect_error(fastglm_marginal_effects(sf, "x1"), "require 'newdata'")
})
