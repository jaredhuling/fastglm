# Regression tests for the final whole-branch review fixes.

review_frame <- function(n, seed = 501L) {
    set.seed(seed)
    d <- data.frame(x1 = rnorm(n), e = runif(n, 1, 3),
                    f = factor(sample(c("u", "v", "w"), n, TRUE)))
    eta <- cbind(0, 0.4 + 0.6 * d$x1, -0.3 * d$x1 + 0.4 * (d$f == "v"))
    pr <- exp(eta) / rowSums(exp(eta))
    d$y <- factor(c("A", "B", "C")[apply(pr, 1L, function(q) sample.int(3L, 1L, prob = q))],
                  levels = c("A", "B", "C"))
    d
}

test_that("marginal effects work with a data-vector offset for n > 8192 and a zero weight", {
    d <- review_frame(9000L)
    fit <- fastglm_multinom(y ~ x1 + f, data = d, offset = log(d$e))
    me <- fastglm_marginal_effects(fit, "x1")
    expect_true(all(is.finite(me$estimate)))
    expect_max_diff(sum(me$estimate), 0, 1e-8)
    # The effect equals a numerical derivative of the fitted probabilities.
    h <- 1e-5
    up <- lo <- d
    up$x1 <- up$x1 + h; lo$x1 <- lo$x1 - h
    off <- cbind(log(d$e), log(d$e))
    num <- colMeans(predict(fit, up, offset = off) - predict(fit, lo, offset = off)) / (2 * h)
    expect_max_diff(me$estimate, num, 1e-6)
    # A zero averaging weight in a small fit.
    ds <- review_frame(300L)
    fs <- fastglm_multinom(y ~ x1 + f, data = ds, offset = log(ds$e))
    w <- rep(1, 300L); w[1:5] <- 0
    mw <- fastglm_marginal_effects(fs, "x1", weights = w)
    expect_true(all(is.finite(mw$estimate)))
    m2 <- fastglm_marginal_effects(fs, "x1", weights = rep(1, 300L))
    expect_false(isTRUE(all.equal(mw$estimate, m2$estimate)))
})

test_that("grouped deviance subtracts the saturated log likelihood", {
    set.seed(502L)
    n <- 40L
    x <- rnorm(n); m <- sample(5:30, n, TRUE)
    s <- rbinom(n, m, plogis(0.3 + 0.7 * x)); f <- m - s
    gl <- fastglm(cbind(s, f) ~ x, family = binomial())
    counts <- cbind(f = f, s = s)
    fit <- fastglm_multinom(cbind(f, s) ~ x, data = data.frame(x = x, f = f, s = s))
    expect_max_diff(fit$deviance, gl$deviance, 1e-8)
    expect_max_diff(fit$null.deviance, gl$null.deviance, 1e-8)
    expect_identical(fit$df.residual, gl$df.residual)
    X <- cbind("(Intercept)" = 1, x = x)
    fm <- fastglm_multinom(X, counts)
    expect_max_diff(fm$deviance, gl$deviance, 1e-8)
    expect_max_diff(fm$null.deviance, gl$null.deviance, 1e-8)
    # The log likelihood and information criteria are unchanged: -2 loglik
    # exceeds the deviance by twice the saturated log likelihood.
    lsat <- sum((counts * log(counts / rowSums(counts)))[counts > 0])
    expect_max_diff(-2 * fit$loglik - fit$deviance, -2 * lsat, 1e-8)
    expect_max_diff(AIC(fit), -2 * fit$loglik + 2 * fit$rank, 1e-8)
    # Weights, sparse, big.matrix, and source inputs agree with the dense fit.
    w <- runif(n, 0.5, 2)
    fw <- fastglm_multinom(X, counts, weights = w)
    gw <- fastglm(cbind(s, f) ~ x, family = binomial(), weights = w)
    expect_max_diff(fw$deviance, gw$deviance, 1e-8)
    skip_if_not_installed("Matrix")
    fs <- fastglm_multinom(methods::as(Matrix::Matrix(X, sparse = TRUE), "generalMatrix"), counts)
    expect_max_diff(fs$deviance, fm$deviance, 1e-8)
    idx <- split(seq_len(n), rep(1:4, each = 10L))
    src <- fastglm_source(function(k) list(X = X[idx[[k]], ], y = counts[idx[[k]], ]), 4L)
    fsrc <- fastglm_multinom(src)
    expect_max_diff(fsrc$deviance, fm$deviance, 1e-8)
    expect_max_diff(fsrc$null.deviance, fm$null.deviance, 1e-8)
    skip_if_not_installed("bigmemory")
    bm <- bigmemory::as.big.matrix(X)
    fb <- fastglm_multinom(bm, counts)
    expect_max_diff(fb$deviance, fm$deviance, 1e-8)
    # Individual data are unaffected: deviance is -2 loglik.
    d <- review_frame(200L)
    fi <- fastglm_multinom(y ~ x1, data = d)
    expect_max_diff(fi$deviance, -2 * fi$loglik, 1e-10)
})

test_that("unused predictor levels are dropped, unused response levels still error", {
    d <- review_frame(400L)
    fit <- fastglm_multinom(y ~ x1 + f, data = d, subset = f != "w")
    expect_false(anyNA(coef(fit)))
    ref <- fastglm_multinom(y ~ x1 + f, data = droplevels(d[d$f != "w", ]))
    expect_max_diff(coef(fit), coef(ref), 1e-10)
    expect_identical(fit$xlevels$f, c("u", "v"))
    expect_true(all(is.finite(fastglm_average(fit)$estimate)))
    expect_true(all(is.finite(fastglm_marginal_effects(fit, "x1")$estimate)))
    expect_error(fastglm_multinom(y ~ x1 + f, data = d, subset = y != "C"),
                 "category 'C' has no observations")
})

test_that("user-set contrasts are kept unless levels are dropped", {
    d <- review_frame(400L)
    contrasts(d$f) <- contr.sum(3L)
    fit <- expect_silent(fastglm_multinom(y ~ x1 + f, data = d))
    ref <- fastglm_multinom(y ~ x1 + f, data = d,
                            contrasts = list(f = "contr.sum"))
    expect_max_diff(coef(fit), coef(ref), 1e-10)
    expect_true(is.matrix(fit$contrasts$f) || identical(fit$contrasts$f, "contr.sum"))
    expect_false(isTRUE(all.equal(coef(fit),
                                  coef(fastglm_multinom(y ~ x1 + f, data = {
                                      d2 <- d; contrasts(d2$f) <- NULL; d2 })))))
    # A factor with unused levels loses a contrasts matrix it carried, with a
    # warning, as model.frame() does.
    expect_warning(fastglm_multinom(y ~ x1 + f, data = d, subset = f != "w"),
                   "contrasts dropped from factor f")
})

test_that("fastglm_ordinal dispatches on a formula named formula", {
    set.seed(91)
    x <- rnorm(150)
    y <- ordered(cut(0.5 * x + rlogis(150), c(-Inf, -0.5, 0.7, Inf),
                     labels = c("low", "middle", "high")))
    a <- fastglm_ordinal(y ~ x)
    b <- fastglm_ordinal(formula = y ~ x)
    expect_max_diff(coef(b), coef(a), 1e-12)
    expect_error(fastglm_ordinal(y = y), "supply a design matrix or model formula")
})

test_that("a reloaded big.matrix fastglm fit stops clearly instead of crashing", {
    skip_if_not_installed("bigmemory")
    set.seed(3)
    x <- cbind(1, matrix(rnorm(400), 200))
    y <- rbinom(200, 1, 0.4)
    f <- fastglm(bigmemory::as.big.matrix(x), y, family = binomial(), method = 3L)
    tf <- tempfile(fileext = ".rds")
    on.exit(unlink(tf))
    saveRDS(f, tf)
    f2 <- readRDS(tf)
    expect_error(fastglm_vcov(f2, "HC0"), "no longer available")
    expect_error(model.matrix(f2), "no longer available")
    expect_equal(coef(f2), coef(f))
})

test_that("individual-data Pearson residuals scale by the square root of the weight", {
    d <- review_frame(150L)
    w <- runif(150L, 0.5, 3)
    fit <- fastglm_multinom(y ~ x1, data = d, weights = w)
    r0 <- residuals(fit, "response")
    expect_max_diff(residuals(fit, "pearson"),
                    r0 * sqrt(w) / sqrt(fit$fitted.values * (1 - fit$fitted.values)), 1e-10)
    f1 <- fastglm_multinom(y ~ x1, data = d)
    p1 <- fitted(f1)
    expect_max_diff(residuals(f1, "pearson"), residuals(f1) / sqrt(p1 * (1 - p1)), 1e-10)
})

test_that("HC1 needs more observations than parameters and carries its type", {
    d <- review_frame(300L)
    fit <- fastglm_multinom(y ~ x1, data = d)
    v <- fastglm_vcov(fit, type = "HC1")
    expect_identical(attr(v, "covariance.type"), "HC1")
    expect_identical(attr(fastglm_vcov(fit, type = "HC0"), "covariance.type"), "HC0")
    # Two grouped rows and four parameters.
    counts <- rbind(c(5, 3, 2), c(1, 4, 6)); colnames(counts) <- c("A", "B", "C")
    tiny <- fastglm_multinom(cbind("(Intercept)" = 1, z = c(0, 1)), counts)
    expect_error(fastglm_vcov(tiny, type = "HC1"),
                 "HC1 requires more positive-weight observations than estimated parameters")
})

test_that("the singular flag is stored and noted in the summary print", {
    d <- review_frame(200L)
    fit <- fastglm_multinom(y ~ x1, data = d)
    expect_false(fit$singular)
    expect_false(any(grepl("numerically singular", capture.output(print(summary(fit))))))
    fit$singular <- TRUE
    expect_true(any(grepl("numerically singular", capture.output(print(summary(fit))))))
})

test_that("a source fit accepts block.size and gives the same fit", {
    d <- review_frame(210L)
    idx <- split(seq_len(210L), rep(1:7, each = 30L))
    mk <- function() fastglm_source(function(k) d[idx[[k]], ], 7L, formula = y ~ x1 + f,
                                    prototype = d)
    a <- fastglm_multinom(mk())
    b <- fastglm_multinom(mk(), block.size = 5L)
    expect_max_diff(coef(b), coef(a), 1e-10)
    expect_error(fastglm_multinom(mk(), block.size = 0), "block.size")
})

test_that("bootstrap print reports inference as unavailable only after failed refits", {
    set.seed(5)
    x <- rnorm(200)
    y <- rbinom(200, 1, plogis(0.5 * x))
    fit <- fastglm(cbind(1, x), y, family = binomial())
    b <- fastglm_bootstrap(fit, R = 19L, seed = 1L)
    expect_false(any(grepl("Unavailable", capture.output(print(b)))))
    b$successful <- b$R - 1L
    expect_true(any(grepl("Unavailable", capture.output(print(b)))))
})

test_that("formula fits without a data argument recover their data for effects", {
    set.seed(8)
    x1 <- rnorm(300)
    x2 <- rnorm(300)
    y <- factor(sample(c("a", "b", "c"), 300, TRUE))
    f <- fastglm_multinom(y ~ x1 + x2)
    g <- fastglm_multinom(y ~ x1 + x2, data = data.frame(y, x1, x2))
    expect_max_diff(fastglm_marginal_effects(f, "x1")$estimate,
                    fastglm_marginal_effects(g, "x1")$estimate, 1e-12)
    expect_max_diff(fastglm_average(f)$estimate, fastglm_average(g)$estimate, 1e-12)
    fs <- fastglm_multinom(y ~ x1 + x2, subset = x2 > 0)
    gs <- fastglm_multinom(y ~ x1 + x2, data = data.frame(y, x1, x2), subset = x2 > 0)
    expect_max_diff(fastglm_marginal_effects(fs, "x1")$estimate,
                    fastglm_marginal_effects(gs, "x1")$estimate, 1e-12)
})

test_that("confint for multinomial fits refuses non-Wald methods", {
    d <- review_frame(150L)
    fit <- fastglm_multinom(y ~ x1, data = d)
    expect_identical(confint(fit, method = "wald"), confint(fit))
    expect_error(confint(fit, method = "profile"), "only Wald intervals")
})
