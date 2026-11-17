scope_data <- function(n = 1500, seed = 21) {
    set.seed(seed)
    d <- data.frame(x = rnorm(n), w = rnorm(n))
    d$yb <- rbinom(n, 1, plogis(-0.3 + 0.8 * d$x - 0.4 * d$x^2 + 0.3 * d$w))
    d$yc <- MASS::rnegbin(n, exp(0.5 + 0.4 * d$x), 1.5)
    d$yz <- ifelse(runif(n) < plogis(0.4 + 0.6 * d$w),
                   MASS::rnegbin(n, exp(0.6 + 0.4 * d$x + 0.2 * d$w), 2), 0)
    d
}

test_that("GLM marginal effects equal the analytic average derivative and its delta-method SE", {
    d <- scope_data()
    fit <- fastglm(yb ~ x + I(x^2) + w, data = d, family = binomial())
    me <- fastglm_marginal_effects(fit, "x")
    X <- model.matrix(~ x + I(x^2) + w, d)
    ame <- function(b) { p <- plogis(drop(X %*% b)); mean(p * (1 - p) * (b[2] + 2 * b[3] * d$x)) }
    b <- coef(fit)
    # Analytic gradient of the average marginal effect.
    p <- plogis(drop(X %*% b)); s <- b[2] + 2 * b[3] * d$x
    g <- colMeans(X * (p * (1 - p) * (1 - 2 * p) * s)) + c(0, mean(p * (1 - p)), mean(p * (1 - p) * 2 * d$x), 0)
    expect_equal(me$estimate, ame(b), tolerance = 1e-8)
    expect_equal(me$std.error, sqrt(drop(g %*% vcov(fit) %*% g)), tolerance = 1e-6)
    # A matrix fit differentiates the named column.
    fm <- fastglm(X, d$yb, family = binomial())
    expect_equal(fastglm_marginal_effects(fm, "w")$estimate, fastglm_marginal_effects(fit, "w")$estimate,
                 tolerance = 1e-10)
    expect_error(fastglm_marginal_effects(fastglm(yb ~ x + factor(w > 0), data = d, family = binomial()), "nope"),
                 "continuous")
})

test_that("count marginal effects match closed forms and the hurdle predictions", {
    d <- scope_data()
    nb <- fastglm_nb(yc ~ x + w, data = d)
    expect_equal(fastglm_marginal_effects(nb, "x")$estimate,
                 mean(fitted(nb)) * coef(nb)[["x"]], tolerance = 1e-8)
    h <- fastglm_hurdle(yz ~ x + w | w + x, data = d, dist = "negbin")
    up <- lo <- d; step <- 1e-5; up$x <- up$x + step; lo$x <- lo$x - step
    expect_equal(fastglm_marginal_effects(h, "x")$estimate,
                 mean((predict(h, up) - predict(h, lo)) / (2 * step)), tolerance = 1e-6)
    for (type in c("count", "zero")) expect_true(is.finite(fastglm_marginal_effects(h, "x", type = type)$std.error))
})

test_that("memory-mapped designs give the dense fits for count, two-part, and ordinal models", {
    skip_if_not_installed("bigmemory")
    set.seed(22); n <- 6000
    X <- cbind("(Intercept)" = 1, x1 = rnorm(n), x2 = rnorm(n)); Z <- X[, 1:2]
    yc <- MASS::rnegbin(n, exp(0.4 + 0.3 * X[, 2]), 1.5)
    yz <- ifelse(runif(n) < plogis(0.3 + 0.5 * X[, 2]), yc, 0)
    ys <- ifelse(yz > 0, rgamma(n, 2, 2 / exp(1 + 0.2 * X[, 3])), 0)
    XB <- bigmemory::as.big.matrix(X); ZB <- bigmemory::as.big.matrix(Z)
    expect_equal(coef(fastglm_nb(XB, yc, block.size = 1000L)), coef(fastglm_nb(X, yc)), tolerance = 1e-7)
    h <- fastglm_hurdle(XB, response = yz, z = ZB, dist = "negbin", block.size = 1000L)
    expect_equal(coef(h), coef(fastglm_hurdle(X, response = yz, z = Z, dist = "negbin")), tolerance = 1e-10)
    expect_equal(coef(fastglm_twopart(XB, response = ys, z = ZB, block.size = 1000L)),
                 coef(fastglm_twopart(X, response = ys, z = Z)), tolerance = 1e-10)
    expect_true(is.finite(fastglm_vcov(h, type = "HC0")[1, 1]))
    yo <- ordered(cut(0.5 * X[, 2] + rlogis(n), c(-Inf, -1, 0, 1, Inf)), labels = 1:4)
    fo <- fastglm_ordinal(bigmemory::as.big.matrix(X[, 2:3]), yo, chunk.size = 1500L)
    expect_s3_class(fo, "fastglm_ordinal_stream")
    expect_equal(coef(fo), coef(fastglm_ordinal(X[, 2:3], yo)), tolerance = 1e-10)
})

test_that("zero-truncated fits agree across dense, sparse, mapped, and streamed designs", {
    set.seed(23); n <- 5000
    X <- cbind("(Intercept)" = 1, x1 = rnorm(n), x2 = rbinom(n, 1, 0.3))
    y <- MASS::rnegbin(n, exp(0.5 + 0.4 * X[, 2] - 0.3 * X[, 3]), 1.2)
    keep <- y > 0; X <- X[keep, ]; y <- y[keep]; m <- nrow(X)
    src <- fastglm_source(function(k) {
        i <- seq.int((k - 1L) * 1000L + 1L, min(m, k * 1000L)); list(X = X[i, ], y = y[i])
    }, ceiling(m / 1000L))
    for (dist in c("negbin", "poisson")) {
        dense <- fastglm_truncated(X, y, dist = dist)
        sparse <- fastglm_truncated(as(X, "dgCMatrix"), y, dist = dist)
        streamed <- fastglm_truncated(src, dist = dist)
        expect_equal(coef(sparse), coef(dense), tolerance = 1e-10)
        expect_equal(coef(streamed), coef(dense), tolerance = 1e-10)
        expect_equal(sparse$loglik, dense$loglik, tolerance = 1e-10)
        expect_equal(sparse$se, dense$se, tolerance = 1e-8)
        if (dist == "negbin") expect_equal(sparse$theta, dense$theta, tolerance = 1e-8)
        else expect_null(sparse$theta)
        if (requireNamespace("bigmemory", quietly = TRUE))
            expect_equal(coef(fastglm_truncated(bigmemory::as.big.matrix(X), y, dist = dist)), coef(dense),
                         tolerance = 1e-10)
    }
    expect_error(fastglm_truncated(as(cbind(1, rnorm(10)), "dgCMatrix"), c(0, rep(1, 9))), "positive")
})

test_that("zero-inflated models accept design matrices and match the formula fit", {
    d <- scope_data(seed = 24)
    d$e <- runif(nrow(d), 0.5, 2)
    ff <- fastglm_zi(yz ~ x + offset(log(e)) | w, data = d, dist = "negbin")
    X <- cbind("(Intercept)" = 1, x = d$x); Z <- cbind("(Intercept)" = 1, w = d$w)
    fm <- fastglm_zi(X, response = d$yz, z = Z, offset = cbind(log(d$e), 0), dist = "negbin")
    expect_equal(coef(fm), coef(ff), tolerance = 1e-12)
    expect_equal(fm$loglik, ff$loglik, tolerance = 1e-12)
    expect_equal(unname(predict(fm)), unname(predict(ff)), tolerance = 1e-10)
    expect_equal(fastglm_marginal_effects(fm, "x")$estimate,
                 fastglm_marginal_effects(ff, "x", newdata = d)$estimate, tolerance = 1e-10)
    expect_error(fastglm_zi(X, dist = "negbin"), "response")
})
