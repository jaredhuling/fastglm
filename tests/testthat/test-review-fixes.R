# Tests for issues identified in the comprehensive code review.
# Each test_that block references the finding ID (C3, C5, I6, etc.)

# ============================================================
# C3: summary.fastglm SE scaling consistency
# ============================================================
test_that("summary.fastglm computes correct SEs from cov.scaled (C3)", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    y <- X %*% c(1, 0.5, -0.3, 0.2) + rnorm(n)

    fit <- fastglm(X, y, family = gaussian())
    ref <- glm.fit(X, y, family = gaussian())

    s <- summary(fit)
    ref_se <- summary.glm(ref)$coefficients[, "Std. Error"]
    expect_equal(unname(s$coefficients[, "Std. Error"]), unname(ref_se), tolerance = 1e-6)
})

test_that("summary.fastglm SEs correct for Gamma family (C3)", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 3), n, 3))
    mu <- exp(X %*% c(1, 0.3, -0.2, 0.1))
    y <- rgamma(n, shape = 2, rate = 2 / mu)

    fit <- fastglm(X, y, family = Gamma(link = "log"))
    ref <- glm(y ~ X[,-1], family = Gamma(link = "log"))

    s_fast <- summary(fit)
    s_ref  <- summary(ref)
    expect_equal(unname(s_fast$coefficients[, "Std. Error"]),
                 unname(s_ref$coefficients[, "Std. Error"]),
                 tolerance = 1e-5)
})

# ============================================================
# C5: ZI/hurdle should not inherit incompatible fastglm methods
# ============================================================
test_that("fastglm_zi class does not inherit broken fastglm methods (C5)", {
    skip_if_not_installed("pscl")
    data("bioChemists", package = "pscl")
    fit <- fastglm_zi(art ~ fem + mar | fem + mar,
                      data = bioChemists, dist = "poisson")
    expect_false("fastglm" %in% class(fit))
    expect_true("fastglm_zi" %in% class(fit))
    expect_s3_class(fit, "fastglm_zi")

    expect_no_error(coef(fit))
    expect_no_error(vcov(fit))
    expect_no_error(logLik(fit))
    expect_no_error(print(fit))
})

test_that("fastglm_hurdle class does not inherit broken fastglm methods (C5)", {
    skip_if_not_installed("pscl")
    data("bioChemists", package = "pscl")
    fit <- fastglm_hurdle(art ~ fem + mar | fem + mar,
                          data = bioChemists, dist = "poisson")
    expect_false("fastglm" %in% class(fit))
    expect_true("fastglm_hurdle" %in% class(fit))

    expect_no_error(coef(fit))
    expect_no_error(vcov(fit))
    expect_no_error(logLik(fit))
    expect_no_error(print(fit))
})

# ============================================================
# I6: predict.fastglm with NULL newdata
# ============================================================
test_that("predict.fastglm returns fitted values when newdata is NULL (I6)", {
    set.seed(123)
    n <- 100
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y <- rbinom(n, 1, plogis(X %*% c(0.2, 0.5, -0.3)))

    fit <- fastglm(X, y, family = binomial())

    pred_link <- predict(fit, type = "link")
    expect_equal(pred_link, fit$linear.predictors)

    pred_resp <- predict(fit, type = "response")
    expect_equal(pred_resp, fit$fitted.values, tolerance = 1e-10)

    pred_se <- predict(fit, se.fit = TRUE)
    expect_true(is.list(pred_se))
    expect_equal(length(pred_se$fit), n)
    expect_equal(length(pred_se$se.fit), n)
    expect_true(all(pred_se$se.fit > 0))
})

# ============================================================
# C4: fastglm_nb mu_init with offset
# ============================================================
test_that("fastglm_nb handles offset correctly (C4)", {
    skip_if_not_installed("MASS")
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    off <- rep(log(2), n)
    mu <- exp(X %*% c(0.5, 0.3) + off)
    y <- MASS::rnegbin(n, mu = mu, theta = 2)

    fit_off  <- fastglm_nb(X, y, offset = off)
    fit_nooff <- fastglm_nb(X, y)

    expect_false(isTRUE(all.equal(coef(fit_off), coef(fit_nooff))))
    expect_true(fit_off$converged)

    ref <- MASS::glm.nb(y ~ X[,-1] + offset(off))
    expect_equal(unname(coef(fit_off)), unname(coef(ref)), tolerance = 1e-3)
})

# ============================================================
# I17: fastglm_nb missing fields
# ============================================================
test_that("fastglm_nb has intercept, aic, and n fields (I17)", {
    skip_if_not_installed("MASS")
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    y <- MASS::rnegbin(n, mu = exp(X %*% c(1, 0.5)), theta = 3)

    fit <- fastglm_nb(X, y)
    expect_true(fit$intercept)
    expect_true(is.finite(fit$aic))
    expect_equal(fit$n, n)

    s <- summary(fit)
    expect_s3_class(s, "summary.glm")
})

# ============================================================
# T1: Rank-deficient / collinear predictors
# ============================================================
test_that("rank-deficient X handled correctly by pivoted QR methods", {
    set.seed(123)
    n <- 200
    X <- cbind(1, x1 = rnorm(n), x2 = rnorm(n))
    X <- cbind(X, x3 = X[, 2])  # x3 is exact copy of x1
    y <- rbinom(n, 1, plogis(X[, 1:3] %*% c(0.2, 0.5, -0.3)))

    for (m in c(0, 4)) {
        fit <- fastglm(X, y, family = binomial(), method = m)
        expect_equal(fit$rank, 3L, info = paste("method", m))
        expect_true(fit$converged, info = paste("method", m))
        expect_true(any(is.nan(diag(vcov(fit)))),
                    info = paste("method", m, "aliased vcov entries should be NaN"))
    }
})

test_that("rank-deficient X: SVD also detects rank", {
    set.seed(123)
    n <- 200
    X <- cbind(1, x1 = rnorm(n), x2 = rnorm(n))
    X <- cbind(X, x3 = X[, 2])
    y <- X[, 1:3] %*% c(1, 0.5, -0.3) + rnorm(n)

    fit <- fastglm(X, y, family = gaussian(), method = 5)
    expect_lte(fit$rank, 3L)
    expect_true(fit$converged)
})

# ============================================================
# T2: Untested native family-link codes
# ============================================================
test_that("gaussian:log link works natively", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    mu <- exp(X %*% c(1, 0.2))
    y <- rnorm(n, mean = mu, sd = 0.5)

    fit_native <- fastglm(X, y, family = gaussian(link = "log"),
                          method = 0, start = c(1, 0.2))
    ref <- glm(y ~ X[, -1], family = gaussian(link = "log"),
               start = c(1, 0.2))
    expect_equal(unname(coef(fit_native)), unname(coef(ref)), tolerance = 1e-5)
})

test_that("poisson:sqrt link works natively", {
    set.seed(123)
    n <- 300
    X <- cbind(1, rnorm(n, mean = 1))
    eta <- X %*% c(2, 0.3)
    y <- rpois(n, lambda = eta^2)

    fit <- fastglm(X, y, family = poisson(link = "sqrt"),
                   method = 0, start = c(2, 0.3))
    ref <- glm(y ~ X[, -1], family = poisson(link = "sqrt"),
               start = c(2, 0.3))
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-5)
})

test_that("Gamma:identity link works natively", {
    set.seed(123)
    n <- 200
    X <- cbind(1, abs(rnorm(n)))
    mu <- X %*% c(2, 0.5)
    y <- rgamma(n, shape = 5, rate = 5 / mu)

    fit <- fastglm(X, y, family = Gamma(link = "identity"),
                   method = 0, start = c(2, 0.5))
    ref <- glm(y ~ X[, -1], family = Gamma(link = "identity"),
               start = c(2, 0.5))
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-4)
})

test_that("inverse.gaussian:identity link works natively", {
    set.seed(123)
    n <- 300
    X <- cbind(1, abs(rnorm(n, 0.5)))
    mu <- X %*% c(1, 0.5)
    y <- 1/rgamma(n, shape = 5, rate = 5 * mu)

    fit <- fastglm(X, y, family = inverse.gaussian(link = "identity"),
                   method = 0, start = c(1, 0.5))
    ref <- glm(y ~ X[, -1], family = inverse.gaussian(link = "identity"),
               start = c(1, 0.5))
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-3)
})

# ============================================================
# T3: fastglm_fit entry point (used as glm method)
# ============================================================
test_that("fastglm_fit works as glm() method for multiple families", {
    set.seed(123)
    n <- 200
    x1 <- rnorm(n)
    x2 <- rnorm(n)

    y_bin <- rbinom(n, 1, plogis(0.5 * x1 - 0.3 * x2))
    fit_bin <- glm(y_bin ~ x1 + x2, family = binomial(), method = fastglm_fit)
    ref_bin <- glm(y_bin ~ x1 + x2, family = binomial())
    expect_equal(unname(coef(fit_bin)), unname(coef(ref_bin)), tolerance = 1e-6)

    y_pois <- rpois(n, exp(0.5 + 0.3 * x1))
    fit_pois <- glm(y_pois ~ x1 + x2, family = poisson(), method = fastglm_fit)
    ref_pois <- glm(y_pois ~ x1 + x2, family = poisson())
    expect_equal(unname(coef(fit_pois)), unname(coef(ref_pois)), tolerance = 1e-6)

    y_gauss <- 1 + 0.5 * x1 - 0.3 * x2 + rnorm(n)
    fit_gauss <- glm(y_gauss ~ x1 + x2, family = gaussian(), method = fastglm_fit)
    ref_gauss <- glm(y_gauss ~ x1 + x2, family = gaussian())
    expect_equal(unname(coef(fit_gauss)), unname(coef(ref_gauss)), tolerance = 1e-6)
})

# ============================================================
# T4: Offsets in non-streaming path
# ============================================================
test_that("offsets work in fastglm and fastglmPure", {
    set.seed(123)
    n <- 300
    X <- cbind(1, rnorm(n))
    off <- rep(log(2), n)
    y <- rpois(n, lambda = exp(X %*% c(0.5, 0.3) + off))

    fit <- fastglm(X, y, family = poisson(), offset = off)
    ref <- glm(y ~ X[, -1], family = poisson(), offset = off)

    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)
    expect_equal(fit$deviance, ref$deviance, tolerance = 1e-6)
})

# ============================================================
# T5: Zero prior weights
# ============================================================
test_that("zero prior weights are handled correctly", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    y <- rbinom(n, 1, plogis(X %*% c(0.2, 0.5)))
    w <- rep(1, n)
    w[1:10] <- 0

    fit <- fastglm(X, y, family = binomial(), weights = w)
    ref <- glm.fit(X, y, family = binomial(), weights = w)

    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)
    expect_true(fit$converged)
})

# ============================================================
# T6: start / etastart / mustart arguments
# ============================================================
test_that("custom start values work", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    y <- rpois(n, exp(X %*% c(0.5, 0.3)))

    start <- c(0.5, 0.3)
    fit <- fastglm(X, y, family = poisson(), start = start)
    ref <- glm.fit(X, y, family = poisson(), start = start)
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
})

test_that("etastart and mustart work", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    y <- rnorm(n, mean = X %*% c(1, 0.5))

    mustart <- rep(mean(y), n)
    fit <- fastglm(X, y, family = gaussian(), mustart = mustart)
    ref <- glm.fit(X, y, family = gaussian(), mustart = mustart)
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
})

# ============================================================
# T8: residuals.fastglm
# ============================================================
test_that("residuals.fastglm returns correct deviance residuals", {
    set.seed(123)
    n <- 200
    x1 <- rnorm(n)
    y <- rpois(n, exp(0.5 + 0.3 * x1))

    fit <- fastglm(cbind(1, x1), y, family = poisson())
    ref <- glm(y ~ x1, family = poisson())

    r_fast <- residuals(fit, type = "deviance")
    r_ref  <- unname(residuals(ref, type = "deviance"))
    expect_equal(unname(r_fast), r_ref, tolerance = 1e-6)
})

test_that("residuals.fastglm returns correct pearson residuals", {
    set.seed(123)
    n <- 200
    x1 <- rnorm(n)
    y <- rbinom(n, 1, plogis(0.2 + 0.5 * x1))

    fit <- fastglm(cbind(1, x1), y, family = binomial())
    ref <- glm(y ~ x1, family = binomial())

    r_fast <- residuals(fit, type = "pearson")
    r_ref  <- unname(residuals(ref, type = "pearson"))
    expect_equal(unname(r_fast), r_ref, tolerance = 1e-6)
})

# ============================================================
# T10: Intercept-only and single-predictor models
# ============================================================
test_that("intercept-only model works", {
    set.seed(123)
    n <- 200
    X <- matrix(1, n, 1)
    y <- rpois(n, exp(0.5))

    fit <- fastglm(X, y, family = poisson())
    ref <- glm(y ~ 1, family = poisson())
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
    expect_true(fit$converged)
})

test_that("single predictor (no intercept) works", {
    set.seed(123)
    n <- 200
    x <- abs(rnorm(n, mean = 1))
    X <- matrix(x, n, 1)
    y <- rnorm(n, mean = 2 * x, sd = 0.5)

    fit <- fastglm(X, y, family = gaussian())
    ref <- glm(y ~ x - 1, family = gaussian())
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
})

# ============================================================
# T9: Firth + big.matrix (if bigmemory available)
# ============================================================
test_that("Firth works with big.matrix", {
    skip_if_not_installed("bigmemory")
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n), rnorm(n))
    y <- rbinom(n, 1, plogis(X %*% c(0.2, 0.5, -0.3)))

    Xbig <- bigmemory::as.big.matrix(X)
    fit_big  <- fastglmPure(Xbig, y, family = binomial(), firth = TRUE, method = 2)
    fit_dense <- fastglmPure(X, y, family = binomial(), firth = TRUE, method = 2)

    expect_equal(unname(fit_big$coefficients), unname(fit_dense$coefficients),
                 tolerance = 1e-6)
})

# ============================================================
# Additional: predict with type="response" and se.fit
# ============================================================
test_that("predict type=response with se.fit works", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n), rnorm(n))
    y <- rbinom(n, 1, plogis(X %*% c(0.2, 0.5, -0.3)))

    fit <- fastglm(X, y, family = binomial())
    pred <- predict(fit, type = "response", se.fit = TRUE)

    expect_true(all(pred$fit >= 0 & pred$fit <= 1))
    expect_true(all(pred$se.fit > 0))
    expect_equal(length(pred$fit), n)
})

# ============================================================
# Additional: predict with newdata
# ============================================================
test_that("predict with newdata still works", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n), rnorm(n))
    y <- rbinom(n, 1, plogis(X %*% c(0.2, 0.5, -0.3)))

    fit <- fastglm(X, y, family = binomial())
    Xnew <- cbind(1, rnorm(10), rnorm(10))
    pred <- predict(fit, newdata = Xnew, type = "link")
    expect_equal(length(pred), 10)
    expect_equal(as.numeric(pred), as.numeric(Xnew %*% coef(fit)))
})
