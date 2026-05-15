# Numerical robustness tests for native C++ family implementations.
# These stress-test edge cases: large mu, extreme eta, overflow-prone
# variance/dev_resids, and verify agreement with glm() under pressure.

rinvgauss <- function(n, mu, lambda) {
    nu <- rnorm(n)^2
    x  <- mu + mu^2 * nu / (2 * lambda) -
        mu / (2 * lambda) * sqrt(4 * mu * lambda * nu + mu^2 * nu^2)
    z  <- runif(n)
    ifelse(z <= mu / (mu + x), x, mu^2 / x)
}

# ---------------------------------------------------------------------------
# Gamma family
# ---------------------------------------------------------------------------

test_that("Gamma log link converges with large coefficients (extreme mu)", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 3), n, 3))
    beta <- c(5, 1.2, -0.8, 0.6)
    mu <- exp(X %*% beta)
    y  <- rgamma(n, shape = 2, rate = 2 / mu)

    fam <- Gamma(link = "log")
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-7)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-7)
})

test_that("Gamma log link: native vs R-callback agree under extreme mu", {
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(6, 1.5, -1.0)
    mu <- exp(X %*% beta)
    y  <- rgamma(n, shape = 1.5, rate = 1.5 / mu)

    fam <- Gamma(link = "log")
    disguise <- function(fam) { fam$family <- paste0(fam$family, "_dis"); fam }

    f_native <- fastglm(X, y, family = fam, method = 2)
    f_cb     <- fastglm(X, y, family = disguise(fam), method = 2)

    expect_true(f_native$converged)
    expect_true(f_cb$converged)
    expect_equal(unname(coef(f_native)), unname(coef(f_cb)), tolerance = 1e-7)
    expect_equal(f_native$deviance, f_cb$deviance, tolerance = 1e-7)
})

test_that("Gamma inverse link converges and matches glm()", {
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(0.5, 0.1, -0.05)
    eta <- X %*% beta
    mu  <- 1 / eta
    ok  <- mu > 0
    X <- X[ok, ]
    eta <- eta[ok]
    mu  <- mu[ok]
    y   <- rgamma(sum(ok), shape = 3, rate = 3 / mu)

    fam <- Gamma(link = "inverse")
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

test_that("Gamma identity link converges and matches glm()", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(5, 0.3, -0.2)
    mu  <- as.vector(X %*% beta)
    ok  <- mu > 0.1
    X <- X[ok, ]
    mu <- mu[ok]
    y  <- rgamma(sum(ok), shape = 2, rate = 2 / mu)

    fam <- Gamma(link = "identity")
    f <- fastglm(X, y, family = fam, method = 2, start = beta)
    g <- glm(y ~ X[, -1], family = fam, start = beta)

    expect_true(f$converged)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

test_that("Gamma dev_resids matches R for log link", {
    set.seed(123)
    n <- 200
    X <- cbind(1, rnorm(n))
    mu_true <- exp(X %*% c(2, 0.5))
    y <- rgamma(n, shape = 2, rate = 2 / mu_true)
    fam <- Gamma(link = "log")

    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# Inverse Gaussian family
# ---------------------------------------------------------------------------

test_that("InvGauss log link converges and matches glm()", {
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(2, 0.3, -0.2)
    mu <- as.vector(exp(X %*% beta))
    y  <- rinvgauss(n, mu = mu, lambda = mu^2 * 2)

    fam <- inverse.gaussian(link = "log")
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

test_that("InvGauss 1/mu^2 link converges and matches glm()", {
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(0.5, 0.05, -0.03)
    eta <- as.vector(X %*% beta)
    ok  <- eta > 0.05
    X <- X[ok, ]
    eta <- eta[ok]
    mu  <- 1 / sqrt(eta)
    y   <- rinvgauss(length(mu), mu = mu, lambda = mu^2 * 5)

    fam <- inverse.gaussian()
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

test_that("InvGauss dev_resids is stable for large mu (no NaN)", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(4, 0.8, -0.5)
    mu <- as.vector(exp(X %*% beta))
    y  <- rinvgauss(n, mu = mu, lambda = mu^2 * 2)

    fam <- inverse.gaussian(link = "log")
    f <- fastglm(X, y, family = fam, method = 2)

    expect_true(f$converged)
    expect_true(is.finite(f$deviance))
    expect_true(all(is.finite(coef(f))))

    g <- glm(y ~ X[, -1], family = fam)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

test_that("InvGauss native vs R-callback agree", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(2, 0.3, -0.2)
    mu <- as.vector(exp(X %*% beta))
    y  <- rinvgauss(n, mu = mu, lambda = mu^2 * 3)

    fam <- inverse.gaussian(link = "log")
    disguise <- function(fam) { fam$family <- paste0(fam$family, "_dis"); fam }

    f_native <- fastglm(X, y, family = fam, method = 2)
    f_cb     <- fastglm(X, y, family = disguise(fam), method = 2)

    expect_true(f_native$converged)
    expect_true(f_cb$converged)
    expect_equal(unname(coef(f_native)), unname(coef(f_cb)), tolerance = 1e-7)
    expect_equal(f_native$deviance, f_cb$deviance, tolerance = 1e-7)
})

test_that("InvGauss inverse link converges and matches glm()", {
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(0.3, 0.05, -0.03)
    eta <- as.vector(X %*% beta)
    ok  <- eta > 0.05
    X <- X[ok, ]
    eta <- eta[ok]
    mu  <- 1 / eta
    y   <- rinvgauss(length(mu), mu = mu, lambda = mu^2 * 3)

    fam <- inverse.gaussian(link = "inverse")
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

# ---------------------------------------------------------------------------
# Tweedie family
# ---------------------------------------------------------------------------

test_that("Tweedie log link converges with large coefficients", {
    skip_if_not_installed("statmod")
    skip_if_not_installed("tweedie")
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(4, 0.8, -0.5)
    mu <- exp(X %*% beta)
    y  <- tweedie::rtweedie(n, mu = mu, phi = 1.5, power = 1.5)

    fam <- statmod::tweedie(var.power = 1.5, link.power = 0)
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})

test_that("Tweedie native vs R-callback agree under stress", {
    skip_if_not_installed("statmod")
    skip_if_not_installed("tweedie")
    set.seed(123)
    n <- 400
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(3, 0.6, -0.4)
    mu <- exp(X %*% beta)
    y  <- tweedie::rtweedie(n, mu = mu, phi = 1.2, power = 1.7)

    fam <- statmod::tweedie(var.power = 1.7, link.power = 0)
    disguise <- function(fam) { fam$family <- paste0(fam$family, "_dis"); fam }

    f_native <- fastglm(X, y, family = fam, method = 2)
    f_cb     <- fastglm(X, y, family = disguise(fam), method = 2)

    expect_true(f_native$converged)
    expect_true(f_cb$converged)
    expect_equal(unname(coef(f_native)), unname(coef(f_cb)), tolerance = 1e-7)
    expect_equal(f_native$deviance, f_cb$deviance, tolerance = 1e-7)
})

test_that("Tweedie p near 1 (Poisson limit) matches Poisson", {
    skip_if_not_installed("statmod")
    set.seed(123)
    n <- 500
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    mu <- exp(X %*% c(1, 0.5, -0.3))
    y  <- rpois(n, mu)

    fam_tw <- statmod::tweedie(var.power = 1.001, link.power = 0)
    fam_p  <- poisson(link = "log")

    f_tw <- fastglm(X, y, family = fam_tw, method = 2)
    f_p  <- fastglm(X, y, family = fam_p,  method = 2)

    expect_true(f_tw$converged)
    expect_equal(unname(coef(f_tw)), unname(coef(f_p)), tolerance = 1e-3)
})

# ---------------------------------------------------------------------------
# Cross-family: all decomposition methods give same result
# ---------------------------------------------------------------------------

test_that("Gamma log link gives consistent results across all methods", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    mu <- exp(X %*% c(2, 0.5, -0.3))
    y  <- rgamma(n, shape = 2, rate = 2 / mu)

    fam <- Gamma(link = "log")
    results <- lapply(0:5, function(m) {
        fastglm(X, y, family = fam, method = m)
    })

    for (m in 2:6) {
        expect_equal(unname(coef(results[[m]])), unname(coef(results[[1]])),
                     tolerance = 1e-7,
                     info = paste0("method ", m - 1, " vs method 0"))
        expect_equal(results[[m]]$deviance, results[[1]]$deviance,
                     tolerance = 1e-7,
                     info = paste0("method ", m - 1, " deviance vs method 0"))
    }
})

test_that("InvGauss log link gives consistent results across methods 0-5", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    mu <- as.vector(exp(X %*% c(2, 0.3, -0.2)))
    y  <- rinvgauss(n, mu = mu, lambda = mu^2 * 3)

    fam <- inverse.gaussian(link = "log")
    results <- lapply(0:5, function(m) {
        fastglm(X, y, family = fam, method = m)
    })

    for (m in 2:6) {
        expect_equal(unname(coef(results[[m]])), unname(coef(results[[1]])),
                     tolerance = 1e-7,
                     info = paste0("method ", m - 1, " vs method 0"))
    }
})

# ---------------------------------------------------------------------------
# Stress: extreme-scale simulation
# ---------------------------------------------------------------------------

test_that("Gamma log link handles very large fitted values without NaN", {
    set.seed(123)
    n <- 200
    X <- cbind(1, matrix(rnorm(n * 3), n, 3))
    beta <- c(8, 2, -1.5, 1)
    mu <- exp(X %*% beta)
    y  <- rgamma(n, shape = 1, rate = 1 / mu)

    fam <- Gamma(link = "log")
    f <- fastglm(X, y, family = fam, method = 2)

    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_true(is.finite(f$deviance))
})

test_that("Poisson handles large fitted values without overflow", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(5, 1, -0.7)
    mu <- exp(X %*% beta)
    y  <- rpois(n, lambda = pmin(mu, 1e8))

    fam <- poisson(link = "log")
    f <- fastglm(X, y, family = fam, method = 2)
    g <- glm(y ~ X[, -1], family = fam)

    expect_true(f$converged)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-6)
})

test_that("InvGauss log link with large coefficients stays finite", {
    set.seed(123)
    n <- 300
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    beta <- c(3, 0.6, -0.4)
    mu <- as.vector(exp(X %*% beta))
    y  <- rinvgauss(n, mu = mu, lambda = mu^2 * 2)

    fam <- inverse.gaussian(link = "log")
    f <- fastglm(X, y, family = fam, method = 2)

    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_true(is.finite(f$deviance))

    g <- glm(y ~ X[, -1], family = fam)
    expect_equal(f$deviance, g$deviance, tolerance = 1e-6)
})
