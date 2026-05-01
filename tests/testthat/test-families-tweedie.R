skip_if_not_installed("statmod")
skip_if_not_installed("tweedie")

sim_tweedie <- function(n, var.power, link.power, seed = 31) {
    set.seed(seed)
    p <- 3
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    beta <- c(1, 0.5, -0.3)
    eta  <- as.vector(X %*% beta)
    mu   <- if (link.power == 0) exp(eta)
            else if (link.power == 1) eta
            else eta^(1 / link.power)
    y    <- tweedie::rtweedie(n, mu = mu, phi = 1.2, power = var.power)
    list(X = X, y = y, beta = beta)
}

test_that("native Tweedie log-link matches glm() across var.power values", {
    for (vp in c(1.3, 1.5, 1.7, 1.9)) {
        d   <- sim_tweedie(500, var.power = vp, link.power = 0, seed = 31 + 10 * vp)
        fam <- statmod::tweedie(var.power = vp, link.power = 0)
        gfit <- glm(d$y ~ d$X[, -1], family = fam)
        ffit <- fastglm(d$X, d$y,    family = fam, method = 2)
        expect_equal(unname(coef(ffit)), unname(coef(gfit)),
                     tolerance = 1e-7,
                     info = paste0("Tweedie log, var.power = ", vp))
        expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-7,
                     info = paste0("Tweedie log dev, var.power = ", vp))
        expect_equal(ffit$dispersion, summary(gfit)$dispersion,
                     tolerance = 1e-6)
    }
})

test_that("native Tweedie sqrt link matches glm()", {
    d   <- sim_tweedie(500, var.power = 1.5, link.power = 0.5, seed = 41)
    fam <- statmod::tweedie(var.power = 1.5, link.power = 0.5)
    gfit <- glm(d$y ~ d$X[, -1], family = fam,
                start = c(sqrt(mean(d$y)), 0, 0))
    ffit <- fastglm(d$X, d$y, family = fam, method = 2,
                    start = c(sqrt(mean(d$y)), 0, 0))
    expect_equal(unname(coef(ffit)), unname(coef(gfit)), tolerance = 1e-6)
    expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-6)
})

test_that("native Tweedie p=2 (Gamma limit) matches Gamma-log", {
    set.seed(47)
    n  <- 400
    X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
    mu <- exp(X %*% c(0.5, 0.3, -0.2))
    y  <- rgamma(n, shape = 2, rate = 2 / mu)
    fam_tw <- statmod::tweedie(var.power = 2, link.power = 0)
    fam_g  <- Gamma(link = "log")
    f_tw   <- fastglm(X, y, family = fam_tw, method = 2)
    f_g    <- fastglm(X, y, family = fam_g,  method = 2)
    expect_equal(unname(coef(f_tw)), unname(coef(f_g)), tolerance = 1e-6)
})

test_that("Tweedie native path matches R-callback fallback", {
    disguise <- function(fam) { fam$family <- paste0(fam$family, "_dis"); fam }
    d   <- sim_tweedie(400, var.power = 1.5, link.power = 0, seed = 53)
    fam <- statmod::tweedie(var.power = 1.5, link.power = 0)
    f_native <- fastglm(d$X, d$y, family = fam,           method = 2)
    f_cb     <- fastglm(d$X, d$y, family = disguise(fam), method = 2)
    expect_equal(unname(coef(f_native)), unname(coef(f_cb)), tolerance = 1e-7)
    expect_equal(f_native$deviance, f_cb$deviance, tolerance = 1e-7)
})
