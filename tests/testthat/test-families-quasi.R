# quasibinomial / quasipoisson share C++ kernels with binomial / poisson.
# The only behavioural differences are (a) the dispersion is estimated from
# the Pearson statistic instead of being clamped to 1, and (b) summary()
# reports it as such.

test_that("native quasi-poisson matches glm(family = quasipoisson())", {
    set.seed(13)
    n <- 400; p <- 3
    X  <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    eta <- X %*% c(0.3, 0.4, -0.2)
    y   <- rpois(n, exp(eta))
    fam <- quasipoisson()
    gfit <- glm(y ~ X[, -1], family = fam)
    ffit <- fastglm(X, y,    family = fam, method = 2)
    expect_equal(unname(coef(ffit)), unname(coef(gfit)), tolerance = 1e-12)
    expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-10)
    # Pearson-based dispersion should match summary.glm exactly.
    expect_equal(ffit$dispersion, summary(gfit)$dispersion, tolerance = 1e-10)
    # Coefficient SEs are scaled by sqrt(dispersion); compare those too.
    expect_equal(unname(ffit$se), unname(sqrt(diag(vcov(gfit)))),
                 tolerance = 1e-8)
})

test_that("native quasi-binomial matches glm(family = quasibinomial())", {
    set.seed(17)
    n <- 600; p <- 3
    X  <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    eta <- X %*% c(0.2, 0.4, -0.1)
    y   <- rbinom(n, 1, plogis(eta))
    fam <- quasibinomial()
    gfit <- glm(y ~ X[, -1], family = fam)
    ffit <- fastglm(X, y,    family = fam, method = 2)
    expect_equal(unname(coef(ffit)), unname(coef(gfit)), tolerance = 1e-12)
    expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-10)
    expect_equal(ffit$dispersion, summary(gfit)$dispersion, tolerance = 1e-10)
    expect_equal(unname(ffit$se), unname(sqrt(diag(vcov(gfit)))),
                 tolerance = 1e-8)
})

test_that("over-dispersed quasi-poisson recovers >1 dispersion", {
    # Generate y ~ NB(mu, theta = 2)  so true dispersion ~ 1 + mu/theta > 1.
    skip_if_not_installed("MASS")
    set.seed(23)
    n <- 500; p <- 3
    X  <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    mu <- exp(X %*% c(1, 0.3, -0.2))
    y  <- MASS::rnegbin(n, mu = mu, theta = 2)
    ffit <- fastglm(X, y, family = quasipoisson(), method = 2)
    gfit <- glm(y ~ X[, -1], family = quasipoisson())
    expect_equal(ffit$dispersion, summary(gfit)$dispersion, tolerance = 1e-10)
    expect_gt(ffit$dispersion, 1.05)
})
