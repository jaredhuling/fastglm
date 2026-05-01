skip_if_not_installed("logistf")

# Compare fastglm(firth = TRUE) against logistf::logistf, which is the
# canonical R implementation of Heinze & Schemper (2002) Firth's penalized
# logistic regression.  Coefficient agreement is to ~1e-7; SE agreement uses
# the looser 1e-2 because logistf reports a slightly different variance
# estimator -- our SE matches the standard (X'WX)^{-1} formula exactly.
.tight_logistf <- function() {
    logistf::logistf.control(lconv = 1e-12, gconv = 1e-12, xconv = 1e-12,
                              maxit = 200L)
}

test_that("Firth coefs match logistf on simulated logistic data", {
    set.seed(123)
    n <- 300
    x <- cbind(1, matrix(rnorm(n * 3), n, 3))
    eta <- x %*% c(0.2, 0.5, -0.4, 0.3)
    y  <- rbinom(n, 1, plogis(eta))

    m <- logistf::logistf(y ~ x[, -1], pl = FALSE, plconf = NULL,
                           control = .tight_logistf())
    f <- fastglm(x, y, family = binomial(), firth = TRUE,
                 tol = 1e-12, maxit = 200L)

    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-7)
    # SE: logistf reports a different variance estimator than the standard
    # (X'WX)^{-1}; allow a looser tolerance here.
    expect_equal(unname(f$se), unname(sqrt(diag(vcov(m)))), tolerance = 1e-2)
    expect_true(f$converged)
    expect_s3_class(f, "fastglm_firth")
})

test_that("Firth converges under perfect separation (where unpenalized glm diverges)", {
    # Albert & Anderson (1984) classic separation case: y is perfectly
    # predicted by sign(x).  Standard glm() diverges; Firth must converge.
    set.seed(7)
    n <- 50
    x_var <- c(rnorm(n / 2, mean = -2), rnorm(n / 2, mean = 2))
    y     <- as.integer(x_var > 0)
    X     <- cbind(1, x_var)

    f <- fastglm(X, y, family = binomial(), firth = TRUE,
                 tol = 1e-12, maxit = 200L)
    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_true(all(is.finite(f$se)))

    m <- logistf::logistf(y ~ x_var, pl = FALSE, plconf = NULL,
                           control = .tight_logistf())
    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-6)
})

test_that("Firth matches logistf on a typical small-sample bias case", {
    # Heinze-Schemper-style: small n, modestly separated.
    set.seed(2024)
    n  <- 60
    z1 <- rnorm(n);  z2 <- rnorm(n)
    eta <- 0.5 + 1.2 * z1 - 0.8 * z2
    y   <- rbinom(n, 1, plogis(eta))
    X   <- cbind(1, z1, z2)

    m <- logistf::logistf(y ~ z1 + z2, pl = FALSE, plconf = NULL,
                           control = .tight_logistf())
    f <- fastglm(X, y, family = binomial(), firth = TRUE,
                 tol = 1e-12, maxit = 200L)

    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-7)
})

test_that("firth = TRUE rejects unsupported families/links", {
    set.seed(1)
    n <- 50
    X <- cbind(1, rnorm(n))
    y <- rbinom(n, 1, 0.5)
    # probit not yet supported
    expect_error(fastglm(X, y, family = binomial("probit"), firth = TRUE),
                 "binomial\\(link = \"logit\"\\)")
    # poisson not allowed
    yp <- rpois(n, 2)
    expect_error(fastglm(X, yp, family = poisson(), firth = TRUE),
                 "binomial\\(link = \"logit\"\\)")
})

test_that("firth result reports unpenalized deviance and penalized.deviance", {
    set.seed(11)
    n <- 200
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y <- rbinom(n, 1, plogis(X %*% c(0.1, 0.5, -0.3)))
    f <- fastglm(X, y, family = binomial(), firth = TRUE)
    expect_true(is.finite(f$deviance))
    expect_true(is.finite(f$penalized.deviance))
    expect_true(is.finite(f$log.det.XtWX))
    # penalized = deviance - log|X'WX|
    expect_equal(f$penalized.deviance, f$deviance - f$log.det.XtWX,
                 tolerance = 1e-9)
    expect_true(isTRUE(f$firth))
})

test_that("fastglm.fit accepts firth = TRUE", {
    set.seed(99)
    n <- 100
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y <- rbinom(n, 1, plogis(X %*% c(0.1, 0.5, -0.3)))
    fit <- fastglm.fit(X, y, family = binomial(), firth = TRUE)
    expect_true(fit$converged)
    expect_true(isTRUE(fit$firth))
})
