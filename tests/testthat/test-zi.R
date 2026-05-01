skip_if_not_installed("pscl")

# fastglm_zi vs pscl::zeroinfl on simulated data and the canonical
# bioChemists data shipped with pscl.  Tolerances are slightly looser
# than hurdle (1e-4 vs 1e-5) because both fastglm_zi and pscl::zeroinfl
# end up at the same MLE but via different optimization paths -- pscl
# uses optim/BFGS on the full joint likelihood, fastglm_zi uses EM with
# an inner Brent for theta.

.tight_pscl <- function() pscl::zeroinfl.control(method = "BFGS",
                                                 reltol = 1e-12,
                                                 maxit  = 5000)

test_that("Poisson ZI: coefs match pscl::zeroinfl on simulated data", {
    set.seed(21)
    n  <- 500
    x1 <- rnorm(n);  x2 <- rnorm(n)
    eta_c <- 0.7 + 0.4 * x1 - 0.3 * x2
    lam   <- exp(eta_c)
    eta_z <- -0.4 + 0.5 * x1 + 0.2 * x2
    p_inf <- plogis(eta_z)
    z     <- rbinom(n, 1, p_inf)
    y     <- ifelse(z == 1, 0L, rpois(n, lam))
    df    <- data.frame(y = y, x1 = x1, x2 = x2)

    f  <- fastglm_zi(y ~ x1 + x2, data = df, dist = "poisson",
                     em.tol = 1e-10, em.maxit = 200L,
                     tol = 1e-10, maxit = 200L)
    p  <- pscl::zeroinfl(y ~ x1 + x2, data = df, dist = "poisson",
                         control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-4)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-4)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-5)
    expect_true(f$converged)
})

test_that("Poisson ZI: SEs agree with pscl::zeroinfl", {
    set.seed(22)
    n  <- 600
    x  <- rnorm(n)
    eta_c <- 0.5 + 0.3 * x
    eta_z <- -0.2 + 0.4 * x
    z     <- rbinom(n, 1, plogis(eta_z))
    y     <- ifelse(z == 1, 0L, rpois(n, exp(eta_c)))
    df    <- data.frame(y = y, x = x)

    f <- fastglm_zi(y ~ x, data = df, dist = "poisson",
                    em.tol = 1e-10, em.maxit = 200L,
                    tol = 1e-10, maxit = 200L)
    p <- pscl::zeroinfl(y ~ x, data = df, dist = "poisson",
                        control = .tight_pscl())

    expect_equal(unname(f$se$count), unname(sqrt(diag(vcov(p)))[1:2]),
                 tolerance = 1e-2)
    expect_equal(unname(f$se$zero),  unname(sqrt(diag(vcov(p)))[3:4]),
                 tolerance = 1e-2)
})

test_that("ZI handles two-RHS formula y ~ x | z (different zero design)", {
    set.seed(23)
    n  <- 500
    x1 <- rnorm(n);  x2 <- rnorm(n);  z1 <- rnorm(n)
    eta_c <- 0.6 + 0.3 * x1 - 0.2 * x2
    eta_z <- -0.3 + 0.7 * z1
    z     <- rbinom(n, 1, plogis(eta_z))
    y     <- ifelse(z == 1, 0L, rpois(n, exp(eta_c)))
    df    <- data.frame(y = y, x1 = x1, x2 = x2, z1 = z1)

    f <- fastglm_zi(y ~ x1 + x2 | z1, data = df, dist = "poisson",
                    em.tol = 1e-10, em.maxit = 200L,
                    tol = 1e-10, maxit = 200L)
    p <- pscl::zeroinfl(y ~ x1 + x2 | z1, data = df, dist = "poisson",
                        control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-4)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-4)
    expect_length(coef(f, "zero"), 2L)         # intercept + z1
    expect_length(coef(f, "count"), 3L)        # intercept + x1 + x2
})

test_that("NB ZI: coefs and theta match pscl::zeroinfl", {
    skip_if_not_installed("MASS")
    set.seed(24)
    n  <- 800
    x1 <- rnorm(n);  x2 <- rnorm(n)
    eta_c <- 0.8 + 0.4 * x1 - 0.3 * x2
    lam   <- exp(eta_c)
    eta_z <- -0.4 + 0.5 * x1
    z     <- rbinom(n, 1, plogis(eta_z))
    y     <- ifelse(z == 1, 0L,
                    MASS::rnegbin(n, mu = lam, theta = 2.0))
    df    <- data.frame(y = y, x1 = x1, x2 = x2)

    f <- fastglm_zi(y ~ x1 + x2, data = df, dist = "negbin",
                    em.tol = 1e-9, em.maxit = 300L,
                    tol = 1e-10, maxit = 200L,
                    theta.tol = 1e-9)
    p <- pscl::zeroinfl(y ~ x1 + x2, data = df, dist = "negbin",
                        control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 5e-3)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 5e-3)
    expect_equal(unname(f$theta), unname(p$theta), tolerance = 5e-2)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-4)
})

test_that("ZI Poisson: probit zero link works", {
    set.seed(25)
    n  <- 400
    x  <- rnorm(n)
    eta_c <- 0.5 + 0.4 * x
    eta_z <- -0.3 + 0.6 * x
    z     <- rbinom(n, 1, pnorm(eta_z))
    y     <- ifelse(z == 1, 0L, rpois(n, exp(eta_c)))
    df    <- data.frame(y = y, x = x)

    f <- fastglm_zi(y ~ x, data = df, dist = "poisson", link = "probit",
                    em.tol = 1e-10, em.maxit = 200L,
                    tol = 1e-10, maxit = 200L)
    p <- pscl::zeroinfl(y ~ x, data = df, dist = "poisson", link = "probit",
                        control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-3)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-3)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-4)
})

test_that("pscl::zeroinfl on bioChemists matches fastglm_zi", {
    skip_if_not_installed("pscl")
    data("bioChemists", package = "pscl")
    f <- fastglm_zi(art ~ ., data = bioChemists, dist = "poisson",
                    em.tol = 1e-10, em.maxit = 300L,
                    tol = 1e-10, maxit = 200L)
    p <- pscl::zeroinfl(art ~ ., data = bioChemists, dist = "poisson",
                        control = .tight_pscl())
    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-4)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-4)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-5)
})
