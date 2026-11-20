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

test_that("zero-inflated NB theta is not trapped at its upper bound", {
    skip_if_not_installed("pscl")
    # Until 0.1.5 the theta score lost its sign to rounding error near theta =
    # 1e8 (a difference of digamma values whose leading terms cancel), theta
    # was pushed to the bound 1e8, and the EM rule reported convergence at the
    # zero-inflated Poisson fit (log likelihood -302.0559). Reference:
    # pscl::zeroinfl, theta 0.92174, log likelihood -300.405901.
    set.seed(20268117); n <- 500
    x <- rnorm(n); w <- rnorm(n)
    yc <- rnbinom(n, size = 1.5, mu = exp(-1 + 0.3 * x))
    y <- ifelse(rbinom(n, 1, plogis(-0.5 + 0.5 * w)) == 1, 0, yc)
    f <- fastglm_zi(y ~ x | w, data = data.frame(y, x, w), dist = "negbin")
    expect_true(f$converged)
    expect_identical(f$convergence.reason, "converged")
    expect_lt(abs(as.numeric(logLik(f)) - (-300.405901)), 1e-5)
    expect_equal(f$theta, 0.92174, tolerance = 1e-3)
})

test_that("the log zero link starts from valid values and reaches the maximum likelihood", {
    skip_if_not_installed("Formula")
    skip_if_not_installed("Matrix")
    set.seed(2)
    n <- 4000L; x <- rnorm(n); z <- rnorm(n)
    pz <- exp(-1.2 + 0.3 * z)                  # below 1 for these z
    y <- rbinom(n, 1L, 1 - pz) * rpois(n, exp(0.5 + 0.3 * x))
    d <- data.frame(y, x, z)
    # The pilot used to start on the logit scale (eta > 0, invalid for the
    # log link) and stopped with "cannot find valid starting values".
    fit <- fastglm_zi(y ~ x | z, data = d, link = "log")
    expect_true(fit$converged)
    nll <- function(p) {
        mu <- exp(p[1] + p[2] * x); pi <- exp(p[3] + p[4] * z)
        if (any(pi >= 1)) return(1e10)
        -sum(ifelse(y == 0, log(pi + (1 - pi) * exp(-mu)), log1p(-pi) + dpois(y, mu, log = TRUE)))
    }
    o <- optim(c(0.4, 0.2, -1, 0.2), nll, method = "BFGS", control = list(reltol = 1e-15, maxit = 1000))
    o <- optim(o$par, nll, method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 5000))
    o <- optim(o$par, nll, method = "BFGS", control = list(reltol = 1e-15, maxit = 1000))
    expect_equal(unname(c(fit$coefficients$count, fit$coefficients$zero)), o$par, tolerance = 1e-5)
    expect_gte(fit$loglik, -o$value - 1e-8)
    # Sparse designs start from a valid intercept as well.
    X <- cbind(1, x); Z <- cbind(1, z)
    sp <- fastglm_zi(Matrix::Matrix(X, sparse = TRUE), response = y,
                     z = Matrix::Matrix(Z, sparse = TRUE), link = "log")
    expect_equal(unname(unlist(sp$coefficients)), unname(c(fit$coefficients$count, fit$coefficients$zero)),
                 tolerance = 1e-5)
    # Hurdle zero part with the log link: the binomial log-link GLM of y > 0.
    h <- fastglm_hurdle(y ~ x | z, data = d, link = "log", tol = 1e-12)
    ref <- suppressWarnings(glm(as.numeric(y > 0) ~ z, family = binomial(link = "log"),
               start = c(log(mean(y > 0)) - 0.1, 0), control = list(epsilon = 1e-14, maxit = 200)))
    expect_equal(unname(h$coefficients$zero), unname(coef(ref)), tolerance = 1e-6)
})

test_that("a log-link zero model whose probabilities approach one keeps them below one", {
    skip_if_not_installed("Formula")
    set.seed(2)
    n <- 4000L; x <- rnorm(n); z <- rnorm(n)
    pz <- pmin(exp(-0.05 + 0.8 * z), 0.999)    # not log-linear: the maximum is on the boundary
    y <- rbinom(n, 1L, 1 - pz) * rpois(n, exp(0.5 + 0.3 * x))
    msgs <- character()
    fit <- withCallingHandlers(fastglm_zi(y ~ x | z, data = data.frame(y, x, z), link = "log"),
                               warning = function(w) {
                                   msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning")
                               })
    eta <- drop(cbind(1, z) %*% fit$coefficients$zero)
    expect_true(all(eta < 0))
    expect_true(is.finite(fit$loglik))
    # The maximum is on the boundary pi = 1: the fit says so rather than
    # stopping or leaving the valid region.
    expect_false(fit$converged)
    expect_true(any(grepl("did not converge", msgs)))
})
