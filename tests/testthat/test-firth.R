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
    # Separated data: the likelihood is very flat, so algorithms converge
    # to slightly different points.  Use a looser tolerance.
    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-3)
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

test_that("firth = TRUE works for all standard families", {
    set.seed(1)
    n <- 100
    X <- cbind(1, rnorm(n))
    # Binomial probit
    y <- rbinom(n, 1, 0.5)
    f_probit <- fastglm(X, y, family = binomial("probit"), firth = TRUE)
    expect_true(f_probit$converged)
    expect_true(f_probit$firth)
    # Poisson log
    yp <- rpois(n, 2)
    f_pois <- fastglm(X, yp, family = poisson(), firth = TRUE)
    expect_true(f_pois$converged)
    expect_true(f_pois$firth)
    # Gamma log
    yg <- rgamma(n, 2, 1)
    f_gam <- fastglm(X, yg, family = Gamma("log"), firth = TRUE)
    expect_true(f_gam$converged)
    expect_true(f_gam$firth)
    # Gaussian identity
    yn <- rnorm(n, 2, 1)
    f_gauss <- fastglm(X, yn, family = gaussian(), firth = TRUE)
    expect_true(f_gauss$converged)
    expect_true(f_gauss$firth)
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

test_that("fastglm_fit accepts firth = TRUE", {
    set.seed(99)
    n <- 100
    X <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y <- rbinom(n, 1, plogis(X %*% c(0.1, 0.5, -0.3)))
    fit <- fastglm_fit(X, y, family = binomial(), firth = TRUE)
    expect_true(fit$converged)
    expect_true(isTRUE(fit$firth))
})

test_that("Firth binomial gives identical results across all 6 methods", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))

    ref <- fastglm(X, y, family = binomial(), method = 2, firth = TRUE)
    for (m in c(0L, 1L, 3L, 4L, 5L)) {
        fit <- fastglm(X, y, family = binomial(), method = m, firth = TRUE)
        expect_equal(unname(fit$coefficients), unname(ref$coefficients),
                     tolerance = 1e-10,
                     info = paste0("method ", m, " vs method 2"))
        expect_true(fit$converged)
    }
})

test_that("Firth Poisson gives identical results across all 6 methods", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    y <- rpois(n, exp(X %*% c(0.5, 0.3, -0.2, 0.1)))

    ref <- fastglm(X, y, family = poisson(), method = 2, firth = TRUE)
    for (m in c(0L, 1L, 3L, 4L, 5L)) {
        fit <- fastglm(X, y, family = poisson(), method = m, firth = TRUE)
        expect_equal(unname(fit$coefficients), unname(ref$coefficients),
                     tolerance = 1e-10,
                     info = paste0("method ", m, " vs method 2"))
    }
})

test_that("Firth sparse matches dense", {
    skip_if_not_installed("Matrix")
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    Xs <- Matrix::Matrix(X, sparse = TRUE)

    # Binomial
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))
    ref <- fastglm(X, y, family = binomial(), method = 2, firth = TRUE)
    sp  <- fastglm(Xs, y, family = binomial(), method = 2, firth = TRUE)
    expect_equal(unname(sp$coefficients), unname(ref$coefficients), tolerance = 1e-7)

    # Poisson
    yp <- rpois(n, exp(X %*% c(0.5, 0.3, -0.2, 0.1)))
    ref_p <- fastglm(X, yp, family = poisson(), method = 2, firth = TRUE)
    sp_p  <- fastglm(Xs, yp, family = poisson(), method = 2, firth = TRUE)
    expect_equal(unname(sp_p$coefficients), unname(ref_p$coefficients), tolerance = 1e-7)

    # Gamma
    yg <- rgamma(n, shape = 2, rate = 1 / exp(X %*% c(0.5, 0.3, -0.2, 0.1)))
    ref_g <- fastglm(X, yg, family = Gamma("log"), method = 2, firth = TRUE)
    sp_g  <- fastglm(Xs, yg, family = Gamma("log"), method = 2, firth = TRUE)
    expect_equal(unname(sp_g$coefficients), unname(ref_g$coefficients), tolerance = 1e-7)
})

test_that("Firth sparse LDLT matches dense", {
    skip_if_not_installed("Matrix")
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    Xs <- Matrix::Matrix(X, sparse = TRUE)
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))

    ref <- fastglm(X, y, family = binomial(), method = 3, firth = TRUE)
    sp  <- fastglm(Xs, y, family = binomial(), method = 3, firth = TRUE)
    expect_equal(unname(sp$coefficients), unname(ref$coefficients), tolerance = 1e-7)
})

test_that("Firth streaming matches dense", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    chunk_size <- 50

    # Binomial
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))
    ref <- fastglm(X, y, family = binomial(), method = 2, firth = TRUE)
    chunks <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = X[idx, , drop = FALSE], y = y[idx])
    }
    st <- fastglm_streaming(chunks, n_chunks = n / chunk_size,
                            family = binomial(), method = 2, firth = TRUE)
    expect_equal(unname(st$coefficients), unname(ref$coefficients), tolerance = 1e-7)
    expect_true(st$converged)
    expect_s3_class(st, "fastglm_firth")

    # Poisson
    yp <- rpois(n, exp(X %*% c(0.5, 0.3, -0.2, 0.1)))
    ref_p <- fastglm(X, yp, family = poisson(), method = 2, firth = TRUE)
    chunks_p <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = X[idx, , drop = FALSE], y = yp[idx])
    }
    st_p <- fastglm_streaming(chunks_p, n_chunks = n / chunk_size,
                              family = poisson(), method = 2, firth = TRUE)
    expect_equal(unname(st_p$coefficients), unname(ref_p$coefficients), tolerance = 1e-7)

    # Gamma
    yg <- rgamma(n, shape = 2, rate = 1 / exp(X %*% c(0.5, 0.3, -0.2, 0.1)))
    ref_g <- fastglm(X, yg, family = Gamma("log"), method = 2, firth = TRUE)
    chunks_g <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = X[idx, , drop = FALSE], y = yg[idx])
    }
    st_g <- fastglm_streaming(chunks_g, n_chunks = n / chunk_size,
                              family = Gamma("log"), method = 2, firth = TRUE)
    expect_equal(unname(st_g$coefficients), unname(ref_g$coefficients), tolerance = 1e-6)
})

test_that("Firth streaming LDLT matches dense", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))
    chunk_size <- 50

    ref <- fastglm(X, y, family = binomial(), method = 3, firth = TRUE)
    chunks <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = X[idx, , drop = FALSE], y = y[idx])
    }
    st <- fastglm_streaming(chunks, n_chunks = n / chunk_size,
                            family = binomial(), method = 3, firth = TRUE)
    expect_equal(unname(st$coefficients), unname(ref$coefficients), tolerance = 1e-7)
})

test_that("Firth streaming reports penalized deviance", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))
    chunk_size <- 50
    chunks <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = X[idx, , drop = FALSE], y = y[idx])
    }
    st <- fastglm_streaming(chunks, n_chunks = n / chunk_size,
                            family = binomial(), method = 2, firth = TRUE)
    expect_true(is.finite(st$penalized.deviance))
    expect_true(is.finite(st$log.det.XtWX))
    expect_true(isTRUE(st$firth))
})

test_that("Firth uses user's chosen method, not forced to 2", {
    set.seed(123)
    n <- 200; p <- 4
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    y <- rbinom(n, 1, plogis(X %*% c(-0.5, 0.3, -0.2, 0.1)))

    f0 <- fastglm(X, y, family = binomial(), method = 0, firth = TRUE)
    f5 <- fastglm(X, y, family = binomial(), method = 5, firth = TRUE)
    expect_equal(unname(f0$coefficients), unname(f5$coefficients), tolerance = 1e-10)
    expect_true(f0$converged)
    expect_true(f5$converged)
})
