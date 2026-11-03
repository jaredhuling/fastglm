test_that("constant-information Firth fits agree across weighted backends", {
    set.seed(81031)
    n <- 320L
    X <- cbind(1, matrix(rnorm(n * 4L, sd = .15), n))
    X[, -1L][abs(X[, -1L]) < .12] <- 0
    offset <- runif(n, -.03, .03)
    families <- list(gaussian(), Gamma("log"), poisson("sqrt"),
                     Gamma("identity"), Gamma("inverse"), poisson("log"))
    weights <- list(rep(1, n), rep(c(0, .25, 1.5, 3), n / 4L),
                    seq(.3, 2, length.out = n))
    for (family in families) {
        beta <- c(if (family$link == "sqrt") 3 else 1.2, rep(.15, 4))
        mu <- family$linkinv(drop(X %*% beta) + offset)
        y <- switch(family$family, gaussian = rnorm(n, mu, .7),
                    Gamma = rgamma(n, shape = 2, scale = mu / 2),
                    poisson = rpois(n, mu))
        for (w in weights) {
            dense <- fastglm(X, y, family, weights = w, offset = offset,
                             start = beta, method = 2L, firth = TRUE, tol = 1e-10)
            for (method in 2:3) {
                sparse <- fastglm(Matrix::Matrix(X, sparse = TRUE), y, family,
                                  weights = w, offset = offset, start = beta,
                                  method = method, firth = TRUE, tol = 1e-10)
                last <- 0L
                reads <- 0L
                callback <- function(k) {
                    if (k != 1L) stopifnot(k == last + 1L)
                    last <<- k
                    reads <<- reads + 1L
                    ii <- ((k - 1L) * 80L + 1L):(k * 80L)
                    list(X = X[ii, , drop = FALSE], y = y[ii],
                         weights = w[ii], offset = offset[ii])
                }
                streamed <- fastglm_streaming(callback, 4L, family,
                    start = beta, method = method, firth = TRUE, tol = 1e-10)
                expect_equal(reads, streamed$chunk.reads)
                expect_equal(reads, 4L * streamed$passes)
                for (fit in list(sparse, streamed)) {
                    expect_true(fit$converged)
                    expect_equal(unname(coef(fit)), unname(coef(dense)), tolerance = 2e-7)
                    expect_equal(fit$dispersion, dense$dispersion, tolerance = 2e-7)
                    eta <- drop(X %*% coef(fit)) + offset
                    muhat <- family$linkinv(eta)
                    working <- w * family$mu.eta(eta)^2 / family$variance(muhat)
                    information <- crossprod(X, working * X)
                    expect_equal(unname(fit$cov.unscaled), unname(solve(information)),
                                 tolerance = 2e-10)
                }
            }
        }
    }
})

test_that("cached Firth information retains rank and convergence safeguards", {
    X <- cbind(1, rep(1, 80L))
    y <- seq(.4, 2, length.out = nrow(X))
    callback <- function(k) list(X = X, y = y)
    expect_error(fastglm(Matrix::Matrix(X, sparse = TRUE), y, Gamma("log"),
                         firth = TRUE, method = 2), "singular|positive definite|ill-conditioned")
    expect_error(fastglm_streaming(callback, 1L, Gamma("log"), firth = TRUE),
                 "rank deficient")
    set.seed(81032)
    X <- cbind(1, rnorm(80L))
    callback <- function(k) list(X = X, y = y)
    expect_warning(fit <- fastglm_streaming(callback, 1L, Gamma("log"),
        firth = TRUE, start = c(0, 0), maxit = 1L), "did not converge")
    expect_false(fit$converged)
    eta <- drop(X %*% coef(fit))
    expect_equal(unname(fit$cov.unscaled), unname(solve(crossprod(X))), tolerance = 1e-12)
})

test_that("Gamma chunk grouping preserves skewed and small dispersion fits", {
    set.seed(81033)
    n <- 240L
    X <- cbind(1, runif(n, -.3, .3))
    w <- c(rep(0, 60L), rep(c(.125, .7, 2), 60L))
    offset <- runif(n, -.05, .05)
    mu <- exp(drop(X %*% c(.6, .2)) + offset)
    for (shape in c(.3, 1e6)) {
        y <- rgamma(n, shape = shape, scale = mu / shape)
        dense <- fastglm(X, y, Gamma("log"), weights = w, offset = offset,
                         firth = TRUE, method = 2L, tol = 1e-10)
        callback <- function(k) {
            ii <- ((k - 1L) * 60L + 1L):(k * 60L)
            list(X = X[ii, , drop = FALSE], y = y[ii], weights = w[ii], offset = offset[ii])
        }
        fit <- fastglm_streaming(callback, 4L, Gamma("log"), firth = TRUE, tol = 1e-10)
        expect_true(fit$converged)
        expect_equal(unname(coef(fit)), unname(coef(dense)), tolerance = 2e-7)
        expect_equal(fit$dispersion, dense$dispersion, tolerance = 2e-7)
    }
})

test_that("Poisson sqrt information respects the derivative floor", {
    X <- matrix(1, 40L, 1L)
    y <- numeric(40L)
    family <- poisson("sqrt")
    callback <- function(k) list(X = X, y = y)
    expect_warning(sparse <- fastglm(Matrix::Matrix(X, sparse = TRUE), y, family,
                     start = 1e-18, method = 2L, firth = TRUE), "numerically 0")
    streamed <- fastglm_streaming(callback, 1L, family, start = 1e-18, firth = TRUE)
    for (fit in list(sparse, streamed)) {
        eta <- drop(X %*% coef(fit))
        working <- pmax(2*eta, .Machine$double.eps)^2 / eta^2
        expect_equal(unname(fit$cov.unscaled), unname(solve(crossprod(X, working * X))),
                     tolerance = 1e-12)
    }
})

test_that("big.matrix Firth information caches retain weighted results", {
    skip_if_not_installed("bigmemory")
    set.seed(81034)
    X <- cbind(1, matrix(rnorm(1200L, sd = .1), 300L))
    xb <- bigmemory::as.big.matrix(X)
    w <- rep(c(0, .3, 1, 2, 3), 60L)
    offset <- runif(300L, -.02, .02)
    for (family in list(gaussian(), Gamma("log"), poisson("sqrt"))) {
        beta <- c(2, rep(.2, 4L))
        mu <- family$linkinv(drop(X %*% beta) + offset)
        y <- switch(family$family, gaussian = rnorm(300L, mu),
                    Gamma = rgamma(300L, shape = 3, scale = mu/3),
                    poisson = rpois(300L, mu))
        dense <- fastglm(X, y, family, weights = w, offset = offset,
                         start = beta, firth = TRUE, method = 2L, tol = 1e-10)
        for (method in 2:3) {
            fit <- fastglm(xb, y, family, weights = w, offset = offset,
                           start = beta, firth = TRUE, method = method, tol = 1e-10)
            expect_true(fit$converged)
            expect_equal(unname(coef(fit)), unname(coef(dense)), tolerance = 2e-7)
            expect_equal(fit$dispersion, dense$dispersion, tolerance = 2e-7)
            expect_equal(unname(fit$cov.unscaled), unname(dense$cov.unscaled), tolerance = 2e-10)
        }
    }
})
