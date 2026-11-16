# Extracted from test-convergence-safeguards.R:25

# prequel ----------------------------------------------------------------------
fisher_decrement <- function(x, y, family, beta, weights = rep(1, length(y))) {
    eta <- drop(x %*% beta); mu <- family$linkinv(eta); me <- family$mu.eta(eta)
    w <- weights * me^2 / family$variance(mu)
    g <- drop(crossprod(x, w * (y - mu) / me))
    sum(g * solve(crossprod(x * sqrt(w)), g))
}
gamma_sqrt_data <- function(seed, n = 2000L, p = 20L) {
    set.seed(seed)
    z <- matrix(rnorm(n * (p - 1L)), n, p - 1L)
    y <- exp(0.25 * z[, 1] - 0.25 * z[, 3] + 0.5 * z[, 4] - 0.5 * z[, 5] + rnorm(n)) + 0.1
    list(x = cbind(1, z), y = y)
}

# test -------------------------------------------------------------------------
d <- gamma_sqrt_data(1400784547L)
fit <- suppressWarnings(fastglm(d$x, d$y, family = Gamma("sqrt")))
expect_true(fit$converged)
expect_equal(fit$convergence.reason, "converged")
