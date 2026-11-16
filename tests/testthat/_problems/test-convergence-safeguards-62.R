# Extracted from test-convergence-safeguards.R:62

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
for (seed in c(814277366L, 814277368L, 814277369L, 814277370L, 814277371L)) {
        set.seed(seed); n <- 100L; x1 <- runif(n)
        y <- rbinom(n, 1, exp(-1.31 * x1)); x <- cbind(1, x1)
        fit <- suppressWarnings(fastglm(x, y, family = binomial("log"), start = c(-1, -1)))
        if (fit$converged) {
            expect_lt(fisher_decrement(x, y, binomial("log"), coef(fit)), 1e-6 * (0.1 + fit$deviance))
        } else {
            expect_true(fit$convergence.reason %in% c("step_halving_failed", "iteration_limit"))
        }
    }
