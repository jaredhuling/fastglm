# Extracted from test-convergence-safeguards.R:38

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
set.seed(9301L)
n <- 1000L
z <- matrix(runif(n * 2), n, 2)
x <- cbind(1, z)
y <- rpois(n, 0.2 + drop(z %*% c(2, 1.5)))
fit <- fastglm(x, y, family = poisson("identity"), tol = 1e-12, maxit = 200L)
ref <- glm(y ~ z, family = poisson("identity"), start = c(1, 1, 1),
               control = glm.control(epsilon = 1e-12, maxit = 200))
expect_true(fit$converged)
expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-8)
