# Convergence is reported only when the Fisher decrement at the estimate is
# small, and the default start is a valid fit (the intercept-only fit), so
# that an invalid first step can be halved back into the parameter space.

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

test_that("an invalid first step is halved toward a valid start and reaches the optimum", {
    # With zero starting coefficients, fastglm 0.1.4 reported convergence
    # after 3 iterations at deviance 5293.08 for these data.
    d <- gamma_sqrt_data(1400784547L)
    fit <- suppressWarnings(fastglm(d$x, d$y, family = Gamma("sqrt")))
    expect_true(fit$converged)
    expect_equal(fit$convergence.reason, "converged")
    expect_lt(fit$deviance, 1786.78)
    expect_lt(fisher_decrement(d$x, d$y, Gamma("sqrt"), coef(fit)), 1e-8 * (0.1 + fit$deviance))
})

test_that("identity-link models fit without user-supplied starting values", {
    set.seed(9301L); n <- 1000L
    z <- matrix(runif(n * 2), n, 2); x <- cbind(1, z)
    y <- rpois(n, 0.2 + drop(z %*% c(2, 1.5)))
    fit <- fastglm(x, y, family = poisson("identity"), tol = 1e-12, maxit = 200L)
    ref <- glm(y ~ z, family = poisson("identity"), start = c(1, 1, 1),
               control = glm.control(epsilon = 1e-12, maxit = 200))
    expect_true(fit$converged)
    expect_equal(fit$deviance, ref$deviance, tolerance = 1e-12)
    expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-6)

    yg <- rgamma(n, shape = 1, scale = 0.1 + drop(z %*% c(2, 1)))
    fitg <- fastglm(x, yg, family = Gamma("identity"), tol = 1e-12, maxit = 200L)
    refg <- glm(yg ~ z, family = Gamma("identity"), start = c(1, 1, 1),
                control = glm.control(epsilon = 1e-12, maxit = 200))
    expect_true(fitg$converged)
    # The Gamma likelihood is flat enough near its optimum that two fits at
    # tol = 1e-12 agree in deviance to about 1e-15 but in coefficients only
    # to about 1e-7.
    expect_equal(fitg$deviance, refg$deviance, tolerance = 1e-12)
    expect_equal(unname(coef(fitg)), unname(coef(refg)), tolerance = 1e-6)
})

test_that("a fit reported as converged has a small Fisher decrement", {
    # Log-binomial models with the maximum likelihood estimate on the
    # boundary: fastglm 0.1.4 reported convergence at suboptimal estimates
    # for several of these seeds.
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
})

test_that("convergence rule 0 reproduces the deviance-only rule of fastglm 0.1.4", {
    old <- fastglm:::set_conv_rule(0L)
    on.exit(fastglm:::set_conv_rule(old), add = TRUE)
    set.seed(5521L); x <- cbind(1, matrix(rnorm(3000), 1000)); y <- rbinom(1000, 1, 0.3)
    f0 <- fastglm(x, y, family = binomial())
    g <- glm.fit(x, y, family = binomial(), control = list(epsilon = 1e-8))
    expect_equal(f0$iter, g$iter)
    expect_equal(unname(coef(f0)), unname(coef(g)), tolerance = 1e-12)
})

test_that("easy fits keep their estimates and iteration counts under the default rule", {
    # Standard errors use the information at the reported estimate (glm.fit
    # uses the weights of the preceding iterate, which differ slightly).
    set.seed(7717L); x <- cbind(1, matrix(rnorm(20000), 2000))
    eta <- drop(x %*% c(0.2, rep(c(0.3, -0.3), 5)))
    for (fam in list(binomial(), poisson(), Gamma("log"))) {
        y <- switch(fam$family, binomial = rbinom(2000, 1, plogis(eta)),
                    poisson = rpois(2000, exp(eta)), Gamma = rgamma(2000, 2, scale = exp(eta) / 2))
        f <- fastglm(x, y, family = fam)
        g <- glm.fit(x, y, family = fam, control = list(epsilon = 1e-8))
        expect_true(f$converged)
        expect_equal(f$iter, g$iter)
        expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-10)
        mu <- fam$linkinv(drop(x %*% coef(f)))
        w <- fam$mu.eta(fam$linkfun(mu))^2 / fam$variance(mu)
        unscaled <- solve(crossprod(x * sqrt(w)))
        expect_equal(unname(f$cov.unscaled), unname(unscaled), tolerance = 1e-10)
    }
})

stream_fit <- function(x, y, family, method = "llt", K = 4L) {
    rows <- split(seq_len(nrow(x)), cut(seq_len(nrow(x)), K, labels = FALSE))
    reads <- 0L
    fit <- fastglm_streaming(function(k) {
        reads <<- reads + 1L
        list(X = x[rows[[k]], , drop = FALSE], y = y[rows[[k]]])
    }, n_chunks = K, family = family, method = method)
    fit$passes <- reads / K
    fit
}

test_that("streamed fits halve an invalid first step toward a valid start", {
    d <- gamma_sqrt_data(1400784547L)
    for (m in c("llt", "qr")) {
        sf <- suppressWarnings(stream_fit(d$x, d$y, Gamma("sqrt"), m))
        df <- suppressWarnings(fastglm(d$x, d$y, family = Gamma("sqrt"), method = m))
        expect_true(sf$converged)
        expect_lt(sf$deviance, 1786.78)
        expect_equal(unname(coef(sf)), unname(coef(df)), tolerance = 1e-10)
    }
    set.seed(9301L); n <- 1000L
    z <- matrix(runif(n * 2), n, 2); x <- cbind(1, z)
    y <- rpois(n, 0.2 + drop(z %*% c(2, 1.5)))
    sf <- stream_fit(x, y, poisson("identity"))
    expect_true(sf$converged)
    expect_equal(sf$deviance, fastglm(x, y, family = poisson("identity"))$deviance, tolerance = 1e-12)
})

test_that("streamed fits match dense fits and skip the separate totals and final passes", {
    set.seed(6603L); x <- cbind(1, matrix(rnorm(40000 * 9), 40000)); eta <- drop(x %*% c(0.2, rep(0.15, 9)))
    for (fam in list(binomial(), poisson())) {
        y <- if (fam$family == "binomial") rbinom(40000, 1, plogis(eta)) else rpois(40000, exp(eta))
        sf <- stream_fit(x, y, fam, "llt")
        df <- fastglm(x, y, family = fam, method = "llt")
        expect_equal(unname(coef(sf)), unname(coef(df)), tolerance = 1e-12)
        expect_equal(unname(sf$se), unname(df$se), tolerance = 1e-10)
        # Validation pass, initial pass, and one pass per iteration; 0.1.4
        # also read the data for the totals and for the final statistics.
        expect_lte(sf$passes, df$iter + 2 + 1)
    }
})
