# Regression tests for the findings of a confirmation study of convergence
# (fastglm-jss, notes/results_e07_stability_confirmation.md). Each data set
# is the one that exposed the problem; reference values are the best
# maxima found there by independent optimizers (CVXR for GLMs, SLSQP for
# negative binomial, BFGS and pscl for zero-inflated models).

test_that("a held boundary row is released when its multiplier is negative (NB, sqrt link)", {
    skip_if_not_installed("MASS")
    # The fit held two rows at eta = 0 and was reported converged at
    # log likelihood -944.984800; the maximum holds one row (-944.942405).
    set.seed(214611522); n <- 800; x <- matrix(runif(n * 3), n, 3)
    mu <- pmax(0.5 + drop(x %*% c(1.2, -0.9, 0.4)), 0.04)^2
    y <- MASS::rnegbin(n, mu = mu, theta = 1.5); X <- cbind(1, x)
    f <- suppressWarnings(fastglm_nb(X, y, link = "sqrt"))
    expect_true(f$converged)
    ll <- sum(dnbinom(y, size = f$theta, mu = f$fitted.values, log = TRUE))
    expect_lt(abs(ll - (-944.9424052879)), 1e-6)
})

test_that("zero-inflated fits near the maximum are certified, including a flat theta", {
    zinb <- function(seed) {
        set.seed(seed); n <- 600; x <- rnorm(n); w <- rnorm(n)
        yc <- stats::rnbinom(n, size = 1.2, mu = exp(-0.8 + 0.4 * x))
        data.frame(y = ifelse(rbinom(n, 1, plogis(-0.6 + 0.6 * w)) == 1, 0, yc), x = x, w = w)
    }
    # The log likelihood is flat and not concave in log theta in the first
    # (certification then searches over log theta), and the zero-inflation
    # probability tends to zero in the second (Newton gains a constant
    # fraction per step, so certification takes more than 20 steps).
    for (case in list(list(seed = 216773810, best = -405.0915936042),
                      list(seed = 216773811, best = -385.7312815860))) {
        f <- suppressWarnings(fastglm_zi(y ~ x | w, data = zinb(case$seed), dist = "negbin"))
        expect_true(f$converged, label = sprintf("seed %d fit converged (reason %s, %d Newton steps, theta %.6g, decrement %.3g)",
                                                 case$seed, f$convergence.reason, f$newton.iter, f$theta, f$em.decrement))
        expect_lt(abs(as.numeric(logLik(f)) - case$best) / (1 + abs(case$best)), 1e-6)
    }
})

test_that("streamed fits with an offset start from a valid intercept-only fit", {
    # Square-root Poisson with an offset: the first update was invalid and no
    # fallback start existed with an offset, so the fit stopped with an error.
    set.seed(214525716); n <- 1200
    x <- matrix(runif(n * 3, -1, 1), n, 3); o <- runif(n, 0, 0.2)
    y <- rpois(n, pmax(o + 0.8 + drop(x %*% c(0.9, -0.7, 0.3)), 0.05)^2); X <- cbind(1, x)
    ix <- split(seq_len(n), cut(seq_len(n), 4, labels = FALSE))
    f <- suppressWarnings(fastglm_streaming(function(k) list(X = X[ix[[k]], ], y = y[ix[[k]]], offset = o[ix[[k]]]),
                                            n_chunks = 4, family = poisson("sqrt")))
    expect_true(f$converged)
    expect_lt(abs(f$deviance - 1339.6366260605) / (1 + 1339.6366260605), 1e-6)
})

test_that("Cholesky failures from the working weights fall back to QR for sparse and streamed fits", {
    skip_if_not_installed("Matrix")
    # Binomial identity link with fitted probabilities near 0 and 1: the
    # working weights 1 / (mu (1 - mu)) make X'WX ill-conditioned, although
    # the design itself has full rank.
    set.seed(187829926); n <- 1500
    x <- matrix(runif(n * 3, -1, 1), n, 3)
    p <- pmin(pmax(0.5 + drop(x %*% c(0.6, -0.5, 0.2)), 0.01), 0.99)
    y <- rbinom(n, 1, p); X <- cbind(1, x)
    ix <- split(seq_len(n), cut(seq_len(n), 4, labels = FALSE))
    expect_no_error(s <- suppressWarnings(fastglm(as(X, "dgCMatrix"), y, family = binomial("identity"), method = "llt",
                                                  maxit = 400L)))
    expect_no_error(st <- suppressWarnings(fastglm_streaming(function(k) list(X = X[ix[[k]], ], y = y[ix[[k]]]),
                                                             n_chunks = 4, family = binomial("identity"), maxit = 400L)))
    for (f in list(s, st)) {
        expect_true(f$converged)
        expect_lt((f$deviance - 1376.9494619297) / (1 + 1376.9494619297), 1e-6)
    }
})

test_that("sparse negative binomial fits report a reason consistent with convergence", {
    skip_if_not_installed("Matrix")
    set.seed(201318074); n <- 400; x <- matrix(rnorm(n * 11), n, 11)
    beta <- c(1.5, rnorm(11, sd = 0.4))
    y <- MASS::rnegbin(n, mu = exp(pmin(drop(cbind(1, x) %*% beta), 12)), theta = 2000)
    f <- suppressWarnings(fastglm_nb(as(cbind(1, x), "dgCMatrix"), y))
    expect_identical(isTRUE(f$converged), f$convergence.reason %in% c("converged", "converged_boundary", "theta_infinite"))
})
