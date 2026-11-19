# The `solver` control of fastglm_multinom(): resolution of "auto", agreement
# of the conjugate gradient and exact solvers, and the fitted object fields.

test_that(".vglm_resolve_solver covers every branch", {
    th <- fastglm:::.vglm_cg_threshold
    r <- fastglm:::.vglm_resolve_solver
    expect_identical(r("exact", 1000L, 50L, FALSE, FALSE), "exact")
    expect_identical(r("cg", 2L, 2L, FALSE, FALSE), "cg")
    expect_identical(r("cg", 2L, 2L, FALSE, TRUE), "cg")
    expect_identical(r("auto", 1000L, 50L, TRUE, FALSE), "exact")
    expect_identical(r("auto", 1000L, 50L, FALSE, TRUE), "exact")
    mp <- fastglm:::.vglm_cg_min_p
    expect_identical(mp, 8L)
    # p * M at the threshold: "cg" only from the floor on p.
    expect_identical(r("auto", mp, ceiling(th / mp) - 1L, FALSE, FALSE), "exact")
    expect_identical(r("auto", mp, ceiling(th / mp), FALSE, FALSE), "cg")
    expect_identical(r("auto", mp - 1L, 1000L, FALSE, FALSE), "exact")
    expect_identical(r("auto", 5L, 100L, FALSE, FALSE), "exact")
    # Intercept-only and null fits (p = 1) stay on the exact solver.
    expect_identical(r("auto", 1L, th, FALSE, FALSE), "exact")
    expect_identical(r("auto", 1L, 10L * th, FALSE, FALSE), "exact")
    expect_identical(r("auto", 10L, 50L, FALSE, FALSE), "cg")
    expect_error(r("cg", 5L, 5L, TRUE, FALSE),
                 "the conjugate gradient solver is not available with bias reduction")
})

test_that("auto picks the solver by p * (K - 1) and the fit reports it", {
    d <- vglm_sim_data(300L, 2L, 3L, seed = 11L)
    f1 <- fastglm_multinom(d$x, d$y)
    expect_identical(f1$solver, "exact")
    expect_true(is.numeric(f1$products))
    expect_true(!is.null(f1$accumulation))
    big <- vglm_sim_data(4000L, 7L, 60L, seed = 12L)
    f2 <- suppressWarnings(fastglm_multinom(big$x, big$y, control = list(solver = "auto"), maxit = 3L))
    expect_identical(f2$solver, "cg")
    expect_identical(f2$refit.controls$control$solver, "cg")
    expect_identical(f1$refit.controls$control$solver, "exact")
    # Firth fits use the exact solver under "auto".
    f3 <- suppressWarnings(fastglm_multinom(big$x[, 1:2], big$y, firth = TRUE, maxit = 2L,
                                            control = list(solver = "auto")))
    expect_identical(f3$solver, "exact")
})

test_that("a formula fit with cg agrees with the exact solver", {
    d <- vglm_sim_data(600L, 3L, 4L, seed = 13L)
    df <- data.frame(y = d$y, d$x[, -1])
    ex <- fastglm_multinom(y ~ x1 + x2 + x3, df, tol = 1e-12,
                           control = list(solver = "exact"))
    cg <- fastglm_multinom(y ~ x1 + x2 + x3, df, tol = 1e-12,
                           control = list(solver = "cg"))
    expect_identical(cg$solver, "cg")
    expect_identical(ex$solver, "exact")
    expect_true(cg$converged)
    expect_max_diff(cg$coefficients, ex$coefficients, 1e-6)
    expect_max_diff(cg$cov.unscaled, ex$cov.unscaled, 1e-6)
    expect_max_diff(cg$deviance, ex$deviance, 1e-10)
})

test_that("cg on a source warns once and agrees; invalid values and firth error", {
    set.seed(21L)
    n <- 400L
    d <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
    d$y <- factor(sample(c("A", "B", "C"), n, TRUE))
    idx <- split(seq_len(n), rep(1:4, each = n / 4))
    src <- fastglm_source(function(k) d[idx[[k]], ], 4L, formula = y ~ x1 + x2,
                          prototype = d[1:20, ])
    de <- fastglm_multinom(y ~ x1 + x2, d, tol = 1e-12)
    msg <- "fastglm_multinom: solver = \"cg\" reads the whole source for every conjugate gradient step"
    w <- character()
    sf <- withCallingHandlers(
        fastglm_multinom(src, tol = 1e-12, control = list(solver = "cg")),
        warning = function(c) { w <<- c(w, conditionMessage(c)); invokeRestart("muffleWarning") })
    expect_identical(sum(w == msg), 1L)
    expect_identical(sf$solver, "cg")
    expect_max_diff(sf$coefficients, de$coefficients, 1e-6)
    expect_identical(fastglm_multinom(src)$solver, "exact")

    expect_error(fastglm_multinom(y ~ x1, d, control = list(solver = "newton")),
                 "fastglm_multinom: 'control$solver' must be \"auto\", \"exact\", or \"cg\"",
                 fixed = TRUE)
    expect_error(fastglm_multinom(y ~ x1, d, control = list(solver = c("cg", "exact"))),
                 "'control$solver' must be", fixed = TRUE)
    expect_error(fastglm_multinom(y ~ x1, d, firth = TRUE, control = list(solver = "cg")),
                 "the conjugate gradient solver is not available with bias reduction")
})

test_that("bootstrap refits of a cg fit use cg", {
    d <- vglm_sim_data(300L, 2L, 3L, seed = 31L)
    fit <- fastglm_multinom(d$x, d$y, control = list(solver = "cg"))
    expect_identical(fit$refit.controls$control$solver, "cg")
    z <- list(X = d$x, y = d$y, weights = rep(1, nrow(d$x)),
              offset = matrix(0, nrow(d$x), 2L))
    r <- fastglm:::.vglm_bootstrap_refit(fit, z, list())
    expect_identical(r$solver, "cg")
    b <- fastglm_bootstrap(fit, R = 3L, seed = 1L)
    expect_equal(b$successful, 3L)
})

test_that("sparse and big.matrix fits record the resolved solver; null fit honors cg", {
    d <- vglm_sim_data(200L, 2L, 3L, seed = 41L)
    sp <- fastglm_multinom(Matrix::Matrix(d$x, sparse = TRUE), d$y,
                           control = list(solver = "cg"))
    expect_identical(sp$solver, "cg")
    expect_identical(sp$refit.controls$control$solver, "cg")
    skip_if_not_installed("bigmemory")
    bm <- fastglm_multinom(bigmemory::as.big.matrix(d$x), d$y,
                           control = list(solver = "cg"))
    expect_identical(bm$solver, "cg")
    expect_identical(bm$refit.controls$control$solver, "cg")
    # The null fit uses cg as well when requested explicitly.
    expect_true(is.finite(sp$null.deviance))
})
