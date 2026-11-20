# Multinomial (vector GLM) fits must be identical for every thread count.
# Fits use n = 20000 rows in kernel blocks of 1000 rows (20 blocks), so the
# reductions run in parallel whenever more than one thread is requested.

multinom_fields <- c("coefficients", "deviance", "loglik", "iter", "passes",
                     "products", "vcov", "information", "score", "fitted.values")

# A multinomial data set with p design columns (intercept included) and K
# categories. Coefficients are small so that no category is empty and the
# fit is well inside the parameter space.
thread_multinom_data <- function(n, p, K, seed = 20261119L, sd = 0.3) {
    set.seed(seed)
    x <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
    colnames(x) <- c("(Intercept)", paste0("x", seq_len(p - 1L)))
    beta <- matrix(rnorm((K - 1L) * p, sd = sd), p, K - 1L)
    eta <- cbind(0, x %*% beta)
    pr <- exp(eta - apply(eta, 1L, max))
    pr <- pr / rowSums(pr)
    u <- runif(n)
    cls <- 1L + rowSums(u > t(apply(pr, 1L, cumsum)))
    cls <- pmin(cls, K)
    lev <- paste0("c", seq_len(K))
    list(x = x, y = factor(lev[cls], levels = lev))
}

# A fit with control$threads = t, with vcov(fit) stored as `vcov`.
multinom_threads_fit <- function(...) {
    function(t) {
        args <- list(...)
        args$control <- c(args$control, list(threads = t))
        fit <- do.call(fastglm_multinom, args)
        fit$vcov <- vcov(fit)
        fit
    }
}

test_that("K = 3, p = 4 (lanes, pairwise) is identical for every thread count", {
    d <- thread_multinom_data(20000L, 4L, 3L)
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = d$x, y = d$y, block.size = 1000L), multinom_fields)
    expect_true(ref$converged)
    expect_identical(ref$accumulation, "pairwise")
    expect_identical(ref$solver, "exact")
    expect_identical(ref$threads, 1L)
})

test_that("K = 9, p = 6 (lanes, low rank) is identical for every thread count", {
    d <- thread_multinom_data(20000L, 6L, 9L)
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = d$x, y = d$y, block.size = 1000L,
                             control = list(solver = "exact")), multinom_fields)
    expect_true(ref$converged)
    expect_identical(ref$accumulation, "lowrank")
})

test_that("K = 80, p = 60 with the cg solver (P = 4740, column tiles) is identical for every thread count", {
    # About 10 s per fit at one thread.
    skip_on_cran()
    # Memory: the full information is P x P = 180 MB, and a full pass holds
    # it twice more (S and the packed G, freed after the pass); each fit
    # keeps information and cov.unscaled (360 MB), and two fits are alive.
    d <- thread_multinom_data(6000L, 60L, 80L, sd = 0.05)
    fields <- c("coefficients", "deviance", "iter", "passes", "products", "vcov")
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = d$x, y = d$y, block.size = 500L,
                             control = list(solver = "cg")), fields)
    expect_true(ref$converged)
    expect_identical(ref$solver, "cg")
    expect_gt(ref$products, 0L)
    expect_identical(ref$accumulation, "lowrank")
})

test_that("pairwise and diagonal column tiles (K = 3, p = 730, two groups) are identical for every thread count", {
    # P = 1460 > 1024, so full passes use column tiles of the pairwise
    # information (K - 1 < 6); the cg solver's diagonal passes hold
    # 2 x 730^2 > 1024^2 entries, so they use column tiles too. Sparse
    # designs add these blocks on the calling thread with the serial kernels,
    # which checks the tile arithmetic. n = 4300 in blocks of 500 rows gives
    # 9 blocks: a full group of 8 and a partial last group with a short block.
    # Memory: S is 1460^2 doubles (17 MB), the design 25 MB.
    skip_on_cran()
    skip_if_not_installed("Matrix")
    d <- thread_multinom_data(4300L, 730L, 3L, sd = 0.02)
    x <- d$x
    x[, -1][abs(x[, -1]) < 1.6] <- 0
    xs <- Matrix::Matrix(x, sparse = TRUE)
    fields <- c("coefficients", "deviance", "iter", "passes", "products", "vcov")
    thr <- c(1L, 3L, 8L)
    exact <- expect_thread_invariant(
        multinom_threads_fit(x = x, y = d$y, block.size = 500L,
                             control = list(solver = "exact")), fields, threads = thr)
    cg <- expect_thread_invariant(
        multinom_threads_fit(x = x, y = d$y, block.size = 500L,
                             control = list(solver = "cg")), fields, threads = thr)
    sparse <- expect_thread_invariant(
        multinom_threads_fit(x = xs, y = d$y, block.size = 500L,
                             control = list(solver = "cg")), fields, threads = thr)
    serial <- fastglm_multinom(xs, d$y, block.size = 500L, control = list(solver = "exact"))
    expect_true(exact$converged && cg$converged && sparse$converged)
    expect_identical(exact$accumulation, "pairwise")
    expect_max_diff(exact$coefficients, serial$coefficients, 1e-12)
    expect_max_diff(exact$cov.unscaled, serial$cov.unscaled, 1e-12)
    expect_max_diff(cg$coefficients, sparse$coefficients, 1e-12)
    expect_max_diff(cg$coefficients, exact$coefficients, 1e-6)
})

test_that("sparse designs with K = 4 are identical for every thread count", {
    skip_if_not_installed("Matrix")
    d <- thread_multinom_data(20000L, 6L, 4L)
    x <- d$x
    x[, 2:6][abs(x[, 2:6]) < 0.8] <- 0
    xs <- Matrix::Matrix(x, sparse = TRUE)
    expect_s4_class(xs, "dgCMatrix")
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = xs, y = d$y, block.size = 1000L), multinom_fields)
    expect_true(ref$converged)
    dense <- fastglm_multinom(x, d$y, block.size = 1000L)
    expect_equal(unname(ref$coefficients), unname(dense$coefficients), tolerance = 1e-10)
})

test_that("big.matrix designs are identical for every thread count", {
    skip_if_not_installed("bigmemory")
    d <- thread_multinom_data(20000L, 4L, 3L)
    xb <- bigmemory::as.big.matrix(d$x)
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = xb, y = d$y, block.size = 1000L),
        setdiff(multinom_fields, "fitted.values"))
    dense <- fastglm_multinom(d$x, d$y, block.size = 1000L)
    expect_equal(unname(ref$coefficients), unname(dense$coefficients), tolerance = 1e-10)
})

test_that("source fits over 10 blocks are identical for every thread count", {
    d <- thread_multinom_data(20000L, 4L, 3L)
    chunk <- function(k) {
        i <- ((k - 1L) * 2000L + 1L):(k * 2000L)
        list(X = d$x[i, , drop = FALSE], y = d$y[i])
    }
    fields <- c("coefficients", "deviance", "loglik", "iter", "passes", "vcov")
    ref <- expect_thread_invariant(function(t) {
        fit <- fastglm_multinom(fastglm_source(chunk, 10L), block.size = 1000L,
                                control = list(threads = t))
        fit$vcov <- vcov(fit)
        fit
    }, fields)
    expect_true(ref$converged)
    dense <- fastglm_multinom(d$x, d$y)
    expect_equal(unname(ref$coefficients), unname(dense$coefficients), tolerance = 1e-8)
})

test_that("bias-reduced fits (firth = TRUE) with K = 3 are identical for every thread count", {
    d <- thread_multinom_data(20000L, 4L, 3L)
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = d$x, y = d$y, block.size = 1000L, firth = TRUE),
        c(multinom_fields, "adjustment", "penalized.loglik"))
    expect_true(ref$converged)
})

test_that("a restricted-step model type gives the same shrink and halving for every thread count", {
    # step_bound (test only) limits every change in eta, so the passes take
    # the shrink path of the engine, with the step check made in parallel.
    d <- thread_multinom_data(20000L, 4L, 3L, sd = 0.6)
    ref <- expect_thread_invariant(
        multinom_threads_fit(x = d$x, y = d$y, block.size = 1000L,
                             control = list(step_bound = 0.2)),
        c("coefficients", "deviance", "iter", "passes", "vcov"))
    expect_true(ref$converged)
    # Steps were shortened: more passes than one per iteration.
    expect_gt(ref$passes, ref$iter + 1L)
})

test_that("multinomial fits run in parallel and record the thread count", {
    skip_if_not(fastglm_threads()$openmp, "built without OpenMP")
    skip_if(fastglm:::.fastglm_check_cap(4L) < 2L || fastglm_threads()$logical.cores < 2L)
    d <- thread_multinom_data(20000L, 4L, 3L)
    session <- fastglm_threads()$threads
    session.option <- getOption("fastglm.threads")
    fastglm:::fastglm_max_team_size_test(TRUE)
    fit <- fastglm_multinom(d$x, d$y, block.size = 1000L, control = list(threads = 2L))
    expect_gt(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
    expect_identical(fit$threads, 2L)
    fastglm:::fastglm_max_team_size_test(TRUE)
    fit1 <- fastglm_multinom(d$x, d$y, block.size = 1000L, control = list(threads = 1L))
    expect_identical(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
    expect_identical(fit1$threads, 1L)
    # The session setting is restored, and threads is not a refit control.
    expect_identical(fastglm_threads()$threads, session)
    expect_identical(getOption("fastglm.threads"), session.option)
    expect_null(fit$refit.controls$control$threads)
    expect_identical(fastglm_diagnostics(fit)$threads, 2L)
})

test_that("control$threads is validated", {
    d <- thread_multinom_data(200L, 3L, 3L)
    expect_error(fastglm_multinom(d$x, d$y, control = list(threads = 0)),
                 "'control\\$threads' must be a positive integer")
    expect_error(fastglm_multinom(d$x, d$y, control = list(threads = "a")),
                 "'control\\$threads' must be a positive integer")
})

test_that("bootstrap refits of a multinomial fit ignore the original fit's threads", {
    d <- thread_multinom_data(400L, 3L, 3L)
    fit <- fastglm_multinom(d$x, d$y, control = list(threads = 2L))
    old <- fastglm_threads()$threads
    seen <- integer(0)
    stat <- function(f) { seen <<- c(seen, f$threads); coef(f) }
    b <- fastglm_bootstrap(fit, R = 2L, statistic = stat, seed = 1L,
                           control = list(threads = 1L))
    expect_identical(b$successful, 2L)
    # The first value is the statistic of the original fit (2 threads, or 1
    # on a build without OpenMP).
    expect_identical(seen, c(if (fastglm_threads()$openmp) 2L else 1L, 1L, 1L))
    expect_identical(fastglm_threads()$threads, old)
})
