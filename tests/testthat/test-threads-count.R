# Count, two-part, batch, and bootstrap fits must give identical results for
# every thread count. Each fit has n = 20000 rows in blocks of 1000 rows (20
# blocks), so the reductions run in parallel whenever more than one thread is
# requested.

thread_count_data <- function(n = 20000L, seed = 20261119L) {
    set.seed(seed)
    x1 <- rnorm(n); x2 <- rnorm(n)
    x <- cbind(1, x1, x2)
    mu <- exp(0.4 + 0.3 * x1 - 0.2 * x2)
    ynb <- MASS::rnegbin(n, mu = mu, theta = 1.5)
    pinf <- plogis(-0.5 + 0.4 * x1)
    yzi <- rbinom(n, 1L, 1 - pinf) * rpois(n, mu)
    yzinb <- rbinom(n, 1L, 1 - pinf) * ynb
    pos <- rbinom(n, 1L, plogis(0.3 + 0.2 * x2))
    yh <- pos * pmax(1, ynb)
    ytp <- pos * rgamma(n, shape = 2, rate = 2 / mu)
    ytr <- rpois(n, mu)
    while (any(ytr == 0)) ytr[ytr == 0] <- rpois(sum(ytr == 0), mu[ytr == 0])
    list(x = x, ynb = ynb, ytr = ytr,
         df = data.frame(ynb = ynb, yzi = yzi, yzinb = yzinb, yh = yh, ytp = ytp, x1 = x1, x2 = x2))
}

# The effective count for a request of t threads (reduced by the
# _R_CHECK_LIMIT_CORES_ cap and the core count, and 1 on a serial build).
effective_threads <- function(t) suppressWarnings(fastglm:::.fastglm_resolve_threads(t))

test_that("negative binomial fits (dense and sparse) are identical for every thread count", {
    skip_if_not_installed("MASS")
    d <- thread_count_data()
    fields <- c("coefficients", "theta", "SE.theta", "twologlik", "iter", "iter.theta",
                "cov.unscaled", "vcov.full")
    ref <- expect_thread_invariant(function(t) fastglm_nb(d$x, d$ynb, threads = t), fields)
    expect_true(ref$converged)
    skip_if_not_installed("Matrix")
    xs <- Matrix::Matrix(d$x, sparse = TRUE)
    sref <- expect_thread_invariant(function(t) fastglm_nb(xs, d$ynb, threads = t), fields)
    expect_true(sref$converged)
    expect_equal(unname(sref$coefficients), unname(ref$coefficients), tolerance = 1e-6)
})

test_that("zero-truncated Poisson and negative binomial fits are identical for every thread count", {
    d <- thread_count_data()
    fields <- c("coefficients", "loglik", "iter", "cov.unscaled", "vcov.full")
    ref <- expect_thread_invariant(function(t)
        fastglm_truncated(d$x, d$ytr, dist = "poisson", threads = t), fields)
    expect_true(ref$converged)
    pos <- d$ynb > 0
    nb <- expect_thread_invariant(function(t)
        fastglm_truncated(d$x[pos, ], d$ynb[pos], dist = "negbin", threads = t),
        c(fields, "theta", "SE.theta"))
    expect_true(nb$converged)
})

test_that("hurdle negative binomial fits are identical for every thread count", {
    skip_if_not_installed("Formula")
    d <- thread_count_data()
    ref <- expect_thread_invariant(function(t)
        fastglm_hurdle(yh ~ x1 + x2, data = d$df, dist = "negbin", threads = t),
        c("coefficients", "theta", "loglik", "vcov.full", "iter_count", "iter_zero", "outer_iter"))
    expect_true(ref$converged)
})

test_that("zero-inflated fits are identical for every thread count", {
    skip_if_not_installed("Formula")
    d <- thread_count_data()
    fields <- c("coefficients", "loglik", "vcov.full", "tau", "em_iter", "newton.iter")
    ref <- expect_thread_invariant(function(t)
        fastglm_zi(yzi ~ x1 + x2 | x1, data = d$df, dist = "poisson", threads = t), fields)
    expect_true(ref$converged)
    # The probit zero link is evaluated serially (R's math library); the rest
    # of the fit is threaded.
    expect_thread_invariant(function(t)
        fastglm_zi(yzi ~ x1 + x2 | x1, data = d$df, link = "probit", threads = t), fields)
})

test_that("zero-inflated NB and cloglog zero-link fits are identical for every thread count", {
    skip_if_not_installed("Formula")
    d <- thread_count_data()
    fields <- c("coefficients", "theta", "loglik", "vcov.full", "tau", "em_iter", "newton.iter")
    nb <- expect_thread_invariant(function(t)
        fastglm_zi(yzinb ~ x1 + x2 | x1, data = d$df, dist = "negbin", threads = t), fields)
    expect_true(nb$converged)
    expect_false(nb$theta.boundary)
    # The cloglog link takes the parallel branch of the inflation score's
    # elementwise loop (shared with the log link).
    expect_thread_invariant(function(t)
        fastglm_zi(yzi ~ x1 + x2 | x1, data = d$df, link = "cloglog", threads = t), fields)
    # The log link (inflation probability exp(-1.2 + 0.3 z) < 1 here).
    set.seed(12)
    z <- runif(nrow(d$x), -1, 1)
    df <- cbind(d$df, z = z, ylog = rbinom(nrow(d$x), 1L, 1 - exp(-1.2 + 0.3 * z)) *
                                    rpois(nrow(d$x), exp(0.4 + 0.3 * d$df$x1)))
    lg <- expect_thread_invariant(function(t)
        fastglm_zi(ylog ~ x1 + x2 | z, data = df, link = "log", threads = t), fields)
    expect_true(lg$converged)
})

test_that("count fits with weights and offsets are identical for every thread count", {
    skip_if_not_installed("MASS")
    skip_if_not_installed("Formula")
    d <- thread_count_data()
    n <- nrow(d$x)
    w <- rep(c(1, 2, 0, 1.5), length.out = n)
    off <- rep(c(0, 0.1, -0.2), length.out = n)
    expect_thread_invariant(function(t)
        fastglm_nb(d$x, d$ynb, weights = w, offset = off, threads = t),
        c("coefficients", "theta", "twologlik", "iter", "vcov.full"))
    df <- cbind(d$df, w = w, off = off)
    fields <- c("coefficients", "theta", "loglik", "vcov.full", "tau")
    expect_thread_invariant(function(t)
        fastglm_zi(yzi ~ x1 + x2 + offset(off) | x1, data = df, weights = w, threads = t), fields)
    expect_thread_invariant(function(t)
        fastglm_zi(yzinb ~ x1 + x2 + offset(off) | x1, data = df, weights = w, dist = "negbin",
                   threads = t), fields)
    pos <- d$ynb > 0
    expect_thread_invariant(function(t)
        fastglm_truncated(d$x[pos, ], d$ynb[pos], dist = "negbin", weights = w[pos],
                          offset = off[pos], threads = t),
        c("coefficients", "theta", "loglik", "vcov.full"))
})

test_that("two-part fits are identical for every thread count", {
    skip_if_not_installed("Formula")
    d <- thread_count_data()
    ref <- expect_thread_invariant(function(t)
        fastglm_twopart(ytp ~ x1 + x2, data = d$df, threads = t),
        c("coefficients", "loglik", "vcov"))
    expect_true(ref$converged)
})

test_that("count and two-part fits record their thread setting", {
    skip_if_not_installed("MASS")
    skip_if_not_installed("Formula")
    d <- thread_count_data(n = 2000L)
    t2 <- effective_threads(2L)
    pos <- d$ynb > 0
    fits <- list(
        nb = fastglm_nb(d$x, d$ynb, threads = 2L),
        nb.formula = fastglm_nb(ynb ~ x1 + x2, data = d$df, threads = 2L),
        truncated = fastglm_truncated(d$x, d$ytr, threads = 2L),
        hurdle = fastglm_hurdle(yh ~ x1 + x2, data = d$df, threads = 2L),
        hurdle.matrix = fastglm_hurdle(d$x, response = d$df$yh, threads = 2L),
        zi = fastglm_zi(yzi ~ x1 | x1, data = d$df, threads = 2L),
        twopart = fastglm_twopart(ytp ~ x1 + x2, data = d$df, threads = 2L))
    for (nm in names(fits)) expect_identical(fits[[nm]]$threads, t2, info = nm)
    old <- fastglm_threads(1L)
    on.exit(fastglm_threads(old$threads), add = TRUE)
    expect_identical(fastglm_nb(d$x, d$ynb)$threads, 1L)
    expect_identical(fastglm_zi(yzi ~ x1 | x1, data = d$df)$threads, 1L)
    expect_error(fastglm_nb(d$x, d$ynb, threads = 0), "'threads' must be a positive integer")
})

test_that("an explicit thread request covers every pass of a count fit and is then restored", {
    skip_if_not_installed("MASS")
    skip_if_not_installed("Formula")
    skip_if_not_installed("Matrix")
    info <- fastglm_threads()
    skip_if_not(info$openmp, "built without OpenMP")
    skip_if(fastglm:::.fastglm_check_cap(min(2L, info$logical.cores)) < 2L, "fewer than 2 threads allowed")
    d <- thread_count_data()
    old.rows <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit(if (is.na(old.rows)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
            else Sys.setenv(FASTGLM_CHUNK_ROWS = old.rows), add = TRUE)
    Sys.setenv(FASTGLM_CHUNK_ROWS = "1000")
    old <- fastglm_threads(2L)
    on.exit(fastglm_threads(old$threads), add = TRUE)
    xs <- Matrix::Matrix(d$x, sparse = TRUE)
    pos <- d$ynb > 0
    calls <- list(
        nb = function(t) fastglm_nb(d$x, d$ynb, threads = t),
        nb.sparse = function(t) fastglm_nb(xs, d$ynb, threads = t),
        truncated = function(t) fastglm_truncated(d$x[pos, ], d$ynb[pos], dist = "negbin", threads = t),
        hurdle = function(t) fastglm_hurdle(yh ~ x1 + x2, data = d$df, dist = "negbin", threads = t),
        zi = function(t) fastglm_zi(yzi ~ x1 + x2 | x1, data = d$df, threads = t),
        twopart = function(t) fastglm_twopart(ytp ~ x1 + x2, data = d$df, threads = t))
    for (nm in names(calls)) {
        # threads = 1 holds for the pilot, component, Poisson-limit, and
        # covariance passes although the session setting is 2.
        fastglm:::fastglm_max_team_size_test(TRUE)
        suppressWarnings(calls[[nm]](1L))
        expect_identical(fastglm:::fastglm_max_team_size_test(TRUE), 1L, info = nm)
        expect_identical(fastglm_threads()$threads, 2L, info = nm)
        expect_identical(getOption("fastglm.threads"), 2L, info = nm)
    }
    fastglm_threads(1L)
    for (nm in names(calls)) {
        fastglm:::fastglm_max_team_size_test(TRUE)
        suppressWarnings(calls[[nm]](2L))
        expect_gt(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
        expect_identical(fastglm_threads()$threads, 1L, info = nm)
        expect_identical(getOption("fastglm.threads"), 1L, info = nm)
    }
})

test_that("batch worker processes fit with one thread and leave the session setting unchanged", {
    skip_if_not_installed("MASS")
    d <- thread_count_data(n = 2000L)
    set.seed(5)
    Y <- cbind(a = d$ynb, b = MASS::rnegbin(2000L, mu = 2, theta = 1.5),
               c = MASS::rnegbin(2000L, mu = 3, theta = 2))
    old <- fastglm_threads(2L)
    on.exit(fastglm_threads(old$threads), add = TRUE)
    before <- list(fastglm_threads()$threads, getOption("fastglm.threads"))
    fits <- fastglm_batch(d$x, Y, model = "negbin", workers = 2L)
    expect_identical(unname(vapply(fits, `[[`, integer(1), "threads")), rep(1L, 3L))
    glms <- fastglm_batch(d$x, Y, family = poisson(), workers = 2L)
    expect_identical(unname(vapply(glms, `[[`, integer(1), "threads")), rep(1L, 3L))
    expect_identical(list(fastglm_threads()$threads, getOption("fastglm.threads")), before)
    # One process: the requested setting, then the session setting again.
    # (Without warm starts, as in the workers, so the fits are comparable.)
    seq.fits <- fastglm_batch(d$x, Y, model = "negbin", threads = 2L, warm.start = FALSE)
    expect_identical(unname(vapply(seq.fits, `[[`, integer(1), "threads")), rep(effective_threads(2L), 3L))
    expect_identical(unname(lapply(seq.fits, coef)), unname(lapply(fits, coef)))
    expect_identical(list(fastglm_threads()$threads, getOption("fastglm.threads")), before)
})

# TRUE when PSOCK worker processes, which load the installed copy of fastglm
# from the caller's library paths, get a build with thread control. Worker
# tests are skipped otherwise: they would test an older installed build.
psock_workers_have_threads <- function() {
    probe <- function(paths) {
        .libPaths(paths)
        "fastglm_threads" %in% getNamespaceExports(loadNamespace("fastglm"))
    }
    environment(probe) <- baseenv()
    cl <- tryCatch(parallel::makePSOCKcluster(1L), error = function(e) NULL)
    if (is.null(cl)) return(FALSE)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    isTRUE(tryCatch(parallel::clusterCall(cl, probe, .libPaths())[[1L]],
                    error = function(e) FALSE))
}

test_that("bootstrap worker processes refit with one thread and leave the session setting unchanged", {
    skip_if_not(psock_workers_have_threads(), "worker processes load a fastglm build without thread control")
    d <- thread_count_data(n = 2000L)
    old <- fastglm_threads(2L)
    on.exit(fastglm_threads(old$threads), add = TRUE)
    before <- list(fastglm_threads()$threads, getOption("fastglm.threads"))
    fit <- fastglm(d$x, d$ytr, family = poisson(), threads = 2L)
    stat <- function(f) c(threads = as.numeric(f$threads))
    b <- fastglm_bootstrap(fit, R = 4L, workers = 2L, seed = 1, statistic = stat)
    expect_identical(as.numeric(b$replicates), rep(1, 4L))
    expect_identical(list(fastglm_threads()$threads, getOption("fastglm.threads")), before)
    # A thread count in 'control' does not override one thread per worker.
    expect_warning(b2 <- fastglm_bootstrap(fit, R = 4L, workers = 2L, seed = 1, statistic = stat,
                                           control = list(threads = 2L)),
                   "control\\$threads is ignored with workers > 1")
    expect_identical(as.numeric(b2$replicates), rep(1, 4L))
    expect_identical(list(fastglm_threads()$threads, getOption("fastglm.threads")), before)
})

test_that("a bootstrap thread count in control applies to refits in this session for every model type", {
    skip_if_not_installed("MASS")
    d <- thread_count_data(n = 600L)
    old <- fastglm_threads(1L)
    on.exit(fastglm_threads(old$threads), add = TRUE)
    t2 <- as.numeric(effective_threads(2L))
    # The statistic runs after each refit, inside the refits' thread scope.
    stat <- function(f) c(b = unname(unlist(coef(f)))[1L], thr = getOption("fastglm.threads"))
    set.seed(9)
    yc <- factor(sample(c("a", "b", "c"), 600L, replace = TRUE))
    yo <- factor(cut(d$df$x1 + rnorm(600L), c(-Inf, -0.5, 0.5, Inf)), ordered = TRUE)
    fits <- list(glm = fastglm(d$x, d$ytr, family = poisson()),
                 nb = fastglm_nb(d$x, d$ynb),
                 multinom = fastglm_multinom(d$x[, -1L], yc),
                 ordinal = fastglm_ordinal(d$x[, -1L], yo))
    for (nm in names(fits)) {
        b <- fastglm_bootstrap(fits[[nm]], R = 2L, workers = 1L, seed = 1, statistic = stat,
                               control = list(threads = 2L))
        expect_identical(b$successful, 2L, info = nm)
        expect_identical(unname(b$replicates[, "thr"]), rep(t2, 2L), info = nm)
        expect_identical(fastglm_threads()$threads, 1L, info = nm)
        expect_identical(getOption("fastglm.threads"), 1L, info = nm)
    }
    # Recorded by the GLM and count fitters themselves as well.
    b <- fastglm_bootstrap(fits$nb, R = 2L, workers = 1L, seed = 1,
                           statistic = function(f) c(threads = as.numeric(f$threads)),
                           control = list(threads = 2L))
    expect_identical(as.numeric(b$replicates), rep(t2, 2L))
})

test_that("control$threads warns with workers > 1 only for a count other than one, after validation", {
    d <- thread_count_data(n = 600L)
    fit <- fastglm(d$x, d$ytr, family = poisson())
    warned <- FALSE
    expect_error(withCallingHandlers(
        fastglm_bootstrap(fit, R = 1L, workers = 2L, control = list(threads = 4L)),
        warning = function(w) warned <<- TRUE), "at least two")
    expect_false(warned)
    skip_if_not(psock_workers_have_threads(), "worker processes load a fastglm build without thread control")
    expect_no_warning(fastglm_bootstrap(fit, R = 2L, workers = 2L, seed = 1,
                                        control = list(threads = 1L)))
})
