# Ordinal (cumulative-link) fits must be identical for every thread count.
# Fits use n = 20000 rows in kernel blocks of 1000 rows (block.size), so the
# reductions run in parallel whenever more than one thread is requested.

ordinal_fields <- c("coefficients", "loglik", "iterations", "observed.iterations",
                    "evaluations", "cov.unscaled", "information", "score",
                    "scaled.score", "predictor.step", "history")

# An ordinal data set with p predictors and K categories (a cumulative
# logit model with moderate slopes, so every category is observed).
thread_ordinal_data <- function(n, p, K, seed = 20261119L) {
    set.seed(seed)
    x <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
    beta <- seq(-0.5, 0.5, length.out = p)
    eta <- drop(x %*% beta) + rlogis(n)
    cuts <- stats::quantile(eta, seq_len(K - 1L) / K)
    y <- factor(findInterval(eta, cuts) + 1L, levels = seq_len(K), ordered = TRUE)
    list(x = x, y = y)
}

# A fit with control$threads = t (the remaining arguments fixed).
ordinal_threads_fit <- function(...) {
    function(t) {
        args <- list(...)
        args$control <- c(args$control, list(threads = t))
        do.call(fastglm_ordinal, args)
    }
}

test_that("proportional odds logit, probit, and cloglog fits are identical for every thread count", {
    d <- thread_ordinal_data(20000L, 4L, 5L)
    for (link in c("logit", "probit", "cloglog")) {
        ref <- expect_thread_invariant(
            ordinal_threads_fit(x = d$x, y = d$y, link = link, block.size = 1000L), ordinal_fields)
        expect_true(ref$converged, info = link)
        expect_gt(ref$observed.iterations, 0L)
        expect_identical(ref$threads, 1L)
    }
})

test_that("expected-information scoring (observed = FALSE) is identical for every thread count", {
    d <- thread_ordinal_data(20000L, 4L, 5L, seed = 7L)
    for (link in c("logit", "probit")) {
        ref <- expect_thread_invariant(
            ordinal_threads_fit(x = d$x, y = d$y, link = link, block.size = 1000L,
                                control = list(observed = FALSE)), ordinal_fields)
        expect_true(ref$converged, info = link)
        expect_identical(ref$observed.iterations, 0L)
    }
})

test_that("grouped count responses (probit and logit) are identical for every thread count", {
    d <- thread_ordinal_data(20000L, 3L, 4L, seed = 11L)
    set.seed(12L)
    counts <- t(vapply(as.integer(d$y), function(k) tabulate(c(k, sample.int(4L, 2L)), 4L), numeric(4)))
    colnames(counts) <- levels(d$y)
    for (link in c("logit", "probit")) {
        ref <- expect_thread_invariant(
            ordinal_threads_fit(x = d$x, y = counts, levels = levels(d$y), link = link,
                                block.size = 1000L), ordinal_fields)
        expect_true(ref$converged, info = link)
    }
})

test_that("bias-reduced (firth = TRUE) logit and probit fits are identical for every thread count", {
    d <- thread_ordinal_data(20000L, 4L, 5L, seed = 3L)
    for (link in c("logit", "probit")) {
        ref <- expect_thread_invariant(
            ordinal_threads_fit(x = d$x, y = d$y, link = link, firth = TRUE, block.size = 1000L),
            c(ordinal_fields, "adjustment"))
        expect_true(ref$converged, info = link)
    }
})

test_that("partial proportional odds (logit and probit) is identical for every thread count", {
    d <- thread_ordinal_data(20000L, 3L, 5L, seed = 5L)
    fields <- c(ordinal_fields, "gamma")
    for (link in c("logit", "probit")) {
        ref <- expect_thread_invariant(
            ordinal_threads_fit(x = d$x, y = d$y, link = link, nonparallel = "x1",
                                block.size = 1000L), fields)
        expect_true(ref$converged, info = link)
        expect_s3_class(ref, "fastglm_ordinal_partial")
        expect_gt(ref$observed.iterations, 0L)
    }
    # Bias reduction uses the expected information only.
    ref <- expect_thread_invariant(
        ordinal_threads_fit(x = d$x, y = d$y, nonparallel = "x1", firth = TRUE,
                            block.size = 1000L), c(fields, "adjustment"))
    expect_true(ref$converged)
})

test_that("a streamed proportional odds fit over 10 source blocks is identical for every thread count", {
    d <- thread_ordinal_data(20000L, 4L, 5L, seed = 9L)
    reader <- function(k) {
        i <- ((k - 1L) * 2000L + 1L):(k * 2000L)
        list(X = d$x[i, , drop = FALSE], y = d$y[i])
    }
    for (link in c("logit", "probit")) {
        ref <- expect_thread_invariant(function(t)
            fastglm_ordinal_streaming(reader, 10L, link = link, block.size = 500L,
                                      control = list(threads = t)),
            c(ordinal_fields, "source.passes"))
        expect_true(ref$converged, info = link)
        expect_identical(ref$threads, 1L)
    }
    # The same partial proportional odds model from the source.
    ref <- expect_thread_invariant(function(t)
        fastglm_ordinal_streaming(reader, 10L, nonparallel = "x2", block.size = 500L,
                                  control = list(threads = t)),
        c(ordinal_fields, "gamma"))
    expect_true(ref$converged)
})

test_that("streamed and in-memory fits agree to rounding", {
    d <- thread_ordinal_data(20000L, 4L, 5L, seed = 9L)
    reader <- function(k) {
        i <- ((k - 1L) * 2000L + 1L):(k * 2000L)
        list(X = d$x[i, , drop = FALSE], y = d$y[i])
    }
    a <- fastglm_ordinal(d$x, d$y, block.size = 500L, control = list(threads = 2L))
    b <- fastglm_ordinal_streaming(reader, 10L, block.size = 500L, control = list(threads = 2L))
    expect_equal(unname(coef(a)), unname(coef(b)), tolerance = 1e-8)
})

test_that("column tiles (p + K - 1 > 1024) give identical results for every thread count", {
    # p = 1030 slopes and K = 3 (d = 1032 > 1024): the expected and observed
    # information use column tiles (17 tiles of up to 64 columns). n = 2700
    # rows in blocks of 300 rows gives 9 blocks: a full group of 8 and a
    # partial last group. Each pass costs about n p^2 flops, so maxit is kept
    # small; the fit need not converge. Memory: the design takes 22 MB; the
    # information 8.5 MB.
    skip_on_cran()
    set.seed(31L)
    n <- 2700L; p <- 1030L
    x <- matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
    y <- factor(findInterval(0.3 * x[, 1L] + rlogis(n), c(-0.5, 0.5)) + 1L, levels = 1:3,
                ordered = TRUE)
    fields <- c("coefficients", "loglik", "iterations", "observed.iterations", "evaluations",
                "information", "score")
    for (link in c("logit", "probit")) {
        ref <- suppressWarnings(expect_thread_invariant(function(t)
            suppressWarnings(fastglm_ordinal(x, y, link = link, maxit = 3L, block.size = 300L,
                                             control = list(threads = t))),
            fields, threads = intersect(threads_to_test(), c(1L, 2L, 3L, 8L))))
        expect_identical(ref$threads, 1L)
    }
})

test_that("control$threads is validated, recorded, and kept out of the fit's controls", {
    d <- thread_ordinal_data(2000L, 2L, 3L)
    expect_error(fastglm_ordinal(d$x, d$y, control = list(threads = 0)), "control\\$threads")
    expect_error(fastglm_ordinal(d$x, d$y, control = list(threads = "2")), "control\\$threads")
    expect_error(fastglm_ordinal(d$x, d$y, control = list(threads = 1.5)), "control\\$threads")
    old <- fastglm_threads()$threads
    fit <- fastglm_ordinal(d$x, d$y, control = list(threads = 1L, trace = FALSE))
    expect_identical(fit$threads, 1L)
    expect_null(fit$control$threads)
    expect_identical(fastglm_threads()$threads, old)
    fit2 <- fastglm_ordinal(d$x, d$y, control = list(threads = 2L))
    expect_identical(fit2$threads, if (fastglm_threads()$openmp) min(2L, fastglm_threads()$logical.cores) else 1L)
    expect_identical(fastglm_threads()$threads, old)
    # The trace argument keeps printing.
    expect_output(fastglm_ordinal(d$x, d$y, control = list(threads = 2L, trace = TRUE)), "iteration")
})

test_that("threads are used for an ordinal fit with several blocks, and not with one", {
    skip_if_not(fastglm_threads()$openmp)
    skip_if(fastglm_threads()$logical.cores < 2L)
    d <- thread_ordinal_data(20000L, 4L, 5L)
    fastglm:::fastglm_max_team_size_test(TRUE)
    fastglm_ordinal(d$x, d$y, link = "probit", block.size = 1000L, control = list(threads = 2L))
    expect_gt(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
    fastglm_ordinal(d$x, d$y, block.size = 50000L, control = list(threads = 2L))
    expect_identical(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
})

test_that("ordinal evaluation errors are the same for every thread count", {
    # Two rows (in blocks 6 and 16 of 1000 rows) whose cumulative predictor
    # overflows: the probit and cloglog rows cannot be evaluated inside the
    # parallel region, and the bias adjustment of the logit fit is not finite.
    set.seed(41L)
    n <- 20000L
    x <- matrix(rnorm(2L * n), n, 2L)
    codes <- sample.int(3L, n, replace = TRUE)
    x[c(5500L, 15000L), 1L] <- 1e306
    outcome <- function(t, link, firth) {
        old <- fastglm_threads(t)
        on.exit(fastglm_threads(old$threads))
        tryCatch({
            fastglm:::ordinal_native_evaluate(x, codes, matrix(numeric(), 0L, 0L), rep(1, n), numeric(n),
                                              3L, c(-1, 1, 1, -1), firth, 1000L, 1e-14, link)
            "no error"
        }, error = function(e) conditionMessage(e))
    }
    for (link in 0:2) for (firth in c(FALSE, TRUE)) {
        info <- paste("link", link, "firth", firth)
        ref <- outcome(1L, link, firth)
        if (link > 0L || firth) expect_false(identical(ref, "no error"), info = info)
        for (t in setdiff(threads_to_test(), 1L))
            expect_identical(outcome(t, link, firth), ref, info = paste(info, "threads", t))
    }
    # R remains usable after an error inside a parallel region.
    expect_true(fastglm_ordinal(x[-c(5500L, 15000L), ], factor(codes[-c(5500L, 15000L)], ordered = TRUE),
                                block.size = 1000L, control = list(threads = 2L))$converged)
})

test_that("bootstrap refits never inherit a thread count from the optimizer controls", {
    d <- thread_ordinal_data(600L, 2L, 3L)
    fit <- fastglm_ordinal(d$x, d$y, control = list(threads = 2L))
    b1 <- fastglm_bootstrap(fit, R = 3L, seed = 1L)
    b2 <- fastglm_bootstrap(fit, R = 3L, seed = 1L, control = list(control = list(threads = 2L)))
    expect_identical(b1$replicates, b2$replicates)
    expect_warning(fastglm_bootstrap(fit, R = 2L, seed = 1L, workers = 2L,
                                     control = list(control = list(threads = 2L))),
                   "ignored with workers > 1")
})
