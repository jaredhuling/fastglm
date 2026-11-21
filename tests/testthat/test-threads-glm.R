# Results of GLM fits must be identical for every thread count. Each fit has
# n = 20000 rows in blocks of 1000 rows (20 blocks), so the reductions run in
# parallel whenever more than one thread is requested.

thread_glm_data <- function(n = 20000L, p = 5L, seed = 20261119L) {
    set.seed(seed)
    x <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
    beta <- c(0.2, 0.5, -0.3, 0.4, -0.2, 0.1, -0.1)[seq_len(p)]
    eta <- drop(x %*% beta)
    list(x = x,
         yb = rbinom(n, 1L, plogis(eta)),
         yp = rpois(n, exp(eta)),
         yg = rgamma(n, shape = 2, rate = 2 / exp(eta)))
}

glm_fields <- c("coefficients", "deviance", "iter", "cov.unscaled", "fitted.values")

test_that("dense logistic, Poisson, and Gamma fits are identical for every thread count", {
    d <- thread_glm_data()
    cases <- list(list(y = d$yb, family = binomial()),
                  list(y = d$yp, family = poisson()),
                  list(y = d$yg, family = Gamma(link = "log")))
    for (cs in cases) for (m in c(2L, 3L, 0L)) {
        ref <- expect_thread_invariant(function(t)
            fastglm(d$x, cs$y, family = cs$family, method = m, threads = t), glm_fields)
        expect_true(ref$converged, info = paste(cs$family$family, m))
    }
})

test_that("big.matrix logistic fits are identical for every thread count", {
    skip_if_not_installed("bigmemory")
    d <- thread_glm_data()
    xb <- bigmemory::as.big.matrix(d$x)
    ref <- expect_thread_invariant(function(t)
        fastglm(xb, d$yb, family = binomial(), method = 3L, threads = t), glm_fields)
    dense <- fastglm(d$x, d$yb, family = binomial(), method = 3L)
    expect_equal(unname(ref$coefficients), unname(dense$coefficients), tolerance = 1e-10)
})

test_that("streamed logistic fits over 10 source blocks are identical for every thread count", {
    d <- thread_glm_data()
    chunks <- function(k) {
        idx <- ((k - 1L) * 2000L + 1L):(k * 2000L)
        list(X = d$x[idx, , drop = FALSE], y = d$yb[idx])
    }
    ref <- expect_thread_invariant(function(t)
        fastglm_streaming(chunks, n_chunks = 10L, family = binomial(), threads = t),
        c("coefficients", "deviance", "iter", "cov.unscaled", "passes"))
    expect_true(ref$converged)
    dense <- fastglm(d$x, d$yb, family = binomial(), method = 2L)
    expect_equal(unname(ref$coefficients), unname(dense$coefficients), tolerance = 1e-8)
})

test_that("probit fits (serial family evaluation) are identical for every thread count", {
    d <- thread_glm_data()
    yb <- rbinom(nrow(d$x), 1L, pnorm(drop(d$x %*% c(0.1, 0.3, -0.2, 0.2, -0.1))))
    expect_thread_invariant(function(t)
        fastglm(d$x, yb, family = binomial(link = "probit"), method = 2L, threads = t), glm_fields)
})

test_that("an R-side family is identical with 1 and 4 threads", {
    d <- thread_glm_data()
    fam <- binomial()
    fam$linkinv <- function(eta) plogis(eta)
    expect_identical(fastglm:::family_code(fam), -1L)
    ref <- expect_thread_invariant(function(t)
        fastglm(d$x, d$yb, family = fam, method = 2L, threads = t), glm_fields,
        threads = c(1L, 4L))
    native <- fastglm(d$x, d$yb, family = binomial(), method = 2L)
    expect_equal(unname(ref$coefficients), unname(native$coefficients), tolerance = 1e-8)
})

test_that("p = 600 (lanes) and p = 1100 (column tiles, 2 groups) are identical for every thread count", {
    skip_on_cran()  # about 5 seconds even with 2 thread counts
    set.seed(6)
    n <- 3000L
    for (cs in list(list(p = 600L, rows = 300L), list(p = 1100L, rows = 256L))) {
        # p = 600: 10 blocks of 300 rows, so lanes 0 and 1 get two blocks.
        # p = 1100: 12 blocks of 256 rows, two groups of kGroupBlocks = 8.
        # Memory: the design is n p 8 bytes (26 MB at p = 1100).
        p <- cs$p
        x <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
        y <- rpois(n, exp(0.5 + 0.02 * x[, 2] - 0.02 * x[, 3]))
        ref <- expect_thread_invariant(function(t)
            fastglm(x, y, family = poisson(), method = 2L, threads = t),
            c("coefficients", "deviance", "iter", "cov.unscaled"), rows = cs$rows)
        expect_true(ref$converged)
        qr <- fastglm(x, y, family = poisson(), method = 0L)
        expect_equal(unname(ref$coefficients), unname(qr$coefficients), tolerance = 1e-7)
    }
})

test_that("bias-reduced fits (dense and big.matrix) are identical for every thread count", {
    d <- thread_glm_data()
    fields <- c("coefficients", "deviance", "iter", "cov.unscaled")
    for (m in c(2L, 0L))
        expect_thread_invariant(function(t)
            fastglm(d$x, d$yp, family = poisson(), method = m, firth = TRUE, threads = t), fields)
    expect_thread_invariant(function(t)
        fastglm(d$x, d$yg, family = Gamma(link = "log"), method = 3L, firth = TRUE, threads = t), fields)
    skip_if_not_installed("bigmemory")
    xb <- bigmemory::as.big.matrix(d$x)
    # Lagged leverages (accumulate_firth_streamed), with and without the
    # retained Gamma-log adjustment.
    expect_thread_invariant(function(t)
        fastglm(xb, d$yp, family = poisson(), method = 2L, firth = TRUE, threads = t), fields)
    expect_thread_invariant(function(t)
        fastglm(xb, d$yg, family = Gamma(link = "log"), method = 3L, firth = TRUE, threads = t), fields)
})

test_that("sparse fits are identical for every thread count", {
    skip_if_not_installed("Matrix")
    d <- thread_glm_data()
    xs <- Matrix::Matrix(d$x * (abs(d$x) > 0.5 | col(d$x) == 1L), sparse = TRUE)
    expect_thread_invariant(function(t)
        fastglm(xs, d$yb, family = binomial(), method = 2L, threads = t), glm_fields)
})

test_that("sparse Poisson, LDLT, weighted, and bias-reduced fits are identical for every thread count", {
    skip_if_not_installed("Matrix")
    d <- thread_glm_data()
    xs <- Matrix::Matrix(d$x * (abs(d$x) > 0.5 | col(d$x) == 1L), sparse = TRUE)
    expect_thread_invariant(function(t)
        fastglm(xs, d$yp, family = poisson(), method = 2L, threads = t), glm_fields)
    expect_thread_invariant(function(t)
        fastglm(xs, d$yb, family = binomial(), method = 3L, threads = t), glm_fields)
    set.seed(13)
    wt <- runif(nrow(xs), 0.5, 2)
    off <- rnorm(nrow(xs), sd = 0.2)
    expect_thread_invariant(function(t)
        fastglm(xs, d$yp, family = poisson(), weights = wt, offset = off,
                method = 2L, threads = t), glm_fields)
    expect_thread_invariant(function(t)
        fastglm(xs, d$yp, family = poisson(), method = 2L, firth = TRUE, threads = t),
        c("coefficients", "deviance", "iter", "cov.unscaled"))
})

test_that("wide sparse fits (row sweep and column scheme) match dense fits and are thread invariant", {
    skip_on_cran()  # about 5 seconds even with 2 thread counts
    skip_if_not_installed("Matrix")
    make <- function(n, p, density, seed) {
        set.seed(seed)
        x <- as(cbind(1, Matrix::rsparsematrix(n, p - 1L, density = density)), "CsparseMatrix")
        eta <- as.numeric(x %*% c(0.3, rnorm(p - 1L, sd = 0.05)))
        list(x = x, y = rpois(n, exp(eta)))
    }
    # p = 600 uses the row sweep (with prior weights and an offset); p = 1100
    # the column scheme. Blocks of 250 rows give several blocks in each.
    d6 <- make(5000L, 600L, 0.03, 15L)
    set.seed(16)
    wt <- runif(5000L, 0.5, 2)
    off <- rnorm(5000L, sd = 0.1)
    d11 <- make(3000L, 1100L, 0.02, 17L)
    cases <- list(
        p600 = list(d = d6, fit = function(x, t) fastglm(x, d6$y, family = poisson(), weights = wt,
                                                         offset = off, method = 2L, threads = t)),
        p1100 = list(d = d11, fit = function(x, t) fastglm(x, d11$y, family = poisson(),
                                                           method = 2L, threads = t)))
    for (nm in names(cases)) {
        cs <- cases[[nm]]
        sp <- expect_thread_invariant(function(t) cs$fit(cs$d$x, t),
                                      c("coefficients", "deviance", "iter", "cov.unscaled"),
                                      rows = 250L)
        dn <- cs$fit(as.matrix(cs$d$x), 1L)
        expect_true(sp$converged, info = nm)
        expect_equal(sp$coefficients, dn$coefficients, tolerance = 1e-10, info = nm)
        expect_equal(sp$cov.unscaled, dn$cov.unscaled, tolerance = 1e-10, info = nm)
        expect_equal(sp$deviance, dn$deviance, tolerance = 1e-10, info = nm)
    }
})

test_that("sparse design products match Matrix and are identical for every thread count", {
    skip_if_not_installed("Matrix")
    old <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
            else Sys.setenv(FASTGLM_CHUNK_ROWS = old), add = TRUE)
    Sys.setenv(FASTGLM_CHUNK_ROWS = "100")
    set.seed(14)
    # p = 40 uses the dense row sweep, p = 1100 the column scheme.
    for (dims in list(c(3000, 40, 0.1), c(2000, 1100, 0.01))) {
        n <- dims[1]; p <- dims[2]
        x <- cbind(1, Matrix::rsparsematrix(n, p - 1, density = dims[3]))
        x <- as(x, "CsparseMatrix")
        v <- runif(n, 0.1, 2); b <- rnorm(p); off <- rnorm(n)
        ref <- fastglm:::fastglm_sparse_products_test(x, v, b, off, FALSE, 1L)
        expect_equal(as.matrix(ref$xtvx), as.matrix(Matrix::crossprod(x, x * v)),
                     tolerance = 1e-12, info = paste("p =", p))
        expect_equal(ref$eta, as.numeric(x %*% b) + off, tolerance = 1e-12)
        expect_equal(ref$xtv, as.numeric(Matrix::crossprod(x, v)), tolerance = 1e-12)
        expect_true(Matrix::isSymmetric(ref$xtvx))
        # The column scheme (used for p > 1024) on the same design.
        cols <- fastglm:::fastglm_sparse_products_test(x, v, b, off, TRUE, 1L)
        expect_equal(cols, ref, tolerance = 1e-13, info = paste("schemes, p =", p))
        for (t in setdiff(threads_to_test(), 1L)) {
            expect_identical(fastglm:::fastglm_sparse_products_test(x, v, b, off, FALSE, t), ref,
                             info = paste("threads =", t, "p =", p))
            expect_identical(fastglm:::fastglm_sparse_products_test(x, v, b, off, TRUE, t), cols,
                             info = paste("column scheme, threads =", t, "p =", p))
        }
    }
})

test_that("prior weights, offsets, and covariance = 'previous' are identical for every thread count", {
    d <- thread_glm_data()
    set.seed(11)
    wt <- runif(nrow(d$x), 0.5, 2)
    off <- rnorm(nrow(d$x), sd = 0.2)
    expect_thread_invariant(function(t)
        fastglm(d$x, d$yp, family = poisson(), weights = wt, offset = off,
                method = 2L, threads = t), glm_fields)
    expect_thread_invariant(function(t)
        fastglm(d$x, d$yb, family = binomial(), method = 3L, covariance = "previous",
                threads = t), glm_fields)
})

test_that("negative binomial, zero-inflated, and hurdle fits are identical for every session thread count", {
    set.seed(12)
    n <- 20000L
    df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
    df$y <- rnbinom(n, mu = exp(0.3 + 0.4 * df$x1 - 0.2 * df$x2), size = 2)
    df$y[sample(n, 3000L)] <- 0L
    x <- cbind(1, df$x1, df$x2)
    old <- fastglm_threads()
    on.exit(fastglm_threads(old$threads), add = TRUE)
    # These entry points take no threads argument yet (Task 4); they use the
    # session setting.
    session <- function(fit) function(t) { fastglm_threads(t); fit() }
    expect_thread_invariant(session(function() fastglm_nb(x, df$y)),
                            c("coefficients", "theta", "twologlik", "iter", "cov.unscaled"))
    expect_thread_invariant(session(function() fastglm_zi(y ~ x1 + x2 | x1, data = df, dist = "poisson")),
                            c("coefficients", "loglik", "vcov"))
    expect_thread_invariant(session(function() fastglm_hurdle(y ~ x1 + x2, data = df)),
                            c("coefficients", "loglik", "vcov"))
})

test_that("threaded fits run in parallel and threads = 1 fits do not", {
    info <- fastglm_threads()
    skip_if_not(info$openmp, "built without OpenMP")
    skip_if(fastglm:::.fastglm_check_cap(min(2L, info$logical.cores)) < 2L, "fewer than 2 threads allowed")
    d <- thread_glm_data()
    old <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
            else Sys.setenv(FASTGLM_CHUNK_ROWS = old), add = TRUE)
    Sys.setenv(FASTGLM_CHUNK_ROWS = "1000")
    fastglm:::fastglm_max_team_size_test(TRUE)
    fastglm(d$x, d$yb, family = binomial(), method = 2L, threads = 2L)
    expect_gt(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
    # An explicit threads = 1 covers the whole call, including the post-fit
    # pass, whatever the session setting.
    old.threads <- fastglm_threads(2L)
    on.exit(fastglm_threads(old.threads$threads), add = TRUE)
    fastglm:::fastglm_max_team_size_test(TRUE)
    fastglm(d$x, d$yb, family = binomial(), method = 2L, threads = 1L)
    dat <- data.frame(y = d$yb, d$x[, -1L])
    glm(y ~ ., data = dat, family = binomial(), method = fastglm_fit,
        control = list(fastmethod = 2L, threads = 1L))
    expect_identical(fastglm:::fastglm_max_team_size_test(TRUE), 1L)
})

test_that("an error inside a parallel region gives the same message for every thread count", {
    ths <- threads_to_test()
    # Elementwise blocks 5 and 12 (1-based) throw; the smallest is reported.
    msgs <- vapply(ths, function(t) tryCatch({
        fastglm:::fastglm_parallel_blocks_test(20000L, 1000L, c(12L, 5L), t); "none"
    }, error = conditionMessage), character(1))
    expect_true(all(msgs == "failure in block 5"))
    expect_identical(fastglm:::fastglm_parallel_blocks_test(20000L, 1000L, integer(), 4L), 20L)
    # A cross-product block that fails to allocate inside the reduction.
    set.seed(13)
    X <- matrix(rnorm(40000 * 4), 40000)
    w <- runif(40000)
    msgs <- vapply(ths, function(t) tryCatch({
        fastglm:::fastglm_block_reduce_test(X, w, 2L, t); "none"
    }, error = conditionMessage), character(1))
    expect_true(all(msgs == msgs[1L]))
    expect_match(msgs[1L], "not enough memory")
    expect_equal(fastglm:::fastglm_block_reduce_test(X, w, 0L, 4L), crossprod(X * w),
                 tolerance = 1e-12)
    # R stays usable.
    d <- thread_glm_data(n = 2000L)
    expect_true(fastglm(d$x, d$yb, family = binomial(), method = 2L, threads = 4L)$converged)
})

test_that("a non-finite design value gives the same error for every thread count", {
    skip_if_not_installed("bigmemory")
    d <- thread_glm_data()
    x <- d$x
    x[15000L, 2L] <- NaN
    xb <- bigmemory::as.big.matrix(x)
    old <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
            else Sys.setenv(FASTGLM_CHUNK_ROWS = old), add = TRUE)
    Sys.setenv(FASTGLM_CHUNK_ROWS = "1000")
    # The NaN passes through the threaded accumulation; the error is raised
    # after it, by the Cholesky check.
    e1 <- tryCatch(fastglm(xb, d$yb, family = binomial(), method = 3L, threads = 1L),
                   error = function(e) e)
    e4 <- tryCatch(fastglm(xb, d$yb, family = binomial(), method = 3L, threads = 4L),
                   error = function(e) e)
    expect_s3_class(e1, "error")
    expect_s3_class(e4, "error")
    expect_identical(conditionMessage(e4), conditionMessage(e1))
    small <- fastglm(d$x[1:200, ], d$yb[1:200], family = binomial(), threads = 4L)
    expect_true(small$converged)
})

test_that("fits record the thread count used and diagnostics print it", {
    d <- thread_glm_data(n = 2000L)
    # A build without OpenMP always uses 1 thread.
    cap <- function(n) if (!fastglm_threads()$openmp) 1L else
        fastglm:::.fastglm_check_cap(min(n, fastglm_threads()$logical.cores,
                                         fastglm:::.fastglm_thread_limit(), na.rm = TRUE))
    for (req in c(1L, 2L, 4L)) {
        fit <- fastglm(d$x, d$yb, family = binomial(), method = 2L, threads = req)
        expect_identical(fit$threads, cap(req))
    }
    fit <- fastglm(d$x, d$yb, family = binomial(), threads = 2L)
    expect_true(any(grepl("Threads", capture.output(print(fastglm_diagnostics(fit))))))
    pure <- fastglmPure(d$x, d$yb, family = binomial(), method = 2L, threads = 2L)
    expect_identical(pure$threads, cap(2L))
    # Dense QR and SVD fits run on one thread.
    for (m in c(0L, 1L, 4L, 5L))
        expect_identical(fastglmPure(d$x, d$yb, family = binomial(), method = m, threads = 2L)$threads, 1L)
    chunks <- function(k) list(X = d$x[(k - 1L) * 1000L + 1:1000, ], y = d$yb[(k - 1L) * 1000L + 1:1000])
    st <- fastglm_streaming(chunks, n_chunks = 2L, family = binomial(), threads = 2L)
    expect_identical(st$threads, cap(2L))
    dat <- data.frame(y = d$yb, d$x[, -1L])
    gf <- glm(y ~ ., data = dat, family = binomial(), method = fastglm_fit,
              control = list(fastmethod = 2L, threads = 2L))
    expect_identical(gf$threads, cap(2L))
    ff <- fastglm(y ~ ., data = dat, family = binomial(), method = 2L, threads = 2L)
    expect_identical(ff$threads, cap(2L))
    # The session setting applies when threads is NULL.
    old <- fastglm_threads(1L)
    on.exit(fastglm_threads(old$threads), add = TRUE)
    expect_identical(fastglm(d$x, d$yb, family = binomial(), method = 2L)$threads, 1L)
    expect_error(fastglm(d$x, d$yb, family = binomial(), threads = 0), "positive integer")
})
