test_that("fastglm_threads reports and sets the effective count", {
    old <- fastglm_threads()
    on.exit(fastglm_threads(old$threads))
    expect_named(old, c("threads", "default", "openmp", "logical.cores"))
    expect_true(old$threads >= 1L)
    cores <- old$logical.cores
    lim <- fastglm:::.fastglm_thread_limit()
    expect_identical(fastglm_threads(cores + 8L)$threads,
                     if (old$openmp) min(cores, fastglm:::.fastglm_check_cap(cores), lim, na.rm = TRUE) else 1L)
    expect_error(fastglm_threads(0), "positive integer")
    expect_error(fastglm_threads(NA), "positive integer")
})

test_that("the default follows the core count and the environment limits", {
    d <- fastglm:::.fastglm_default_threads
    expect_identical(d(physical = 14L, logical = 14L, env = list()), 7L)
    expect_identical(d(physical = 32L, logical = 64L, env = list()), 8L)
    expect_identical(d(physical = 1L, logical = 2L, env = list()), 1L)
    expect_identical(d(physical = NA, logical = 10L, env = list()), 5L)
    expect_identical(d(physical = 14L, logical = 14L, env = list(OMP_THREAD_LIMIT = "3")), 3L)
    expect_identical(d(physical = 14L, logical = 14L, env = list(OMP_NUM_THREADS = "2")), 2L)
    expect_identical(d(physical = 14L, logical = 14L, env = list(`_R_CHECK_LIMIT_CORES_` = "TRUE")), 2L)
    expect_identical(d(physical = 14L, logical = 14L, env = list(`_R_CHECK_LIMIT_CORES_` = "false")), 7L)
})

test_that("a build without OpenMP warns once and uses 1 thread", {
    skip_if(fastglm_threads()$openmp, "built with OpenMP")
    state <- fastglm:::.fastglm_state
    old_flag <- state$warned_serial
    old <- fastglm_threads()$threads
    on.exit({
        state$warned_serial <- old_flag
        fastglm_threads(old)
    }, add = TRUE)
    state$warned_serial <- NULL
    expect_warning(res <- fastglm_threads(4L), "without OpenMP")
    expect_identical(res$threads, 1L)
    expect_no_warning(res <- fastglm_threads(4L))
    expect_identical(res$threads, 1L)
    expect_identical(fastglm_threads()$threads, 1L)
})

test_that("R CMD check caps the count at 2 with or without --as-cran", {
    d <- fastglm:::.fastglm_default_threads
    cap <- fastglm:::.fastglm_check_cap
    # A plain R CMD check sets _R_CHECK_PACKAGE_NAME_ only; --as-cran also
    # sets _R_CHECK_LIMIT_CORES_.
    expect_identical(d(physical = 14L, logical = 14L, env = list(`_R_CHECK_PACKAGE_NAME_` = "fastglm")), 2L)
    expect_identical(d(physical = 14L, logical = 14L,
                       env = list(`_R_CHECK_PACKAGE_NAME_` = "fastglm", `_R_CHECK_LIMIT_CORES_` = "TRUE")), 2L)
    expect_identical(d(physical = 14L, logical = 14L, env = list(`_R_CHECK_PACKAGE_NAME_` = "")), 7L)
    expect_identical(cap(8L, limit = "", package = "fastglm"), 2L)
    expect_identical(cap(8L, limit = "TRUE", package = ""), 2L)
    expect_identical(cap(8L, limit = "false", package = ""), 8L)
    expect_identical(cap(8L, limit = "", package = ""), 8L)
    # An explicit request is capped the same way during a check.
    skip_if_not(fastglm_threads()$openmp)
    old <- fastglm_threads()$threads
    on.exit(fastglm_threads(old), add = TRUE)
    old_env <- Sys.getenv(c("_R_CHECK_PACKAGE_NAME_", "_R_CHECK_LIMIT_CORES_"), unset = NA)
    on.exit({
        for (v in names(old_env))
            if (is.na(old_env[[v]])) Sys.unsetenv(v) else do.call(Sys.setenv, as.list(old_env[v]))
    }, add = TRUE)
    Sys.setenv(`_R_CHECK_PACKAGE_NAME_` = "fastglm")
    Sys.unsetenv("_R_CHECK_LIMIT_CORES_")
    expect_lte(fastglm_threads(8L)$threads, 2L)
    expect_lte(fastglm:::.fastglm_resolve_threads(NULL), 2L)
})

test_that("an allocation failure gives the memory message for every thread count", {
    msgs <- vapply(c(1L, threads_to_test()), function(t) tryCatch({
        fastglm:::fastglm_parallel_blocks_test(20000L, 1000L, -4L, t); "none"
    }, error = conditionMessage), character(1))
    expect_true(all(msgs == "not enough memory: an allocation failed inside a parallel computation"))
})

test_that("a request above OMP_THREAD_LIMIT is reduced before it reaches OpenMP", {
    skip_on_cran()
    skip_if_not(fastglm_threads()$openmp)
    skip_if(fastglm_threads()$logical.cores < 2L)
    # The runtime reads OMP_THREAD_LIMIT when it starts, so use a new process.
    script <- tempfile(fileext = ".R")
    on.exit(unlink(script), add = TRUE)
    writeLines(c(
        "suppressMessages(library(fastglm))",
        "set.seed(1); n <- 40000L",
        "X <- cbind(1, matrix(rnorm(n * 3), n)); y <- rbinom(n, 1, 0.4)",
        "f <- fastglm(X, y, family = binomial(), method = 2L, threads = 4L)",
        "cat('THREADS', f$threads, fastglm_threads(4L)$threads, '\\n')"), script)
    out <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"), c("--vanilla", shQuote(script)),
        stdout = TRUE, stderr = TRUE,
        env = c("OMP_THREAD_LIMIT=1", paste0("R_LIBS=", paste(.libPaths(), collapse = .Platform$path.sep)))))
    expect_true(any(grepl("^THREADS 1 1", out)), info = paste(out, collapse = "\n"))
    expect_false(any(grepl("Warning #96|Cannot form a team", out)), info = paste(out, collapse = "\n"))
})

test_that("forked workers use one thread unless they set a count", {
    skip_on_os("windows")
    skip_on_cran()
    skip_if_not(fastglm_threads()$openmp)
    skip_if(fastglm_threads()$logical.cores < 2L)
    skip_if(fastglm:::.fastglm_check_cap(2L) < 2L)
    old <- fastglm_threads()$threads
    on.exit(fastglm_threads(old), add = TRUE)
    fastglm_threads(2L)
    old_rows <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit(if (is.na(old_rows)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
            else Sys.setenv(FASTGLM_CHUNK_ROWS = old_rows), add = TRUE)
    Sys.setenv(FASTGLM_CHUNK_ROWS = "1000")
    set.seed(11); n <- 6000L
    X <- cbind(1, matrix(rnorm(n * 3), n)); y <- rbinom(n, 1, 0.4)
    team <- fastglm:::fastglm_max_team_size_test
    # The parent uses its 2 threads (and so has started its OpenMP threads).
    team(TRUE)
    expect_identical(fastglm(X, y, family = binomial(), method = 2L)$threads, 2L)
    expect_identical(team(TRUE), 2L)
    child <- function(mode) {
        if (mode == "set") fastglm_threads(2L)
        cxx <- fastglm:::fastglm_set_threads_cpp(1L)  # the C++ setting in effect
        if (mode == "set") fastglm:::fastglm_set_threads_cpp(cxx)
        team(TRUE)
        f <- if (mode == "explicit") fastglm(X, y, family = binomial(), method = 2L, threads = 2L)
             else fastglm(X, y, family = binomial(), method = 2L)
        c(fit = f$threads, team = team(TRUE), session = fastglm_threads()$threads, cxx = cxx)
    }
    res <- parallel::mclapply(c("default", "set", "explicit"), child, mc.cores = 2L)
    expect_false(any(vapply(res, inherits, logical(1), "try-error")))
    names(res) <- c("default", "set", "explicit")
    expect_identical(unname(res$default), c(1L, 1L, 1L, 1L))
    expect_identical(unname(res$set), c(2L, 2L, 2L, 2L))
    expect_identical(unname(res$explicit[c("fit", "team", "session")]), c(2L, 2L, 1L))
    # The parent is unchanged.
    expect_identical(fastglm_threads()$threads, 2L)
    team(TRUE)
    fastglm(X, y, family = binomial(), method = 2L)
    expect_identical(team(TRUE), 2L)
})

test_that("ordinal score and proportional-odds tests use the fit's thread count", {
    skip_if_not(fastglm_threads()$openmp)
    skip_if(fastglm_threads()$logical.cores < 2L)
    skip_if(fastglm:::.fastglm_check_cap(2L) < 2L)
    old <- fastglm_threads()$threads
    old_opt <- getOption("fastglm.threads")
    on.exit({ fastglm_threads(old); options(fastglm.threads = old_opt) }, add = TRUE)
    set.seed(2); n <- 20000L
    df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
    df$yo <- cut(df$x1 + rlogis(n), c(-Inf, -1, 0, 1, Inf), ordered_result = TRUE)
    team <- fastglm:::fastglm_max_team_size_test
    fastglm_threads(2L)
    f2 <- fastglm_ordinal(yo ~ x1 + x2, data = df, block.size = 2000L)
    f1 <- fastglm_ordinal(yo ~ x1 + x2, data = df, block.size = 2000L, control = list(threads = 1L))
    expect_identical(f2$threads, 2L)
    expect_identical(f1$threads, 1L)
    # Positive control: a 2-thread fit in a 2-thread session runs a team of 2.
    team(TRUE); s2 <- fastglm_score_test(f2, "x2"); expect_identical(team(TRUE), 2L)
    team(TRUE); p2 <- fastglm_po_test(f2); expect_identical(team(TRUE), 2L)
    # The fit's own count limits the tests.
    team(TRUE); s1 <- fastglm_score_test(f1, "x2"); expect_identical(team(TRUE), 1L)
    team(TRUE); p1 <- fastglm_po_test(f1); expect_identical(team(TRUE), 1L)
    # The option set after load (the C++ setting is still 2 here) gives a
    # 1-thread fit, and the tests on it follow the fit.
    options(fastglm.threads = 1L)
    f0 <- fastglm_ordinal(yo ~ x1 + x2, data = df, block.size = 2000L)
    expect_identical(f0$threads, 1L)
    team(TRUE); fastglm_score_test(f0, "x2"); expect_identical(team(TRUE), 1L)
    team(TRUE); fastglm_po_test(f0); expect_identical(team(TRUE), 1L)
    team(TRUE); vcov(f0); predict(f0, newdata = df, type = "prob"); expect_identical(team(TRUE), 1L)
    s1$data.name <- s2$data.name <- p1$data.name <- p2$data.name <- NULL
    expect_identical(s1, s2)
    expect_identical(p1, p2)
})

test_that("post-fit calls use at most the session setting", {
    skip_if_not(fastglm_threads()$openmp)
    skip_if(fastglm_threads()$logical.cores < 2L)
    skip_if(fastglm:::.fastglm_check_cap(2L) < 2L)
    old <- fastglm_threads()$threads
    on.exit(fastglm_threads(old), add = TRUE)
    set.seed(5); n <- 20000L
    df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
    df$yo <- cut(df$x1 + rlogis(n), c(-Inf, -1, 0, 1, Inf), ordered_result = TRUE)
    team <- fastglm:::fastglm_max_team_size_test
    # A fit at 4 threads (reduced by the cores and any check cap), then a
    # session at 1: the post-fit calls run one thread.
    fastglm_threads(4L)
    f4 <- fastglm_ordinal(yo ~ x1 + x2, data = df, block.size = 2000L, control = list(threads = 4L))
    expect_gt(f4$threads, 1L)
    team(TRUE); s4 <- fastglm_score_test(f4, "x2"); expect_gt(team(TRUE), 1L)
    fastglm_threads(1L)
    team(TRUE); s41 <- fastglm_score_test(f4, "x2"); expect_identical(team(TRUE), 1L)
    team(TRUE); fastglm_po_test(f4); confint(f4, method = "profile"); expect_identical(team(TRUE), 1L)
    expect_identical(fastglm_threads()$threads, 1L)
    # A fit at 1 thread in a session at 4: the post-fit calls run one thread.
    fastglm_threads(4L)
    f1 <- fastglm_ordinal(yo ~ x1 + x2, data = df, block.size = 2000L, control = list(threads = 1L))
    team(TRUE); s1 <- fastglm_score_test(f1, "x2"); fastglm_po_test(f1); expect_identical(team(TRUE), 1L)
    s4$data.name <- s41$data.name <- s1$data.name <- NULL
    expect_identical(s41, s4)
    expect_identical(s1, s4)
})

test_that("a fit's recorded thread count never warns on a build without OpenMP", {
    state <- fastglm:::.fastglm_state
    old_flag <- state$warned_serial
    on.exit(state$warned_serial <- old_flag, add = TRUE)
    state$warned_serial <- NULL
    set.seed(6); n <- 300L
    X <- cbind(1, rnorm(n)); y <- rbinom(n, 1, 0.4)
    f <- fastglm(X, y, family = binomial(), method = 2L, threads = 1L)
    f$threads <- 4L   # as recorded by a fit made on an OpenMP build
    expect_no_warning(predict(f, newdata = X, se.fit = TRUE))
    expect_no_warning(vcov(f))
    expect_identical(fastglm:::.fastglm_fit_threads(f),
                     min(4L, fastglm:::.fastglm_resolve_threads(NULL)))
})
