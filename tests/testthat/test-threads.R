test_that("fastglm_threads reports and sets the effective count", {
    old <- fastglm_threads()
    on.exit(fastglm_threads(old$threads))
    expect_named(old, c("threads", "default", "openmp", "logical.cores"))
    expect_true(old$threads >= 1L)
    cores <- old$logical.cores
    expect_identical(fastglm_threads(cores + 8L)$threads,
                     if (old$openmp) min(cores, fastglm:::.fastglm_check_cap(cores)) else 1L)
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
