# Thread counts used by the bit-identity tests: the full set when NOT_CRAN is
# "true", otherwise 1 and 2 only (R CMD check limits cores to 2).
threads_to_test <- function() {
    if (identical(Sys.getenv("NOT_CRAN"), "true")) c(1L, 2L, 3L, 7L, 8L) else 1:2
}

# Fits fit(threads) for each thread count with FASTGLM_CHUNK_ROWS set to rows,
# so that the fit has several row blocks, and checks that each named field is
# identical() to the threads = 1 fit. fit is a function of the thread count
# returning a fit. Returns the threads = 1 fit invisibly.
expect_thread_invariant <- function(fit, fields, threads = threads_to_test(), rows = 1000L) {
    old <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit(if (is.na(old)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
            else Sys.setenv(FASTGLM_CHUNK_ROWS = old), add = TRUE)
    Sys.setenv(FASTGLM_CHUNK_ROWS = as.character(rows))
    ref <- fit(1L)
    for (field in fields)
        testthat::expect_false(is.null(ref[[field]]), info = paste0("field '", field, "' is missing"))
    for (t in setdiff(threads, 1L)) {
        res <- fit(t)
        for (field in fields)
            testthat::expect_identical(res[[field]], ref[[field]],
                                       info = paste0("field '", field, "', threads = ", t))
    }
    invisible(ref)
}

# On a build without OpenMP, the first request for more than one thread warns
# once per session. Trigger that warning here, silently, so that it does not
# surface in whichever test happens to request threads first; the warning
# itself is tested in test-threads.R.
if (!fastglm::fastglm_threads()$openmp)
    invisible(suppressWarnings(fastglm:::.fastglm_resolve_threads(2L)))
