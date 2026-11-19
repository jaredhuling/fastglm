cp <- function(X, w, threads, scheme, rows = 97L)
    fastglm:::fastglm_crossprod_test(X, w, rows, threads, scheme)

# Checks bit identity across thread counts and accuracy against crossprod.
check_reduce <- function(X, w, rows) {
    n <- nrow(X); z <- seq_len(n) / n
    lo <- lower.tri(diag(ncol(X)), TRUE)
    exact <- crossprod(X * w, X)
    for (scheme in c("lanes", "tiles")) {
        ref <- cp(X, w, 1L, scheme, rows)
        for (t in threads_to_test()[-1]) expect_identical(cp(X, w, t, scheme, rows), ref)
        expect_lt(max(abs(ref$xtwx[lo] - exact[lo])), 1e-10 * max(abs(ref$xtwx)))
        expect_lt(max(abs(ref$xtwz - drop(crossprod(X * w, z)))),
                  1e-10 * max(abs(ref$xtwz)))
    }
}

test_that("lane and tile reductions are bit-identical across thread counts", {
    set.seed(11)
    for (d in c(5L, 40L, 600L)) {
        X <- matrix(rnorm(1500 * d), 1500, d); w <- runif(1500)
        check_reduce(X, w, 97L)
    }
})

test_that("few blocks, a partial last tile group, and a single block are covered", {
    set.seed(12)
    # (n, d, rows): 6 blocks; 5 blocks with 2 tiles; 11 blocks (last group of
    # 3) with 3 tiles; a single block with 2 tiles.
    cases <- list(c(300L, 7L, 50L), c(450L, 65L, 100L), c(1000L, 130L, 97L),
                  c(80L, 70L, 100L))
    for (cs in cases) {
        X <- matrix(rnorm(cs[1] * cs[2]), cs[1], cs[2]); w <- runif(cs[1])
        check_reduce(X, w, cs[3])
    }
})

test_that("a single block or fewer rows than a block is serial and exact", {
    X <- matrix(rnorm(50 * 3), 50, 3); w <- rep(1, 50)
    expect_identical(cp(X, w, 8L, "lanes", rows = 100L), cp(X, w, 1L, "lanes", rows = 100L))
})

test_that("a failed block is reported after the region for any thread count", {
    X <- matrix(rnorm(500 * 4), 500, 4); w <- runif(500); w[333] <- -1
    for (scheme in c("lanes", "tiles"))
        for (t in threads_to_test())
            expect_error(cp(X, w, t, scheme, rows = 50L),
                         "negative or non-finite weight in block 7$")
})

test_that("the smallest failing block is reported for every thread count", {
    # 10 blocks of 50 rows; blocks 5 and 10 (1-based) fail. Block 10 is in
    # lane 1, which a serial lane pass reaches before lane 4 (block 5).
    X <- matrix(rnorm(500 * 4), 500, 4); w <- runif(500)
    w[230] <- -1; w[480] <- NaN
    for (scheme in c("lanes", "tiles"))
        for (t in threads_to_test())
            expect_error(cp(X, w, t, scheme, rows = 50L),
                         "negative or non-finite weight in block 5$")
})
