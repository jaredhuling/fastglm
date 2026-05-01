test_that("big.matrix path matches dense path", {
    skip_if_not_installed("bigmemory")
    d <- make_glm_data(n = 800, p = 5, response = "binomial")
    Xb <- bigmemory::as.big.matrix(d$X)

    f_dense <- fastglm(d$X, d$y, family = binomial(), method = 2)
    f_big   <- fastglm(Xb,  d$y, family = binomial(), method = 2)

    expect_equal(unname(coef(f_big)), unname(coef(f_dense)), tolerance = 1e-10)
    expect_equal(f_big$deviance, f_dense$deviance, tolerance = 1e-10)
})

test_that("big.matrix rejects QR / SVD methods (would defeat streaming)", {
    skip_if_not_installed("bigmemory")
    d <- make_glm_data(n = 200, p = 4, response = "binomial")
    Xb <- bigmemory::as.big.matrix(d$X)
    for (m in c(0, 1, 4, 5))
        expect_error(fastglm(Xb, d$y, family = binomial(), method = m),
                     regexp = "LLT|LDLT|big.matrix")
})

test_that("FASTGLM_CHUNK_ROWS environment variable does not change results", {
    skip_if_not_installed("bigmemory")
    d <- make_glm_data(n = 1200, p = 4, response = "poisson")
    Xb <- bigmemory::as.big.matrix(d$X)
    f_default <- fastglm(Xb, d$y, family = poisson(), method = 2)

    old <- Sys.getenv("FASTGLM_CHUNK_ROWS", unset = NA)
    on.exit({
        if (is.na(old)) Sys.unsetenv("FASTGLM_CHUNK_ROWS")
        else Sys.setenv(FASTGLM_CHUNK_ROWS = old)
    })
    Sys.setenv(FASTGLM_CHUNK_ROWS = "100")
    f_small <- fastglm(Xb, d$y, family = poisson(), method = 2)
    expect_equal(unname(coef(f_default)), unname(coef(f_small)), tolerance = 1e-12)
})
