test_that("fits without an offset store a NULL offset", {
    set.seed(123)
    n <- 200
    x <- cbind(1, rnorm(n))
    y <- rnbinom(n, mu = exp(0.5 + 0.3 * x[, 2]), size = 2)
    off <- rnorm(n, sd = 0.1)

    expect_null(fastglm(x, y, family = poisson())$offset)
    expect_null(fastglm(x, y, family = poisson(), offset = rep(0, n))$offset)
    expect_null(fastglm_nb(x, y)$offset)
    expect_equal(fastglm(x, y, family = poisson(), offset = off)$offset, off)
    expect_equal(fastglm_nb(x, y, offset = off)$offset, off)

    fit <- fastglm(x, y, family = poisson())
    expect_equal(unname(predict(fit)), unname(fit$linear.predictors))
})
