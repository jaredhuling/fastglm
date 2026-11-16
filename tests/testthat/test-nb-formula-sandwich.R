test_that("fastglm_nb accepts a formula with data, subset, weights, and offsets", {
    skip_if_not_installed("MASS")
    set.seed(3301L); n <- 800L
    d <- data.frame(x1 = rnorm(n), g = factor(sample(c("a", "b", "c"), n, TRUE)), e = runif(n, 0.5, 2))
    d$y <- MASS::rnegbin(n, mu = exp(0.3 + 0.4 * d$x1 + (d$g == "b") * 0.3 + log(d$e)), theta = 2)
    d$w <- sample(1:3, n, TRUE)
    f <- fastglm_nb(y ~ x1 + g + offset(log(e)), data = d, weights = w, subset = x1 > -2)
    m <- MASS::glm.nb(y ~ x1 + g + offset(log(e)), data = d, weights = w, subset = x1 > -2)
    expect_identical(names(coef(f)), names(coef(m)))
    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-6)
    expect_equal(f$theta, m$theta, tolerance = 1e-6)
    nd <- d[1:5, ]
    expect_equal(unname(predict(f, newdata = nd, type = "response")),
                 unname(predict(m, newdata = nd, type = "response")), tolerance = 1e-6)
    # The matrix interface gives the same fit.
    X <- model.matrix(~ x1 + g, d[d$x1 > -2, ])
    fm <- fastglm_nb(X, d$y[d$x1 > -2], weights = d$w[d$x1 > -2], offset = log(d$e[d$x1 > -2]))
    expect_equal(unname(coef(fm)), unname(coef(f)), tolerance = 1e-12)
})

test_that("sandwich::vcovHC defaults to HC0 for count models", {
    skip_if_not_installed("sandwich"); skip_if_not_installed("MASS")
    set.seed(3302L); n <- 600L
    d <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
    d$y <- MASS::rnegbin(n, mu = exp(0.5 + 0.3 * d$x1), theta = 1.5)
    f <- fastglm_nb(y ~ x1 + x2, data = d)
    expect_equal(sandwich::vcovHC(f), fastglm_vcov(f, "HC0"))
    expect_error(sandwich::vcovHC(f, type = "HC3"), "HC0 or HC1")
    h <- fastglm_hurdle(y ~ x1 + x2 | x1, data = d)
    expect_equal(sandwich::vcovHC(h), fastglm_vcov(h, "HC0"))
    expect_equal(sandwich::vcovHC(h, type = "HC1"), fastglm_vcov(h, "HC1"))
    z <- fastglm_zi(y ~ x1 + x2 | x1, data = d)
    expect_equal(sandwich::vcovHC(z), fastglm_vcov(z, "HC0"))
})
