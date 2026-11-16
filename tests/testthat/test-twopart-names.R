test_that("two-part coefficients are labeled positive_ and accept positive/binary selectors", {
    set.seed(5150L); n <- 600L
    d <- data.frame(x = rnorm(n), g = factor(sample(c("a", "b"), n, TRUE)))
    d$y <- rbinom(n, 1, plogis(0.3 + 0.4 * d$x)) * rgamma(n, 2, scale = exp(0.5 + 0.2 * d$x) / 2)
    fit <- fastglm_twopart(y ~ x + g, data = d)
    expect_true(all(startsWith(names(coef(fit)), c(rep("positive_", 3), rep("zero_", 3)))))
    expect_identical(rownames(vcov(fit)), names(coef(fit)))
    expect_identical(coef(fit, model = "positive"), coef(fit, model = "count"))
    expect_identical(coef(fit, model = "binary"), coef(fit, model = "zero"))
    expect_identical(vcov(fit, model = "positive"), vcov(fit, model = "count"))
    expect_identical(predict(fit, d[1:4, ], type = "positive"), predict(fit, d[1:4, ], type = "count"))
    expect_identical(model.matrix(fit, model = "positive"), model.matrix(fit, model = "count"))
    expect_equal(fastglm_wald_test(fit, contrast = c(positive_x = 1))$statistic,
                 fastglm_wald_test(fit, terms = "x", component = "positive")$statistic, tolerance = 1e-12)
    X <- model.matrix(~ x + g, d)
    expect_identical(unname(coef(fastglm_twopart(X, response = d$y))), unname(coef(fit)))
    expect_true(startsWith(names(coef(fastglm_twopart(X, response = d$y)))[1], "positive_"))
    # Hurdle models keep the count_ labels.
    d$k <- rpois(n, 2) * rbinom(n, 1, 0.6)
    expect_true(startsWith(names(coef(fastglm_hurdle(k ~ x, data = d)))[1], "count_"))
})
