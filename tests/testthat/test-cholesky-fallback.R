test_that("Cholesky methods fall back to pivoted QR on a singular dense design", {
    set.seed(123)
    n <- 400
    z <- rnorm(n)
    x <- cbind(1, z, z)
    yb <- rbinom(n, 1, plogis(z))
    yg <- z + rnorm(n)
    for (family in list(binomial(), gaussian())) {
        y <- if (family$family == "binomial") yb else yg
        qr <- fastglm(x, y, family = family, method = 0L)
        expect_true(is.na(coef(qr)[3]))
        for (m in 2:3) {
            expect_warning(fit <- fastglm(x, y, family = family, method = m),
                           "fit with pivoted QR")
            expect_equal(coef(fit), coef(qr))
            expect_identical(fit$solver, "qr")
            expect_warning(pure <- fastglmPure(x, y, family = family, method = m),
                           "fit with pivoted QR")
            expect_equal(pure$coefficients, qr$coefficients)
        }
        expect_silent(fastglm(x, y, family = family, method = "auto"))
    }
})

test_that("sparse Cholesky fits still report a singular design", {
    skip_if_not_installed("Matrix")
    set.seed(123)
    z <- rnorm(200)
    x <- Matrix::Matrix(cbind(1, z, z), sparse = TRUE)
    y <- z + rnorm(200)
    expect_error(fastglm(x, y, method = 2L), "singular")
})
