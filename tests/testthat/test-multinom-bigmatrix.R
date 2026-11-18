# Memory-mapped (bigmemory::big.matrix) designs for multinomial fits: the
# big.matrix path must agree with the dense one to 1e-10.

skip_if_not_installed("bigmemory")

big_sim_data <- function(n = 6000L, p = 8L, K = 4L, seed = 91L) {
    d <- vglm_sim_data(n = n, p = p, K = K, seed = seed)
    d$xb <- bigmemory::as.big.matrix(d$x)
    d
}

test_that("big.matrix and dense fits agree (coefficients, covariance, loglik, HC0)", {
    d <- big_sim_data()
    de <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    bg <- fastglm_multinom(d$xb, d$y, tol = 1e-12, block.size = 1000L)
    expect_s3_class(bg, "fastglm_multinom")
    expect_true(bg$converged)
    expect_identical(names(bg$coefficients), names(de$coefficients))
    expect_max_diff(bg$coefficients, de$coefficients, 1e-10)
    expect_max_diff(bg$cov.unscaled, de$cov.unscaled, 1e-10)
    expect_max_diff(bg$loglik, de$loglik, 1e-10)
    expect_max_diff(bg$deviance, de$deviance, 1e-10)
    expect_max_diff(bg$null.deviance, de$null.deviance, 1e-10)
    expect_max_diff(bg$fitted.values, de$fitted.values, 1e-10)
    expect_identical(bg$nobs, de$nobs)
    expect_identical(bg$rank, de$rank)
    expect_max_diff(fastglm_vcov(bg, "HC0"), fastglm_vcov(de, "HC0"), 1e-10)
    expect_max_diff(fastglm_vcov(bg, "HC1", block.size = 777L),
                    fastglm_vcov(de, "HC1"), 1e-10)
})

test_that("the fit keeps a reference to the big.matrix, not a copy", {
    d <- big_sim_data(n = 500L)
    bg <- fastglm_multinom(d$xb, d$y)
    expect_s4_class(bg$x, "big.matrix")
    expect_identical(bigmemory::describe(bg$x)@description$sharedName,
                     bigmemory::describe(d$xb)@description$sharedName)
    rows <- fastglm:::.vglm_rows(bg)
    de <- fastglm_multinom(d$x, d$y)
    expect_max_diff(rows$estfun, fastglm:::.vglm_rows(de)$estfun, 1e-10)
    expect_max_diff(rows$loglik_rows, fastglm:::.vglm_rows(de)$loglik_rows, 1e-10)
})

test_that("grouped data, weights, offsets, and bias reduction agree", {
    d <- big_sim_data(n = 2000L, p = 4L, K = 3L)
    set.seed(5L)
    w <- runif(nrow(d$x), 0.5, 2)
    off <- matrix(rnorm(nrow(d$x) * 2L, sd = 0.1), ncol = 2L)
    de <- fastglm_multinom(d$x, d$counts, weights = w, offset = off, tol = 1e-12)
    bg <- fastglm_multinom(d$xb, d$counts, weights = w, offset = off, tol = 1e-12,
                           block.size = 333L)
    expect_max_diff(bg$coefficients, de$coefficients, 1e-10)
    expect_max_diff(bg$loglik, de$loglik, 1e-10)
    expect_max_diff(fastglm_vcov(bg, "HC0"), fastglm_vcov(de, "HC0"), 1e-10)
    de <- fastglm_multinom(d$x, d$y, firth = TRUE, tol = 1e-12)
    bg <- fastglm_multinom(d$xb, d$y, firth = TRUE, tol = 1e-12)
    expect_max_diff(bg$coefficients, de$coefficients, 1e-10)
    expect_max_diff(bg$penalized.loglik, de$penalized.loglik, 1e-10)
})

test_that("a sub-matrix view and a design without column names work", {
    d <- big_sim_data(n = 1500L, p = 3L, K = 3L)
    big <- bigmemory::big.matrix(nrow(d$x), ncol(d$x) + 2L, type = "double",
                                 dimnames = list(NULL, c("junk", colnames(d$x), "junk2")))
    big[, 2:(ncol(d$x) + 1L)] <- d$x
    big[, 1L] <- 7
    big[, ncol(d$x) + 2L] <- rnorm(nrow(d$x))
    sub <- bigmemory::sub.big.matrix(big, firstCol = 2L, lastCol = ncol(d$x) + 1L)
    expect_identical(colnames(sub), colnames(d$x))
    de <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    bg <- fastglm_multinom(sub, d$y, tol = 1e-12)
    expect_max_diff(bg$coefficients, de$coefficients, 1e-10)
    nm <- bigmemory::as.big.matrix(unname(d$x))
    fit <- fastglm_multinom(nm, d$y)
    expect_identical(names(fit$coefficients)[1:2], c("B:x1", "C:x1"))
    expect_null(colnames(nm))
    expect_max_diff(predict(fit, bigmemory::as.big.matrix(unname(d$x[1:50, ]))),
                    predict(fastglm_multinom(unname(d$x), d$y), unname(d$x[1:50, ])), 1e-10)
})

test_that("predict accepts matrix or big.matrix newdata, with and without se.fit", {
    d <- big_sim_data(n = 3000L)
    de <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    bg <- fastglm_multinom(d$xb, d$y, tol = 1e-12)
    nd <- d$x[1:400, ]
    ndb <- bigmemory::as.big.matrix(nd)
    for (type in c("probability", "link")) {
        expect_max_diff(predict(bg, nd, type = type), predict(de, nd, type = type), 1e-10)
        expect_max_diff(predict(bg, ndb, type = type), predict(de, nd, type = type), 1e-10)
    }
    expect_identical(predict(bg, ndb, type = "class"), predict(de, nd, type = "class"))
    a <- predict(bg, ndb, se.fit = TRUE)
    b <- predict(de, nd, se.fit = TRUE)
    expect_max_diff(a$fit, b$fit, 1e-10)
    expect_max_diff(a$se.fit, b$se.fit, 1e-10)
    a <- predict(bg, se.fit = TRUE, type = "link")
    b <- predict(de, se.fit = TRUE, type = "link")
    expect_max_diff(a$se.fit, b$se.fit, 1e-10)
    # Columns of newdata given in another order are matched by name.
    expect_max_diff(predict(bg, bigmemory::as.big.matrix(nd[, rev(colnames(nd))])),
                    predict(de, nd), 1e-10)
})

test_that("big.matrix designs are validated", {
    d <- big_sim_data(n = 300L, p = 2L, K = 3L)
    bad <- bigmemory::as.big.matrix(d$x)
    bad[5L, 2L] <- NA
    expect_error(fastglm_multinom(bad, d$y), "finite")
    expect_error(fastglm_multinom(d$xb, d$y[-1L]), "same number of rows")
    xi <- suppressWarnings(bigmemory::as.big.matrix(round(d$x * 10), type = "integer"))
    expect_error(fastglm_multinom(xi, d$y), "double")
})

test_that("diagnostics and Wald inference run on a big.matrix fit", {
    d <- big_sim_data(n = 1500L, p = 3L, K = 3L)
    de <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    bg <- fastglm_multinom(d$xb, d$y, tol = 1e-12)
    expect_max_diff(summary(bg)$coefficients, summary(de)$coefficients, 1e-8)
    expect_max_diff(unname(fastglm_diagnostics(bg)$score),
                    unname(fastglm_diagnostics(de)$score), 1e-8)
})

test_that("a serialized and reloaded big.matrix fit errors clearly instead of crashing", {
    d <- big_sim_data(n = 800L, p = 3L, K = 3L)
    f <- fastglm_multinom(d$xb, d$y)
    f2 <- unserialize(serialize(f, NULL))
    expect_true(bigmemory::is.nil(f2$x@address))
    expect_identical(coef(f2), coef(f))
    expect_identical(coef(f2, matrix = TRUE), coef(f, matrix = TRUE))
    expect_no_error(summary(f2))
    expect_identical(vcov(f2), vcov(f))
    msg <- "no longer available"
    expect_error(predict(f2), msg)
    expect_error(predict(f2, se.fit = TRUE), msg)
    expect_error(fastglm_vcov(f2, "HC0"), msg)
    expect_error(fastglm:::.vglm_rows(f2), msg)
    expect_error(fastglm_diagnostics(f2), msg)
    # Wald tests and tidy need only coefficients, covariance, and names.
    w1 <- fastglm_wald_test(f, terms = "x1")
    w2 <- fastglm_wald_test(f2, terms = "x1")
    w2$data.name <- w1$data.name
    expect_identical(w2, w1)
    skip_if_not_installed("broom")
    expect_identical(broom::tidy(f2), broom::tidy(f))
    # newdata does not need the design, and works on the reloaded fit.
    expect_max_diff(predict(f2, d$x[1:20, ]), predict(f, d$x[1:20, ]), 1e-12)
    # The compiled entry points also stop on a null pointer.
    expect_error(fastglm:::vglm_evaluate_big(f2$x@address, 1L, 5L, as.integer(d$y[1:5]),
                                             matrix(0, 0L, 0L), rep(1, 5), matrix(0, 5L, 2L),
                                             f2$constraints, "multinomial", 3L, 1L,
                                             unname(f2$coefficients), 5L), msg)
})

test_that("a row-offset sub.big.matrix agrees with the dense fit on the same rows", {
    d <- big_sim_data(n = 2000L, p = 3L, K = 3L)
    rows <- 301:1700
    sub <- bigmemory::sub.big.matrix(d$xb, firstRow = min(rows), lastRow = max(rows))
    de <- fastglm_multinom(d$x[rows, ], d$y[rows], tol = 1e-12)
    bg <- fastglm_multinom(sub, d$y[rows], tol = 1e-12, block.size = 300L)
    expect_max_diff(bg$coefficients, de$coefficients, 1e-10)
    expect_max_diff(bg$cov.unscaled, de$cov.unscaled, 1e-10)
    expect_max_diff(bg$loglik, de$loglik, 1e-10)
    expect_max_diff(fastglm_vcov(bg, "HC0", block.size = 250L), fastglm_vcov(de, "HC0"), 1e-10)
})
