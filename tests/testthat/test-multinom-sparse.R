# Sparse (dgCMatrix) designs for multinomial fits: the sparse engine path
# must agree with the dense one to 1e-10.

skip_if_not_installed("Matrix")

# n by p design with an intercept column and about 40% zeros overall.
sparse_sim_data <- function(n = 20000L, p = 30L, K = 4L, seed = 81L) {
    set.seed(seed)
    z <- matrix(rnorm(n * (p - 1L)), n, p - 1L)
    z[runif(n * (p - 1L)) < 0.42] <- 0
    x <- cbind(1, z)
    colnames(x) <- c("(Intercept)", paste0("x", seq_len(p - 1L)))
    beta <- matrix(rnorm((K - 1L) * p, sd = 0.3), K - 1L, p)
    eta <- cbind(0, x %*% t(beta))
    pr <- exp(eta - apply(eta, 1, max))
    pr <- pr / rowSums(pr)
    u <- runif(n)
    cls <- 1L + rowSums(u > t(apply(pr, 1, cumsum)))
    y <- factor(LETTERS[pmin(cls, K)], levels = LETTERS[seq_len(K)])
    list(x = x, xs = Matrix::Matrix(x, sparse = TRUE), y = y)
}

expect_fits_agree <- function(sp, de, tol = 1e-10) {
    expect_identical(names(sp$coefficients), names(de$coefficients))
    expect_identical(sp$aliased, de$aliased)
    expect_max_diff(sp$coefficients[!is.na(de$coefficients)],
                    de$coefficients[!is.na(de$coefficients)], tol)
    good <- !is.na(de$coefficients)
    expect_max_diff(sp$cov.unscaled[good, good], de$cov.unscaled[good, good], tol)
    expect_max_diff(sp$loglik, de$loglik, tol)
    expect_max_diff(sp$deviance, de$deviance, tol)
    expect_max_diff(sp$null.deviance, de$null.deviance, tol)
    expect_max_diff(sp$fitted.values, de$fitted.values, tol)
    expect_identical(sp$nobs, de$nobs)
    expect_identical(sp$rank, de$rank)
    expect_identical(sp$converged, de$converged)
}

test_that("sparse and dense fits agree (n = 20000, p = 30, K = 4, 40% zeros)", {
    d <- sparse_sim_data()
    expect_s4_class(d$xs, "dgCMatrix")
    expect_gt(mean(d$x == 0), 0.38)
    de <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    sp <- fastglm_multinom(d$xs, d$y, tol = 1e-12)
    expect_s3_class(sp, "fastglm_multinom")
    expect_true(sp$converged)
    expect_fits_agree(sp, de)
    expect_s4_class(sp$x, "dgCMatrix")
})

test_that("sparse and dense bias-reduced fits agree", {
    d <- sparse_sim_data()
    de <- fastglm_multinom(d$x, d$y, firth = TRUE, tol = 1e-12)
    sp <- fastglm_multinom(d$xs, d$y, firth = TRUE, tol = 1e-12)
    expect_true(sp$firth)
    expect_fits_agree(sp, de)
    expect_max_diff(sp$penalized.loglik, de$penalized.loglik, 1e-10)
    expect_max_diff(sp$adjustment, de$adjustment, 1e-10)
})

test_that("predict and fastglm_vcov agree between sparse and dense fits", {
    d <- sparse_sim_data()
    de <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    sp <- fastglm_multinom(d$xs, d$y, tol = 1e-12)
    for (type in c("probability", "link")) {
        expect_max_diff(predict(sp, type = type), predict(de, type = type), 1e-10)
        nd <- d$x[1:500, ]
        ref <- predict(de, newdata = nd, type = type, se.fit = TRUE)
        a <- predict(sp, newdata = d$xs[1:500, ], type = type, se.fit = TRUE)
        b <- predict(sp, newdata = nd, type = type, se.fit = TRUE)
        expect_max_diff(a$fit, ref$fit, 1e-10)
        expect_max_diff(a$se.fit, ref$se.fit, 1e-10)
        expect_max_diff(b$fit, ref$fit, 1e-10)
        expect_max_diff(b$se.fit, ref$se.fit, 1e-10)
    }
    expect_identical(predict(sp, newdata = d$xs[1:500, ], type = "class"),
                     predict(de, newdata = d$x[1:500, ], type = "class"))
    expect_max_diff(fastglm_vcov(sp, "HC0"), fastglm_vcov(de, "HC0"), 1e-10)
    expect_max_diff(fastglm_vcov(sp, "HC1"), fastglm_vcov(de, "HC1"), 1e-10)
    cl <- rep(seq_len(400L), length.out = nrow(d$x))
    expect_max_diff(fastglm_vcov(sp, "HC1", cluster = cl, block.size = 3000L),
                    fastglm_vcov(de, "HC1", cluster = cl), 1e-10)
    ds <- fastglm_diagnostics(sp); dd <- fastglm_diagnostics(de)
    expect_max_diff(ds$score, dd$score, 1e-10)

    # Block size does not change sparse inference (only the order of the
    # block sums, so agreement is to rounding).
    ref0 <- fastglm_vcov(sp, "HC0", block.size = nrow(d$x))
    refc <- fastglm_vcov(sp, "HC1", cluster = cl, block.size = nrow(d$x))
    for (bs in c(777L, 4096L, 8192L)) {
        expect_max_diff(fastglm_vcov(sp, "HC0", block.size = bs), ref0, 1e-12)
        expect_max_diff(fastglm_vcov(sp, "HC1", cluster = cl, block.size = bs), refc, 1e-12)
    }
    # The sparse block evaluator maps a row range of the transposed design;
    # every range gives exactly the rows of the dense evaluation.
    dat_s <- fastglm:::.vglm_eval_data(sp)
    dat_d <- fastglm:::.vglm_eval_data(de)
    for (i in list(1L, 1:3, 5000:5003, 19990:20000, 1:20000)) {
        a <- fastglm:::.vglm_eval_block(sp, dat_s, i)
        b <- fastglm:::.vglm_eval_block(de, dat_d, i)
        for (nm in c("eta", "u", "W", "P", "loglik_rows", "estfun", "x"))
            expect_max_diff(a[[nm]], b[[nm]], 1e-12)
    }
    expect_error(fastglm:::.vglm_eval_block(sp, dat_s, c(1L, 3L)), "contiguous")
})

test_that("sparse fits handle weights, offsets, grouped counts, and aliasing like dense fits", {
    d <- vglm_sim_data(n = 400L, p = 4L, K = 3L, seed = 7L)
    x <- d$x
    x[abs(x) < 0.6] <- 0
    x[, 1] <- 1
    x <- cbind(x, dup = x[, 2] + x[, 3])   # structurally aliased
    xs <- Matrix::Matrix(x, sparse = TRUE)
    set.seed(3L)
    w <- runif(nrow(x), 0, 2); w[1:5] <- 0
    off <- cbind(rnorm(nrow(x), sd = 0.2), rnorm(nrow(x), sd = 0.2))
    de <- fastglm_multinom(x, d$y, weights = w, offset = off, tol = 1e-12)
    sp <- fastglm_multinom(xs, d$y, weights = w, offset = off, tol = 1e-12)
    expect_true(any(de$aliased))
    expect_fits_agree(sp, de)
    expect_max_diff(predict(sp, newdata = xs[1:20, ], offset = off[1:20, ]),
                    predict(de, newdata = x[1:20, ], offset = off[1:20, ]), 1e-10)

    counts <- d$counts * sample(1:3, nrow(x), TRUE)
    counts[6:8, ] <- 0
    de <- fastglm_multinom(x[, 1:5], counts, tol = 1e-12)
    sp <- fastglm_multinom(xs[, 1:5], counts, tol = 1e-12)
    expect_fits_agree(sp, de)
    expect_max_diff(fastglm_vcov(sp), fastglm_vcov(de), 1e-10)
    for (bs in c(1L, 7L, 100L))
        expect_max_diff(fastglm_vcov(sp, block.size = bs), fastglm_vcov(de), 1e-10)

    # Block size does not change the sparse fit.
    sb <- fastglm_multinom(xs[, 1:5], counts, tol = 1e-12, block.size = 37L)
    expect_fits_agree(sb, sp)
})

test_that("sparse designs are checked like dense ones", {
    d <- vglm_sim_data(n = 50L, p = 2L, K = 3L, seed = 2L)
    xs <- Matrix::Matrix(d$x, sparse = TRUE)
    bad <- xs; bad@x[1] <- NA
    expect_error(fastglm_multinom(bad, d$y), "finite")
    expect_error(fastglm_multinom(xs, d$y[-1]), "same number of rows")
    expect_error(fastglm_multinom(xs[, 2:3], factor(rep("A", 50), levels = c("A", "B"))),
                 "no observations")
})
