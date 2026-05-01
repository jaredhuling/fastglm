test_that("sparse dgCMatrix path matches dense path on a binomial logit fit", {
    skip_if_not_installed("Matrix")
    d <- make_glm_data(n = 600, p = 5, response = "binomial")
    Xs <- methods::as(d$X, "CsparseMatrix")

    f_dense  <- fastglm(d$X, d$y, family = binomial(), method = 2)
    f_sparse <- fastglm(Xs,  d$y, family = binomial(), method = 2)

    expect_equal(unname(coef(f_sparse)), unname(coef(f_dense)), tolerance = 1e-10)
    expect_equal(f_sparse$deviance, f_dense$deviance, tolerance = 1e-10)
})

test_that("sparse rejects unsupported decomposition methods", {
    skip_if_not_installed("Matrix")
    d <- make_glm_data(n = 200, p = 4, response = "binomial")
    Xs <- methods::as(d$X, "CsparseMatrix")
    for (m in c(0, 1, 4, 5))
        expect_error(fastglm(Xs, d$y, family = binomial(), method = m),
                     regexp = "(LLT|LDLT|sparse)")
})

test_that("one-hot encoded categorical recovery: sparse matches dense", {
    skip_if_not_installed("Matrix")
    set.seed(2)
    n <- 500; k <- 8
    g <- sample.int(k, n, replace = TRUE)
    Xd <- model.matrix(~ factor(g))      # dense one-hot with intercept
    beta_true <- c(0.3, rep(c(0.6, -0.4), length.out = ncol(Xd) - 1))
    eta <- Xd %*% beta_true
    y <- rbinom(n, 1, plogis(eta))

    Xs <- methods::as(Xd, "CsparseMatrix")
    f_dense  <- fastglm(Xd, y, family = binomial(), method = 3)
    f_sparse <- fastglm(Xs, y, family = binomial(), method = 3)
    expect_equal(unname(coef(f_sparse)), unname(coef(f_dense)), tolerance = 1e-10)
})
