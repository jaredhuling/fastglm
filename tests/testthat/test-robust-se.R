test_that("vcovHC matches sandwich::vcovHC for binomial logit", {
    skip_if_not_installed("sandwich")
    d <- make_glm_data(n = 500, p = 5, response = "binomial")
    df <- data.frame(d$X, y = d$y)
    gl <- glm(y ~ . - 1, data = df, family = binomial(), control = list(epsilon = 1e-12))
    gf <- fastglm(d$X, d$y, family = binomial(), method = 2, tol = 1e-12)

    for (tp in c("HC0", "HC1", "HC2", "HC3")) {
        v_fast <- sandwich::vcovHC(gf, type = tp)
        v_sw   <- sandwich::vcovHC(gl, type = tp)
        expect_equal(unname(v_fast), unname(v_sw), tolerance = 1e-6,
                     info = paste0("binomial ", tp))
    }
})

test_that("vcovHC matches sandwich for gaussian and poisson", {
    skip_if_not_installed("sandwich")
    for (resp in c("gaussian", "poisson")) {
        d <- make_glm_data(n = 500, p = 4, response = resp)
        df <- data.frame(d$X, y = d$y)
        gl <- glm(y ~ . - 1, data = df, family = family_for(resp),
                  control = list(epsilon = 1e-12))
        gf <- fastglm(d$X, d$y, family = family_for(resp), method = 2, tol = 1e-12)
        for (tp in c("HC0", "HC2", "HC3")) {
            v_fast <- sandwich::vcovHC(gf, type = tp)
            v_sw   <- sandwich::vcovHC(gl, type = tp)
            expect_equal(unname(v_fast), unname(v_sw), tolerance = 1e-6,
                         info = paste0(resp, " ", tp))
        }
    }
})

test_that("vcovCL matches sandwich::vcovCL", {
    skip_if_not_installed("sandwich")
    d <- make_glm_data(n = 500, p = 4, response = "binomial")
    df <- data.frame(d$X, y = d$y)
    gl <- glm(y ~ . - 1, data = df, family = binomial(), control = list(epsilon = 1e-12))
    gf <- fastglm(d$X, d$y, family = binomial(), method = 2, tol = 1e-12)
    cluster <- rep(seq_len(25), length.out = d$n)

    v_fast <- sandwich::vcovCL(gf, cluster = cluster, type = "HC1")
    v_sw   <- sandwich::vcovCL(gl, cluster = cluster, type = "HC1")
    # Use absolute-error comparison: scale by overall matrix norm so a single
    # near-zero off-diagonal entry doesn't dominate the relative tolerance.
    expect_lt(max(abs(v_fast - v_sw)) / max(abs(v_sw)), 1e-5)
})

test_that("vcovHC works on sparse and big.matrix fits", {
    skip_if_not_installed("sandwich")
    skip_if_not_installed("Matrix")
    skip_if_not_installed("bigmemory")
    d <- make_glm_data(n = 400, p = 4, response = "binomial")
    df <- data.frame(d$X, y = d$y)
    gl <- glm(y ~ . - 1, data = df, family = binomial())

    f_sparse <- fastglm(methods::as(d$X, "CsparseMatrix"), d$y,
                        family = binomial(), method = 2)
    f_big    <- fastglm(bigmemory::as.big.matrix(d$X), d$y,
                        family = binomial(), method = 2)
    v_sw <- sandwich::vcovHC(gl, type = "HC0")
    expect_equal(unname(sandwich::vcovHC(f_sparse, type = "HC0")), unname(v_sw),
                 tolerance = 1e-6)
    expect_equal(unname(sandwich::vcovHC(f_big,    type = "HC0")), unname(v_sw),
                 tolerance = 1e-6)
})
