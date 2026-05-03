test_that("vcov(fastglm) matches vcov(glm) across all decomposition methods", {
    d <- make_glm_data(n = 500, p = 5, response = "binomial")
    gl <- glm.fit(d$X, d$y, family = binomial())

    for (m in 0:5) {
        gf <- fastglm(d$X, d$y, family = binomial(), method = m)
        expect_equal(unname(coef(gf)), unname(gl$coefficients), tolerance = 1e-6,
                     info = sprintf("method = %d coefficients", m))
        v_glm <- summary(glm(y ~ . - 1,
                             data = data.frame(d$X, y = d$y),
                             family = binomial()))$cov.scaled
        expect_equal(unname(vcov(gf)), unname(v_glm), tolerance = 1e-6,
                     info = sprintf("method = %d vcov", m))
    }
})

test_that("vcov(fastglm) for gaussian, poisson, gamma agrees with glm", {
    for (resp in c("gaussian", "poisson", "gamma_log")) {
        d <- make_glm_data(n = 500, p = 4, response = resp)
        fam <- family_for(resp)
        df <- data.frame(d$X, y = d$y)
        gl_full <- glm(y ~ . - 1, data = df, family = fam)
        gf <- fastglm(d$X, d$y, family = fam, method = 2)
        expect_equal(unname(coef(gf)), unname(coef(gl_full)), tolerance = 1e-6,
                     info = resp)
        expect_equal(unname(vcov(gf)), unname(vcov(gl_full)), tolerance = 1e-6,
                     info = resp)
    }
})

test_that("predict(se.fit = TRUE) matches predict.glm", {
    d <- make_glm_data(n = 300, p = 4, response = "binomial")
    df <- data.frame(d$X, y = d$y)
    gl <- glm(y ~ . - 1, data = df, family = binomial())
    gf <- fastglm(d$X, d$y, family = binomial(), method = 2)

    new_X <- d$X[1:50, , drop = FALSE]
    p_glm  <- predict(gl, newdata = data.frame(new_X), se.fit = TRUE)
    p_fast <- predict(gf, newdata = new_X, se.fit = TRUE)

    expect_equal(unname(p_fast$fit),    unname(p_glm$fit),    tolerance = 1e-6)
    expect_equal(unname(p_fast$se.fit), unname(p_glm$se.fit), tolerance = 1e-6)
})

test_that("vcov.fastglmFit no longer requires a refit", {
    d <- make_glm_data(n = 200, p = 4, response = "gaussian")
    df <- data.frame(d$X, y = d$y)
    gl <- glm(y ~ . - 1, data = df, family = gaussian(), method = fastglm_fit)
    expect_s3_class(gl, "fastglmFit")
    v <- vcov(gl)
    v_ref <- vcov(glm(y ~ . - 1, data = df, family = gaussian()))
    expect_equal(unname(v), unname(v_ref), tolerance = 1e-6)
})
