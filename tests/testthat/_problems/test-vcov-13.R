# Extracted from test-vcov.R:13

# test -------------------------------------------------------------------------
d <- make_glm_data(n = 500, p = 5, response = "binomial")
gl <- glm.fit(d$X, d$y, family = binomial())
for (m in 0:5) {
        gf <- fastglm(d$X, d$y, family = binomial(), method = m)
        expect_equal(unname(coef(gf)), unname(gl$coefficients), tolerance = 1e-6,
                     info = sprintf("method = %d coefficients", m))
        v_glm <- summary(glm(y ~ . - 1,
                             data = data.frame(d$X, y = d$y),
                             family = binomial(), control = list(epsilon = 1e-12)))$cov.scaled
        expect_equal(unname(vcov(gf)), unname(v_glm), tolerance = 1e-6,
                     info = sprintf("method = %d vcov", m))
    }
