skip_if_not_installed("MASS")

# Match MASS::glm.nb(y ~ x[,-1]) on simulated NB data.

test_that("fastglm_nb matches MASS::glm.nb across realistic theta", {
    for (th_true in c(0.5, 2.0)) {
        set.seed(2024 + th_true * 10)
        n  <- 500
        X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
        mu <- exp(X %*% c(0.3, 0.5, -0.2))
        y  <- MASS::rnegbin(n, mu = mu, theta = th_true)

        m  <- MASS::glm.nb(y ~ X[, -1])
        f  <- fastglm_nb(X, y)

        expect_equal(unname(coef(f)), unname(coef(m)),
                     tolerance = 1e-7,
                     info = paste0("theta_true = ", th_true))
        expect_equal(f$theta, m$theta, tolerance = 1e-6)
        expect_equal(f$SE.theta, m$SE.theta, tolerance = 1e-4)
        expect_equal(f$twologlik, m$twologlik, tolerance = 1e-6)
        # Coefficient SE should match too.
        expect_equal(unname(f$se), unname(sqrt(diag(vcov(m)))),
                     tolerance = 1e-6)
    }
})

test_that("fastglm_nb matches MASS::glm.nb on the quine dataset", {
    skip_if_not(requireNamespace("MASS", quietly = TRUE))
    quine <- tryCatch({
        utils::data("quine", package = "MASS", envir = environment())
        get("quine", envir = environment())
    }, error = function(e) NULL)
    skip_if(is.null(quine), "quine dataset unavailable")
    f <- fastglm_nb(model.matrix(~ Sex + Age, data = quine), quine$Days)
    m <- MASS::glm.nb(Days ~ Sex + Age, data = quine)
    expect_equal(unname(coef(f)), unname(coef(m)), tolerance = 1e-6)
    expect_equal(f$theta, m$theta, tolerance = 1e-5)
    expect_equal(f$twologlik, m$twologlik, tolerance = 1e-6)
})

test_that("init.theta is honored", {
    set.seed(2025)
    n  <- 300
    X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y  <- MASS::rnegbin(n, mu = exp(X %*% c(0.3, 0.5, -0.2)), theta = 1.5)
    f1 <- fastglm_nb(X, y)
    f2 <- fastglm_nb(X, y, init.theta = 5)
    expect_equal(f1$theta, f2$theta, tolerance = 1e-6)
    expect_equal(unname(coef(f1)), unname(coef(f2)), tolerance = 1e-7)
})

test_that("fastglm_nb converges on overdispersed data where MASS::glm.nb fails", {
    set.seed(4)
    n  <- 200
    p  <- 9
    X  <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    beta <- c(4, 1.5, -1.2, 1.5, -1.2, 1.5, -1.2, 1.5, -1.2)
    mu <- exp(X %*% beta)
    y  <- MASS::rnegbin(n, mu = mu, theta = 1)

    f <- fastglm_nb(X, y, link = "log")
    expect_true(f$converged)
    expect_true(all(is.finite(coef(f))))
    expect_true(is.finite(f$theta))
    expect_equal(f$theta, 1, tolerance = 0.2)

    # Verify log-likelihood against glm at the converged theta.
    fam <- MASS::negative.binomial(theta = f$theta)
    g <- glm(y ~ X[, -1], family = fam)
    expect_equal(unname(coef(f)), unname(coef(g)), tolerance = 1e-5)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(g)),
                 tolerance = 1e-3)
})

test_that("fastglm_nb returns a fastglm-classed object with the expected slots", {
    set.seed(33)
    n  <- 200
    X  <- cbind(1, matrix(rnorm(n * 2), n, 2))
    y  <- MASS::rnegbin(n, mu = exp(X %*% c(0.3, 0.4, -0.2)), theta = 2)
    f  <- fastglm_nb(X, y)
    expect_s3_class(f, "fastglm_nb")
    expect_s3_class(f, "fastglm")
    expect_named(f$coefficients)
    expect_true(is.finite(f$theta))
    expect_true(is.finite(f$SE.theta))
    expect_true(f$converged)
    # vcov / SE work via the standard fastglm S3 dispatch.
    v <- vcov(f)
    expect_equal(dim(v), rep(ncol(X), 2))
})
