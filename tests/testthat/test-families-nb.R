skip_if_not_installed("MASS")

# Helper: simulate NB(mu, theta) data with a known design
sim_nb_data <- function(n = 500, p = 3, theta = 2.5, seed = 7L) {
    set.seed(seed)
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    colnames(X) <- c("(Intercept)", paste0("x", seq_len(p - 1)))
    beta_true <- c(0.3, 0.4, -0.2, 0.25, 0.1)[seq_len(p)]
    eta <- as.vector(X %*% beta_true)
    mu  <- exp(eta)
    y   <- MASS::rnegbin(n, mu = mu, theta = theta)
    list(X = X, y = y, beta_true = beta_true)
}

test_that("native NB log link agrees with glm() on MASS::negative.binomial", {
    for (theta in c(0.5, 2.0, 50)) {
        d   <- sim_nb_data(n = 600, p = 3, theta = theta)
        fam <- MASS::negative.binomial(theta = theta, link = "log")
        gfit <- glm(d$y ~ d$X[, -1], family = fam)
        ffit <- fastglm(d$X, d$y, family = fam, method = 2)
        expect_equal(unname(coef(ffit)), unname(coef(gfit)),
                     tolerance = 1e-8,
                     info = paste0("NB log link, theta = ", theta))
        expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-8)
        # fastglm follows summary.glm()'s convention: NB uses the Pearson
        # dispersion estimate (theta is *known* on the family object, but
        # summary.glm doesn't bake that in).
        expect_equal(ffit$dispersion, summary(gfit)$dispersion,
                     tolerance = 1e-8)
    }
})

test_that("native NB sqrt link agrees with glm()", {
    d   <- sim_nb_data(n = 600, p = 3, theta = 2.5)
    fam <- MASS::negative.binomial(theta = 2.5, link = "sqrt")
    eta_pos <- pmax(d$X %*% c(0.3, 0.4, -0.2), 0.5)
    mu_pos  <- as.vector(eta_pos)^2
    set.seed(11)
    yp <- MASS::rnegbin(length(mu_pos), mu = mu_pos, theta = 2.5)
    gfit <- glm(yp ~ d$X[, -1], family = fam)
    ffit <- fastglm(d$X, yp,    family = fam, method = 2)
    expect_equal(unname(coef(ffit)), unname(coef(gfit)), tolerance = 1e-7)
    expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-7)
})

test_that("built-in negbin() matches MASS::negative.binomial()", {
    d <- sim_nb_data(n = 500, p = 3, theta = 1.7)
    f1 <- fastglm(d$X, d$y, family = negbin(theta = 1.7), method = 2)
    f2 <- fastglm(d$X, d$y, family = MASS::negative.binomial(theta = 1.7),
                  method = 2)
    expect_equal(unname(coef(f1)), unname(coef(f2)), tolerance = 1e-12)
    expect_equal(f1$deviance, f2$deviance, tolerance = 1e-12)
})

test_that("NB native path matches R-callback fallback", {
    # Disguise the family name so family_code() falls back to -1.
    disguise <- function(fam) { fam$family <- paste0(fam$family, "_disguised"); fam }
    d   <- sim_nb_data(n = 400, p = 3, theta = 2)
    fam <- MASS::negative.binomial(theta = 2)
    f_native <- fastglm(d$X, d$y, family = fam,            method = 2)
    f_cb     <- fastglm(d$X, d$y, family = disguise(fam),  method = 2)
    expect_equal(unname(coef(f_native)), unname(coef(f_cb)), tolerance = 1e-8)
    expect_equal(f_native$deviance, f_cb$deviance, tolerance = 1e-8)
})

test_that("NB SE / vcov are unaffected by Pearson-disp override (disp = 1)", {
    d   <- sim_nb_data(n = 500, p = 3, theta = 2)
    fam <- MASS::negative.binomial(theta = 2)
    ffit <- fastglm(d$X, d$y, family = fam, method = 2)
    gfit <- glm(d$y ~ d$X[, -1], family = fam)
    # Coefficient SEs should match those of glm() to high precision.
    expect_equal(unname(ffit$se), unname(sqrt(diag(vcov(gfit)))),
                 tolerance = 1e-7)
})
