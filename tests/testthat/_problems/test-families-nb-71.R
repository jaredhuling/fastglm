# Extracted from test-families-nb.R:71

# prequel ----------------------------------------------------------------------
skip_if_not_installed("MASS")
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

# test -------------------------------------------------------------------------
d   <- sim_nb_data(n = 500, p = 3, theta = 2)
fam <- MASS::negative.binomial(theta = 2)
ffit <- fastglm(d$X, d$y, family = fam, method = 2)
gfit <- glm(d$y ~ d$X[, -1], family = fam, control = list(epsilon = 1e-12))
expect_equal(unname(ffit$se), unname(sqrt(diag(vcov(gfit, dispersion = 1)))),
                 tolerance = 1e-7)
