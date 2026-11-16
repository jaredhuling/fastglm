# Extracted from test-families-nb.R:40

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
d   <- sim_nb_data(n = 600, p = 3, theta = 2.5)
fam <- MASS::negative.binomial(theta = 2.5, link = "sqrt")
eta_pos <- pmax(d$X %*% c(0.3, 0.4, -0.2), 0.5)
mu_pos  <- as.vector(eta_pos)^2
set.seed(11)
yp <- MASS::rnegbin(length(mu_pos), mu = mu_pos, theta = 2.5)
gfit <- glm(yp ~ d$X[, -1], family = fam)
ffit <- fastglm(d$X, yp,    family = fam, method = 2)
expect_equal(unname(coef(ffit)), unname(coef(gfit)), tolerance = 1e-7)
