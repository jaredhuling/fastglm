# Extracted from test-families-quasi.R:39

# test -------------------------------------------------------------------------
set.seed(17)
n <- 600
p <- 3
X  <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
eta <- X %*% c(0.2, 0.4, -0.1)
y   <- rbinom(n, 1, plogis(eta))
fam <- quasibinomial()
gfit <- glm(y ~ X[, -1], family = fam)
ffit <- fastglm(X, y,    family = fam, method = 2)
expect_equal(unname(coef(ffit)), unname(coef(gfit)), tolerance = 1e-12)
expect_equal(ffit$deviance, gfit$deviance, tolerance = 1e-10)
expect_equal(ffit$dispersion, sum(gfit$prior.weights * (gfit$y - fitted(gfit))^2 / gfit$family$variance(fitted(gfit))) / gfit$df.residual, tolerance = 1e-10)
reference_cov <- solve(crossprod(X, (gfit$prior.weights * fam$mu.eta(gfit$linear.predictors)^2 / fam$variance(fitted(gfit))) * X)) * ffit$dispersion
expect_equal(unname(ffit$se), unname(sqrt(diag(reference_cov))),
                 tolerance = 1e-8)
