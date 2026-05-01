make_glm_data <- function(n = 500, p = 5, seed = 1L,
                          response = c("binomial", "gaussian", "poisson", "gamma_log"))
{
    response <- match.arg(response)
    set.seed(seed)
    X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
    colnames(X) <- c("(Intercept)", paste0("x", seq_len(p - 1)))
    beta_true <- c(0.2, 0.5, -0.3, 0.4, -0.2)[seq_len(p)]
    eta <- as.vector(X %*% beta_true)
    y <- switch(response,
        binomial   = rbinom(n, 1, plogis(eta)),
        gaussian   = eta + rnorm(n, sd = 0.5),
        poisson    = rpois(n, exp(eta)),
        gamma_log  = rgamma(n, shape = 2, rate = 2 / exp(eta)))
    list(X = X, y = y, beta_true = beta_true, n = n, p = p)
}

family_for <- function(name) {
    switch(name,
        binomial  = binomial(),
        gaussian  = gaussian(),
        poisson   = poisson(),
        gamma_log = Gamma(link = "log"))
}
