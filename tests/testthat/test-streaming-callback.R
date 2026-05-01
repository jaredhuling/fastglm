test_that("fastglm_streaming with K chunks matches single-shot fastglm", {
    d <- make_glm_data(n = 1000, p = 5, response = "binomial")
    K <- 4
    chunk_size <- d$n / K
    chunks <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = d$X[idx, , drop = FALSE], y = d$y[idx])
    }
    f_stream <- fastglm_streaming(chunks, n_chunks = K, family = binomial(),
                                  tol = 1e-12)
    f_full   <- fastglm(d$X, d$y, family = binomial(), method = 2, tol = 1e-12)

    expect_equal(unname(coef(f_stream)), unname(coef(f_full)), tolerance = 1e-6)
    expect_lt(max(abs(f_stream$cov.unscaled - f_full$cov.unscaled)) /
              max(abs(f_full$cov.unscaled)), 1e-5)
})

test_that("streaming honors offset and prior weights", {
    set.seed(3)
    n <- 1000; K <- 5
    X <- cbind(1, matrix(rnorm(n * 3), n, 3))
    eta <- X %*% c(0.1, 0.4, -0.2, 0.3)
    ofs <- runif(n, -0.1, 0.1)
    pw  <- runif(n, 0.5, 1.5)
    yp <- rpois(n, exp(eta + ofs))

    chunk_size <- n / K
    chunks <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = X[idx, , drop = FALSE], y = yp[idx],
             offset = ofs[idx], weights = pw[idx])
    }
    f_stream <- fastglm_streaming(chunks, n_chunks = K, family = poisson())
    f_full   <- fastglm(X, yp, family = poisson(), offset = ofs, weights = pw,
                       method = 2)
    expect_equal(unname(coef(f_stream)), unname(coef(f_full)), tolerance = 1e-7)
})

test_that("streaming gaussian recovers exact OLS", {
    d <- make_glm_data(n = 800, p = 4, response = "gaussian")
    K <- 4; chunk_size <- d$n / K
    chunks <- function(k) {
        idx <- ((k - 1) * chunk_size + 1):(k * chunk_size)
        list(X = d$X[idx, , drop = FALSE], y = d$y[idx])
    }
    f_stream <- fastglm_streaming(chunks, n_chunks = K, family = gaussian())
    f_full   <- fastglm(d$X, d$y, family = gaussian(), method = 2)
    expect_equal(unname(coef(f_stream)), unname(coef(f_full)), tolerance = 1e-12)
})

test_that("streaming rejects bad chunk shapes", {
    chunks_bad <- function(k) {
        list(X = matrix(rnorm(20), 5, 4), y = rnorm(4))   # nrow != length(y)
    }
    expect_error(fastglm_streaming(chunks_bad, n_chunks = 1, family = gaussian()),
                 regexp = "rows")
})
