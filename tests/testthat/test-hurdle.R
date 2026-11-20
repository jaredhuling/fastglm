skip_if_not_installed("pscl")

# fastglm_hurdle vs pscl::hurdle on simulated data and the canonical
# bioChemists data shipped with pscl.  Coefficient agreement to 1e-4
# (pscl uses optim/BFGS, fastglm uses Fisher-scoring IRLS — both reach
# the same MLE but with slightly different numerical paths).

.tight_pscl <- function() pscl::hurdle.control(method = "BFGS",
                                               reltol = 1e-12,
                                               maxit  = 5000)

test_that("Poisson hurdle: coefs match pscl::hurdle on simulated data", {
    set.seed(11)
    n  <- 400
    x1 <- rnorm(n);  x2 <- rnorm(n)
    eta_c <- 0.7 + 0.4 * x1 - 0.3 * x2
    lam   <- exp(eta_c)
    eta_z <- -0.4 + 0.5 * x1 + 0.2 * x2
    p_pos <- plogis(eta_z)
    is_pos <- rbinom(n, 1, p_pos)
    # zero-truncated Poisson draws on the positives
    yt <- integer(n)
    for (i in seq_len(n)) {
        repeat {
            v <- rpois(1, lam[i])
            if (v > 0) { yt[i] <- v; break }
        }
    }
    y <- ifelse(is_pos == 1, yt, 0L)
    df <- data.frame(y = y, x1 = x1, x2 = x2)

    f  <- fastglm_hurdle(y ~ x1 + x2, data = df, dist = "poisson",
                         tol = 1e-12, maxit = 200L)
    p  <- pscl::hurdle(y ~ x1 + x2, data = df, dist = "poisson",
                       control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-4)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-4)
    # Joint log-likelihood agreement
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-6)
    expect_true(f$converged)
})

test_that("Poisson hurdle: SEs agree with pscl::hurdle", {
    set.seed(12)
    n  <- 500
    x  <- rnorm(n)
    eta_c <- 0.5 + 0.3 * x
    lam   <- exp(eta_c)
    eta_z <- -0.2 + 0.4 * x
    is_pos <- rbinom(n, 1, plogis(eta_z))
    yt <- integer(n)
    for (i in seq_len(n)) {
        repeat { v <- rpois(1, lam[i]); if (v > 0) { yt[i] <- v; break } }
    }
    y <- ifelse(is_pos == 1, yt, 0L)
    df <- data.frame(y = y, x = x)

    f <- fastglm_hurdle(y ~ x, data = df, dist = "poisson",
                        tol = 1e-12, maxit = 200L)
    p <- pscl::hurdle(y ~ x, data = df, dist = "poisson",
                      control = .tight_pscl())

    expect_equal(unname(f$se$count), unname(sqrt(diag(vcov(p)))[1:2]),
                 tolerance = 1e-3)
    expect_equal(unname(f$se$zero),  unname(sqrt(diag(vcov(p)))[3:4]),
                 tolerance = 1e-3)
})

test_that("Hurdle handles two-RHS formula y ~ x | z (different zero design)", {
    set.seed(13)
    n  <- 400
    x1 <- rnorm(n);  x2 <- rnorm(n);  z1 <- rnorm(n)
    eta_c <- 0.6 + 0.3 * x1 - 0.2 * x2
    lam   <- exp(eta_c)
    eta_z <- -0.3 + 0.7 * z1
    is_pos <- rbinom(n, 1, plogis(eta_z))
    yt <- integer(n)
    for (i in seq_len(n)) {
        repeat { v <- rpois(1, lam[i]); if (v > 0) { yt[i] <- v; break } }
    }
    y <- ifelse(is_pos == 1, yt, 0L)
    df <- data.frame(y = y, x1 = x1, x2 = x2, z1 = z1)

    f <- fastglm_hurdle(y ~ x1 + x2 | z1, data = df, dist = "poisson",
                        tol = 1e-12, maxit = 200L)
    p <- pscl::hurdle(y ~ x1 + x2 | z1, data = df, dist = "poisson",
                      control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-4)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-4)
    expect_length(coef(f, "zero"), 2L)         # intercept + z1
    expect_length(coef(f, "count"), 3L)        # intercept + x1 + x2
})

test_that("NB hurdle: coefs and theta match pscl::hurdle", {
    skip_if_not_installed("MASS")
    set.seed(14)
    n  <- 600
    x1 <- rnorm(n);  x2 <- rnorm(n)
    eta_c <- 0.8 + 0.4 * x1 - 0.3 * x2
    lam   <- exp(eta_c)
    eta_z <- -0.4 + 0.5 * x1
    is_pos <- rbinom(n, 1, plogis(eta_z))
    # zero-truncated NB draws (theta = 1.5)
    yt <- integer(n)
    for (i in seq_len(n)) {
        repeat {
            v <- MASS::rnegbin(1, mu = lam[i], theta = 1.5)
            if (v > 0) { yt[i] <- v; break }
        }
    }
    y <- ifelse(is_pos == 1, yt, 0L)
    df <- data.frame(y = y, x1 = x1, x2 = x2)

    f <- fastglm_hurdle(y ~ x1 + x2, data = df, dist = "negbin",
                        tol = 1e-10, maxit = 200L,
                        outer.tol = 1e-9, outer.maxit = 50L,
                        theta.tol = 1e-10)
    p <- pscl::hurdle(y ~ x1 + x2, data = df, dist = "negbin",
                      control = .tight_pscl())

    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-3)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-3)
    expect_equal(unname(f$theta), unname(p$theta), tolerance = 1e-2)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-4)
})

test_that("Hurdle Poisson: log-link returns valid block-diagonal vcov", {
    set.seed(15)
    n  <- 300
    x  <- rnorm(n)
    eta_c <- 0.6 + 0.3 * x
    is_pos <- rbinom(n, 1, plogis(-0.2 + 0.5 * x))
    yt <- integer(n)
    for (i in seq_len(n)) {
        repeat { v <- rpois(1, exp(eta_c[i])); if (v > 0) { yt[i] <- v; break } }
    }
    y <- ifelse(is_pos == 1, yt, 0L)
    df <- data.frame(y = y, x = x)
    f <- fastglm_hurdle(y ~ x, data = df, dist = "poisson")
    V <- vcov(f, "full")
    p_c <- length(coef(f, "count"))
    # off-diagonal blocks must be zero -- the two parts factorize
    expect_true(all(V[seq_len(p_c), -seq_len(p_c)] == 0))
    expect_true(all(diag(V) > 0))
})

test_that("pscl::hurdle on bioChemists matches fastglm_hurdle", {
    skip_if_not_installed("pscl")
    data("bioChemists", package = "pscl")
    f <- fastglm_hurdle(art ~ ., data = bioChemists, dist = "poisson",
                        tol = 1e-12, maxit = 200L)
    p <- pscl::hurdle(art ~ ., data = bioChemists, dist = "poisson",
                      control = .tight_pscl())
    expect_equal(unname(coef(f, "count")), unname(p$coefficients$count),
                 tolerance = 1e-4)
    expect_equal(unname(coef(f, "zero")),  unname(p$coefficients$zero),
                 tolerance = 1e-4)
    expect_equal(as.numeric(logLik(f)), as.numeric(logLik(p)),
                 tolerance = 1e-6)
})

test_that("Hurdle with zero.dist != binomial errors clearly", {
    set.seed(1)
    n <- 50
    df <- data.frame(y = rpois(n, 1) + 1, x = rnorm(n))
    expect_error(fastglm_hurdle(y ~ x, data = df,
                                zero.dist = "poisson"),
                 regexp = "binomial")
})

test_that("log-link hurdle zero parts with a varying offset fit for every design representation", {
    skip_if_not_installed("Formula")
    skip_if_not_installed("Matrix")
    set.seed(4)
    n <- 4000L; x <- rnorm(n); zc <- rnorm(n); oo <- rnorm(n, 0, 0.3)
    pp <- exp(-2 + 0.3 * zc + oo)                      # below one for these data
    cnt <- rpois(n, exp(0.5 + 0.3 * x))
    while (any(cnt == 0)) cnt[cnt == 0] <- rpois(sum(cnt == 0), exp(0.5 + 0.3 * x[cnt == 0]))
    yo <- rbinom(n, 1L, pp) * cnt
    X <- cbind(1, x); Z <- cbind(1, zc); b <- as.numeric(yo > 0)
    ref <- suppressWarnings(glm(b ~ zc + offset(oo), family = binomial(link = "log"),
                                start = c(log(mean(b)) - 1, 0),
                                control = list(epsilon = 1e-15, maxit = 500)))
    off <- list(count = rep(0, n), zero = oo)
    src <- fastglm_source(function(k) {
        i <- ((k - 1L) * 1000L + 1L):(k * 1000L)
        list(X = X[i, ], Z = Z[i, ], y = yo[i], offset = list(count = rep(0, 1000L), zero = oo[i]))
    }, 4L)
    fits <- list(
        formula = function() fastglm_hurdle(yo ~ x | zc + offset(oo), data = data.frame(yo, x, zc, oo),
                                            link = "log", tol = 1e-12),
        matrix = function() fastglm_hurdle(X, response = yo, z = Z, offset = off, link = "log", tol = 1e-12),
        sparse = function() fastglm_hurdle(Matrix::Matrix(X, sparse = TRUE), response = yo,
                                           z = Matrix::Matrix(Z, sparse = TRUE), offset = off,
                                           link = "log", tol = 1e-12),
        source = function() fastglm_hurdle(src, link = "log", tol = 1e-12))
    if (requireNamespace("bigmemory", quietly = TRUE))
        fits$big.matrix <- function() fastglm_hurdle(bigmemory::as.big.matrix(X), response = yo,
                                                     z = bigmemory::as.big.matrix(Z), offset = off,
                                                     link = "log", tol = 1e-12)
    for (nm in names(fits)) {
        f <- fits[[nm]]()
        expect_equal(unname(f$coefficients$zero), unname(coef(ref)), tolerance = 1e-6, info = nm)
    }
    # The null deviance of a log-link binomial GLM with a varying offset
    # (previously glm.fit's default start failed there).
    g <- fastglm(Z, b, family = binomial(link = "log"), offset = oo)
    expect_equal(g$null.deviance, ref$null.deviance, tolerance = 1e-8)
    expect_equal(unname(coef(g)), unname(coef(ref)), tolerance = 1e-5)
    # Two-part binary part with the log link, matrix interface.
    yp <- b * rgamma(n, 2, 2)
    tp <- fastglm_twopart(X, response = yp, z = Z, offset = off, link = "log", tol = 1e-12)
    expect_equal(unname(tp$coefficients$zero), unname(coef(ref)), tolerance = 1e-6)
})
