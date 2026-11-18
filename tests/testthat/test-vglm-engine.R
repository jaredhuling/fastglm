# Tests of the compiled vector GLM engine on dense in-memory data, through the
# internal export vglm_fit_dense(). Coefficients are term-major: all K - 1
# coefficients of design column 1, then those of column 2, and so on.

engine_fit <- function(x, y = NULL, counts = NULL, weights = NULL,
                       offset = NULL, constraints = NULL, K = NULL,
                       reference = 1L, tol = 1e-10, maxit = 100L,
                       block_size = 777L, control = list()) {
    x <- as.matrix(x)
    n <- nrow(x)
    if (is.null(K)) K <- if (is.null(counts)) nlevels(y) else ncol(counts)
    M <- K - 1L
    codes <- if (is.null(counts)) as.integer(y) else integer(0)
    if (is.null(counts)) counts <- matrix(0, 0, 0)
    if (is.null(weights)) weights <- rep(1, n)
    if (is.null(offset)) offset <- matrix(0, n, M)
    if (is.null(constraints))
        constraints <- rep(list(diag(M)), ncol(x))
    fastglm:::vglm_fit_dense(x, codes, counts, as.numeric(weights),
                             as.matrix(offset), constraints, "multinomial",
                             as.integer(K), as.integer(reference), FALSE, tol,
                             as.integer(maxit), as.integer(block_size), FALSE,
                             control)
}

numeric_outputs <- c("coefficients", "cov.unscaled", "information", "score",
                     "loglik", "deviance", "iter", "passes", "rank",
                     "fisher.decrement")

expect_same_fit <- function(a, b, tol) {
    for (nm in numeric_outputs) expect_max_diff(a[[nm]], b[[nm]], tol)
    expect_identical(a$converged, b$converged)
    expect_identical(a$convergence.reason, b$convergence.reason)
    expect_identical(a$aliased, b$aliased)
    expect_identical(a$separation, b$separation)
}

test_that("engine agrees with the reference fit (n = 5000, p = 6, K = 5)", {
    d <- vglm_sim_data(5000L, 6L, 5L, seed = 3L)
    ref <- .vglm_reference_fit(d$x, d$counts)
    fit <- engine_fit(d$x, d$y)
    expect_true(fit$converged)
    expect_identical(fit$convergence.reason, "converged")
    expect_identical(fit$passes, fit$iter + 1L)
    expect_max_diff(fit$loglik, ref$loglik, 1e-10)
    expect_max_diff(fit$deviance, -2 * ref$loglik, 1e-10)
    expect_max_diff(fit$coefficients, as.vector(ref$coefficients), 1e-8)
    expect_max_diff(fit$cov.unscaled, ref$cov.unscaled, 1e-8)
    expect_max_diff(solve(fit$information), ref$cov.unscaled, 1e-8)
    expect_identical(fit$rank, length(ref$coefficients))
    expect_false(any(fit$aliased))
    expect_false(fit$separation)
    expect_lt(fit$fisher.decrement, 1e-8)

    # Grouped form of the same data gives the same fit.
    grp <- engine_fit(d$x, counts = d$counts)
    expect_max_diff(grp$coefficients, fit$coefficients, 1e-10)
    expect_max_diff(grp$loglik, fit$loglik, 1e-10)

    # Prior weights, offsets, and another reference category.
    set.seed(5L)
    w <- runif(nrow(d$x), 0.5, 2)
    off <- matrix(rnorm(nrow(d$x) * 4L, sd = 0.3), ncol = 4L)
    ref2 <- .vglm_reference_fit(d$x, d$counts, weights = w,
                                          offset = off, reference = 3L)
    fit2 <- engine_fit(d$x, d$y, weights = w, offset = off, reference = 3L)
    expect_max_diff(fit2$loglik, ref2$loglik, 1e-10)
    expect_max_diff(fit2$coefficients, as.vector(ref2$coefficients), 1e-8)
    expect_max_diff(fit2$cov.unscaled, ref2$cov.unscaled, 1e-8)
})

test_that("intercept-only fit equals the log category proportions", {
    d <- vglm_sim_data(3000L, 1L, 4L, seed = 8L)
    x <- d$x[, 1L, drop = FALSE]
    tab <- tabulate(as.integer(d$y), 4L)
    fit <- engine_fit(x, d$y)
    expect_true(fit$converged)
    expect_max_diff(fit$coefficients, log(tab[-1] / tab[1]), 1e-10)
    fit3 <- engine_fit(x, d$y, reference = 3L)
    expect_max_diff(fit3$coefficients, log(tab[-3] / tab[3]), 1e-10)
    # Weighted totals.
    w <- rep(c(1, 3), length.out = nrow(x))
    wt <- tapply(w, d$y, sum)
    fitw <- engine_fit(x, d$y, weights = w)
    expect_max_diff(fitw$coefficients, log(wt[-1] / wt[1]), 1e-10)
})

test_that("K = 2 equals fastglm(binomial())", {
    d <- vglm_sim_data(2000L, 4L, 2L, seed = 4L)
    fit <- engine_fit(d$x, d$y, tol = 1e-12)
    gl <- fastglm::fastglm(d$x, as.numeric(d$y == "B"), family = binomial(),
                           tol = 1e-12)
    expect_max_diff(fit$coefficients, unname(coef(gl)), 1e-10)
    expect_max_diff(sqrt(diag(fit$cov.unscaled)), unname(gl$se), 1e-10)
})

test_that("a duplicated design column is aliased", {
    d <- vglm_sim_data(1500L, 3L, 4L, seed = 6L)
    M <- 3L
    xd <- cbind(d$x, dup = d$x[, "x2"])
    fit <- engine_fit(xd, d$y)
    base <- engine_fit(d$x, d$y)
    P <- ncol(xd) * M
    expect_identical(fit$rank, P - M)
    expect_identical(sum(fit$aliased), M)
    expect_true(all(is.na(fit$coefficients[fit$aliased])))
    expect_true(all(is.na(fit$cov.unscaled[fit$aliased, ])))
    expect_true(fit$converged)
    # The later column is the aliased one, as glm reports it.
    expect_true(all(fit$aliased[(P - M + 1L):P]))
    expect_max_diff(fit$coefficients[!fit$aliased], base$coefficients, 1e-10)
    expect_max_diff(fit$cov.unscaled[!fit$aliased, !fit$aliased],
                    base$cov.unscaled, 1e-10)
    expect_max_diff(fit$loglik, base$loglik, 1e-10)

    # K = 5, duplicate placed before the original: the original (later)
    # column is aliased as a whole block.
    d5 <- vglm_sim_data(5000L, 6L, 5L, seed = 3L)
    M5 <- 4L
    x5 <- cbind(d5$x[, 1:2], dup = d5$x[, "x2"], d5$x[, -(1:2)])
    f5 <- engine_fit(x5, d5$y)
    b5 <- engine_fit(d5$x, d5$y)
    expect_identical(which(f5$aliased), 3L * M5 + seq_len(M5))
    expect_max_diff(f5$coefficients[!f5$aliased], b5$coefficients, 1e-10)
    x6 <- cbind(d5$x, dup = d5$x[, "x2"])
    f6 <- engine_fit(x6, d5$y)
    expect_identical(which(f6$aliased), 7L * M5 + seq_len(M5))
})

test_that("rows with zero weight or zero total change nothing", {
    d <- vglm_sim_data(800L, 3L, 4L, seed = 9L)
    n <- nrow(d$x)
    set.seed(2L)
    w <- runif(n, 0.5, 2)
    drop <- sort(sample.int(n, 120L))

    # Zero-weight rows, with extreme designs that would matter if used.
    xz <- d$x
    xz[drop, -1] <- 50
    wz <- w
    wz[drop] <- 0
    full <- engine_fit(xz, d$y, weights = wz)
    sub <- engine_fit(d$x[-drop, ], d$y[-drop], weights = w[-drop])
    for (nm in numeric_outputs) expect_false(anyNA(full[[nm]]))
    expect_same_fit(full, sub, 1e-12)

    # Grouped rows whose counts are all zero.
    cz <- d$counts
    cz[drop, ] <- 0
    gfull <- engine_fit(xz, counts = cz, weights = w)
    gsub <- engine_fit(d$x[-drop, ], counts = d$counts[-drop, ],
                       weights = w[-drop])
    for (nm in numeric_outputs) expect_false(anyNA(gfull[[nm]]))
    expect_same_fit(gfull, gsub, 1e-12)
    expect_max_diff(gfull$coefficients, sub$coefficients, 1e-10)
})

test_that("separation ends with finite output and the separation flag", {
    d <- vglm_separated_data()
    fit <- engine_fit(d$x, d$y, tol = 1e-8)
    expect_true(all(is.finite(fit$coefficients)))
    expect_true(all(is.finite(fit$cov.unscaled)))
    expect_true(is.finite(fit$loglik))
    expect_false(any(fit$aliased))
    # x1 is the third column; its coefficient for level C is the second of M.
    b <- matrix(fit$coefficients, 2L)
    expect_gt(abs(b[2L, 2L]), 10)
    expect_true(fit$separation)
    expect_true(fit$convergence.reason %in%
                c("converged", "iteration_limit", "step_halving_failed"))
})

test_that("the constraint mapping is exact", {
    d <- vglm_sim_data(1200L, 3L, 4L, seed = 12L)
    a <- engine_fit(d$x, d$y)
    b <- engine_fit(d$x, d$y, control = list(force_mapping = TRUE))
    expect_same_fit(a, b, 1e-12)

    # Invertible constraint matrices on two columns reparameterize them:
    # H_k beta_k = beta_k(identity), so beta = T^{-1} beta(identity) and
    # cov = T^{-1} cov(identity) T^{-T} with T = blockdiag(I, I, H3, H4).
    # Two non-identity columns check an off-diagonal information block
    # with H != I on both sides.
    M <- 3L
    set.seed(1L)
    H3 <- diag(M) + matrix(runif(M * M, -0.3, 0.3), M)
    H4 <- diag(M) + matrix(runif(M * M, -0.4, 0.4), M)
    cons <- rep(list(diag(M)), ncol(d$x))
    cons[[3L]] <- H3
    cons[[4L]] <- H4
    r <- engine_fit(d$x, d$y, constraints = cons)
    Tinv <- diag(4L * M)
    Tinv[7:9, 7:9] <- solve(H3)
    Tinv[10:12, 10:12] <- solve(H4)
    expect_true(r$converged)
    expect_max_diff(r$loglik, a$loglik, 1e-10)
    expect_max_diff(r$coefficients, Tinv %*% a$coefficients, 1e-8)
    expect_max_diff(r$cov.unscaled, Tinv %*% a$cov.unscaled %*% t(Tinv), 1e-8)
    Tm <- solve(Tinv)
    expect_max_diff(r$information, t(Tm) %*% a$information %*% Tm, 1e-8)
})

test_that("a common slope (H = 1_M) agrees with independent fits", {
    d <- vglm_sim_data(1200L, 3L, 4L, seed = 12L)
    M <- 3L
    cons1 <- rep(list(diag(M)), ncol(d$x))
    cons1[[2L]] <- matrix(1, M, 1L)
    r1 <- engine_fit(d$x, d$y, constraints = cons1)
    expect_true(r1$converged)
    expect_identical(length(r1$coefficients), 3L * M + 1L)

    # Direct maximization from 0 with the analytic gradient.
    Y <- d$counts
    expand <- function(theta) {
        B <- matrix(0, ncol(d$x), M)
        B[1, ] <- theta[1:3]
        B[2, ] <- theta[4]
        B[3, ] <- theta[5:7]
        B[4, ] <- theta[8:10]
        B
    }
    probs <- function(theta) {
        eta <- cbind(0, d$x %*% expand(theta))
        e <- exp(eta - apply(eta, 1, max))
        e / rowSums(e)
    }
    nll <- function(theta) -sum(Y * log(probs(theta)))
    grad <- function(theta) {
        G <- -crossprod(d$x, (Y - probs(theta))[, -1])
        c(G[1, ], sum(G[2, ]), G[3, ], G[4, ])
    }
    opt <- stats::optim(numeric(10), nll, grad, method = "BFGS",
                        control = list(reltol = 1e-16, maxit = 5000))
    expect_identical(opt$convergence, 0L)
    expect_max_diff(-r1$loglik, opt$value, 1e-10)
    expect_max_diff(r1$coefficients, opt$par, 1e-5)

    skip_if_not_installed("VGAM")
    dat <- data.frame(y = d$y, d$x[, -1])
    vg <- suppressWarnings(VGAM::vglm(
        y ~ x1 + x2 + x3, VGAM::multinomial(refLevel = 1), data = dat,
        constraints = list("(Intercept)" = diag(M), x1 = matrix(1, M, 1L),
                           x2 = diag(M), x3 = diag(M)),
        control = VGAM::vglm.control(epsilon = 1e-12, maxit = 200)))
    expect_max_diff(r1$coefficients, unname(coef(vg)), 1e-6)
    expect_max_diff(r1$cov.unscaled, unname(VGAM::vcov(vg)), 1e-6)
    expect_max_diff(r1$loglik, VGAM::logLik(vg), 1e-8)
})

test_that("a near-collinear but estimable design is not aliased", {
    d <- vglm_sim_data(3000L, 3L, 4L, seed = 14L)
    set.seed(15L)
    xc <- cbind(d$x, x4 = d$x[, "x1"] + rnorm(nrow(d$x), sd = 1e-4))
    fit <- engine_fit(xc, d$y)
    ref <- .vglm_reference_fit(xc, d$counts)
    expect_true(fit$converged)
    expect_false(any(fit$aliased))
    expect_identical(fit$rank, length(fit$coefficients))
    expect_max_diff(fit$loglik, ref$loglik, 1e-10)
    expect_max_diff(fit$coefficients, as.vector(ref$coefficients), 1e-8)

    # Noise of sd 3e-7: the information has condition number near 1e14, past
    # the LLT cutoff, yet the design (condition near 1e7) has full rank. The
    # individual coefficients are determined only to about cond * eps, so
    # the comparison is on the log likelihood and the linear predictors.
    xs <- cbind(d$x, x4 = d$x[, "x1"] + rnorm(nrow(d$x), sd = 3e-7))
    fs <- engine_fit(xs, d$y)
    rs <- .vglm_reference_fit(xs, d$counts)
    expect_false(any(fs$aliased))
    expect_identical(fs$rank, length(fs$coefficients))
    expect_max_diff(fs$loglik, rs$loglik, 1e-10)
    expect_max_diff(xs %*% t(matrix(fs$coefficients, 3L)),
                    xs %*% t(rs$coefficients), 1e-6)

    d2 <- vglm_sim_data(3000L, 3L, 2L, seed = 16L)
    x2 <- cbind(d2$x, x4 = d2$x[, "x1"] + rnorm(nrow(d2$x), sd = 1e-4))
    f2 <- engine_fit(x2, d2$y, tol = 1e-12)
    gl <- fastglm::fastglm(x2, as.numeric(d2$y == "B"), family = binomial(),
                           tol = 1e-12)
    expect_false(any(f2$aliased))
    expect_max_diff(f2$coefficients, unname(coef(gl)), 1e-8)
})

test_that("columns with no data on the rows used are aliased", {
    d <- vglm_sim_data(600L, 2L, 3L, seed = 17L)
    n <- nrow(d$x)
    w <- rep(1, n)
    w[1:20] <- 0
    z <- numeric(n)
    z[1:20] <- 1            # nonzero only on zero-weight rows
    fit <- engine_fit(cbind(d$x, z = z, zero = 0), d$y, weights = w)
    base <- engine_fit(d$x, d$y, weights = w)
    expect_identical(which(fit$aliased), 7:10)
    expect_max_diff(fit$coefficients[!fit$aliased], base$coefficients, 1e-10)
})

test_that("restricted steps are scaled, and a zero step is a failure", {
    d <- vglm_sim_data(1500L, 3L, 4L, seed = 18L)
    free <- engine_fit(d$x, d$y)
    bounded <- engine_fit(d$x, d$y, control = list(step_bound = 0.05))
    expect_true(bounded$converged)
    expect_gt(bounded$iter, free$iter)
    expect_max_diff(bounded$coefficients, free$coefficients, 1e-8)
    stuck <- engine_fit(d$x, d$y, control = list(step_bound = 0))
    expect_false(stuck$converged)
    expect_identical(stuck$convergence.reason, "step_halving_failed")
    expect_identical(stuck$iter, 0L)
})

test_that("engine agrees with the reference fit for K = 20", {
    d <- vglm_sim_data(10000L, 5L, 20L, seed = 20L)
    ref <- .vglm_reference_fit(d$x, d$counts)
    fit <- engine_fit(d$x, d$y, block_size = 4096L)
    expect_true(fit$converged)
    expect_max_diff(fit$coefficients, as.vector(ref$coefficients), 1e-8)
    expect_max_diff(fit$loglik, ref$loglik, 1e-10)
})
