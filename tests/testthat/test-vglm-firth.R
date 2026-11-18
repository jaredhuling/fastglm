# Tests of mean bias reduction (firth = TRUE) in the compiled vector GLM
# engine, through the internal export vglm_fit_dense(). Coefficients are
# term-major: all K - 1 coefficients of design column 1, then column 2, ...

firth_fit <- function(x, y = NULL, counts = NULL, weights = NULL,
                      offset = NULL, constraints = NULL, K = NULL,
                      reference = 1L, firth = TRUE, tol = 1e-13,
                      maxit = 100L, block_size = 97L, control = list()) {
    x <- as.matrix(x)
    n <- nrow(x)
    if (is.null(K)) K <- if (is.null(counts)) nlevels(y) else ncol(counts)
    M <- K - 1L
    codes <- if (is.null(counts)) as.integer(y) else integer(0)
    if (is.null(counts)) counts <- matrix(0, 0, 0)
    if (is.null(weights)) weights <- rep(1, n)
    if (is.null(offset)) offset <- matrix(0, n, M)
    if (is.null(constraints)) constraints <- rep(list(diag(M)), ncol(x))
    fastglm:::vglm_fit_dense(x, codes, counts, as.numeric(weights),
                             as.matrix(offset), constraints, "multinomial",
                             as.integer(K), as.integer(reference), firth, tol,
                             as.integer(maxit), as.integer(block_size), FALSE,
                             control)
}

# log det of a symmetric positive definite matrix.
logdet <- function(A) as.numeric(determinant(A, logarithm = TRUE)$modulus)

# Pure-R oracle run to an absolute Fisher decrement of 1e-20. Its rule is
# the decrement alone, which keeps falling geometrically until rounding
# (about 1e-30 here), so 1e-20 is reachable and leaves the oracle within
# about 1e-10 sqrt(V_jj) of its fixed point (see coef_bound).
tight_reference <- function(d, ...)
    .vglm_reference_fit(d$x, d$counts, firth = TRUE, tol = 1e-20,
                                  maxit = 500L, ...)

# Penalized log likelihood l + (1/2) log det I of a reference fit.
reference_pll <- function(ref) ref$loglik - 0.5 * logdet(ref$cov.unscaled)

# Distance from a stopped fit to the fixed point that its stopping rule
# guarantees. Fisher scoring on the adjusted score converges linearly: the
# steps delta_k = I^{-1} U*_k shrink by a factor rho per iteration, in the
# norm ||v||_I = sqrt(v' I v). The error at the stopping iterate is the sum of
# the remaining steps, so ||e||_I <= lambda / (1 - rho) with lambda^2 = U*'
# I^{-1} U* the final Fisher decrement, and by Cauchy-Schwarz each coordinate
# satisfies |e_j| <= sqrt(V_jj) ||e||_I with V = I^{-1} (cov.unscaled). The
# rates measured on the data sets below (from the ratio of successive
# decrements) are 0.42 on the separated data and below 0.04 on the others;
# rho = 1/2 bounds them all.
coef_bound <- function(fit, decrement = fit$fisher.decrement, rho = 0.5)
    sqrt(diag(fit$cov.unscaled)) * sqrt(decrement) / (1 - rho)

# Coefficients of a fit stopped at tol agree with the oracle's within the
# sum of both stopping bounds.
expect_near_reference <- function(fit, ref) {
    bound <- coef_bound(fit) + coef_bound(fit, decrement = 1e-20)
    expect_true(all(abs(fit$coefficients - as.vector(ref$coefficients)) <= bound))
    # The penalized log likelihood is stationary at the fixed point, so its
    # error is second order: about (1/2) ||e||_I^2 <= 2 lambda^2, plus the
    # rounding of the sums. The log likelihood itself is not stationary there
    # (its gradient is minus the adjustment) and is not compared.
    expect_lte(abs(fit$penalized.loglik - reference_pll(ref)),
               2 * fit$fisher.decrement +
                   1e-10 * max(1, abs(fit$penalized.loglik)))
}

check_bias_reduced <- function(fit, tol = 1e-13) {
    expect_true(fit$converged)
    expect_identical(fit$convergence.reason, "converged")
    expect_false(fit$singular)
    # Aliased coefficients are held at 0 and do not enter I or the score.
    keep <- !fit$aliased
    info <- fit$information[keep, keep, drop = FALSE]
    expect_max_diff(fit$penalized.loglik,
                    fit$loglik + 0.5 * logdet(info), 1e-10)
    # Size of the adjusted score at the stopping point, from the stopping
    # rule: with r = D^{-1} (U + adjustment), D^2 = diag(I), and
    # C = D^{-1} I D^{-1} (unit diagonal, so its largest eigenvalue is at
    # most P), ||r||^2 <= lambda_max(C) lambda^2 <= P lambda^2, and
    # lambda^2 < tol (0.1 + |D|) with D = -2 penalized.loglik.
    s <- (fit$score + fit$adjustment)[keep]
    r2 <- sum(s^2 / diag(info))
    dev <- 0.1 + abs(2 * fit$penalized.loglik)
    expect_lt(fit$fisher.decrement, tol * dev)
    expect_lt(r2, length(s) * tol * dev)
}

test_that("bias reduction agrees with brglm2 and the reference fit", {
    skip_if_not_installed("brglm2")
    # The internal comparison with the reference oracle is tight; the external
    # comparison with brglm2 depends on brglm2's own stopping rule.
    compare_brglm2 <- function(d, fit, ref) {
        dat <- data.frame(y = d$y, d$x[, -1])
        fm <- stats::reformulate(colnames(d$x)[-1], "y")
        br <- brglm2::brmultinom(fm, data = dat, type = "AS_mean",
                                 epsilon = 1e-12)
        bc <- as.vector(unname(coef(br)))
        # brglm2 stops by its own rule, about 1e-6 from the fixed point here.
        expect_max_diff(bc, as.vector(ref$coefficients), 1e-6)
        bound <- coef_bound(fit) + 1e-6 * max(1, abs(bc))
        expect_true(all(abs(fit$coefficients - bc) <= bound))
    }
    internal <- function(d) {
        fit <- firth_fit(d$x, d$y)
        check_bias_reduced(fit)
        ref <- tight_reference(d)
        expect_true(ref$converged)
        expect_near_reference(fit, ref)
        list(fit = fit, ref = ref)
    }
    # brmultinom is slow at large n (one nuisance parameter per row), so the
    # simulated check uses n = 300, as in the reference tests.
    d1 <- vglm_sim_data(300L, 3L, 4L, seed = 1L)
    r1 <- internal(d1)
    compare_brglm2(d1, r1$fit, r1$ref)
    d2 <- vglm_separated_data()
    r2 <- internal(d2)
    expect_lt(max(abs(r2$fit$coefficients)), 10)
    # On separated data brglm2's agreement margin is tied to its stopping rule
    # (it varies across brglm2 versions), so this external check is local only.
    skip_on_cran()
    compare_brglm2(d2, r2$fit, r2$ref)
})

test_that("separated data: the bias-reduced score is zero, with finite estimates", {
    d <- vglm_separated_data()
    fit <- firth_fit(d$x, d$y, block_size = 7L)
    check_bias_reduced(fit)
    expect_true(all(is.finite(fit$coefficients)))
    expect_true(all(is.finite(fit$cov.unscaled)))
    # The ML fit of the same data diverges; bias reduction keeps it bounded.
    ml <- suppressWarnings(firth_fit(d$x, d$y, firth = FALSE, tol = 1e-10))
    expect_gt(max(abs(ml$coefficients)), 10)
    expect_lt(max(abs(fit$coefficients)), 10)
    # Without bias reduction the adjustment is zero and the penalized log
    # likelihood is the log likelihood.
    expect_identical(ml$adjustment, rep(0, length(ml$coefficients)))
    expect_identical(ml$penalized.loglik, ml$loglik)
})

test_that("bias reduction with weights, offsets, grouped data, and another reference", {
    d <- vglm_sim_data(400L, 2L, 3L, seed = 2L)
    n <- nrow(d$x)
    set.seed(7L)
    w <- runif(n, 0.5, 2)
    w[1:15] <- 0
    off <- matrix(rnorm(n * 2L, sd = 0.3), n, 2L)
    ref <- tight_reference(d, weights = w, offset = off, reference = 3L)
    fit <- firth_fit(d$x, d$y, weights = w, offset = off, reference = 3L)
    check_bias_reduced(fit)
    expect_near_reference(fit, ref)
    grp <- firth_fit(d$x, counts = d$counts, weights = w, offset = off,
                     reference = 3L)
    expect_max_diff(grp$coefficients, fit$coefficients, 1e-10)
    # The constraint mapping path gives the same fit.
    fm <- firth_fit(d$x, d$y, weights = w, offset = off, reference = 3L,
                    control = list(force_mapping = TRUE))
    expect_max_diff(fm$coefficients, fit$coefficients, 1e-10)
    expect_max_diff(fm$adjustment, fit$adjustment, 1e-10)
})

test_that("K = 2 bias reduction equals fastglm(binomial(), firth = TRUE)", {
    d <- vglm_sim_data(500L, 3L, 2L, seed = 4L)
    fit <- firth_fit(d$x, d$y)
    check_bias_reduced(fit)
    expect_near_reference(fit, tight_reference(d))
    gl <- fastglm::fastglm(d$x, as.numeric(d$y == "B"), family = binomial(),
                           firth = TRUE, tol = 1e-12)
    # 1e-8 allows for the GLM fit's own stopping error (about 4e-11 here).
    bound <- coef_bound(fit) + 1e-8
    expect_true(all(abs(fit$coefficients - unname(coef(gl))) <= bound))
})

test_that("a common slope: the bias-reduced estimate maximizes the penalized likelihood", {
    d <- vglm_sim_data(250L, 2L, 4L, seed = 12L)
    M <- 3L
    H <- list(diag(M), matrix(1, M, 1L), diag(M))
    fit <- firth_fit(d$x, d$y, constraints = H)
    check_bias_reduced(fit)
    # Penalized log likelihood l + (1/2) log det I computed in R from the
    # expanded rows X_i* = [x_i1 H_1, ..., x_ip H_p].
    r <- vapply(H, ncol, integer(1))
    Hbig <- matrix(0, M * length(H), sum(r))
    for (k in seq_along(H))
        Hbig[(k - 1L) * M + seq_len(M), sum(r[seq_len(k - 1L)]) + seq_len(r[k])] <- H[[k]]
    pll <- function(beta) {
        B <- matrix(Hbig %*% beta, M)            # M by p, column k = H_k beta_k
        eta <- d$x %*% t(B)
        lp <- cbind(0, eta)
        lse <- apply(lp, 1, function(v) max(v) + log(sum(exp(v - max(v)))))
        pr <- exp(lp - lse)
        ll <- sum(lp[cbind(seq_len(nrow(lp)), as.integer(d$y))] - lse)
        I <- matrix(0, length(beta), length(beta))
        for (i in seq_len(nrow(d$x))) {
            Xi <- kronecker(t(d$x[i, ]), diag(M)) %*% Hbig
            p <- pr[i, -1]
            I <- I + t(Xi) %*% (diag(p, M) - tcrossprod(p)) %*% Xi
        }
        c(ll + 0.5 * logdet(I), ll)
    }
    beta <- fit$coefficients
    at <- pll(beta)
    expect_max_diff(fit$penalized.loglik, at[1], 1e-10)
    expect_max_diff(fit$loglik, at[2], 1e-10)
    h <- 1e-5
    grad <- vapply(seq_along(beta), function(j) {
        e <- replace(numeric(length(beta)), j, h)
        (pll(beta + e)[1] - pll(beta - e)[1]) / (2 * h)
    }, numeric(1))
    # The gradient of the penalized likelihood is the adjusted score for the
    # canonical link, at any coefficients; this checks the adjustment itself,
    # independently of where the fit stopped.
    expect_max_diff(grad, fit$score + fit$adjustment, 1e-6)
})

test_that("bias reduction with a duplicated design column aliases the later column", {
    d <- vglm_sim_data(300L, 3L, 4L, seed = 1L)
    M <- 3L
    xd <- cbind(d$x, dup = d$x[, "x2"])
    fit <- firth_fit(xd, d$y)
    base <- firth_fit(d$x, d$y)
    check_bias_reduced(fit)
    P <- ncol(xd) * M
    expect_identical(which(fit$aliased), (P - M + 1L):P)
    expect_true(all(is.na(fit$coefficients[fit$aliased])))
    expect_max_diff(fit$coefficients[!fit$aliased], base$coefficients, 1e-8)
    expect_max_diff(fit$penalized.loglik, base$penalized.loglik, 1e-10)
})

test_that("bias reduction makes one adjustment pass per iteration", {
    d <- vglm_sim_data(300L, 3L, 4L, seed = 1L)
    fit <- firth_fit(d$x, d$y)
    # One pass at the start, at least one trial pass per accepted step (one
    # exactly when no step is halved), and one adjustment pass at each
    # iterate, the last included.
    expect_gte(fit$passes, 2L * fit$iter + 2L)
    nb <- firth_fit(d$x, d$y, firth = FALSE)
    expect_gte(nb$passes, nb$iter + 1L)
    expect_false(nb$singular)
})
