# Large-K accumulation of the information: the diagonal-minus-low-rank path
# (the model type writes W_i = diag(d_i) - C_i C_i') must agree with the
# pairwise path (control force_pairwise = TRUE, test only) on every data
# representation.

largek_inputs <- function(x, y = NULL, counts = NULL, weights = NULL,
                          offset = NULL, constraints = NULL, K = NULL) {
    n <- nrow(x)
    if (is.null(K)) K <- if (is.null(counts)) nlevels(y) else ncol(counts)
    M <- K - 1L
    list(codes = if (is.null(counts)) as.integer(y) else integer(0),
         counts = if (is.null(counts)) matrix(0, 0, 0) else counts,
         weights = if (is.null(weights)) rep(1, n) else as.numeric(weights),
         offset = if (is.null(offset)) matrix(0, n, M) else as.matrix(offset),
         constraints = if (is.null(constraints)) rep(list(diag(M)), ncol(x)) else constraints,
         K = as.integer(K))
}

# Fit through the internal export for the representation of x: a dense
# matrix, a dgCMatrix (passed transposed), or a big.matrix.
largek_fit <- function(x, ..., control = list(), block_size = 173L) {
    a <- largek_inputs(if (inherits(x, "big.matrix")) matrix(0, nrow(x), ncol(x)) else x, ...)
    fn <- if (inherits(x, "dgCMatrix")) fastglm:::vglm_fit_sparse
          else if (inherits(x, "big.matrix")) fastglm:::vglm_fit_big
          else fastglm:::vglm_fit_dense
    xin <- if (inherits(x, "dgCMatrix")) Matrix::t(x)
           else if (inherits(x, "big.matrix")) x@address else x
    fn(xin, a$codes, a$counts, a$weights, a$offset, a$constraints, "multinomial",
       a$K, 1L, FALSE, 1e-10, 100L, as.integer(block_size), FALSE, control)
}

# The engine's vglm_lowrank_min_M (inst/include/vglm_engine.h): by default the
# low-rank accumulation is used from this many linear predictors M = K - 1 on.
largek_min_M <- 6L

# The accumulation a dense or big.matrix fit with default controls reports.
largek_default <- function(K) if (K - 1L >= largek_min_M) "lowrank" else "pairwise"

expect_lowrank_pairwise <- function(lr, pw, accumulation = "lowrank") {
    expect_identical(lr$accumulation, accumulation)
    expect_identical(pw$accumulation, "pairwise")
    expect_max_diff(lr$information, pw$information, 1e-10)
    expect_max_diff(lr$coefficients, pw$coefficients, 1e-10)
    expect_max_diff(lr$deviance, pw$deviance, 1e-10)
    expect_identical(lr$iter, pw$iter)
    expect_identical(lr$converged, pw$converged)
}

test_that("lowrank_matches_pairwise", {
    for (K in c(3L, 6L, 15L)) {
        d <- vglm_sim_data(n = 500L, p = 3L, K = K, seed = K)
        designs <- list(dense = d$x)
        if (requireNamespace("Matrix", quietly = TRUE))
            designs$sparse <- Matrix::Matrix(d$x, sparse = TRUE)
        if (requireNamespace("bigmemory", quietly = TRUE))
            designs$big <- bigmemory::as.big.matrix(d$x)
        for (nm in names(designs)) {
            sparse <- nm == "sparse"
            pw <- largek_fit(designs[[nm]], y = d$y, control = list(force_pairwise = TRUE))
            # The low-rank path, forced below the threshold.
            lr <- largek_fit(designs[[nm]], y = d$y, control = list(force_lowrank = TRUE))
            expect_true(lr$converged)
            expect_lowrank_pairwise(lr, pw, if (sparse) "pairwise" else "lowrank")
            # Default controls: the path chosen by the threshold.
            de <- largek_fit(designs[[nm]], y = d$y)
            expect_lowrank_pairwise(de, pw, if (sparse) "pairwise" else largek_default(K))
        }
    }
})

test_that("lowrank_constraints_offsets", {
    K <- 6L
    M <- K - 1L
    d <- vglm_sim_data(n = 500L, p = 3L, K = K, seed = 21L)
    n <- nrow(d$x)
    set.seed(22L)
    off <- matrix(rnorm(n * M, sd = 0.3), n, M)
    w <- runif(n, 0.5, 2)
    w[sample.int(n, 40L)] <- 0
    H <- rep(list(diag(M)), ncol(d$x))
    H[[3L]] <- matrix(1, M, 1L)
    base <- list(force_mapping = TRUE, force_lowrank = TRUE)
    lr <- largek_fit(d$x, y = d$y, weights = w, offset = off, constraints = H,
                     control = base)
    pw <- largek_fit(d$x, y = d$y, weights = w, offset = off, constraints = H,
                     control = c(base, force_pairwise = TRUE))
    expect_true(lr$converged)
    expect_identical(length(lr$coefficients), ncol(d$x) * M - M + 1L)
    expect_lowrank_pairwise(lr, pw)

    # Grouped response with some rows of zero total count.
    counts <- d$counts
    counts[sample.int(n, 30L), ] <- 0
    lr <- largek_fit(d$x, counts = counts, weights = w, offset = off,
                     constraints = H, control = base)
    pw <- largek_fit(d$x, counts = counts, weights = w, offset = off,
                     constraints = H, control = c(base, force_pairwise = TRUE))
    expect_true(lr$converged)
    expect_lowrank_pairwise(lr, pw)
})

test_that("lowrank_family_terms", {
    set.seed(31L)
    K <- 7L
    M <- K - 1L
    b <- 40L
    eta <- matrix(rnorm(b * M, sd = 1.5), b, M)
    codes <- sample.int(K, b, replace = TRUE)
    counts <- matrix(rpois(b * K, 2), b, K)
    counts[5L, ] <- 0
    w <- runif(b, 0.5, 2)
    w[3L] <- 0
    empty <- matrix(numeric(0), 0, 0)
    for (reference in c(0L, 3L)) {
        cases <- list(
            fastglm:::vglm_family_eval("multinomial", K, reference, eta, codes,
                                       empty, w, empty),
            fastglm:::vglm_family_eval("multinomial", K, reference, eta, integer(0),
                                       counts, w, empty))
        for (ev in cases) {
            expect_identical(dim(ev$d), c(b, M))
            expect_max_diff(ev$u_lowrank, ev$u, 1e-12)
            expect_identical(nrow(ev$C), b)
            r <- ncol(ev$C) %/% M
            expect_identical(ncol(ev$C), r * M)
            expect_identical(r, 1L)
            for (i in seq_len(b)) {
                Ci <- matrix(ev$C[i, ], M, r)
                Wi <- diag(ev$d[i, ], M) - Ci %*% t(Ci)
                S <- matrix(0, M, M)
                S[upper.tri(S, diag = TRUE)] <- ev$W[i, ]
                S[lower.tri(S)] <- t(S)[lower.tri(S)]
                expect_max_diff(Wi, S, 1e-12)
            }
        }
    }
})

test_that("lowrank_cap_subblocks", {
    # p = 4 design columns and M = 5, so one row of Z has pM = 20 entries.
    d <- vglm_sim_data(n = 500L, p = 3L, K = 6L, seed = 41L)
    pw <- largek_fit(d$x, y = d$y, control = list(force_pairwise = TRUE))
    # block_size 173 with a cap of 1000 entries: 50 rows per sub-block, so
    # every full block splits into 4 row groups.
    lr <- largek_fit(d$x, y = d$y, control = list(lowrank_cap = 1000L, force_lowrank = TRUE))
    expect_true(lr$converged)
    expect_lowrank_pairwise(lr, pw)
    # A cap below one row's pM entries processes one row at a time.
    lr1 <- largek_fit(d$x, y = d$y, control = list(lowrank_cap = 7L, force_lowrank = TRUE))
    expect_lowrank_pairwise(lr1, pw)
    expect_error(largek_fit(d$x, y = d$y, control = list(lowrank_cap = 0)),
                 "lowrank_cap")
    expect_error(largek_fit(d$x, y = d$y, control = list(lowrank_cap = 2.5)),
                 "lowrank_cap")
})

test_that("lowrank_threshold", {
    # Dense default fits switch to the low-rank path at M = largek_min_M.
    for (K in largek_min_M + 0:1) {
        if (K < 2L) next
        d <- vglm_sim_data(n = 300L, p = 2L, K = K, seed = 50L + K)
        expect_identical(largek_fit(d$x, y = d$y)$accumulation, largek_default(K))
    }
    # force_pairwise takes precedence over force_lowrank.
    d <- vglm_sim_data(n = 300L, p = 2L, K = 4L, seed = 55L)
    both <- largek_fit(d$x, y = d$y, control = list(force_lowrank = TRUE, force_pairwise = TRUE))
    expect_identical(both$accumulation, "pairwise")
})

test_that("lowrank_source", {
    # A streamed fastglm_source (callback over 5 blocks) takes the low-rank
    # path and agrees with the pairwise path.
    set.seed(61L)
    K <- 6L
    n <- 600L
    dd <- data.frame(x1 = rnorm(n), x2 = rnorm(n), w = runif(n, 0.5, 2))
    eta <- cbind(0, outer(dd$x1, seq(-0.5, 0.5, length.out = K - 1L)) +
                     outer(dd$x2, rep(0.3, K - 1L)))
    pr <- exp(eta) / rowSums(exp(eta))
    dd$y <- factor(LETTERS[apply(pr, 1L, function(q) sample.int(K, 1L, prob = q))],
                   levels = LETTERS[seq_len(K)])
    blk <- split(seq_len(n), ceiling(seq_len(n) / 120))
    src <- fastglm_source(function(k) dd[blk[[k]], ], length(blk), formula = y ~ x1 + x2,
                          prototype = dd[1:20, ], weights = "w")
    read <- fastglm:::.vglm_stream_reader(src, NULL)
    totals <- as.numeric(tapply(dd$w, dd$y, sum))
    run <- function(control)
        fastglm:::vglm_fit_source(read, length(blk), 3L, totals, 1L, 1, rep(list(diag(K - 1L)), 3L),
                                  "multinomial", K, 1L, FALSE, 1e-10, 100L, 97L, FALSE, control)
    pw <- run(list(force_pairwise = TRUE))
    lr <- run(list(force_lowrank = TRUE))
    expect_true(lr$converged)
    expect_lowrank_pairwise(lr, pw)
    expect_identical(run(list())$accumulation, largek_default(K))

    # The public interface gives the same fit either way.
    a <- fastglm_multinom(src, tol = 1e-10, control = list(force_lowrank = TRUE))
    b <- fastglm_multinom(src, tol = 1e-10, control = list(force_pairwise = TRUE))
    expect_max_diff(a$coefficients, b$coefficients, 1e-10)
    expect_max_diff(a$information, b$information, 1e-10)
    expect_max_diff(a$deviance, b$deviance, 1e-10)
    expect_identical(a$accumulation, "lowrank")
    expect_identical(b$accumulation, "pairwise")
})

test_that("aliasing_identity_matches_mapped", {
    d <- vglm_sim_data(n = 300L, p = 4L, K = 4L, seed = 31L)
    x <- d$x
    # x5 duplicates x1; x6 is the sum of x2 and x3.
    x <- cbind(x, dup = x[, "x1"], sumcol = x[, "x2"] + x[, "x3"])
    id <- largek_fit(x, y = d$y)
    mp <- largek_fit(x, y = d$y, control = list(force_mapping = TRUE))
    M <- 3L
    expected <- rep(FALSE, ncol(x) * M)
    # Coefficient (k, a) sits at (k - 1) * M + a; columns 6 and 7 are aliased.
    expected[(5L * M + 1L):(7L * M)] <- TRUE
    expect_identical(id$aliased, expected)
    expect_identical(mp$aliased, id$aliased)
    expect_true(id$converged)
    expect_identical(id$rank, ncol(x) * M - 2L * M)
    expect_identical(is.na(id$coefficients), id$aliased)
    expect_identical(is.na(mp$coefficients), id$aliased)
    expect_max_diff(id$coefficients[!id$aliased], mp$coefficients[!mp$aliased], 1e-10)
    expect_max_diff(id$deviance, mp$deviance, 1e-10)
})

test_that("covariance_unchanged", {
    d <- vglm_sim_data(n = 300L, p = 3L, K = 5L)
    a <- largek_inputs(d$x, y = d$y)
    for (firth in c(FALSE, TRUE)) {
        fit <- fastglm:::vglm_fit_dense(d$x, a$codes, a$counts, a$weights, a$offset,
                                        a$constraints, "multinomial", a$K, 1L, firth,
                                        1e-10, 100L, 173L, FALSE, list())
        expect_true(fit$converged)
        expect_false(fit$singular)
        expect_max_diff(fit$cov.unscaled, solve(fit$information), 1e-10)
        expect_max_diff(fit$cov.unscaled, t(fit$cov.unscaled), 1e-12)
    }
})

test_that("covariance_na_at_aliased_and_unconverged", {
    d <- vglm_sim_data(n = 200L, p = 2L, K = 3L, seed = 5L)
    x <- cbind(d$x, dup = d$x[, "x1"])
    a <- largek_inputs(x, y = d$y)
    run <- function(maxit) fastglm:::vglm_fit_dense(x, a$codes, a$counts, a$weights, a$offset,
                                                    a$constraints, "multinomial", a$K, 1L, FALSE,
                                                    1e-10, maxit, 173L, FALSE, list())
    for (maxit in c(100L, 2L)) {
        fit <- run(maxit)
        al <- which(fit$aliased)
        expect_length(al, 2L)
        expect_true(all(is.na(fit$cov.unscaled[al, ])))
        expect_true(all(is.na(fit$cov.unscaled[, al])))
        keep <- which(!fit$aliased)
        expect_max_diff(fit$cov.unscaled[keep, keep], solve(fit$information[keep, keep]), 1e-10)
    }
    expect_identical(run(2L)$convergence.reason, "iteration_limit")
})

test_that("covariance_singular_iterate_pseudoinverse", {
    d <- vglm_sim_data(n = 300L, p = 2L, K = 3L, seed = 3L)
    set.seed(1L)
    # Not structurally aliased (relative residual 1e-7 is above the 1e-11
    # rule), but the scaled information has two singular values near 1e-14.
    x <- cbind(d$x, near = d$x[, "x1"] + 1e-7 * rnorm(nrow(d$x)))
    a <- largek_inputs(x, y = d$y)
    fit <- fastglm:::vglm_fit_dense(x, a$codes, a$counts, a$weights, a$offset,
                                    a$constraints, "multinomial", a$K, 1L, FALSE,
                                    1e-10, 100L, 173L, FALSE, list())
    expect_true(fit$singular)
    expect_false(any(fit$aliased))
    I <- fit$information
    s <- 1 / sqrt(diag(I))
    Ci <- fit$cov.unscaled / outer(s, s)     # pseudoinverse of the scaled information
    C <- I * outer(s, s)
    scale <- max(abs(Ci))
    # The pseudoinverse is ill determined at this conditioning, so the
    # comparison with MASS::ginv is at 5e-2 of the largest entry.
    expect_max_diff(Ci / scale, MASS::ginv(C, tol = 1e-15) / scale, 5e-2)
    expect_max_diff(Ci / scale, t(Ci) / scale, 1e-12)
    expect_max_diff(fit$cov.unscaled, t(fit$cov.unscaled) , 1e-8 * max(1, max(abs(fit$cov.unscaled))))
})

test_that("aliasing_zero_and_near_collinear_columns", {
    d <- vglm_sim_data(n = 300L, p = 3L, K = 4L, seed = 41L)
    set.seed(42L)
    n <- nrow(d$x)
    rel <- function(col, eps) col + eps * sqrt(sum(col^2) / n) * rnorm(n)
    x <- cbind(d$x,
               zero = 0,
               near13 = rel(d$x[, "x1"], 1e-13),   # aliased
               near9 = rel(d$x[, "x2"], 1e-9))     # kept
    M <- 3L
    id <- largek_fit(x, y = d$y)
    mp <- largek_fit(x, y = d$y, control = list(force_mapping = TRUE))
    expected <- rep(FALSE, ncol(x) * M)
    expected[(4L * M + 1L):(6L * M)] <- TRUE   # columns 5 (zero) and 6 (near13)
    expect_identical(id$aliased, expected)
    expect_identical(mp$aliased, id$aliased)
    expect_identical(is.na(id$coefficients), id$aliased)
})

# Information-vector products, the diagonal pass, and the preconditioner
# through the test-only export vglm_info_product_dense.
largek_product <- function(x, beta, v, ..., control = list(), r = NULL,
                           block_size = 97L) {
    sparse <- inherits(x, "dgCMatrix")
    a <- largek_inputs(if (sparse) as.matrix(x) else x, ...)
    fn <- if (sparse) fastglm:::vglm_info_product_sparse else fastglm:::vglm_info_product_dense
    fn(if (sparse) Matrix::t(x) else x, a$codes, a$counts, a$weights, a$offset,
       a$constraints, "multinomial", a$K, 1L, beta, v, as.integer(block_size), control, r)
}

# The information sum_i X_i*' W_i X_i* computed in R, with W_i from
# vglm_family_eval at eta = X B + offset (reference category 1, that is,
# 0-based 0) and X_i* = (x_i' kron I_M) blockdiag(H_1, ..., H_p). With
# diagonal = TRUE, W_i is replaced by diag(W_i): the information with the
# off-diagonal predictor blocks dropped.
largek_information_r <- function(x, beta, y, weights, offset, constraints, K,
                                 diagonal = FALSE) {
    M <- K - 1L
    n <- nrow(x)
    r <- vapply(constraints, ncol, integer(1))
    start <- cumsum(c(0L, r))[seq_along(r)]
    B <- t(vapply(seq_along(constraints), function(k)
        as.numeric(constraints[[k]] %*% beta[start[k] + seq_len(r[k])]), numeric(M)))
    if (M == 1L) B <- t(B)
    eta <- x %*% B + offset
    ev <- fastglm:::vglm_family_eval("multinomial", K, 0L, eta, as.integer(y),
                                     matrix(numeric(0), 0, 0), weights,
                                     matrix(numeric(0), 0, 0))
    Hbig <- matrix(0, length(constraints) * M, sum(r))
    for (k in seq_along(constraints))
        Hbig[(k - 1L) * M + seq_len(M), start[k] + seq_len(r[k])] <- constraints[[k]]
    I <- matrix(0, sum(r), sum(r))
    for (i in seq_len(n)) {
        S <- matrix(0, M, M)
        S[upper.tri(S, diag = TRUE)] <- ev$W[i, ]
        S[lower.tri(S)] <- t(S)[lower.tri(S)]
        if (diagonal) S <- diag(diag(S), M)
        Xi <- kronecker(t(x[i, ]), diag(M)) %*% Hbig
        I <- I + t(Xi) %*% S %*% Xi
    }
    I
}

test_that("info_product_matches_information", {
    for (K in c(3L, 10L)) {
        M <- K - 1L
        d <- vglm_sim_data(n = 300L, p = 3L, K = K, seed = 70L + K)
        n <- nrow(d$x)
        set.seed(71L + K)
        off <- matrix(rnorm(n * M, sd = 0.3), n, M)
        w <- runif(n, 0.5, 2)
        w[sample.int(n, 25L)] <- 0
        Hid <- rep(list(diag(M)), ncol(d$x))
        Hpar <- Hid
        Hpar[[3L]] <- matrix(1, M, 1L)
        for (H in list(Hid, Hpar)) {
            P <- sum(vapply(H, ncol, integer(1)))
            beta <- rnorm(P, sd = 0.2)
            Iref <- largek_information_r(d$x, beta, d$y, w, off, H, K)
            for (fm in c(FALSE, TRUE)) for (fp in c(FALSE, TRUE)) {
                control <- list(force_mapping = fm, force_pairwise = fp)
                for (j in 1:3) {
                    v <- rnorm(P)
                    out <- largek_product(d$x, beta, v, y = d$y, weights = w, offset = off,
                                          constraints = H, control = control)
                    expect_max_diff(out$product, out$information %*% v, 1e-10)
                    expect_max_diff(out$information, Iref, 1e-10)
                }
                # The diagonal pass changes only the accumulated blocks.
                expect_identical(out$loglik.diagonal, out$loglik)
                expect_identical(out$score.diagonal, out$score)
                expect_identical(out$pmin.diagonal, out$pmin)
                expect_identical(out$pmax.diagonal, out$pmax)
                expect_identical(out$structure.diagonal, out$structure)
                # Under an identity layout the diagonal pass forms no P by P matrix.
                expect_identical(out$diagonal.I.rows,
                                 if (!fm && identical(H, Hid)) 0L else P)
            }
        }
    }
})

test_that("diagonal_pass_sparse_matches_dense", {
    # The diagonal pass of a sparse design uses the low-rank terms (no packed
    # W) and gives the same blocks as that of the dense design.
    for (K in c(3L, 10L)) {
        M <- K - 1L
        d <- vglm_sim_data(n = 300L, p = 3L, K = K, seed = 75L + K)
        n <- nrow(d$x)
        set.seed(76L + K)
        off <- matrix(rnorm(n * M, sd = 0.3), n, M)
        w <- runif(n, 0.5, 2)
        w[sample.int(n, 25L)] <- 0
        xs <- Matrix::Matrix(d$x, sparse = TRUE)
        Hid <- rep(list(diag(M)), ncol(d$x))
        Hpar <- Hid
        Hpar[[3L]] <- matrix(1, M, 1L)
        for (H in list(Hid, Hpar)) {
            P <- sum(vapply(H, ncol, integer(1)))
            beta <- rnorm(P, sd = 0.2)
            v <- rnorm(P)
            for (fp in c(FALSE, TRUE)) {
                control <- list(force_pairwise = fp)
                de <- largek_product(d$x, beta, v, y = d$y, weights = w, offset = off,
                                     constraints = H, control = control)
                sp <- largek_product(xs, beta, v, y = d$y, weights = w, offset = off,
                                     constraints = H, control = control)
                expect_max_diff(sp$information.diagonal, de$information.diagonal, 1e-10)
            }
        }
    }
})

test_that("info_product_zero_rows", {
    # Rows of zero weight, or grouped rows of zero total, contribute nothing.
    K <- 8L
    M <- K - 1L
    d <- vglm_sim_data(n = 200L, p = 2L, K = K, seed = 81L)
    n <- nrow(d$x)
    w <- runif(n, 0.5, 2)
    drop <- sort(sample.int(n, 30L))
    w[drop] <- 0
    counts <- d$counts
    counts[drop, ] <- 0
    set.seed(82L)
    P <- ncol(d$x) * M
    beta <- rnorm(P, sd = 0.2)
    v <- rnorm(P)
    for (fp in c(FALSE, TRUE)) {
        control <- list(force_pairwise = fp)
        a <- largek_product(d$x, beta, v, y = d$y, weights = w, control = control)
        b <- largek_product(d$x[-drop, ], beta, v, y = d$y[-drop], weights = w[-drop],
                            control = control)
        g <- largek_product(d$x, beta, v, counts = counts, control = control)
        expect_max_diff(a$product, b$product, 1e-10)
        expect_max_diff(g$product, largek_product(d$x[-drop, ], beta, v,
                                                  counts = counts[-drop, ],
                                                  control = control)$product, 1e-10)
    }
})

test_that("diagonal_pass_blocks", {
    for (K in c(3L, 10L)) {
        M <- K - 1L
        d <- vglm_sim_data(n = 300L, p = 3L, K = K, seed = 90L + K)
        set.seed(91L + K)
        P <- ncol(d$x) * M
        beta <- rnorm(P, sd = 0.2)
        a_of <- (seq_len(P) - 1L) %% M
        for (fp in c(FALSE, TRUE)) {
            out <- largek_product(d$x, beta, rnorm(P), y = d$y,
                                  control = list(force_pairwise = fp))
            expected <- out$information * outer(a_of, a_of, "==")
            expect_max_diff(out$information.diagonal, expected, 1e-12)
        }
    }
})

test_that("preconditioner_apply", {
    K <- 7L
    M <- K - 1L
    d <- vglm_sim_data(n = 300L, p = 3L, K = K, seed = 101L)
    x <- cbind(d$x, dup = d$x[, "x1"])
    set.seed(102L)
    P <- ncol(x) * M
    beta <- rnorm(P, sd = 0.2)
    al <- (4L * M + 1L):(5L * M)   # the coefficients of the duplicated column
    beta[al] <- 0
    for (fm in c(FALSE, TRUE)) {
        r <- rnorm(P)
        out <- largek_product(x, beta, rnorm(P), y = d$y,
                              control = list(force_mapping = fm), r = r)
        expect_identical(which(out$aliased), al)
        expect_true(out$precond.ok)
        keep <- which(!out$aliased)
        expect_max_diff(out$precond.apply[keep],
                        solve(out$information.diagonal[keep, keep], r[keep]), 1e-10)
        expect_identical(out$precond.apply[al], rep(0, M))
        # Entries of r at aliased coefficients are not read.
        r2 <- r
        r2[al] <- NaN
        out2 <- largek_product(x, beta, rnorm(P), y = d$y,
                               control = list(force_mapping = fm), r = r2)
        expect_identical(out2$precond.apply, out$precond.apply)
    }
})

test_that("info_product_force_lowrank_small_K", {
    # The product uses the low-rank form at every M unless force_pairwise is
    # set; at K = 3 the information is accumulated pairwise by default, so
    # this checks one path against the other.
    K <- 3L
    M <- K - 1L
    d <- vglm_sim_data(n = 300L, p = 3L, K = K, seed = 111L)
    n <- nrow(d$x)
    set.seed(112L)
    off <- matrix(rnorm(n * M, sd = 0.3), n, M)
    w <- runif(n, 0.5, 2)
    w[sample.int(n, 20L)] <- 0
    P <- ncol(d$x) * M
    beta <- rnorm(P, sd = 0.2)
    for (control in list(list(), list(force_lowrank = TRUE),
                         list(force_lowrank = TRUE, force_mapping = TRUE))) {
        for (j in 1:3) {
            v <- rnorm(P)
            out <- largek_product(d$x, beta, v, y = d$y, weights = w, offset = off,
                                  control = control)
            expect_max_diff(out$product, out$information %*% v, 1e-10)
        }
    }
})

test_that("diagonal_pass_mapped", {
    # Under a non-identity layout (force_mapping, a parallel constraint, and
    # a two-column constraint) the diagonal pass is mapped to P by P; it is
    # checked against sum_i X_i*' diag(W_i) X_i* computed in R.
    for (K in c(3L, 8L)) {
        M <- K - 1L
        d <- vglm_sim_data(n = 200L, p = 3L, K = K, seed = 120L + K)
        n <- nrow(d$x)
        set.seed(121L + K)
        off <- matrix(rnorm(n * M, sd = 0.3), n, M)
        w <- runif(n, 0.5, 2)
        w[sample.int(n, 15L)] <- 0
        H <- rep(list(diag(M)), ncol(d$x))
        H[[3L]] <- matrix(1, M, 1L)
        H[[4L]] <- cbind(1, seq_len(M))
        P <- sum(vapply(H, ncol, integer(1)))
        beta <- rnorm(P, sd = 0.2)
        ref <- largek_information_r(d$x, beta, d$y, w, off, H, K, diagonal = TRUE)
        for (fp in c(FALSE, TRUE)) {
            out <- largek_product(d$x, beta, rnorm(P), y = d$y, weights = w, offset = off,
                                  constraints = H,
                                  control = list(force_mapping = TRUE, force_pairwise = fp))
            expect_identical(out$diagonal.I.rows, P)
            expect_max_diff(out$information.diagonal, ref, 1e-10)
        }
        # Identity constraints with force_mapping: the same reference.
        Hid <- rep(list(diag(M)), ncol(d$x))
        Pid <- ncol(d$x) * M
        bid <- rnorm(Pid, sd = 0.2)
        ref <- largek_information_r(d$x, bid, d$y, w, off, Hid, K, diagonal = TRUE)
        for (fm in c(FALSE, TRUE)) {
            out <- largek_product(d$x, bid, rnorm(Pid), y = d$y, weights = w, offset = off,
                                  control = list(force_mapping = fm))
            expect_max_diff(out$information.diagonal, ref, 1e-10)
        }
    }
})

test_that("info_product_sparse", {
    skip_if_not_installed("Matrix")
    # Sparse designs take the low-rank product (no Z is formed) and agree
    # with the dense results.
    for (K in c(3L, 10L)) {
        M <- K - 1L
        d <- vglm_sim_data(n = 300L, p = 3L, K = K, seed = 130L + K)
        x <- d$x
        x[abs(x) < 0.5] <- 0
        x[, 1L] <- 1
        xs <- Matrix::Matrix(x, sparse = TRUE)
        n <- nrow(x)
        set.seed(131L + K)
        off <- matrix(rnorm(n * M, sd = 0.3), n, M)
        w <- runif(n, 0.5, 2)
        w[sample.int(n, 20L)] <- 0
        H <- rep(list(diag(M)), ncol(x))
        H[[3L]] <- matrix(1, M, 1L)
        P <- sum(vapply(H, ncol, integer(1)))
        beta <- rnorm(P, sd = 0.2)
        for (control in list(list(), list(force_pairwise = TRUE), list(force_mapping = TRUE))) {
            v <- rnorm(P)
            sp <- largek_product(xs, beta, v, y = d$y, weights = w, offset = off,
                                 constraints = H, control = control)
            de <- largek_product(x, beta, v, y = d$y, weights = w, offset = off,
                                 constraints = H, control = control)
            expect_max_diff(sp$product, de$information %*% v, 1e-10)
            expect_max_diff(sp$information, de$information, 1e-10)
            expect_max_diff(sp$information.diagonal, de$information.diagonal, 1e-10)
        }
    }
})

# Truncated Newton (preconditioned conjugate gradient) solver: control
# solver = "cg" against the exact solver, at tol = 1e-12.
cg_fit <- function(x, ..., control = list(), tol = 1e-12, maxit = 100L,
                   firth = FALSE, block_size = 173L) {
    a <- largek_inputs(if (inherits(x, "big.matrix")) matrix(0, nrow(x), ncol(x)) else x, ...)
    fn <- if (inherits(x, "dgCMatrix")) fastglm:::vglm_fit_sparse
          else if (inherits(x, "big.matrix")) fastglm:::vglm_fit_big
          else fastglm:::vglm_fit_dense
    xin <- if (inherits(x, "dgCMatrix")) Matrix::t(x)
           else if (inherits(x, "big.matrix")) x@address else x
    fn(xin, a$codes, a$counts, a$weights, a$offset, a$constraints, "multinomial",
       a$K, 1L, firth, tol, as.integer(maxit), as.integer(block_size), FALSE, control)
}

expect_cg_exact <- function(cg, ex) {
    expect_identical(cg$solver, "cg")
    expect_identical(ex$solver, "exact")
    expect_identical(ex$products, 0L)
    expect_true(cg$products > 0L)
    expect_identical(cg$aliased, ex$aliased)
    expect_identical(is.na(cg$coefficients), is.na(ex$coefficients))
    keep <- !ex$aliased
    expect_max_diff(cg$coefficients[keep], ex$coefficients[keep], 1e-6)
    expect_max_diff(cg$cov.unscaled[keep, keep], ex$cov.unscaled[keep, keep], 1e-6)
    expect_max_diff(cg$deviance, ex$deviance, 1e-10)
    expect_identical(cg$converged, ex$converged)
}

test_that("cg_matches_exact", {
    for (K in c(3L, 10L, 30L)) {
        d <- vglm_sim_data(n = 3000L, p = 4L, K = K, seed = 200L + K)
        designs <- list(dense = d$x)
        if (requireNamespace("Matrix", quietly = TRUE))
            designs$sparse <- Matrix::Matrix(d$x, sparse = TRUE)
        if (requireNamespace("bigmemory", quietly = TRUE))
            designs$big <- bigmemory::as.big.matrix(d$x)
        for (nm in names(designs)) {
            ex <- cg_fit(designs[[nm]], y = d$y)
            cg <- cg_fit(designs[[nm]], y = d$y, control = list(solver = "cg"))
            expect_true(ex$converged)
            expect_true(cg$converged)
            expect_cg_exact(cg, ex)
            # Every product is a pass over the data.
            expect_true(cg$passes > cg$products)
        }
    }
})

test_that("cg_constraints_offsets", {
    K <- 6L
    M <- K - 1L
    d <- vglm_sim_data(n = 800L, p = 3L, K = K, seed = 221L)
    n <- nrow(d$x)
    set.seed(222L)
    off <- matrix(rnorm(n * M, sd = 0.3), n, M)
    w <- runif(n, 0.5, 2)
    H <- rep(list(diag(M)), ncol(d$x))
    H[[3L]] <- matrix(1, M, 1L)
    for (fm in c(FALSE, TRUE)) {
        base <- list(force_mapping = fm)
        ex <- cg_fit(d$x, y = d$y, weights = w, offset = off, constraints = H, control = base)
        cg <- cg_fit(d$x, y = d$y, weights = w, offset = off, constraints = H,
                     control = c(base, solver = "cg"))
        expect_true(cg$converged)
        expect_cg_exact(cg, ex)
        # Identity constraints with force_mapping: the mapped preconditioner.
        Hid <- rep(list(diag(M)), ncol(d$x))
        ex <- cg_fit(d$x, y = d$y, weights = w, offset = off, constraints = Hid, control = base)
        cg <- cg_fit(d$x, y = d$y, weights = w, offset = off, constraints = Hid,
                     control = c(base, solver = "cg"))
        expect_true(cg$converged)
        expect_cg_exact(cg, ex)
    }
})

test_that("cg_zero_weight_rows", {
    K <- 8L
    d <- vglm_sim_data(n = 600L, p = 3L, K = K, seed = 231L)
    n <- nrow(d$x)
    set.seed(232L)
    w <- runif(n, 0.5, 2)
    drop <- sort(sample.int(n, 60L))
    w[drop] <- 0
    counts <- d$counts
    counts[drop, ] <- 0
    ctl <- list(solver = "cg")
    full <- cg_fit(d$x, y = d$y, weights = w, control = ctl)
    reduced <- cg_fit(d$x[-drop, ], y = d$y[-drop], weights = w[-drop], control = ctl)
    expect_true(full$converged)
    expect_max_diff(full$coefficients, reduced$coefficients, 1e-6)
    expect_max_diff(full$cov.unscaled, reduced$cov.unscaled, 1e-6)
    expect_max_diff(full$deviance, reduced$deviance, 1e-10)
    expect_cg_exact(full, cg_fit(d$x, y = d$y, weights = w))
    gfull <- cg_fit(d$x, counts = counts, control = ctl)
    gred <- cg_fit(d$x[-drop, ], counts = counts[-drop, ], control = ctl)
    expect_true(gfull$converged)
    expect_max_diff(gfull$coefficients, gred$coefficients, 1e-6)
    expect_max_diff(gfull$cov.unscaled, gred$cov.unscaled, 1e-6)
    expect_max_diff(gfull$deviance, gred$deviance, 1e-10)
    expect_cg_exact(gfull, cg_fit(d$x, counts = counts))
})

test_that("cg_aliased_columns", {
    d <- vglm_sim_data(n = 600L, p = 3L, K = 7L, seed = 241L)
    x <- cbind(d$x, dup = d$x[, "x1"], dupint = 1)
    for (fm in c(FALSE, TRUE)) {
        ex <- cg_fit(x, y = d$y, control = list(force_mapping = fm))
        cg <- cg_fit(x, y = d$y, control = list(force_mapping = fm, solver = "cg"))
        expect_true(any(cg$aliased))
        expect_true(cg$converged)
        expect_cg_exact(cg, ex)
        expect_true(all(is.na(cg$cov.unscaled[cg$aliased, ])))
    }
})

test_that("cg_near_separation", {
    d <- vglm_separated_data()
    for (maxit in c(25L, 100L)) {
        ex <- cg_fit(d$x, y = d$y, maxit = maxit)
        cg <- cg_fit(d$x, y = d$y, maxit = maxit, control = list(solver = "cg"))
        expect_identical(cg$converged, ex$converged)
        expect_identical(cg$separation, ex$separation)
        expect_max_diff(cg$deviance, ex$deviance, 1e-8)
    }
})

test_that("cg_maxit_state", {
    K <- 6L
    d <- vglm_sim_data(n = 500L, p = 3L, K = K, seed = 251L)
    a <- largek_inputs(d$x, y = d$y)
    # (maxit, max_products): a CG step then the exit; with max_products = 1
    # at maxit = 3, the fit leaves right after an exact fallback step, whose
    # trial pass is diagonal, so the returned state needs the final full pass.
    for (case in list(c(1L, -1L), c(1L, 1L), c(3L, 1L))) {
        ctl <- list(solver = "cg", max_products = case[2])
        out <- capture.output(
            fit <- fastglm:::vglm_fit_dense(d$x, a$codes, a$counts, a$weights, a$offset,
                                            a$constraints, "multinomial", a$K, 1L, FALSE,
                                            1e-12, case[1], 173L, TRUE, ctl))
        expect_identical(fit$iter, case[1])
        expect_false(fit$converged)
        expect_identical(fit$convergence.reason, "iteration_limit")
        # The trace prints the common one-line format; the solver path is
        # recorded in solver.steps.
        expect_match(out, "^iter ")
        if (case[1] == 3L) expect_identical(fit$solver.steps[length(fit$solver.steps)], "fallback")
        ref <- largek_product(d$x, fit$coefficients, rep(0, length(fit$coefficients)),
                              y = d$y)
        expect_max_diff(fit$information, ref$information, 1e-10)
        expect_max_diff(fit$score, ref$score, 1e-10)
        expect_max_diff(fit$deviance, -2 * ref$loglik, 1e-12)
        expect_max_diff(fit$cov.unscaled, solve(ref$information), 1e-10)
    }
    # max_products = 1 forces exact fallbacks after the first step; the fit
    # still converges to the exact solution.
    fb <- cg_fit(d$x, y = d$y, control = list(solver = "cg", max_products = 1L))
    expect_cg_exact(fb, cg_fit(d$x, y = d$y))
})

test_that("cg_retry_and_halving_failed", {
    K <- 6L
    d <- vglm_sim_data(n = 500L, p = 3L, K = K, seed = 291L)
    a <- largek_inputs(d$x, y = d$y)
    ex <- cg_fit(d$x, y = d$y)
    # CG steps whose halving is treated as failed (test only) are repeated
    # with the exact step, and the fit still reaches the exact solution.
    out <- capture.output(
        rt <- fastglm:::vglm_fit_dense(d$x, a$codes, a$counts, a$weights, a$offset,
                                       a$constraints, "multinomial", a$K, 1L, FALSE,
                                       1e-12, 100L, 173L, TRUE,
                                       list(solver = "cg", cg_reject_steps = 2L)))
    expect_identical(sum(rt$solver.steps == "retry"), 2L)
    expect_true(rt$converged)
    expect_cg_exact(rt, ex)
    # Every CG step rejected: only exact steps are taken, as in the exact
    # solver (same iterations).
    all <- cg_fit(d$x, y = d$y, control = list(solver = "cg", cg_reject_steps = 1000L))
    expect_true(all$converged)
    expect_identical(all$iter, ex$iter)
    expect_max_diff(all$coefficients, ex$coefficients, 1e-6)
    expect_max_diff(all$deviance, ex$deviance, 1e-10)
    # A step bound of 0 (test only) blocks every step: the blocked CG step is
    # retried with the exact step, whose block ends the fit as in the exact
    # solver, with the state of the starting values.
    exb <- cg_fit(d$x, y = d$y, control = list(step_bound = 0))
    cgb <- cg_fit(d$x, y = d$y, control = list(step_bound = 0, solver = "cg"))
    expect_identical(exb$convergence.reason, "step_halving_failed")
    expect_identical(cgb$convergence.reason, exb$convergence.reason)
    expect_identical(cgb$converged, exb$converged)
    expect_identical(cgb$iter, 0L)
    expect_max_diff(cgb$coefficients, exb$coefficients, 1e-12)
    expect_max_diff(cgb$information, exb$information, 1e-10)
    expect_max_diff(cgb$cov.unscaled, exb$cov.unscaled, 1e-10)
    # A positive bound restricts the steps; CG and exact agree.
    exr <- cg_fit(d$x, y = d$y, control = list(step_bound = 0.05))
    cgr <- cg_fit(d$x, y = d$y, control = list(step_bound = 0.05, solver = "cg"))
    expect_identical(cgr$converged, exr$converged)
    expect_identical(cgr$convergence.reason, exr$convergence.reason)
    expect_max_diff(cgr$deviance, exr$deviance, 1e-10)
})

test_that("cg_firth_rejected", {
    d <- vglm_sim_data(n = 200L, p = 2L, K = 4L, seed = 261L)
    expect_error(cg_fit(d$x, y = d$y, firth = TRUE, control = list(solver = "cg")),
                 "the conjugate gradient solver is not available with bias reduction")
    expect_error(cg_fit(d$x, y = d$y, control = list(solver = "newton")),
                 "solver must be \"exact\" or \"cg\"")
    ex <- cg_fit(d$x, y = d$y, firth = TRUE, control = list(solver = "exact"))
    expect_identical(ex$solver, "exact")
})

test_that("cg_product_cache", {
    # A product read from the cache of per-row terms equals the product that
    # evaluates the model type, for the low-rank and pairwise forms, dense
    # and sparse, with constraints, offsets, and zero-weight rows.
    K <- 7L
    M <- K - 1L
    d <- vglm_sim_data(n = 400L, p = 3L, K = K, seed = 271L)
    n <- nrow(d$x)
    set.seed(272L)
    off <- matrix(rnorm(n * M, sd = 0.3), n, M)
    w <- runif(n, 0.5, 2)
    w[sample.int(n, 30L)] <- 0
    H <- rep(list(diag(M)), ncol(d$x))
    H[[3L]] <- matrix(1, M, 1L)
    P <- sum(vapply(H, ncol, integer(1)))
    beta <- rnorm(P, sd = 0.2)
    v <- rnorm(P)
    designs <- list(d$x)
    if (requireNamespace("Matrix", quietly = TRUE))
        designs <- c(designs, Matrix::Matrix(d$x, sparse = TRUE))
    for (x in designs) for (fp in c(FALSE, TRUE)) {
        on <- largek_product(x, beta, v, y = d$y, weights = w, offset = off,
                             constraints = H, control = list(force_pairwise = fp))
        off0 <- largek_product(x, beta, v, y = d$y, weights = w, offset = off,
                               constraints = H,
                               control = list(force_pairwise = fp, product_cache_cap = 0))
        expect_true(on$cache.used)
        expect_false(off0$cache.used)
        expect_max_diff(on$product.cached, on$product, 1e-12)
        expect_max_diff(off0$product.cached, on$product, 1e-12)
    }
    # A cap below the size of the terms disables the cache.
    small <- largek_product(d$x, beta, v, y = d$y, constraints = H,
                            control = list(product_cache_cap = 100))
    expect_false(small$cache.used)
    expect_max_diff(small$product.cached, small$product, 1e-12)

    # Whole fits with and without the cache agree.
    for (K in c(3L, 10L)) {
        d <- vglm_sim_data(n = 1000L, p = 3L, K = K, seed = 273L + K)
        a <- cg_fit(d$x, y = d$y, control = list(solver = "cg"))
        b <- cg_fit(d$x, y = d$y, control = list(solver = "cg", product_cache_cap = 0))
        expect_true(a$converged)
        expect_identical(a$iter, b$iter)
        expect_identical(a$products, b$products)
        expect_max_diff(a$coefficients, b$coefficients, 1e-12)
        expect_max_diff(a$deviance, b$deviance, 1e-12)
    }
})

test_that("cg_reject_confirmation", {
    # A confirmation treated as failed (test only) continues with exact steps
    # and converges to the exact solution after a later confirmation.
    d <- vglm_sim_data(n = 800L, p = 3L, K = 8L, seed = 281L)
    ex <- cg_fit(d$x, y = d$y)
    base <- cg_fit(d$x, y = d$y, control = list(solver = "cg"))
    rej <- cg_fit(d$x, y = d$y, control = list(solver = "cg", cg_reject_confirmation = 1L))
    expect_true(rej$converged)
    expect_identical(rej$convergence.reason, "converged")
    expect_identical(rej$iter, base$iter + 1L)
    expect_cg_exact(rej, ex)
    expect_cg_exact(base, ex)
})
