# Tests of the multinomial model type through the internal vglm_family_eval.

vglm_family_case <- function(b = 50L, K = 5L, seed = 2L) {
    set.seed(seed)
    M <- K - 1L
    eta <- matrix(rnorm(b * M, sd = 1.5), b, M)
    codes <- sample.int(K, b, replace = TRUE)
    counts <- matrix(rpois(b * K, 2), b, K)
    weights <- runif(b, 0.5, 2)
    weights[3] <- 0
    A <- matrix(rnorm(b * M * (M + 1L) / 2L), b, M * (M + 1L) / 2L)
    list(eta = eta, codes = codes, counts = counts, weights = weights, A = A,
         K = K, M = M, b = b)
}

empty_matrix <- function() matrix(numeric(0), 0, 0)

eval_individual <- function(cs, reference, eta = cs$eta, rows = seq_len(nrow(eta))) {
    fastglm:::vglm_family_eval("multinomial", cs$K, reference,
                               eta, cs$codes[rows], empty_matrix(),
                               cs$weights[rows], cs$A[rows, , drop = FALSE])
}

eval_grouped <- function(cs, reference, counts = cs$counts, eta = cs$eta,
                         rows = seq_len(nrow(eta))) {
    fastglm:::vglm_family_eval("multinomial", cs$K, reference,
                               eta, integer(0), counts[rows, , drop = FALSE],
                               cs$weights[rows], cs$A[rows, , drop = FALSE])
}

# Full K-category softmax with the reference linear predictor fixed at 0.
direct_softmax <- function(eta, reference) {
    K <- ncol(eta) + 1L
    full <- matrix(0, nrow(eta), K)
    full[, -(reference + 1L)] <- eta
    e <- exp(full)
    e / rowSums(e)
}

pack_upper <- function(S) S[upper.tri(S, diag = TRUE)]

test_that("score is the derivative of the row log likelihood", {
    skip_if_not_installed("numDeriv")
    cs <- vglm_family_case()
    for (reference in c(0L, 2L)) {
        ind <- eval_individual(cs, reference)
        grp <- eval_grouped(cs, reference)
        num_ind <- num_grp <- matrix(0, cs$b, cs$M)
        for (i in seq_len(cs$b)) {
            num_ind[i, ] <- numDeriv::jacobian(function(e)
                eval_individual(cs, reference, matrix(e, 1L), i)$loglik_rows,
                cs$eta[i, ])
            num_grp[i, ] <- numDeriv::jacobian(function(e)
                eval_grouped(cs, reference, eta = matrix(e, 1L), rows = i)$loglik_rows,
                cs$eta[i, ])
        }
        expect_equal(dim(ind$u), c(cs$b, cs$M))
        expect_max_diff(ind$u, num_ind, 1e-7)
        expect_max_diff(grp$u, num_grp, 1e-7)
        expect_true(ind$valid)
    }
})

test_that("information equals the expected information computed from P", {
    cs <- vglm_family_case()
    for (reference in c(0L, 2L)) {
        for (grouped in c(FALSE, TRUE)) {
            out <- if (grouped) eval_grouped(cs, reference) else eval_individual(cs, reference)
            n_i <- if (grouped) rowSums(cs$counts) else rep(1, cs$b)
            expect_equal(dim(out$W), c(cs$b, cs$M * (cs$M + 1L) / 2L))
            expected <- t(vapply(seq_len(cs$b), function(i) {
                p <- out$P[i, -(reference + 1L)]
                pack_upper(cs$weights[i] * n_i[i] * (diag(p) - tcrossprod(p)))
            }, numeric(ncol(out$W))))
            expect_max_diff(out$W, expected, 1e-12)
        }
    }
})

test_that("probabilities sum to one and match a direct softmax", {
    cs <- vglm_family_case()
    for (reference in c(0L, 2L, 4L)) {
        out <- eval_individual(cs, reference)
        expect_equal(dim(out$P), c(cs$b, cs$K))
        expect_max_diff(rowSums(out$P), rep(1, cs$b), 1e-14)
        expect_max_diff(out$P, direct_softmax(cs$eta, reference), 1e-14)
    }
})

test_that("bias terms equal half the A-weighted derivative of W", {
    skip_if_not_installed("numDeriv")
    cs <- vglm_family_case()
    Q <- cs$M * (cs$M + 1L) / 2L
    S <- matrix(2, cs$M, cs$M)
    diag(S) <- 1
    multiplicity <- pack_upper(S)  # off-diagonal pairs appear twice in sum_ab
    for (reference in c(0L, 2L)) {
        for (grouped in c(FALSE, TRUE)) {
            f <- if (grouped) eval_grouped else eval_individual
            out <- f(cs, reference)
            expected <- matrix(0, cs$b, cs$M)
            for (i in seq_len(cs$b)) {
                J <- numDeriv::jacobian(function(e)
                    as.vector(f(cs, reference, eta = matrix(e, 1L), rows = i)$W),
                    cs$eta[i, ])
                expected[i, ] <- 0.5 * colSums(multiplicity * cs$A[i, ] * J)
            }
            expect_equal(dim(out$t), c(cs$b, cs$M))
            expect_max_diff(out$t, expected, 1e-6)
        }
    }
})

test_that("individual and grouped forms of the same rows agree exactly", {
    cs <- vglm_family_case()
    counts <- outer(cs$codes, seq_len(cs$K), "==") * 1
    for (reference in c(0L, 3L)) {
        ind <- eval_individual(cs, reference)
        grp <- eval_grouped(cs, reference, counts = counts)
        expect_identical(ind$loglik_rows, grp$loglik_rows)
        expect_identical(ind$u, grp$u)
        expect_identical(ind$W, grp$W)
        expect_identical(ind$P, grp$P)
        expect_identical(ind$t, grp$t)
    }
})

test_that("zero-weight rows contribute nothing", {
    cs <- vglm_family_case()
    out <- eval_grouped(cs, 0L)
    expect_identical(out$loglik_rows[3], 0)
    expect_true(all(out$u[3, ] == 0))
    expect_true(all(out$W[3, ] == 0))
    expect_true(all(out$t[3, ] == 0))
})

test_that("linear predictors of magnitude 700 give finite outputs", {
    cs <- vglm_family_case()
    set.seed(3L)
    cs$eta <- matrix(sample(c(-700, 700), cs$b * cs$M, replace = TRUE) +
                     rnorm(cs$b * cs$M), cs$b, cs$M)
    for (reference in c(0L, 2L)) {
        for (out in list(eval_individual(cs, reference), eval_grouped(cs, reference))) {
            for (nm in c("loglik_rows", "u", "W", "P", "t"))
                expect_true(all(is.finite(out[[nm]])), info = nm)
            expect_max_diff(rowSums(out$P), rep(1, cs$b), 1e-14)
            expect_true(out$valid)
        }
    }
})

test_that("invalid inputs are rejected", {
    cs <- vglm_family_case()
    expect_error(fastglm:::vglm_family_eval("nonesuch", 3L, 0L, cs$eta[, 1:2],
                                            cs$codes, empty_matrix(), cs$weights,
                                            empty_matrix()), "nonesuch")
    expect_error(fastglm:::vglm_family_eval("multinomial", cs$K, cs$K, cs$eta,
                                            cs$codes, empty_matrix(), cs$weights,
                                            empty_matrix()), "reference")
    bad <- cs
    bad$codes[1] <- cs$K + 1L
    expect_error(eval_individual(bad, 0L), "categor")
})
