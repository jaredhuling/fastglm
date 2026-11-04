test_that("partial ordinal mean-bias adjustment agrees with generic score cumulants", {
    skip_if_not_installed("numDeriv")
    x <- cbind(c(-.8, -.2, .35, .75, 1.1, -.45))
    z <- cbind(c(0, 1, 0, 1, 0, 1), c(-.4, .6, -.7, .2, .5, -.3))
    counts <- rbind(c(3, 2, 4, 1), c(1, 5, 2, 3), c(2, 1, 3, 4),
                    c(4, 3, 1, 2), c(2, 4, 1, 3), c(3, 1, 5, 2))
    w <- c(1, 2, 0, 1, 3, 2); off <- c(.1, -.2, 0, .1, -.1, .2)
    b <- c(-1.1, .2, 1.4, .25, .2, -.1, -.15, .12, .08, -.18)
    d <- length(b)
    for (link in 0:2) {
        G <- switch(link + 1L, plogis, pnorm, function(t) -expm1(-exp(t)))
        probability <- function(theta, i) {
            eta <- theta[1:3] - x[i, 1L] * theta[4L] -
                drop(matrix(theta[5:10], 3L, byrow = TRUE) %*% z[i, ]) - off[i]
            diff(c(0, G(eta), 1))
        }
        I <- matrix(0, d, d); EH <- I; U <- numeric(d)
        PQ <- array(0, c(d, d, d))
        for (i in which(w > 0)) {
            lp <- function(theta) log(probability(theta, i))
            pr <- probability(b, i); m <- sum(counts[i, ]) * w[i]
            u <- numDeriv::jacobian(lp, b)
            I <- I + m * crossprod(u, pr * u)
            U <- U + drop(crossprod(u, counts[i, ] * w[i]))
            for (k in 1:4) {
                H <- numDeriv::hessian(function(theta) lp(theta)[k], b)
                EH <- EH - m * pr[k] * H
                # Generic Firth adjustment: P_t=E[UU' U_t], Q_t=E[H U_t].
                for (t in seq_len(d)) PQ[, , t] <- PQ[, , t] +
                    m * pr[k] * (tcrossprod(u[k, ]) + H) * u[k, t]
            }
        }
        V <- solve(I)
        A <- vapply(seq_len(d), function(t) sum(V * t(PQ[, , t])) / 2, numeric(1))
        got <- fastglm:::ordinal_partial_native_evaluate(x, z, integer(), counts,
            w, off, 4L, b, TRUE, block_size = 2L, link = link)
        expect_equal(got$information, I, tolerance = 2e-6)
        expect_equal(got$information, EH, tolerance = 2e-6)
        expect_equal(drop(got$ordinary.score), U, tolerance = 2e-6)
        expect_equal(drop(got$adjustment), A, tolerance = 2e-5)
        expect_equal(drop(got$score), U + A, tolerance = 2e-5)
    }
})

test_that("partial ordinal adjustment respects frequency and linear recoding semantics", {
    x <- cbind(c(-.8, -.2, .35, .75, 1.1, -.45))
    z <- cbind(c(0, 1, 0, 1, 0, 1))
    counts <- rbind(c(3, 2, 4), c(1, 5, 2), c(2, 1, 3),
                    c(4, 3, 1), c(2, 4, 1), c(3, 1, 5))
    w <- c(1, 2, 0, 1, 3, 2); off <- c(.1, -.2, 0, .1, -.1, .2)
    b <- c(-.8, 1.1, .25, .3, -.15)
    T <- diag(c(1, 1, 1 / 1.7, 1 / .8, 1 / .8))
    T[1:2, 3L] <- .4 / 1.7; T[1L, 4L] <- -.3 / .8; T[2L, 5L] <- -.3 / .8
    inv <- solve(T)
    rows <- rep(seq_len(nrow(x)), rowSums(counts * w))
    y <- unlist(lapply(seq_len(nrow(x)), function(i) rep(1:3, counts[i, ] * w[i])))
    for (link in 0:2) {
        ev <- function(x, z, counts, w, off, b) fastglm:::ordinal_partial_native_evaluate(
            x, z, integer(), counts, w, off, ncol(counts), b, TRUE, link = link)
        a <- ev(x, z, counts, w, off, b)
        changed <- ev(x * 1.7 + .4, z * .8 - .3, counts, w, off, drop(T %*% b))
        expect_equal(drop(changed$score), drop(t(inv) %*% a$score), tolerance = 2e-9)
        expect_equal(drop(changed$adjustment), drop(t(inv) %*% a$adjustment), tolerance = 2e-9)
        expect_equal(changed$information, t(inv) %*% a$information %*% inv, tolerance = 2e-9)
        more <- ev(x, z, counts, 3 * w, off, b)
        expect_equal(more$adjustment, a$adjustment, tolerance = 2e-9)
        expect_equal(more$ordinary.score, 3 * a$ordinary.score, tolerance = 2e-9)
        expect_equal(more$information, 3 * a$information, tolerance = 2e-9)
        expanded <- fastglm:::ordinal_partial_native_evaluate(x[rows, , drop = FALSE],
            z[rows, , drop = FALSE], as.integer(y), matrix(numeric(), 0, 0),
            rep(1, length(rows)), off[rows], 3L, b, TRUE, block_size = 7L, link = link)
        expect_equal(expanded$score, a$score, tolerance = 2e-9)
        expect_equal(expanded$adjustment, a$adjustment, tolerance = 2e-9)
        z[3L, 1L] <- -1e5
        zero <- ev(x, z, counts, w, off, b)
        expect_equal(zero$score, a$score, tolerance = 2e-9)
        expect_equal(zero$information, a$information, tolerance = 2e-9)
        expect_error(ev(x, z, counts, replace(w, 3L, 1), off, b), "probabilit|order|cross|gap")
        z[3L, 1L] <- 0
    }
})

test_that("partial ordinal equations reduce to common and binary ordinal equations", {
    x <- cbind(c(-.8, -.2, .35, .75, 1.1, -.45))
    z <- cbind(c(0, 1, 0, 1, 0, 1))
    counts <- rbind(c(3, 2, 4), c(1, 5, 2), c(2, 1, 3),
                    c(4, 3, 1), c(2, 4, 1), c(3, 1, 5))
    w <- c(1, 2, 0, 1, 3, 2); off <- c(.1, -.2, 0, .1, -.1, .2)
    for (link in 0:2) for (binary in c(FALSE, TRUE)) {
        if (binary) {
            cc <- cbind(counts[, 1L], rowSums(counts[, 2:3])); b <- c(-.8, .25, .3)
            zz <- z; xx <- cbind(x, z)
        } else {
            cc <- counts; b <- c(-.8, 1.1, .25)
            zz <- matrix(numeric(), nrow(x), 0L); xx <- x
        }
        partial <- fastglm:::ordinal_partial_native_evaluate(x, zz, integer(), cc,
            w, off, ncol(cc), b, TRUE, link = link)
        ordinary <- fastglm:::ordinal_native_evaluate(xx, integer(), cc, w, off,
            ncol(cc), b, TRUE, link = link)
        for (field in c("score", "ordinary.score", "information", "adjustment", "cov.unscaled"))
            expect_equal(partial[[field]], ordinary[[field]], tolerance = 2e-9)
    }
})
