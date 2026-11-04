# Independent expected-cumulant calculation. This deliberately differentiates
# category log probabilities, rather than using adjusted cumulative counts.
partial_probability <- function(theta, x, z, offset, categories, link) {
    q <- categories - 1L; pc <- length(x); pn <- length(z)
    eta <- theta[seq_len(q)] - offset
    if (pc) eta <- eta - sum(x * theta[q + seq_len(pc)])
    if (pn) eta <- eta - drop(matrix(theta[q + pc + seq_len(q * pn)],
                                      nrow = q, byrow = TRUE) %*% z)
    cumulative <- switch(link, logit = plogis(eta), probit = pnorm(eta),
                         cloglog = -expm1(-exp(eta)))
    diff(c(0, cumulative, 1))
}

partial_cumulant_reference <- function(theta, x, z, counts, weights, offset,
                                       link = "logit", map = diag(length(theta))) {
    d <- length(theta); information <- matrix(0, d, d)
    expected_hessian <- matrix(0, d, d); score <- numeric(d)
    cumulants <- array(0, c(d, d, d)); loglik <- 0
    for (i in seq_len(nrow(x))) {
        total <- weights[i] * sum(counts[i, ])
        if (total == 0) next
        lp <- function(b) log(partial_probability(drop(map %*% b), x[i, ], z[i, ],
                                                 offset[i], ncol(counts), link))
        probabilities <- exp(lp(theta))
        stopifnot(all(is.finite(probabilities)), all(probabilities > 0))
        u <- numDeriv::jacobian(lp, theta)
        information <- information + total * crossprod(u, probabilities * u)
        score <- score + drop(crossprod(u, counts[i, ] * weights[i]))
        loglik <- loglik + sum(counts[i, ] * weights[i] * lp(theta))
        for (k in seq_len(ncol(counts))) {
            H <- numDeriv::hessian(function(b) lp(b)[k], theta)
            expected_hessian <- expected_hessian - total * probabilities[k] * H
            # Q_t = E[H U_t], with H the Hessian of the log likelihood.
            joint <- tcrossprod(u[k, ]) + H
            for (t in seq_len(d)) cumulants[, , t] <- cumulants[, , t] +
                total * probabilities[k] * joint * u[k, t]
        }
    }
    covariance <- solve(information)
    adjustment <- vapply(seq_len(d), function(t)
        sum(covariance * t(cumulants[, , t])) / 2, numeric(1))
    list(loglik = loglik, ordinary.score = score, information = information,
         expected.hessian = expected_hessian, adjustment = adjustment,
         score = score + adjustment, cov.unscaled = covariance)
}

partial_confirmation_fixture <- function(wide = FALSE) {
    x <- cbind(common = c(-.85, -.15, .45, .8, 1.05, -.5, .2, .65))
    z <- cbind(varying = c(0, 1, 0, 1, 0, 1, 0, 1))
    counts <- rbind(c(3, 2, 4), c(1, 5, 2), c(2, 1, 3), c(4, 3, 1),
                    c(2, 4, 1), c(3, 1, 5), c(4, 2, 1), c(1, 3, 4))
    theta <- c(-.8, 1.1, .25, .3, -.15)
    if (wide) {
        z <- cbind(z, c(-.4, .7, -.6, .2, .5, -.3, .8, -.2))
        counts <- cbind(counts, c(2, 1, 3, 1, 4, 2, 3, 2))
        theta <- c(-1.15, .2, 1.45, .25, .2, -.1, -.15, .12, .08, -.18)
    }
    list(x = x, z = z, counts = counts, weights = c(1, 2, 0, 1, 3, 2, 1, 1),
         offset = c(.12, -.18, .04, .09, -.13, .2, -.08, .03), theta = theta)
}
