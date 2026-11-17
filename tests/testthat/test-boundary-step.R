# Boundary-aware step (inst/include/boundary_step.h): fits whose optimum lies
# on the boundary of the family's valid region converge to that optimum and
# are reported as "converged_boundary".
#
# One-way layouts give the optimum in closed form. Their weighted least
# squares update can land exactly on the bound (to rounding, on the valid
# side), in which case the full step is valid and the boundary step is not
# needed; those tests check the optimum and accept either reason. The
# reason "converged_boundary" is tested on designs that need the step.
#
# One-way layouts give the optimum in closed form. With an identity link,
# each group's fitted mean is its sample mean, so a group of zeros (or ones)
# puts the optimum on the bound, and the infimum of the deviance is the sum of
# the other groups' within-group deviances: the boundary group contributes
# 2 * mu -> 0 (Poisson, y = 0), -2 log(1 - mu) -> 0 (binomial, y = 0), or
# -2 log(mu) -> 0 (binomial, y = 1).

oneway <- function(groups) {
    g <- factor(rep(seq_along(groups), lengths(groups)))
    list(x = stats::model.matrix(~ g), y = unlist(groups), g = g)
}

group_deviance <- function(y, g, family) {
    sum(vapply(split(y, g), function(v) {
        m <- mean(v)
        if (m <= 0 || (family$family == "binomial" && m >= 1)) return(0)
        sum(family$dev.resids(v, rep(m, length(v)), rep(1, length(v))))
    }, numeric(1)))
}

fit_quietly <- function(...) {
    warned <- character()
    fit <- withCallingHandlers(fastglm(...), warning = function(w) {
        warned <<- c(warned, conditionMessage(w)); invokeRestart("muffleWarning")
    })
    fit$warned <- warned
    fit
}

test_that("Poisson identity fit with a zero group converges to the closed-form optimum", {
    set.seed(5101L)
    d <- oneway(list(rpois(40, 3), rpois(40, 6), rep(0, 40)))
    fam <- poisson("identity")
    target <- group_deviance(d$y, d$g, fam)
    for (m in c("qr", "llt", "ldlt", "svd", "auto")) {
        f <- fit_quietly(d$x, d$y, family = fam, method = m)
        expect_true(f$converged)
        expect_true(f$convergence.reason %in% c("converged", "converged_boundary"))
        expect_lt(abs(f$deviance - target) / (1 + target), 1e-7)
        means <- tapply(f$fitted.values, d$g, mean)
        expect_equal(unname(means[1:2]), unname(tapply(d$y, d$g, mean)[1:2]), tolerance = 1e-6)
        expect_lt(means[[3]], 1e-3)
    }
})

test_that("binomial identity fit with groups of zeros and ones reaches both bounds", {
    set.seed(5102L)
    d <- oneway(list(rep(0, 30), rep(1, 30), rbinom(60, 1, 0.4)))
    fam <- binomial("identity")
    target <- group_deviance(d$y, d$g, fam)
    f <- fit_quietly(d$x, d$y, family = fam)
    expect_true(f$converged)
    expect_lt(abs(f$deviance - target) / (1 + target), 1e-7)
    means <- tapply(f$fitted.values, d$g, mean)
    expect_lt(means[[1]], 1e-3)
    expect_gt(means[[2]], 1 - 1e-3)
})

test_that("log-binomial fit with a group of ones converges to the closed-form optimum", {
    set.seed(5103L)
    d <- oneway(list(rbinom(80, 1, 0.3), rep(1, 25), rbinom(80, 1, 0.6)))
    fam <- binomial("log")
    target <- group_deviance(d$y, d$g, fam)
    f <- fit_quietly(d$x, d$y, family = fam)
    expect_true(f$converged)
    expect_lt(abs(f$deviance - target) / (1 + target), 1e-7)
})

test_that("Poisson square-root fit reaches the optimum found by a convex solver", {
    # The deviance is convex in beta here (per observation, -4 y log(eta) +
    # 2 eta^2 up to constants, with second derivative 4 y / eta^2 + 4 > 0).
    # Reference: infimum 652.3875188 under X beta >= 0, computed with CVXR
    # 1.8.2 (CLARABEL) and evaluated with poisson("sqrt")$dev.resids at the
    # solver point shifted 1e-10 into the region (fastglm-jss,
    # dev/boundary). fastglm 0.1.5 stopped above it, reported as not converged.
    set.seed(20261116L); n <- 600L
    z <- matrix(runif(n * 3), n, 3)
    y <- rpois(n, pmax(0.3 + drop(z %*% c(1.5, 1.0, -0.8)), 0.05)^2)
    x <- cbind(1, z)
    for (m in c("qr", "llt", "ldlt", "svd", "auto")) {
        f <- fit_quietly(x, y, family = poisson("sqrt"), method = m)
        expect_identical(f$convergence.reason, "converged_boundary")
        expect_true(any(grepl("boundary of the valid region", f$warned)))
        expect_lt(abs(f$deviance - 652.3875188) / (1 + 652.3875188), 1e-6)
    }
    f <- fit_quietly(x, y, family = poisson("sqrt"))
    expect_lt(abs(f$deviance - 652.3875188) / (1 + 652.3875188), 1e-6)
})

test_that("streamed boundary fits equal dense fits", {
    set.seed(20261116L); n <- 600L
    z <- matrix(runif(n * 3), n, 3)
    y <- rpois(n, pmax(0.3 + drop(z %*% c(1.5, 1.0, -0.8)), 0.05)^2)
    x <- cbind(1, z)
    ix <- split(seq_len(n), cut(seq_len(n), 4, labels = FALSE))
    for (m in c("llt", "qr")) {
        df <- suppressWarnings(fastglm(x, y, family = poisson("sqrt"), method = m))
        sf <- suppressWarnings(fastglm_streaming(function(k) list(X = x[ix[[k]], , drop = FALSE], y = y[ix[[k]]]),
                                                 n_chunks = 4L, family = poisson("sqrt"), method = m))
        expect_identical(sf$convergence.reason, "converged_boundary")
        expect_identical(sf$convergence.reason, df$convergence.reason)
        expect_identical(sf$iter, df$iter)
        expect_equal(unname(coef(sf)), unname(coef(df)), tolerance = 1e-10)
    }
})

test_that("bounded-family fits with an interior optimum are reported as converged, without a warning", {
    set.seed(5105L); n <- 500L
    z <- matrix(runif(n * 2), n, 2); x <- cbind(1, z)
    y <- rpois(n, 1 + drop(z %*% c(2, 1.5)))
    expect_no_warning(f <- fastglm(x, y, family = poisson("identity")))
    expect_identical(f$convergence.reason, "converged")
})

test_that("printed fits describe convergence on the boundary", {
    set.seed(20261116L); n <- 600L
    z <- matrix(runif(n * 3), n, 3)
    y <- rpois(n, pmax(0.3 + drop(z %*% c(1.5, 1.0, -0.8)), 0.05)^2)
    f <- suppressWarnings(fastglm(cbind(1, z), y, family = poisson("sqrt")))
    expect_output(print(f), "on the boundary of the valid region")
})

test_that("streamed fits make at most maxit coefficient updates, as dense fits do", {
    set.seed(20261116L); n <- 600L
    z <- matrix(runif(n * 3), n, 3)
    y <- rpois(n, pmax(0.3 + drop(z %*% c(1.5, 1.0, -0.8)), 0.05)^2)
    x <- cbind(1, z)
    ix <- split(seq_len(n), cut(seq_len(n), 4, labels = FALSE))
    for (mx in c(3L, 5L)) {
        df <- suppressWarnings(fastglm(x, y, family = poisson("sqrt"), method = "llt", maxit = mx))
        sf <- suppressWarnings(fastglm_streaming(function(k) list(X = x[ix[[k]], , drop = FALSE], y = y[ix[[k]]]),
                                                 n_chunks = 4L, family = poisson("sqrt"), method = "llt", maxit = mx))
        expect_false(sf$converged)
        expect_identical(sf$convergence.reason, "iteration_limit")
        expect_identical(sf$iter, mx)
        expect_identical(sf$iter, df$iter)
        expect_equal(unname(coef(sf)), unname(coef(df)), tolerance = 1e-10)
    }
})

test_that("sparse designs use the decrement rule and the boundary step", {
    # Until 0.1.5 the sparse solver used the deviance-change rule only and
    # reported this fit as converged at deviance 654.32, above the optimum
    # 652.3875188 (see the convex-solver reference above).
    set.seed(20261116L); n <- 600L
    z <- matrix(runif(n * 3), n, 3)
    y <- rpois(n, pmax(0.3 + drop(z %*% c(1.5, 1.0, -0.8)), 0.05)^2)
    x <- cbind(1, z)
    dense <- suppressWarnings(fastglm(x, y, family = poisson("sqrt"), method = "llt"))
    for (m in c("llt", "ldlt")) {
        f <- suppressWarnings(fastglm(as(x, "dgCMatrix"), y, family = poisson("sqrt"), method = m))
        expect_identical(f$convergence.reason, "converged_boundary")
        expect_lt(abs(f$deviance - 652.3875188) / (1 + 652.3875188), 1e-6)
        expect_equal(unname(coef(f)), unname(coef(dense)), tolerance = 1e-8)
    }
})

test_that("bias-reduced fits are not reported as converged after a halved step", {
    # Identity-link Poisson with its optimum on the boundary mu = 0. Under the
    # identity link the mean bias-reducing adjustment (proportional to the
    # second derivative of the inverse link) is zero, so the bias-reduced
    # estimate is the maximum likelihood estimate. Bias-reduced fits do not use
    # the boundary step, so the iteration can stall short of that optimum
    # (where it stalls depends on the platform's rounding); it must then not
    # be reported as converged, as it was in 0.1.4 (deviance 913.601 against
    # the optimum 913.487).
    set.seed(5); n <- 800
    z <- matrix(runif(n * 2), n, 2); x <- cbind(1, z)
    y <- rpois(n, pmax(0.2 + 3 * z[, 1] - 2.5 * z[, 2], 0.02))
    ml <- suppressWarnings(fastglm(x, y, family = poisson("identity")))
    f <- suppressWarnings(fastglm(x, y, family = poisson("identity"), firth = TRUE, maxit = 1000L))
    if (isTRUE(f$converged)) expect_lt(f$deviance - ml$deviance, 1e-6 * (1 + ml$deviance))
    else expect_false(f$converged)
})
