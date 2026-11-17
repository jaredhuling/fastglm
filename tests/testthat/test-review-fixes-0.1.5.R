test_that("a partial proportional odds maximum with a shared touching constraint is certified", {
    # A probit design in which no observation with z = 1 falls in the third
    # category, so at the maximum the second and third cumulative
    # probabilities coincide for every z = 1 row (one constraint shared by all
    # of them). fastglm 0.1.5 before this fix reached the maximum but stopped
    # at the iteration limit.
    set.seed(1985040346); n <- 250
    X <- matrix(rnorm(n * 2), n, 2, dimnames = list(NULL, c("x1", "x2"))); z <- rbinom(n, 1, 0.3)
    E <- outer(rep(1, n), c(-1, 0, 1)) - drop(X %*% c(0.4, -0.4)) - outer(z, c(0, 0, 0.9))
    y <- 1L + rowSums(runif(n) > pnorm(E))
    expect_false(any(y[z == 1] == 3))
    fit <- suppressWarnings(fastglm_ordinal(cbind(X, z = z), ordered(y, levels = 1:4),
                                            link = "probit", nonparallel = "z"))
    expect_true(fit$converged)
    expect_identical(fit$convergence.reason, "converged_boundary")
    expect_equal(as.numeric(logLik(fit)), -281.958498079597, tolerance = 1e-10)
})

two_part_data <- function() {
    set.seed(11); n <- 400
    d <- data.frame(x = rnorm(n), exposure = runif(n, 0.2, 1))
    p <- 1 - exp(-d$exposure * exp(-1 + 0.5 * d$x))
    d$y <- ifelse(runif(n) < p, rgamma(n, 2, 2 / exp(1 + 0.3 * d$x)), 0)
    d
}

test_that("count prediction rejects an offset that evaluates to the fitting rows", {
    d <- two_part_data()
    nd <- data.frame(x = c(-1, 0, 1), exposure = 1)
    good <- fastglm_twopart(y ~ x, data = d, offset = cbind(0, log(exposure)), link = "cloglog")
    expect_length(predict(good, newdata = nd), 3L)
    bad <- fastglm_twopart(y ~ x, data = d, offset = cbind(0, log(d$exposure)), link = "cloglog")
    expect_error(predict(bad, newdata = nd), "offset")
})

test_that("predict type 'binary' is the probability of a positive outcome", {
    d <- two_part_data()
    fit <- fastglm_twopart(y ~ x, data = d)
    nd <- data.frame(x = c(-1, 0, 1))
    pos <- predict(fit, newdata = nd, type = "binary")
    expect_equal(pos, 1 - predict(fit, newdata = nd, type = "zero"))
    expect_equal(unname(pos), unname(plogis(cbind(1, nd$x) %*% coef(fit, model = "binary"))[, 1]))
    se <- predict(fit, newdata = nd, type = "binary", se.fit = TRUE)
    expect_equal(se$fit, pos)
    expect_true(all(se$se.fit > 0))
})

test_that("diagnostics cover truncated, hurdle, zero-inflated, and two-part fits", {
    set.seed(3); n <- 500
    d <- data.frame(x = rnorm(n), w = rnorm(n))
    d$y <- ifelse(runif(n) < plogis(0.5 + 0.5 * d$w), MASS::rnegbin(n, exp(0.6 + 0.4 * d$x), 2), 0)
    fits <- list(
        truncated = fastglm_truncated(y ~ x, data = subset(d, y > 0), dist = "negbin"),
        hurdle = fastglm_hurdle(y ~ x | w, data = d, dist = "negbin"),
        zi = fastglm_zi(y ~ x | w, data = d, dist = "negbin"),
        twopart = fastglm_twopart(y ~ x | w, data = transform(d, y = y * 1.5)))
    for (nm in names(fits)) {
        dg <- fastglm_diagnostics(fits[[nm]])
        expect_s3_class(dg, "fastglm_diagnostics")
        expect_true(dg$ordinary.score.available, label = nm)
        expect_lt(dg$standardized.score, 1e-3)
        expect_true(is.finite(dg$covariance.condition))
        expect_output(print(dg), "Standardized ordinary score")
    }
})

test_that("sources print a summary rather than their internals", {
    d <- data.frame(y = rbinom(20, 1, 0.5), x = rnorm(20), f = factor(rep(c("a", "b"), 10)))
    src <- fastglm_source(function(k) d[seq(k, 20, by = 2), ], 2L, formula = y ~ x + f, prototype = d)
    out <- capture.output(print(src))
    expect_true(any(grepl("Blocks: 2", out)))
    expect_true(any(grepl("Design columns: 3", out)))
    expect_false(any(grepl("function", out)))
})

test_that("formula fits keep the argument order in the printed call", {
    fit <- fastglm(mpg ~ wt, data = mtcars)
    expect_identical(names(as.list(fit$call))[2:3], c("formula", "data"))
    expect_equal(coef(update(fit, . ~ . + hp)), coef(glm(mpg ~ wt + hp, data = mtcars)))
})

test_that("tidy and glance support truncated, hurdle, zero-inflated, and two-part fits", {
    skip_if_not_installed("broom")
    set.seed(4); n <- 400
    d <- data.frame(x = rnorm(n), w = rnorm(n))
    d$y <- ifelse(runif(n) < plogis(0.5 + 0.5 * d$w), rpois(n, exp(0.6 + 0.4 * d$x)), 0)
    h <- fastglm_hurdle(y ~ x | w, data = d)
    td <- broom::tidy(h, conf.int = TRUE)
    expect_identical(td$component, c("count", "count", "zero", "zero"))
    expect_equal(td$estimate, unname(coef(h)))
    expect_true(all(td$conf.low < td$estimate & td$estimate < td$conf.high))
    gl <- broom::glance(h)
    expect_equal(gl$logLik, as.numeric(logLik(h)))
    tr <- broom::tidy(fastglm_truncated(y ~ x, data = subset(d, y > 0)))
    expect_false("component" %in% names(tr))
    expect_identical(nrow(broom::tidy(fastglm_zi(y ~ x | w, data = d))), 4L)
})
