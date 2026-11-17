test_that("partial proportional odds models support the refitting bootstrap", {
    set.seed(31); n <- 400
    d <- data.frame(x1 = rnorm(n), z = rnorm(n))
    eta <- 0.5 * d$x1; a <- c(-1, 0, 1); g <- c(0, 0.3, 0.6); u <- runif(n)
    d$y <- ordered(1 + (u > plogis(a[1] - eta - g[1] * d$z)) + (u > plogis(a[2] - eta - g[2] * d$z)) +
                   (u > plogis(a[3] - eta - g[3] * d$z)), levels = 1:4)
    fit <- suppressWarnings(fastglm_ordinal(y ~ x1 + z, data = d, nonparallel = ~ z))
    boot <- fastglm_bootstrap(fit, R = 20, seed = 3)
    # Each replicate is the partial fit to that resample.
    set.seed(3); seeds <- sample.int(.Machine$integer.max, 20L)
    direct <- t(sapply(seeds, function(s) {
        set.seed(s); i <- sample.int(n, n, TRUE)
        coef(suppressWarnings(fastglm_ordinal(y ~ x1 + z, data = d[i, ], nonparallel = ~ z)))
    }))
    expect_equal(unname(boot$replicates), unname(direct), tolerance = 1e-10)
    expect_true(all(is.finite(boot$std.error)))
    expect_length(boot$estimate, length(coef(fit)))
})

glm_inference_cases <- function() {
    set.seed(32); n <- 300
    d <- data.frame(x = rnorm(n), w = rnorm(n), f = factor(sample(letters[1:3], n, TRUE)))
    list(d = d, cases = list(
        binomial = list(y = rbinom(n, 1, plogis(-0.3 + 0.8 * d$x)), family = binomial()),
        poisson = list(y = rpois(n, exp(0.2 + 0.4 * d$x)), family = poisson()),
        gamma = list(y = rgamma(n, 2, 2 / exp(0.5 + 0.3 * d$x)), family = Gamma("log")),
        gaussian = list(y = 1 + 0.5 * d$x + rnorm(n), family = gaussian())))
}

test_that("GLM profile intervals solve the profile likelihood equation exactly", {
    z <- glm_inference_cases(); d <- z$d
    for (nm in names(z$cases)) {
        d$y <- z$cases[[nm]]$y; fam <- z$cases[[nm]]$family
        fit <- fastglm(y ~ x + w + f, data = d, family = fam)
        ref <- glm(y ~ x + w + f, data = d, family = fam, control = glm.control(epsilon = 1e-14, maxit = 200))
        ci <- unclass(confint(fit, method = "profile"))[, 1:2]
        X <- model.matrix(ref); phi <- summary(ref)$dispersion; D <- deviance(ref)
        for (j in c(2L, 4L)) {
            signed <- function(v) sign(v - coef(ref)[j]) * sqrt(max(0, glm.fit(X[, -j], d$y, family = fam,
                offset = X[, j] * v, control = glm.control(epsilon = 1e-14, maxit = 200))$deviance - D) / phi)
            expect_equal(unname(signed(ci[j, 2])), qnorm(0.975), tolerance = 1e-6, label = paste(nm, j, "upper"))
            expect_equal(unname(signed(ci[j, 1])), -qnorm(0.975), tolerance = 1e-6, label = paste(nm, j, "lower"))
        }
        # Agreement with stats::confint() up to its spline interpolation.
        expect_equal(ci, suppressMessages(confint(ref)), tolerance = 1e-3, ignore_attr = TRUE)
    }
})

test_that("GLM likelihood-ratio and score tests match anova()", {
    z <- glm_inference_cases(); d <- z$d
    for (nm in names(z$cases)) {
        d$y <- z$cases[[nm]]$y; fam <- z$cases[[nm]]$family
        fit <- fastglm(y ~ x + w + f, data = d, family = fam)
        ctl <- glm.control(epsilon = 1e-14, maxit = 200)
        big <- glm(y ~ x + w + f, data = d, family = fam, control = ctl)
        small <- glm(y ~ x, data = d, family = fam, control = ctl)
        estimated <- nm %in% c("gamma", "gaussian")
        lr <- fastglm_profile_test(fit, parm = c("w", "fb", "fc"))
        av <- anova(small, big, test = if (estimated) "F" else "Chisq")
        expect_equal(lr$p.value, av[2, ncol(av)], tolerance = 1e-5, label = nm)
        sc <- fastglm_score_test(fit, terms = c("w", "f"))
        rao <- anova(small, big, test = "Rao")
        expect_equal(sc$p.value, rao[2, "Pr(>Chi)"], tolerance = 1e-5, label = nm)
    }
    # Matrix fits give the same intervals as formula fits.
    d$y <- z$cases$poisson$y
    ff <- fastglm(y ~ x + w, data = d, family = poisson())
    fm <- fastglm(model.matrix(~ x + w, d), d$y, family = poisson())
    expect_equal(unclass(confint(fm, method = "profile"))[, 1:2],
                 unclass(confint(ff, method = "profile"))[, 1:2], tolerance = 1e-10)
})

zi_case <- function(seed, n = 1500, zip_truth = FALSE) {
    set.seed(seed); x <- rnorm(n); w <- rnorm(n)
    X <- cbind("(Intercept)" = 1, x = x); Z <- cbind("(Intercept)" = 1, w = w)
    mu <- exp(0.5 + 0.4 * x)
    count <- if (zip_truth) rpois(n, mu) else MASS::rnegbin(n, mu, 1.2)
    list(X = X, Z = Z, y = ifelse(runif(n) < plogis(-0.6 + 0.6 * w), 0, count), n = n)
}

test_that("blockwise zero-inflated fits reproduce the dense fit", {
    for (seed in c(41, 42)) for (dist in c("negbin", "poisson")) {
        z <- zi_case(seed)
        dense <- fastglm_zi(z$X, response = z$y, z = z$Z, dist = dist)
        sparse <- fastglm_zi(as(z$X, "dgCMatrix"), response = z$y, z = as(z$Z, "dgCMatrix"), dist = dist)
        src <- fastglm_source(function(k) {
            i <- seq.int((k - 1L) * 400L + 1L, min(z$n, k * 400L)); list(X = z$X[i, ], Z = z$Z[i, ], y = z$y[i])
        }, ceiling(z$n / 400L))
        streamed <- fastglm_zi(src, dist = dist)
        for (fit in list(sparse, streamed)) {
            expect_true(fit$converged)
            expect_equal(coef(fit), coef(dense), tolerance = 1e-6)
            expect_equal(fit$loglik, dense$loglik, tolerance = 1e-9)
            if (dist == "negbin") expect_equal(fit$theta, dense$theta, tolerance = 1e-6)
        }
        expect_equal(sparse$vcov.full, dense$vcov.full, tolerance = 1e-6)
        expect_equal(predict(sparse), predict(dense), tolerance = 1e-6)
        if (requireNamespace("bigmemory", quietly = TRUE))
            expect_equal(coef(fastglm_zi(bigmemory::as.big.matrix(z$X), response = z$y,
                z = bigmemory::as.big.matrix(z$Z), dist = dist)), coef(dense), tolerance = 1e-6)
    }
})

test_that("a blockwise zero-inflated NB fit reaches the Poisson limit as the dense fit does", {
    z <- zi_case(7, zip_truth = TRUE)
    dense <- suppressWarnings(fastglm_zi(z$X, response = z$y, z = z$Z, dist = "negbin"))
    sparse <- suppressWarnings(fastglm_zi(as(z$X, "dgCMatrix"), response = z$y, z = as(z$Z, "dgCMatrix"),
                                          dist = "negbin"))
    expect_identical(dense$convergence.reason, "theta_boundary")
    expect_identical(sparse$convergence.reason, "theta_boundary")
    expect_equal(coef(sparse), coef(dense), tolerance = 1e-6)
    expect_equal(sparse$loglik, dense$loglik, tolerance = 1e-9)
})

test_that("sparse hurdle and two-part fits predict", {
    z <- zi_case(43, n = 800)
    hs <- fastglm_hurdle(as(z$X, "dgCMatrix"), response = z$y, z = as(z$Z, "dgCMatrix"))
    hd <- fastglm_hurdle(z$X, response = z$y, z = z$Z)
    expect_equal(predict(hs), predict(hd), tolerance = 1e-10)
    expect_equal(residuals(hs), residuals(hd), tolerance = 1e-10)
    pos <- ifelse(z$y > 0, z$y + 0.5, 0)
    ts <- fastglm_twopart(as(z$X, "dgCMatrix"), response = pos, z = as(z$Z, "dgCMatrix"))
    expect_equal(predict(ts), predict(fastglm_twopart(z$X, response = pos, z = z$Z)), tolerance = 1e-10)
})
