# Likelihood ratio anova(m0, m1, ...) across the fitted classes.

anova_data <- function(n = 500L, seed = 20261121L) {
    set.seed(seed)
    d <- data.frame(a = rnorm(n), c = rnorm(n), b = factor(sample(c("p", "q", "r"), n, TRUE)))
    eta <- 0.3 + 0.4 * d$a + c(0, 0.3, -0.2)[d$b]
    d$yp <- rpois(n, exp(eta))
    d$ynb <- rnbinom(n, mu = exp(eta), size = 1.5)
    d$yzi <- ifelse(runif(n) < plogis(-1 + 0.6 * d$a), 0, rpois(n, exp(eta)))
    d$yt <- d$ynb + 1
    d$yg <- rgamma(n, shape = 2, rate = 2 / exp(0.2 + 0.3 * d$a))
    d$yn <- 1 + 0.5 * d$a + 0.1 * d$c + rnorm(n)
    d$yo <- factor(cut(eta + rlogis(n), c(-Inf, -0.5, 0.5, 1.5, Inf), labels = c("l1", "l2", "l3", "l4")),
                   ordered = TRUE)
    d$ym <- factor(c("u", "v", "w")[apply(cbind(0, 0.5 * d$a, -0.3 * d$a + 0.2) +
                                          matrix(-log(-log(runif(3 * n))), n, 3), 1, which.max)])
    d$yb <- rbinom(n, 1, plogis(-0.5 + d$a))
    d$w <- runif(n, 0.5, 2)
    d
}

anova_source <- function(d, formula, n_chunks = 4L, ...) {
    idx <- split(seq_len(nrow(d)), rep(seq_len(n_chunks), length.out = nrow(d)))
    fastglm_source(function(k) d[idx[[k]], , drop = FALSE], n_chunks,
                   formula = formula, prototype = d, ...)
}

# The last row's statistic and p-value of an anova table, by column position
# of the statistic.
last_row <- function(tab, stat, p) unname(unlist(tab[nrow(tab), c(stat, p)]))

test_that("GLM comparisons agree with anova.glm (matrix, streamed, big.matrix)", {
    d <- anova_data()
    X0 <- model.matrix(~ a, d); X1 <- model.matrix(~ a + b + c, d)
    g0 <- glm(yp ~ a, poisson, d); g1 <- glm(yp ~ a + b + c, poisson, d)
    ref <- anova(g0, g1, test = "Chisq")
    m0 <- fastglm(X0, d$yp, family = poisson()); m1 <- fastglm(X1, d$yp, family = poisson())
    tab <- anova(m0, m1)
    expect_s3_class(tab, "anova")
    expect_named(tab, names(ref))
    expect_equal(unname(as.matrix(tab)), unname(as.matrix(ref)), tolerance = 1e-8)
    # Order of the arguments does not matter: models are sorted by size.
    expect_equal(unname(as.matrix(anova(m1, m0))), unname(as.matrix(ref)), tolerance = 1e-8)
    # Streamed fits of the same data.
    s0 <- fastglm_streaming(anova_source(d, yp ~ a), family = poisson())
    s1 <- fastglm_streaming(anova_source(d, yp ~ a + b + c), family = poisson())
    expect_warning(tab <- anova(s0, s1), "names only")
    expect_equal(unname(as.matrix(tab)), unname(as.matrix(ref)), tolerance = 1e-8)
    # A streamed fit against a dense one.
    expect_equal(unname(as.matrix(suppressWarnings(anova(m0, s1)))), unname(as.matrix(ref)), tolerance = 1e-8)
    # Three models, weights.
    gw <- lapply(list(~ a, ~ a + b, ~ a + b + c), function(f)
        glm(update(f, yp ~ .), poisson, d, weights = w))
    mw <- lapply(list(~ a, ~ a + b, ~ a + b + c), function(f)
        fastglm(model.matrix(f, d), d$yp, weights = d$w, family = poisson()))
    expect_equal(unname(as.matrix(anova(mw[[1]], mw[[3]], mw[[2]]))),
                 unname(as.matrix(anova(gw[[1]], gw[[2]], gw[[3]], test = "Chisq"))), tolerance = 1e-8)
    skip_if_not_installed("bigmemory")
    b1 <- fastglm(bigmemory::as.big.matrix(X1), d$yp, family = poisson(), method = 2L)
    expect_warning(tab <- anova(m0, b1), "names only")
    expect_equal(unname(as.matrix(tab)), unname(as.matrix(ref)), tolerance = 1e-8)
})

test_that("Gaussian and Gamma fits use the F test of anova.glm", {
    d <- anova_data()
    X0 <- model.matrix(~ a, d); X1 <- model.matrix(~ a + b + c, d)
    for (fam in list(gaussian(), Gamma("log"))) {
        y <- if (fam$family == "gaussian") d$yn else d$yg
        g0 <- glm(y ~ a, fam, d); g1 <- glm(y ~ a + b + c, fam, d)
        m0 <- fastglm(X0, y, family = fam); m1 <- fastglm(X1, y, family = fam)
        refF <- anova(g0, g1, test = "F")
        tab <- anova(m0, m1, test = "F")
        expect_named(tab, names(refF))
        expect_equal(unname(as.matrix(tab)), unname(as.matrix(refF)), tolerance = 1e-8, info = fam$family)
        # The default for a moment dispersion is F, as in anova.glm.
        expect_equal(unname(as.matrix(anova(m0, m1))), unname(as.matrix(anova(g0, g1))), tolerance = 1e-8)
        refC <- anova(g0, g1, test = "Chisq")
        expect_equal(unname(as.matrix(anova(m0, m1, test = "Chisq"))), unname(as.matrix(refC)),
                     tolerance = 1e-8, info = fam$family)
        expect_equal(unname(as.matrix(anova(m0, m1, test = "LRT"))), unname(as.matrix(refC)),
                     tolerance = 1e-8, info = fam$family)
    }
    # Streamed Gamma fits with the F test.
    d$y <- d$yg
    s0 <- fastglm_streaming(anova_source(d, y ~ a), family = Gamma("log"))
    s1 <- fastglm_streaming(anova_source(d, y ~ a + b + c), family = Gamma("log"))
    ref <- anova(glm(yg ~ a, Gamma("log"), d), glm(yg ~ a + b + c, Gamma("log"), d), test = "F")
    expect_warning(tab <- anova(s0, s1, test = "F"), "names only")
    expect_equal(unname(as.matrix(tab)), unname(as.matrix(ref)), tolerance = 1e-8)
    # The F test is refused for other classes.
    expect_error(anova(fastglm_nb(ynb ~ a, data = d), fastglm_nb(ynb ~ a + b, data = d), test = "F"),
                 "F test")
})

test_that("negative binomial comparisons agree with MASS::anova.negbin", {
    skip_if_not_installed("MASS")
    d <- anova_data()
    r0 <- MASS::glm.nb(ynb ~ a, data = d); r1 <- MASS::glm.nb(ynb ~ a + b + c, data = d)
    ref <- anova(r0, r1)
    f0 <- fastglm_nb(ynb ~ a, data = d); f1 <- fastglm_nb(ynb ~ a + b + c, data = d)
    tab <- anova(f0, f1)
    expect_named(tab, c("#Df", "LogLik", "Df", "Chisq", "Pr(>Chisq)"))
    expect_equal(last_row(tab, "Chisq", "Pr(>Chisq)"), last_row(ref, "LR stat.", "Pr(Chi)"), tolerance = 1e-8)
    expect_identical(tab$Df[2], 3)
    # Matrix-interface NB fits.
    m0 <- fastglm_nb(model.matrix(~ a, d), d$ynb); m1 <- fastglm_nb(model.matrix(~ a + b + c, d), d$ynb)
    expect_equal(last_row(anova(m0, m1), "Chisq", "Pr(>Chisq)"), last_row(ref, "LR stat.", "Pr(Chi)"),
                 tolerance = 1e-8)
})

test_that("ordinal comparisons agree with anova.polr", {
    skip_if_not_installed("MASS")
    d <- anova_data()
    r0 <- MASS::polr(yo ~ a, data = d, Hess = TRUE); r1 <- MASS::polr(yo ~ a + b + c, data = d, Hess = TRUE)
    ref <- anova(r0, r1)
    f0 <- fastglm_ordinal(yo ~ a, data = d); f1 <- fastglm_ordinal(yo ~ a + b + c, data = d)
    tab <- anova(f0, f1)
    expect_equal(last_row(tab, "Chisq", "Pr(>Chisq)"), last_row(ref, "LR stat.", "Pr(Chi)"), tolerance = 1e-8)
    expect_identical(tab$Df[2], 3)
    # Proportional odds is nested in partial proportional odds.
    p1 <- fastglm_ordinal(yo ~ a + b + c, data = d, nonparallel = ~ a)
    tp <- anova(f1, p1)
    expect_identical(tp$Df[2], 2)
    expect_equal(tp$Chisq[2], 2 * (as.numeric(logLik(p1)) - as.numeric(logLik(f1))), tolerance = 1e-10)
    # Streamed ordinal fits against dense ones.
    s1 <- fastglm_ordinal_streaming(anova_source(d, yo ~ a + b + c))
    expect_equal(last_row(suppressWarnings(anova(f0, s1)), "Chisq", "Pr(>Chisq)"), last_row(ref, "LR stat.", "Pr(Chi)"),
                 tolerance = 1e-8)
})

test_that("multinomial comparisons agree with nnet's anova.multinom", {
    skip_if_not_installed("nnet")
    d <- anova_data()
    r0 <- nnet::multinom(ym ~ a, data = d, trace = FALSE, reltol = 1e-14, maxit = 1000)
    r1 <- nnet::multinom(ym ~ a + b + c, data = d, trace = FALSE, reltol = 1e-14, maxit = 1000)
    ref <- anova(r0, r1)
    f0 <- fastglm_multinom(ym ~ a, data = d); f1 <- fastglm_multinom(ym ~ a + b + c, data = d)
    tab <- anova(f0, f1)
    expect_equal(last_row(tab, "Chisq", "Pr(>Chisq)"), last_row(ref, "LR stat.", "Pr(Chi)"), tolerance = 1e-8)
    expect_identical(tab$Df[2], 6)
})

test_that("zero-inflated and hurdle comparisons agree with pscl", {
    skip_if_not_installed("pscl")
    d <- anova_data()
    ctl <- pscl::zeroinfl.control(reltol = 1e-14, maxit = 10000)
    z0 <- pscl::zeroinfl(yzi ~ a | a, data = d, control = ctl)
    z1 <- pscl::zeroinfl(yzi ~ a + b + c | a + c, data = d, control = ctl)
    f0 <- fastglm_zi(yzi ~ a | a, data = d); f1 <- fastglm_zi(yzi ~ a + b + c | a + c, data = d)
    tab <- anova(f0, f1)
    stat <- 2 * (as.numeric(logLik(z1)) - as.numeric(logLik(z0)))
    expect_equal(tab$Chisq[2], stat, tolerance = 1e-8)
    expect_equal(tab$`Pr(>Chisq)`[2], pchisq(stat, 4, lower.tail = FALSE), tolerance = 1e-8)
    expect_identical(tab$Df[2], 4)
    hctl <- pscl::hurdle.control(reltol = 1e-14, maxit = 10000)
    for (dist in c("poisson", "negbin")) {
        h0 <- pscl::hurdle(ynb ~ a | a, data = d, dist = dist, control = hctl)
        h1 <- pscl::hurdle(ynb ~ a + b + c | a + c, data = d, dist = dist, control = hctl)
        k0 <- fastglm_hurdle(ynb ~ a | a, data = d, dist = dist)
        k1 <- fastglm_hurdle(ynb ~ a + b + c | a + c, data = d, dist = dist)
        th <- anova(k0, k1)
        stat <- 2 * (as.numeric(logLik(h1)) - as.numeric(logLik(h0)))
        expect_equal(th$Chisq[2], stat, tolerance = 1e-8, info = dist)
        expect_equal(th$`Pr(>Chisq)`[2], pchisq(stat, 4, lower.tail = FALSE), tolerance = 1e-8, info = dist)
    }
    # A zero part not nested: error.
    expect_error(anova(fastglm_zi(yzi ~ a | c, data = d), fastglm_zi(yzi ~ a + b | a, data = d)),
                 "not nested")
})

test_that("Firth fits use the penalized likelihood ratio of logistf", {
    skip_if_not_installed("logistf")
    d <- anova_data()
    f0 <- fastglm(yb ~ a, data = d, family = binomial(), firth = TRUE)
    f1 <- fastglm(yb ~ a + b, data = d, family = binomial(), firth = TRUE)
    l0 <- logistf::logistf(yb ~ a, data = d); l1 <- logistf::logistf(yb ~ a + b, data = d)
    nested <- anova(l1, l0, method = "nested")
    tab <- anova(f0, f1)
    expect_equal(tab$Chisq[2], unname(nested$chisq), tolerance = 1e-8)
    expect_equal(tab$Chisq[2], unname(fastglm_profile_test(f1, parm = c("bq", "br"))$statistic))
    expect_equal(tab$`Pr(>Chisq)`[2], pchisq(tab$Chisq[2], 2, lower.tail = FALSE))
    expect_true(any(grepl("penalized likelihood ratio", tolower(attr(tab, "heading")))))
    expect_output(print(tab), "Penalized likelihood ratio")
    # Matrix-interface Firth fits with names give the same statistic.
    m0 <- fastglm(model.matrix(~ a, d), d$yb, family = binomial(), firth = TRUE)
    m1 <- fastglm(model.matrix(~ a + b, d), d$yb, family = binomial(), firth = TRUE)
    expect_equal(anova(m1, m0)$Chisq[2], tab$Chisq[2], tolerance = 1e-8)
    # Firth against maximum likelihood: error, from either side.
    ml1 <- fastglm(yb ~ a + b, data = d, family = binomial())
    expect_error(anova(f0, ml1), "penalized and an unpenalized")
    expect_error(anova(ml1, f0), "penalized and an unpenalized")
    # Bias reduction without a penalized likelihood comparison: error naming
    # the alternatives.
    p0 <- fastglm(yb ~ a, data = d, family = binomial("probit"), firth = TRUE)
    p1 <- fastglm(yb ~ a + b, data = d, family = binomial("probit"), firth = TRUE)
    expect_error(anova(p0, p1), "fastglm_wald_test")
    s1 <- fastglm_streaming(model.matrix(~ a + b, d), y = d$yb, family = binomial(), firth = TRUE)
    s0 <- fastglm_streaming(model.matrix(~ a, d), y = d$yb, family = binomial(), firth = TRUE)
    expect_error(anova(s0, s1), "fastglm_profile_test")
})

test_that("Poisson against negative binomial uses the boundary-corrected p-value", {
    d <- anova_data()
    X <- model.matrix(~ a + b, d)
    nb <- fastglm_nb(ynb ~ a + b, data = d)
    pm <- fastglm(X, d$ynb, family = poisson())
    pf <- fastglm(ynb ~ a + b, data = d, family = poisson())   # formula fit (fastglmFit)
    expect_s3_class(pf, "glm")
    for (p in list(pm, pf)) {
        for (tab in list(anova(p, nb), anova(nb, p))) {
            stat <- 2 * (as.numeric(logLik(nb)) - as.numeric(logLik(p)))
            expect_equal(tab$Chisq[2], stat, tolerance = 1e-10)
            expect_identical(tab$Df[2], 1)
            expect_equal(tab$`Pr(>Chisq)`[2], 0.5 * pchisq(stat, 1, lower.tail = FALSE))
            expect_output(print(tab), "boundary")
        }
    }
    # A streamed Poisson fit.
    ps <- fastglm_streaming(anova_source(d, ynb ~ a + b), family = poisson())
    expect_warning(tab <- anova(ps, nb), "names only")
    expect_equal(tab$`Pr(>Chisq)`[2], 0.5 * pchisq(tab$Chisq[2], 1, lower.tail = FALSE))
    # Extra mean parameters in the negative binomial model: the 50:50 mixture
    # of chi-square(r) and chi-square(r + 1), here r = 2.
    p0 <- fastglm(ynb ~ a, data = d, family = poisson())
    tab <- anova(p0, nb)
    stat <- 2 * (as.numeric(logLik(nb)) - as.numeric(logLik(p0)))
    expect_identical(tab$Df[2], 3)
    expect_equal(tab$`Pr(>Chisq)`[2], 0.5 * pchisq(stat, 2, lower.tail = FALSE) +
                     0.5 * pchisq(stat, 3, lower.tail = FALSE))
    expect_output(print(tab), "r = 2")
    # The negative binomial model must be the larger one.
    expect_error(anova(fastglm_nb(ynb ~ a, data = d), fastglm(ynb ~ a + b + c, data = d, family = poisson())),
                 "must be the smaller model")
    # Truncated and hurdle count components.
    t0 <- fastglm_truncated(yt ~ a + b, data = d, dist = "poisson")
    t1 <- fastglm_truncated(yt ~ a + b, data = d, dist = "negbin")
    tt <- anova(t0, t1)
    expect_equal(tt$`Pr(>Chisq)`[2], 0.5 * pchisq(tt$Chisq[2], 1, lower.tail = FALSE))
    expect_output(print(tt), "boundary")
    h0 <- fastglm_hurdle(ynb ~ a + b | a, data = d, dist = "poisson")
    h1 <- fastglm_hurdle(ynb ~ a + b | a, data = d, dist = "negbin")
    th <- anova(h1, h0)
    expect_equal(th$`Pr(>Chisq)`[2], 0.5 * pchisq(th$Chisq[2], 1, lower.tail = FALSE))
    expect_output(print(th), "boundary")
})

test_that("zero-inflated against non-inflated is refused with a pointer to AIC()", {
    d <- anova_data()
    z <- fastglm_zi(yzi ~ a + b | a, data = d)
    p <- fastglm(yzi ~ a + b, data = d, family = poisson())
    expect_error(anova(z, p), "AIC")
    expect_error(anova(p, z), "AIC")
    expect_error(anova(fastglm_nb(yzi ~ a + b, data = d), z), "AIC")
})

test_that("data, family, link, and nesting checks", {
    d <- anova_data()
    # Review focus 2: different subsets. Formula fits alone keep the error of
    # stats::anova.glm; other calls name the numbers of observations.
    a1 <- fastglm(yp ~ a + b, data = d, family = poisson())
    a0 <- fastglm(yp ~ a, data = d, family = poisson(), subset = c > 0)
    n0 <- sum(d$c > 0)
    expect_error(anova(a0, a1), "same size of dataset")
    m1 <- fastglm(model.matrix(~ a + b, d), d$yp, family = poisson())
    m0 <- fastglm(model.matrix(~ a, d)[d$c > 0, ], d$yp[d$c > 0], family = poisson())
    expect_error(anova(m0, m1), paste0("uses ", n0, " observations.*uses ", nrow(d)))
    nb0 <- fastglm_nb(ynb ~ a, data = d, subset = c > 0)
    expect_error(anova(nb0, fastglm_nb(ynb ~ a + b, data = d)), paste0("uses ", n0, " observations"))
    # Different weights.
    mw <- fastglm(model.matrix(~ a, d), d$yp, weights = d$w, family = poisson())
    expect_error(anova(mw, m1), "prior weights")
    # Same weights total, different weights.
    w2 <- rev(d$w)
    mw2 <- fastglm(model.matrix(~ a + b, d), d$yp, weights = w2, family = poisson())
    expect_error(anova(mw, mw2), "different responses or weights")
    # Different response, same number of observations.
    expect_error(anova(fastglm(model.matrix(~ a, d), d$yzi, family = poisson()), m1), "different responses")
    # Rows in another order: the data checks do not depend on row order;
    # nesting is checked by names, with a warning, and the test is the same.
    perm <- sample(nrow(d))
    dp <- d[perm, ]
    ref <- anova(fastglm_nb(ynb ~ a, data = d), fastglm_nb(ynb ~ a + b, data = d))
    expect_warning(tab <- anova(fastglm_nb(ynb ~ a, data = d), fastglm_nb(ynb ~ a + b, data = dp)),
                   "names only")
    expect_equal(tab$Chisq[2], ref$Chisq[2], tolerance = 1e-8)
    expect_warning(tab <- anova(fastglm(model.matrix(~ a, d), d$yp, family = poisson()),
                                fastglm(model.matrix(~ a + b, dp), dp$yp, family = poisson())), "names only")
    expect_equal(tab$Deviance[2], anova(glm(yp ~ a, poisson, d), glm(yp ~ a + b, poisson, d))$Deviance[2],
                 tolerance = 1e-8)
    # Different families and links.
    yb <- d$yb
    expect_error(anova(fastglm(model.matrix(~ a, d), yb, family = binomial()),
                       fastglm(model.matrix(~ a + b, d), yb, family = poisson())), "different families")
    expect_error(anova(fastglm(model.matrix(~ a, d), yb, family = binomial()),
                       fastglm(model.matrix(~ a + b, d), yb, family = binomial("probit"))), "different links")
    # Different kinds of models.
    expect_error(anova(fastglm_ordinal(yo ~ a, data = d), fastglm_multinom(factor(yo, ordered = FALSE) ~ a, data = d)),
                 "different kinds")
    # Not nested.
    expect_error(anova(fastglm(model.matrix(~ a, d), d$yp, family = poisson()),
                       fastglm(model.matrix(~ b + c, d), d$yp, family = poisson())), "not nested")
    expect_error(anova(fastglm_nb(ynb ~ a + c, data = d), fastglm_nb(ynb ~ a + b, data = d)), "not nested")
    # No column names: in memory the designs settle nesting; streamed fits
    # cannot be checked (warning), and the test is given.
    u0 <- fastglm(unname(model.matrix(~ a, d)), d$yp, family = poisson())
    u1 <- fastglm(unname(model.matrix(~ a + b, d)), d$yp, family = poisson())
    expect_silent(tab <- anova(u0, u1))
    expect_equal(tab$Deviance[2], u0$deviance - u1$deviance)
    # Unnamed columns that disagree: the row order cannot be confirmed, and
    # generated names cannot be matched.
    expect_warning(anova(u0, fastglm(unname(model.matrix(~ b + c, d)), d$yp, family = poisson())),
                   "cannot be verified")
    s0 <- fastglm_streaming(unname(model.matrix(~ a, d)), y = d$yp, family = poisson())
    s1 <- fastglm_streaming(unname(model.matrix(~ a + b, d)), y = d$yp, family = poisson())
    expect_warning(tab <- anova(s0, s1), "cannot be verified")
    expect_equal(tab$Deviance[2], u0$deviance - u1$deviance, tolerance = 1e-8)
    # Streamed formula fits: nesting by names, with a warning.
    expect_warning(anova(fastglm_streaming(anova_source(d, yp ~ a), family = poisson()),
                         fastglm_streaming(anova_source(d, yp ~ a + b), family = poisson())), "names only")
    # A single model without terms has no sequential table (see
    # test-compare-drop1.R for single models with terms).
    expect_error(anova(m1), "sequential table")
})

test_that("formula fits alone keep stats::anova.glm results", {
    d <- anova_data()
    f0 <- fastglm(yp ~ a, data = d, family = poisson())
    f1 <- fastglm(yp ~ a + b + c, data = d, family = poisson())
    fb <- fastglm(yp ~ a + b, data = d, family = poisson())
    # Identical to the path before the comparison engine existed: the refit
    # controls, then stats::anova.glm.
    old <- function(object, ...) {
        object <- fastglm:::.fastglm_stats_refit(object, "anova", glm.fit = FALSE)
        getS3method("anova", "glm")(object, ...)
    }
    expect_identical(anova(f0, f1), old(f0, f1))
    for (test in c("Chisq", "F", "Rao", "LRT"))
        expect_identical(suppressWarnings(anova(f0, f1, test = test)), suppressWarnings(old(f0, f1, test = test)))
    expect_identical(anova(f1, f0, fb), old(f1, f0, fb))
    expect_identical(anova(f0, f1, dispersion = 2), old(f0, f1, dispersion = 2))
    expect_identical(anova(f1), old(f1))
    # Reordered rows and different data go to anova.glm unchanged.
    fp <- fastglm(yp ~ a + b, data = d[order(d$a), ], family = poisson())
    expect_identical(anova(f0, fp), old(f0, fp))
    g0 <- glm(yp ~ a, poisson, d); g1 <- glm(yp ~ a + b + c, poisson, d)
    for (test in list(NULL, "Chisq", "F")) {
        ours <- suppressWarnings(if (is.null(test)) anova(f0, f1) else anova(f0, f1, test = test))
        ref <- suppressWarnings(if (is.null(test)) anova(g0, g1) else anova(g0, g1, test = test))
        expect_identical(attr(ours, "heading"), attr(ref, "heading"))
        expect_identical(names(ours), names(ref))
        expect_equal(unname(as.matrix(ours)), unname(as.matrix(ref)), tolerance = 1e-8)
    }
    # Rao: the score refits of anova.glm use the fit's method (fastglm_fit),
    # so agreement is to the fitting tolerance.
    ours <- anova(f0, f1, test = "Rao"); ref <- anova(g0, g1, test = "Rao")
    expect_identical(attr(ours, "heading"), attr(ref, "heading"))
    expect_equal(unname(as.matrix(ours)), unname(as.matrix(ref)), tolerance = 1e-5)
    # The anova.glm path: unsorted order, as anova.glm keeps it.
    expect_identical(attr(anova(f1, f0), "heading"), attr(anova(g1, g0), "heading"))
    expect_equal(unname(as.matrix(anova(f1, f0))), unname(as.matrix(anova(g1, g0))), tolerance = 1e-8)
    # Against a stats::glm fit too.
    expect_equal(unname(as.matrix(anova(f0, g1))), unname(as.matrix(anova(g0, g1))), tolerance = 1e-8)
    # Single-model sequential anova is unchanged.
    expect_equal(unname(as.matrix(anova(f1))), unname(as.matrix(anova(g1))), tolerance = 1e-8)
    # Gaussian with the F test.
    n0 <- fastglm(yn ~ a, data = d); n1 <- fastglm(yn ~ a + c, data = d)
    expect_equal(unname(as.matrix(anova(n0, n1, test = "F"))),
                 unname(as.matrix(anova(glm(yn ~ a, data = d), glm(yn ~ a + c, data = d), test = "F"))),
                 tolerance = 1e-8)
})

test_that("offsets must be the same in every model", {
    d <- anova_data()
    d$off <- runif(nrow(d), -0.3, 0.3)
    # Poisson with an offset against NB without: different mean models.
    expect_error(anova(fastglm(ynb ~ a + offset(off), data = d, family = poisson()), fastglm_nb(ynb ~ a, data = d)),
                 "different offsets")
    X0 <- model.matrix(~ a, d); X1 <- model.matrix(~ a + b, d)
    expect_error(anova(fastglm(X0, d$yp, offset = d$off, family = poisson()), fastglm(X1, d$yp, family = poisson())),
                 "different offsets")
    # Same offsets: the test of anova.glm.
    ref <- anova(glm(yp ~ a + offset(off), poisson, d), glm(yp ~ a + b + offset(off), poisson, d), test = "Chisq")
    tab <- anova(fastglm(X0, d$yp, offset = d$off, family = poisson()), fastglm(X1, d$yp, offset = d$off, family = poisson()))
    expect_equal(unname(as.matrix(tab)), unname(as.matrix(ref)), tolerance = 1e-8)
    # Offsets with rows in another order: the fingerprints agree.
    dp <- d[rev(seq_len(nrow(d))), ]
    expect_warning(tab2 <- anova(fastglm_nb(ynb ~ a + offset(off), data = d), fastglm_nb(ynb ~ a + b + offset(off), data = dp)),
                   "names only")
    expect_equal(tab2$Chisq[2], anova(fastglm_nb(ynb ~ a + offset(off), data = d),
                                      fastglm_nb(ynb ~ a + b + offset(off), data = d))$Chisq[2], tolerance = 1e-8)
    # Same sums but different offsets row by row (aligned rows): error.
    d$off2 <- rev(d$off)
    expect_error(anova(fastglm_nb(ynb ~ a + offset(off), data = d), fastglm_nb(ynb ~ a + b + offset(off2), data = d)),
                 "different offsets")
    # Streamed fits record offset fingerprints.
    s0 <- fastglm_streaming(anova_source(d, yp ~ a + offset(off)), family = poisson())
    s1 <- fastglm_streaming(anova_source(d, yp ~ a + b), family = poisson())
    expect_error(anova(s0, s1), "different offsets")
    s1o <- fastglm_streaming(anova_source(d, yp ~ a + b + offset(off)), family = poisson())
    expect_equal(unname(as.matrix(suppressWarnings(anova(s0, s1o)))), unname(as.matrix(ref)), tolerance = 1e-8)
    # Zero-inflated fits compare offsets per part.
    expect_error(anova(fastglm_zi(yzi ~ a | a + offset(off), data = d), fastglm_zi(yzi ~ a + b | a, data = d)),
                 "different offsets")
})

test_that("nesting is checked by the column spaces of the designs", {
    skip_if_not_installed("MASS")
    d <- anova_data()
    d$x <- runif(nrow(d), 0, 2)
    pair <- function(f0, f1) {
        ours <- anova(fastglm_nb(f0, data = d), fastglm_nb(f1, data = d))
        ref <- anova(MASS::glm.nb(f0, data = d), MASS::glm.nb(f1, data = d))
        expect_equal(last_row(ours, "Chisq", "Pr(>Chisq)"), last_row(ref, "LR stat.", "Pr(Chi)"), tolerance = 1e-8,
                     info = deparse(f1))
    }
    expect_silent(pair(ynb ~ a * c, ynb ~ c * a + b))
    pair(ynb ~ poly(x, 2), ynb ~ poly(x, 3))
    pair(ynb ~ poly(x, 2, raw = TRUE), ynb ~ x + I(x^2) + I(x^3))
    pair(ynb ~ b, ynb ~ 0 + b + a)
    # Matrix fits with the same names and different columns: with informative
    # row names the order is confirmed and the pair is not nested; without
    # them the order cannot be confirmed, and the names check gives a warning.
    di <- d; rownames(di) <- paste0("id", seq_len(nrow(d)))
    X0 <- model.matrix(~ a, di); X1 <- model.matrix(~ a + c, di); X1[, "a"] <- di$a^2
    expect_error(anova(fastglm(X0, d$yp, family = poisson()), fastglm(X1, d$yp, family = poisson())),
                 "column space")
    expect_error(anova(fastglm_nb(X0, d$ynb), fastglm_nb(X1, d$ynb)), "column space")
    Xa <- model.matrix(~ a, d); Xb <- model.matrix(~ a + c, d); Xb[, "a"] <- d$a^2
    expect_warning(anova(fastglm(Xa, d$yp, family = poisson()), fastglm(Xb, d$yp, family = poisson())),
                   "names only")
    # Ordinal: proportional odds in partial proportional odds, by designs.
    o1 <- fastglm_ordinal(yo ~ a + b, data = d); p1 <- fastglm_ordinal(yo ~ a + b, data = d, nonparallel = ~ a)
    expect_silent(anova(o1, p1))
    expect_error(anova(fastglm_ordinal(yo ~ c, data = d), p1), "not nested")
    # Multinomial with another reference level: the same data (categories by
    # label) and the same model.
    m <- fastglm_multinom(ym ~ a, data = d)
    d$ym2 <- relevel(d$ym, "w")
    m2 <- fastglm_multinom(ym2 ~ a, data = d)
    tab <- anova(m, m2)
    expect_identical(tab$Df[2], 0)
    expect_equal(tab$LogLik[1], tab$LogLik[2], tolerance = 1e-8)
})

test_that("boundary comparisons agree with MASS and pscl, including three models", {
    skip_if_not_installed("MASS")
    skip_if_not_installed("pscl")
    d <- anova_data()
    mix <- function(x, r) 0.5 * pchisq(x, r, lower.tail = FALSE) + 0.5 * pchisq(x, r + 1, lower.tail = FALSE)
    expect_equal(fastglm:::.fastglm_boundary_pvalue(5, 1), mix(5, 1))
    expect_equal(fastglm:::.fastglm_boundary_pvalue(5, 0), 0.5 * pchisq(5, 1, lower.tail = FALSE))
    expect_equal(fastglm:::.fastglm_boundary_pvalue(5, 1), 0.5 * (pchisq(5, 1, lower.tail = FALSE) + exp(-5 / 2)))
    nbr <- MASS::glm.nb(ynb ~ a + b, data = d); po <- glm(ynb ~ a, poisson, d)
    stat <- 2 * (as.numeric(logLik(nbr)) - as.numeric(logLik(po)))
    tab <- anova(fastglm(ynb ~ a, data = d, family = poisson()), fastglm_nb(ynb ~ a + b, data = d))
    expect_equal(last_row(tab, "Chisq", "Pr(>Chisq)"), c(stat, mix(stat, 2)), tolerance = 1e-8)
    # Three models with the boundary step in the middle.
    tab3 <- anova(fastglm_nb(ynb ~ a + b, data = d), fastglm(ynb ~ a, data = d, family = poisson()),
                  fastglm_nb(ynb ~ a, data = d))
    nba <- MASS::glm.nb(ynb ~ a, data = d)
    s2 <- 2 * (as.numeric(logLik(nba)) - as.numeric(logLik(po)))
    s3 <- 2 * (as.numeric(logLik(nbr)) - as.numeric(logLik(nba)))
    expect_equal(tab3$Chisq[2:3], c(s2, s3), tolerance = 1e-8)
    expect_equal(tab3$`Pr(>Chisq)`[2:3], c(mix(s2, 0), pchisq(s3, 2, lower.tail = FALSE)), tolerance = 1e-8)
    expect_output(print(tab3), "model 2 against model 1")
    # Zero-inflated Poisson against zero-inflated NB, against pscl.
    ctl <- pscl::zeroinfl.control(reltol = 1e-14, maxit = 10000)
    zp <- pscl::zeroinfl(yzi ~ a + b | a, data = d, dist = "poisson", control = ctl)
    zn <- pscl::zeroinfl(yzi ~ a + b | a, data = d, dist = "negbin", control = ctl)
    sz <- 2 * (as.numeric(logLik(zn)) - as.numeric(logLik(zp)))
    tz <- anova(fastglm_zi(yzi ~ a + b | a, data = d, dist = "negbin"), fastglm_zi(yzi ~ a + b | a, data = d))
    expect_equal(last_row(tz, "Chisq", "Pr(>Chisq)"), c(sz, mix(sz, 0)), tolerance = 1e-8)
    expect_output(print(tz), "boundary")
    # Hurdle Poisson against hurdle NB, against pscl.
    hc <- pscl::hurdle.control(reltol = 1e-14, maxit = 10000)
    hp <- pscl::hurdle(ynb ~ a + b | a, data = d, dist = "poisson", control = hc)
    hn <- pscl::hurdle(ynb ~ a + b | a, data = d, dist = "negbin", control = hc)
    sh <- 2 * (as.numeric(logLik(hn)) - as.numeric(logLik(hp)))
    th <- anova(fastglm_hurdle(ynb ~ a + b | a, data = d), fastglm_hurdle(ynb ~ a + b | a, data = d, dist = "negbin"))
    expect_equal(last_row(th, "Chisq", "Pr(>Chisq)"), c(sh, mix(sh, 0)), tolerance = 1e-8)
})

test_that("two-part comparisons use logLik(), maximized over the Gamma shape", {
    d <- anova_data()
    d$ytp <- ifelse(d$ynb > 0, d$yg * (1 + d$ynb), 0)
    d$wi <- rep(1:3, length.out = nrow(d))
    t0 <- fastglm_twopart(ytp ~ a | a, data = d, weights = wi)
    t1 <- fastglm_twopart(ytp ~ a + b + c | a + c, data = d, weights = wi)
    profile <- function(fit) {
        pos <- d$ytp > 0
        y <- d$ytp[pos]; w <- d$wi[pos]; mu <- fit$positive$fitted.values
        g <- function(lnu) { nu <- exp(lnu); sum(w * dgamma(y, shape = nu, scale = mu / nu, log = TRUE)) }
        pz <- fit$binary$fitted.values; z <- as.numeric(d$ytp > 0)
        optimize(g, c(-10, 10), maximum = TRUE, tol = 1e-12)$objective +
            sum(d$wi * (z * log(pz) + (1 - z) * log(1 - pz)))
    }
    tab <- anova(t1, t0)
    stat <- 2 * (profile(t1) - profile(t0))
    expect_equal(tab$Chisq[2], stat, tolerance = 1e-8)
    expect_equal(tab$`Pr(>Chisq)`[2], pchisq(stat, 4, lower.tail = FALSE), tolerance = 1e-8)
    expect_equal(tab$LogLik, c(profile(t0), profile(t1)), tolerance = 1e-10)
    # logLik() is the maximized value (the Gamma shape at its ML estimate).
    expect_equal(as.numeric(logLik(t1)), t1$loglik)
    expect_equal(as.numeric(logLik(t1)), profile(t1), tolerance = 1e-10)
    expect_equal(AIC(t1), -2 * profile(t1) + 2 * attr(logLik(t1), "df"), tolerance = 1e-10)
    # Streamed and in-memory fits with non-integer weights give the same
    # statistic, alone or mixed.
    m0 <- suppressWarnings(fastglm_twopart(ytp ~ a | a, data = d, weights = w))
    m1 <- suppressWarnings(fastglm_twopart(ytp ~ a + b + c | a + c, data = d, weights = w))
    s1 <- suppressWarnings(fastglm_twopart(anova_source(d, ytp ~ a + b + c | a + c, weights = "w")))
    s0 <- suppressWarnings(fastglm_twopart(anova_source(d, ytp ~ a | a, weights = "w")))
    ref <- anova(m0, m1)$Chisq[2]
    expect_equal(suppressWarnings(anova(m0, s1))$Chisq[2], ref, tolerance = 1e-8)
    expect_equal(suppressWarnings(anova(s0, s1))$Chisq[2], ref, tolerance = 1e-8)
    expect_equal(as.numeric(logLik(s1)), as.numeric(logLik(m1)), tolerance = 1e-10)
    expect_equal(s1$shape, m1$shape, tolerance = 1e-10)
})

test_that("the two-part Gamma shape is its maximum likelihood estimate", {
    d <- anova_data()
    d$ytp <- ifelse(d$ynb > 0, d$yg * (1 + d$ynb), 0)
    t1 <- fastglm_twopart(ytp ~ a + b + c | a, data = d)
    # Against MASS::gamma.shape() on the Gamma GLM of the positive part.
    skip_if_not_installed("MASS")
    pos <- d$ytp > 0
    gm <- glm(ytp ~ a + b + c, family = Gamma("log"), data = d[pos, ],
              control = glm.control(epsilon = 1e-14, maxit = 100))
    expect_equal(t1$shape, MASS::gamma.shape(gm, it.lim = 1000, eps.max = 1e-14)$alpha, tolerance = 1e-6)
    # logLik() of the positive part at that shape, plus the binary part.
    mu <- t1$positive$fitted.values
    ll.pos <- sum(dgamma(d$ytp[pos], shape = t1$shape, rate = t1$shape / mu, log = TRUE))
    expect_equal(t1$loglik_count, ll.pos, tolerance = 1e-10)
    expect_equal(as.numeric(logLik(t1)), ll.pos + as.numeric(logLik(t1$binary)), tolerance = 1e-10)
    # The plug-in log likelihood (stats::glm's Gamma AIC) is smaller.
    expect_lt(as.numeric(logLik(gm)), ll.pos)
    expect_identical(attr(logLik(t1), "df"), length(coef(t1)) + 1L)
    # The shape equation is solved to rounding for very small and very large
    # shapes (D / (2 W) from 1e-12 to 1e4).
    for (t in 10^seq(-12, 4, by = 2)) {
        nu <- fastglm:::.gamma_ml_shape(2 * t * 100, 100)
        expect_true(is.finite(nu) && nu > 0)
        expect_lt(abs(fastglm:::.gamma_shape_h(nu) - t), 1e-12 * t)
    }
    # The stable q(nu) agrees with its direct form where both are accurate.
    for (nu in c(0.01, 0.5, 3, 9.9, 10, 50, 500))
        expect_equal(fastglm:::.gamma_shape_q(nu), nu * log(nu) - nu - lgamma(nu), tolerance = 1e-12)
})

test_that("negative statistics, Rao, and stats::glm fits in an engine call", {
    d <- anova_data()
    recs <- list(list(class_family = "nb", dispersion = "ml", df = 3L, loglik = -100),
                 list(class_family = "nb", dispersion = "ml", df = 4L, loglik = -101))
    expect_warning(tab <- fastglm:::.fastglm_lrt_loglik(list(NULL, NULL), recs,
                                                        list(boundary = c(FALSE, FALSE), r = c(NA, NA)), "m"),
                   "negative")
    expect_true(is.na(tab$Chisq[2]) && is.na(tab$`Pr(>Chisq)`[2]))
    pf <- fastglm(ynb ~ a, data = d, family = poisson()); nb <- fastglm_nb(ynb ~ a, data = d)
    expect_error(anova(pf, nb, test = "Rao"), "Rao")
    expect_error(anova(nb, glm(ynb ~ a, poisson, d)), "refit it with fastglm")
})

test_that("row order is confirmed before a design check can fail", {
    d <- anova_data()
    # Binary data sorted by the response: the response and weights agree row
    # by row in the two orders, but the rows are different subjects.
    ds <- d[order(d$yb), ]; rownames(ds) <- NULL
    d2 <- ds[order(ds$yb, ds$b), ]; rownames(d2) <- NULL
    ref <- anova(glm(yb ~ a, binomial, ds), glm(yb ~ a + b, binomial, ds), test = "Chisq")
    expect_warning(tab <- anova(fastglm(model.matrix(~ a, ds), ds$yb, family = binomial()),
                                fastglm(model.matrix(~ a + b, d2), d2$yb, family = binomial())), "names only")
    expect_equal(unname(as.matrix(tab)), unname(as.matrix(ref)), tolerance = 1e-8)
    # NB counts sorted by the response, with an offset.
    ds$off <- runif(nrow(ds), -0.2, 0.2)
    ds <- ds[order(ds$ynb), ]; rownames(ds) <- NULL
    d3 <- ds[order(ds$ynb, ds$b), ]; rownames(d3) <- NULL
    expect_warning(t3 <- anova(fastglm_nb(ynb ~ a + offset(off), data = ds), fastglm_nb(ynb ~ a + b + offset(off), data = d3)),
                   "names only")
    expect_equal(t3$Chisq[2], anova(fastglm_nb(ynb ~ a + offset(off), data = ds),
                                    fastglm_nb(ynb ~ a + b + offset(off), data = ds))$Chisq[2], tolerance = 1e-8)
    # The blockwise factorization: a rare level in the larger design only.
    set.seed(3)
    n <- 3000
    x <- rnorm(n); f <- rep(0, n); f[n] <- 1
    X1 <- cbind(1, x, f); rows <- rep(TRUE, n)
    expect_true(fastglm:::.fastglm_cmp_in_span(cbind(1, x), X1, rows))
    expect_false(fastglm:::.fastglm_cmp_in_span(cbind(1, x, f), cbind(1, x), rows))
    # Columns of very different magnitudes.
    expect_true(fastglm:::.fastglm_cmp_in_span(cbind(1e160 * x), cbind(1, 1e-170 * x), rows))
    expect_false(fastglm:::.fastglm_cmp_in_span(cbind(1e160 * x), cbind(1, 1e-170 * rnorm(n)), rows))
})

test_that("streamed fits of one callback and prototype need no names-only warning", {
    d <- anova_data()
    idx <- split(seq_len(nrow(d)), rep(1:4, length.out = nrow(d)))
    cb <- function(k) d[idx[[k]], , drop = FALSE]
    src <- function(f) fastglm_source(cb, 4L, formula = f, prototype = d)
    s0 <- fastglm_streaming(src(yp ~ a), family = poisson())
    s1 <- fastglm_streaming(src(yp ~ a + b), family = poisson())
    expect_silent(tab <- anova(s0, s1))
    expect_equal(tab$Deviance[2], anova(glm(yp ~ a, poisson, d), glm(yp ~ a + b, poisson, d))$Deviance[2],
                 tolerance = 1e-8)
    one <- src(ynb ~ a + b)
    expect_silent(anova(fastglm_streaming(one, family = poisson()), fastglm_nb(one)))
    # The names check still applies.
    expect_error(anova(fastglm_streaming(src(yp ~ c), family = poisson()), s1), "not nested")
})
