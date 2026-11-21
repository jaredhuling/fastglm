## emmeans support: reference grids, modes, and estimates for every in-memory
## class, checked against the reference packages (MASS, pscl, nnet) or
## against averages of predict() over the reference grid.

emm_data <- function(n = 200L, seed = 11L) {
    set.seed(seed)
    d <- data.frame(b = factor(sample(c("p", "q", "r"), n, TRUE)),
                    g = factor(sample(c("G1", "G2"), n, TRUE)), x = rnorm(n))
    d$y <- rpois(n, exp(0.3 + 0.4 * (d$b == "q") + 0.3 * d$x)) * rbinom(n, 1, 0.8)
    d$yo <- cut(d$x + 0.5 * (d$b == "q") + rlogis(n), c(-Inf, -0.5, 0.5, 1.5, Inf),
                labels = c("A", "B", "C", "D"), ordered_result = TRUE)
    d$ym <- factor(sample(c("u", "v", "w"), n, TRUE))
    d$yc <- ifelse(rbinom(n, 1, 0.7) == 1, rgamma(n, 2, 1), 0)
    d$ynb <- MASS::rnegbin(n, exp(0.5 + 0.4 * (d$b == "q") + 0.3 * d$x), 2)
    d
}

emm_frame <- function(e) as.data.frame(summary(e))

# The estimate column of an emmeans summary: the column before "SE".
emm_est <- function(s) s[[match("SE", names(s)) - 1L]]

expect_emm_equal <- function(a, b, tol = 1e-6, se = TRUE) {
    a <- emm_frame(a); b <- emm_frame(b)
    expect_equal(nrow(a), nrow(b))
    keys <- intersect(c("b", "g", "yo", "ym", "cut"), names(a))
    for (k in keys) expect_identical(as.character(a[[k]]), as.character(b[[k]]))
    expect_equal(emm_est(a), emm_est(b), tolerance = tol)
    if (se) expect_equal(a$SE, b$SE, tolerance = tol)
    expect_equal(a$df, b$df)
}

# Central-difference delta-method standard errors of g(coef) for each grid row.
delta_se <- function(g, cf, V, h = 1e-6) {
    G <- vapply(seq_along(cf), function(j) {
        a <- b <- cf; a[j] <- a[j] + h; b[j] <- b[j] - h
        (g(a) - g(b)) / (2 * h)
    }, numeric(length(g(cf))))
    G <- matrix(G, ncol = length(cf))
    sqrt(rowSums((G %*% V) * G))
}

skip_emm <- function(...) {
    skip_if_not_installed("emmeans")
    for (p in c("MASS", ...)) skip_if_not_installed(p)
    suppressPackageStartupMessages(requireNamespace("emmeans", quietly = TRUE))
}

test_that("formula fastglm fits still use the glm support of emmeans", {
    skip_emm("MASS")
    d <- emm_data()
    f <- fastglm(y ~ b + g + x, data = d, family = poisson())
    r <- glm(y ~ b + g + x, data = d, family = poisson(), control = glm.control(epsilon = 1e-12))
    expect_emm_equal(emmeans::emmeans(f, "b"), emmeans::emmeans(r, "b"))
    expect_emm_equal(emmeans::emmeans(f, "b", type = "response"), emmeans::emmeans(r, "b", type = "response"))
    fg <- fastglm(x ~ b + g, data = d)
    rg <- glm(x ~ b + g, data = d)
    expect_emm_equal(emmeans::emmeans(fg, "b"), emmeans::emmeans(rg, "b"))
})

test_that("negative binomial fits match MASS::glm.nb", {
    skip_emm("MASS")
    d <- emm_data()
    f <- fastglm_nb(ynb ~ b + g + x, data = d)
    r <- MASS::glm.nb(ynb ~ b + g + x, data = d, control = glm.control(epsilon = 1e-12, maxit = 100))
    expect_emm_equal(emmeans::emmeans(f, "b"), emmeans::emmeans(r, "b"))
    expect_emm_equal(emmeans::emmeans(f, "b", type = "response"), emmeans::emmeans(r, "b", type = "response"))
    expect_emm_equal(emmeans::emmeans(f, ~ b | g), emmeans::emmeans(r, ~ b | g))
})

test_that("zero-inflated fits match pscl::zeroinfl in every mode", {
    skip_emm("pscl")
    d <- emm_data()
    f <- fastglm_zi(y ~ b + x | b + g, data = d)
    r <- pscl::zeroinfl(y ~ b + x | b + g, data = d, control = pscl::zeroinfl.control(reltol = 1e-14, maxit = 1e4))
    for (m in c("response", "count", "zero"))
        expect_emm_equal(emmeans::emmeans(f, "b", mode = m), emmeans::emmeans(r, "b", mode = m, data = d))
    for (m in c("count", "zero"))
        expect_emm_equal(emmeans::emmeans(f, "b", mode = m, lin.pred = TRUE),
                         emmeans::emmeans(r, "b", mode = m, lin.pred = TRUE, data = d))
    # prob0: the estimates of emmeans for pscl, with the complete delta method.
    a <- emm_frame(emmeans::emmeans(f, "b", mode = "prob0"))
    expect_equal(emm_est(a), emm_est(emm_frame(emmeans::emmeans(r, "b", mode = "prob0", data = d))), tolerance = 1e-6)
    grid <- data.frame(b = factor(c("p", "q", "r")), x = mean(d$x))
    cells <- expand.grid(b = levels(d$b), g = levels(d$g))
    cells$x <- mean(d$x)
    X1 <- model.matrix(~ b + x, cells); X2 <- model.matrix(~ b + g, cells)
    gfun <- function(cf) {
        mu <- exp(drop(X1 %*% cf[1:4])); p <- plogis(drop(X2 %*% cf[5:8]))
        v <- (1 - p) * exp(-mu) + p
        tapply(v, cells$b, mean)
    }
    cf <- coef(f)
    expect_equal(emm_est(a), as.vector(gfun(cf)), tolerance = 1e-10)
    expect_equal(a$SE, unname(delta_se(gfun, cf, vcov(f))), tolerance = 1e-6)
})

test_that("hurdle fits match pscl::hurdle", {
    skip_emm("pscl")
    d <- emm_data()
    f <- fastglm_hurdle(y ~ b + x, data = d)
    r <- pscl::hurdle(y ~ b + x, data = d, control = pscl::hurdle.control(reltol = 1e-14, maxit = 1e4))
    for (m in c("response", "count", "prob0"))
        expect_emm_equal(emmeans::emmeans(f, "b", mode = m), emmeans::emmeans(r, "b", mode = m, data = d))
    for (m in c("count", "zero"))
        expect_emm_equal(emmeans::emmeans(f, "b", mode = m, lin.pred = TRUE),
                         emmeans::emmeans(r, "b", mode = m, lin.pred = TRUE, data = d))
    # zero: the estimates of emmeans for pscl, with the complete delta method.
    a <- emm_frame(emmeans::emmeans(f, "b", mode = "zero"))
    expect_equal(emm_est(a), emm_est(emm_frame(emmeans::emmeans(r, "b", mode = "zero", data = d))), tolerance = 1e-6)
    X <- model.matrix(~ b + x, data.frame(b = factor(levels(d$b)), x = mean(d$x)))
    gfun <- function(cf) plogis(drop(X %*% cf[5:8])) / (1 - exp(-exp(drop(X %*% cf[1:4]))))
    expect_equal(a$SE, delta_se(gfun, coef(f), vcov(f)), tolerance = 1e-6)
    expect_equal(a$df, rep(f$df.residual, 3L))
    # Different predictors in the two parts.
    f2 <- fastglm_hurdle(y ~ b + x | b + g, data = d)
    r2 <- pscl::hurdle(y ~ b + x | b + g, data = d, control = pscl::hurdle.control(reltol = 1e-14, maxit = 1e4))
    for (m in c("response", "count"))
        expect_emm_equal(emmeans::emmeans(f2, "b", mode = m), emmeans::emmeans(r2, "b", mode = m, data = d))
    cells <- expand.grid(b = levels(d$b), g = levels(d$g)); cells$x <- mean(d$x)
    p0 <- tapply(predict(f2, newdata = cells, type = "zero"), cells$b, mean)
    expect_equal(emm_est(emm_frame(emmeans::emmeans(f2, "b", mode = "prob0"))), as.vector(p0), tolerance = 1e-8)
    lp <- predict(f2, newdata = cells, type = "link")[, "count"]
    ratio <- tapply(predict(f2, newdata = cells, type = "binary") / (1 - exp(-exp(lp))), cells$b, mean)
    expect_equal(emm_est(emm_frame(emmeans::emmeans(f2, "b", mode = "zero"))), as.vector(ratio), tolerance = 1e-8)
})

test_that("cumulative-link fits match MASS::polr in every mode", {
    skip_emm("MASS")
    d <- emm_data()
    f <- fastglm_ordinal(yo ~ b + g + x, data = d)
    r <- MASS::polr(yo ~ b + g + x, data = d, Hess = TRUE, control = list(reltol = 1e-14, maxit = 1000))
    # fastglm uses expected information and polr the observed Hessian, so
    # standard errors are compared with polr given the fastglm covariance.
    nm <- c(names(coef(r)), paste0("threshold:", names(r$zeta)))
    Vf <- unname(vcov(f)[nm, nm])
    for (m in c("latent", "linear.predictor", "cum.prob", "exc.prob")) {
        expect_emm_equal(emmeans::emmeans(f, "b", mode = m), emmeans::emmeans(r, "b", mode = m), se = FALSE)
        expect_emm_equal(emmeans::emmeans(f, "b", mode = m), emmeans::emmeans(r, "b", mode = m, vcov. = Vf))
    }
    for (m in c("prob", "mean.class")) {
        spec <- if (m == "prob") ~ b | yo else ~ b
        expect_emm_equal(emmeans::emmeans(f, spec, mode = m), emmeans::emmeans(r, spec, mode = m), se = FALSE)
        expect_emm_equal(emmeans::emmeans(f, spec, mode = m), emmeans::emmeans(r, spec, mode = m, vcov. = Vf))
    }
    expect_emm_equal(emmeans::emmeans(f, "b", mode = "latent", rescale = c(1, 2)),
                     emmeans::emmeans(r, "b", mode = "latent", rescale = c(1, 2), vcov. = Vf))
    fp <- fastglm_ordinal(yo ~ b + x, data = d, link = "probit")
    rp <- MASS::polr(yo ~ b + x, data = d, method = "probit", Hess = TRUE, control = list(reltol = 1e-14, maxit = 1000))
    expect_emm_equal(emmeans::emmeans(fp, ~ b | yo, mode = "prob"), emmeans::emmeans(rp, ~ b | yo, mode = "prob"),
                     se = FALSE)
})

test_that("multinomial fits match nnet::multinom for any reference category", {
    skip_emm("nnet")
    d <- emm_data()
    r <- nnet::multinom(ym ~ b + g + x, data = d, reltol = 1e-14, maxit = 1000, trace = FALSE)
    for (ref in c("u", "w")) {
        f <- fastglm_multinom(ym ~ b + g + x, data = d, reference = ref)
        for (m in c("prob", "latent"))
            expect_emm_equal(emmeans::emmeans(f, ~ b | ym, mode = m), emmeans::emmeans(r, ~ b | ym, mode = m))
    }
})

test_that("truncated modes equal averages of predict() over the grid", {
    skip_emm()
    d <- emm_data()
    dp <- d[d$y > 0, ]
    cells <- expand.grid(b = levels(d$b), g = levels(d$g)); cells$x <- mean(dp$x)
    for (dist in c("poisson", "negbin")) {
        f <- fastglm_truncated(y ~ b + g + x, data = dp, dist = dist)
        for (m in c("response", "count")) {
            e <- emm_frame(emmeans::emmeans(f, "b", mode = m))
            expect_equal(emm_est(e), as.vector(tapply(predict(f, newdata = cells, type = m), cells$b, mean)),
                         tolerance = 1e-8)
        }
        e <- emm_frame(emmeans::emmeans(f, "b", mode = "link", type = "response"))
        expect_equal(emm_est(e), as.vector(tapply(predict(f, newdata = cells, type = "link"), cells$b,
                                               function(v) exp(mean(v)))), tolerance = 1e-8)
    }
    # Standard errors of single cells agree with predict() for the Poisson case.
    f <- fastglm_truncated(y ~ b + x, data = dp)
    one <- data.frame(b = factor(levels(d$b)), x = mean(dp$x))
    for (m in c("response", "count")) {
        e <- emm_frame(emmeans::emmeans(f, "b", mode = m))
        p <- predict(f, newdata = one, type = m, se.fit = TRUE)
        expect_equal(e$SE, unname(p$se.fit), tolerance = 1e-6)
    }
})

test_that("two-part modes equal averages of predict() over the grid", {
    skip_emm()
    d <- emm_data()
    f <- fastglm_twopart(yc ~ b + x | b + g, data = d)
    cells <- expand.grid(b = levels(d$b), g = levels(d$g)); cells$x <- mean(d$x)
    one <- data.frame(b = factor(levels(d$b)), x = mean(d$x), g = "G1")
    # Each mode estimates the predict() type of the same name.
    for (m in c("response", "zero", "positive")) {
        e <- emm_frame(emmeans::emmeans(f, "b", mode = m))
        expect_equal(emm_est(e), as.vector(tapply(predict(f, newdata = cells, type = m), cells$b, mean)),
                     tolerance = 1e-8)
        e <- emm_frame(emmeans::emmeans(f, "b", mode = m, at = list(g = "G1")))
        expect_equal(e$SE, unname(predict(f, newdata = one, type = m, se.fit = TRUE)$se.fit), tolerance = 1e-6)
    }
    expect_error(emmeans::emmeans(f, "b", mode = "binary"), "should be one of")
})

test_that("partial ordinal modes equal averages of predict() over the grid", {
    skip_emm()
    d <- emm_data()
    f <- fastglm_ordinal(yo ~ b + g + x, data = d, nonparallel = ~ x)
    cells <- expand.grid(b = levels(d$b), g = levels(d$g)); cells$x <- mean(d$x)
    pr <- predict(f, newdata = cells, type = "probability")
    cu <- predict(f, newdata = cells, type = "cumulative")
    avg <- function(m) apply(m, 2L, function(v) tapply(v, cells$b, mean))
    e <- emm_frame(emmeans::emmeans(f, ~ b | yo, mode = "prob"))
    expect_equal(emm_est(e), as.vector(avg(pr)), tolerance = 1e-8)
    e <- emm_frame(emmeans::emmeans(f, ~ b | cut, mode = "cum.prob"))
    expect_equal(emm_est(e), as.vector(avg(cu)), tolerance = 1e-8)
    e <- emm_frame(emmeans::emmeans(f, ~ b | cut, mode = "exc.prob"))
    expect_equal(emm_est(e), as.vector(1 - avg(cu)), tolerance = 1e-8)
    e <- emm_frame(emmeans::emmeans(f, "b", mode = "mean.class"))
    expect_equal(emm_est(e), as.vector(tapply(drop(pr %*% seq_len(ncol(pr))), cells$b, mean)), tolerance = 1e-8)
    expect_error(emmeans::emmeans(f, "b", mode = "latent"), "should be one of")
})

test_that("streamed, big.matrix, and matrix fits point to fastglm_marginal_effects()", {
    skip_emm()
    d <- emm_data(120L)
    X <- model.matrix(~ b + x, d)
    msg <- "fastglm_marginal_effects\\(\\) or fastglm_average\\(\\)"
    quiet <- function(e) suppressMessages(suppressWarnings(e))
    expect_error(quiet(emmeans::emmeans(fastglm(X, d$y, family = poisson()), "x")), msg)
    expect_error(quiet(emmeans::emmeans(fastglm_nb(X, d$ynb), "x")), msg)
    expect_error(quiet(emmeans::emmeans(fastglm_ordinal(X[, -1], d$yo), "x")), msg)
    expect_error(quiet(emmeans::emmeans(fastglm_multinom(X, d$ym), "x")), msg)
    skip_if_not_installed("bigmemory")
    expect_error(quiet(emmeans::emmeans(fastglm(bigmemory::as.big.matrix(X), d$y, family = poisson(), method = 2L), "x")), msg)
    src <- fastglm_source(function(k) d[((k - 1) * 40 + 1):(k * 40), ], 3, formula = y ~ b + x, prototype = d)
    sp <- fastglm_streaming(src, family = poisson())
    # A streamed fit: fastglm_marginal_effects() needs newdata, which the
    # message says; both suggested calls then work.
    smsg <- "fastglm_average\\(\\) \\(which reads the source again\\).*fastglm_marginal_effects\\(\\) with 'newdata'"
    expect_error(quiet(emmeans::emmeans(sp, "b")), smsg)
    expect_error(quiet(emmeans::emmeans(fastglm_nb(src), "b")), smsg)
    expect_equal(fastglm_average(sp)$estimate, fastglm_average(fastglm(y ~ b + x, data = d, family = poisson()))$estimate,
                 tolerance = 1e-8)
    expect_error(fastglm_marginal_effects(sp, "x"), "newdata")
    expect_true(is.finite(fastglm_marginal_effects(sp, "x", newdata = d)$estimate))
    # A big.matrix fit: both work without newdata, and agree with the dense fit.
    fb <- fastglm(bigmemory::as.big.matrix(X), d$y, family = poisson(), method = 2L)
    fm <- fastglm(X, d$y, family = poisson())
    expect_equal(fastglm_average(fb), fastglm_average(fm), tolerance = 1e-8)
    shift <- function(x) { x[, "x"] <- x[, "x"] + 1; x }
    expect_equal(fastglm_average(fb, contrast = shift), fastglm_average(fm, contrast = shift), tolerance = 1e-8)
    expect_equal(fastglm_average(fb, weights = seq_len(nrow(X))),
                 fastglm_average(fm, weights = seq_len(nrow(X))), tolerance = 1e-8)
    expect_equal(fastglm_marginal_effects(fb, "x"), fastglm_marginal_effects(fm, "x"), tolerance = 1e-8)
})

test_that("a fit made without data = gives the reference grid", {
    skip_emm("MASS", "nnet")
    d <- emm_data()
    env <- list2env(as.list(d), envir = new.env())
    f <- eval(quote(fastglm_nb(ynb ~ b + x)), env)
    r <- eval(quote(MASS::glm.nb(ynb ~ b + x, control = glm.control(epsilon = 1e-12))), env)
    expect_emm_equal(emmeans::emmeans(f, "b"), emmeans::emmeans(r, "b"))
    fo <- eval(quote(fastglm_ordinal(yo ~ b + log(x + 5))), env)
    ro <- eval(quote(MASS::polr(yo ~ b + log(x + 5), Hess = TRUE, control = list(reltol = 1e-14))), env)
    expect_emm_equal(emmeans::emmeans(fo, "b"), emmeans::emmeans(ro, "b"), se = FALSE)
    fm <- eval(quote(fastglm_multinom(ym ~ b + x)), env)
    rm <- eval(quote(nnet::multinom(ym ~ b + x, reltol = 1e-14, maxit = 1000, trace = FALSE)), env)
    expect_emm_equal(emmeans::emmeans(fm, ~ b | ym), emmeans::emmeans(rm, ~ b | ym))
})

test_that("changing the data after fitting never changes the grid silently", {
    skip_emm("MASS", "nnet", "pscl")
    d <- emm_data()
    r <- MASS::glm.nb(ynb ~ b + x, data = d, control = glm.control(epsilon = 1e-12))
    rm <- nnet::multinom(ym ~ b + x, data = d, reltol = 1e-14, maxit = 1000, trace = FALSE)
    rz <- pscl::zeroinfl(y ~ b + x, data = d, control = pscl::zeroinfl.control(reltol = 1e-14, maxit = 1e4))
    e.nb <- emmeans::emmeans(r, "b"); e.m <- emmeans::emmeans(rm, ~ b | ym)
    e.z <- emmeans::emmeans(rz, "b", data = d)
    # Formulas without function calls: the stored model frame gives the grid
    # of the fit, whatever happens to the data frame afterwards.
    rg <- glm(y ~ b + x, data = d, family = poisson(), control = glm.control(epsilon = 1e-12))
    e.g <- emmeans::emmeans(rg, "b")
    dd <- d
    fg <- fastglm(y ~ b + x, data = dd, family = poisson())
    f <- fastglm_nb(ynb ~ b + x, data = dd)
    fm <- fastglm_multinom(ym ~ b + x, data = dd)
    fz <- fastglm_zi(y ~ b + x, data = dd)
    dd$b <- factor(sample(c("p", "q", "r", "s"), nrow(dd), TRUE))
    dd$x <- dd$x + 10
    dd <- dd[-(1:20), ]
    expect_emm_equal(emmeans::emmeans(fg, "b"), e.g)
    expect_emm_equal(emmeans::emmeans(f, "b"), e.nb)
    expect_emm_equal(emmeans::emmeans(fm, ~ b | ym), e.m)
    expect_emm_equal(emmeans::emmeans(fz, "b"), e.z)
    # Formulas with function calls need the data from the call: a changed data
    # frame is an error, and supplying the original data with data = works.
    d2 <- d
    f2 <- fastglm_nb(ynb ~ b + log(x + 5), data = d2)
    r2 <- MASS::glm.nb(ynb ~ b + log(x + 5), data = d, control = glm.control(epsilon = 1e-12))
    expect_emm_equal(emmeans::emmeans(f2, "b"), emmeans::emmeans(r2, "b"))
    d2$x <- d2$x + 1
    expect_error(suppressMessages(emmeans::emmeans(f2, "b")), "variable 'log\\(x \\+ 5\\)' has changed")
    d2$x <- d$x
    d2$b <- factor(as.character(d$b), levels = c("r", "q", "p"))
    expect_error(suppressMessages(emmeans::emmeans(f2, "b")), "differ from the data it was fitted to")
    d2 <- d2[-(1:5), ]
    expect_error(suppressMessages(emmeans::emmeans(f2, "b")), "195 rows instead of 200")
    expect_emm_equal(emmeans::emmeans(f2, "b", data = d), emmeans::emmeans(r2, "b"))
    fo <- fastglm_ordinal(yo ~ factor(as.integer(b)) + x, data = d2)
    d2$b <- factor(sample(c("p", "q"), nrow(d2), TRUE))
    expect_error(suppressMessages(emmeans::emmeans(fo, "b")), "differ from the data it was fitted to")
})

test_that("offsets follow the glm convention, and the emmeans offset argument overrides them", {
    skip_emm("pscl")
    d <- emm_data()
    set.seed(21); d$t <- runif(nrow(d), 1, 5)
    d$y <- rpois(nrow(d), d$t * exp(-0.5 + 0.4 * (d$b == "q") + 0.3 * d$x)) * rbinom(nrow(d), 1, 0.8)
    # glm itself: the mean of an offset = argument, replaced by emmeans' offset.
    f <- fastglm(y ~ b + x, data = d, family = poisson(), offset = log(t))
    r <- glm(y ~ b + x, data = d, family = poisson(), offset = log(t), control = glm.control(epsilon = 1e-12))
    for (o in list(NULL, 0, log(2))) {
        a <- if (is.null(o)) emmeans::emmeans(f, "b", type = "response") else
            emmeans::emmeans(f, "b", type = "response", offset = o)
        b <- if (is.null(o)) emmeans::emmeans(r, "b", type = "response") else
            emmeans::emmeans(r, "b", type = "response", offset = o)
        expect_emm_equal(a, b)
    }
    fz <- fastglm_zi(y ~ b + x | b, data = d, offset = log(t))
    rz <- pscl::zeroinfl(y ~ b + x | b, data = d, offset = log(t),
                         control = pscl::zeroinfl.control(reltol = 1e-14, maxit = 1e4))
    b1 <- coef(fz, model = "count"); b2 <- coef(fz, model = "zero")
    X1 <- model.matrix(~ b + x, data.frame(b = factor(levels(d$b)), x = mean(d$x)))
    X2 <- model.matrix(~ b, data.frame(b = factor(levels(d$b))))
    pi <- unname(plogis(drop(X2 %*% b2)))
    mu <- function(o) unname(exp(drop(X1 %*% b1) + o))
    est <- function(m, ...) emm_est(emm_frame(emmeans::emmeans(fz, "b", mode = m, ...)))
    # Default: the mean of log(t) enters the count linear predictor, as for
    # glm and as emmeans does for pscl with lin.pred = TRUE.
    expect_equal(est("count"), mu(mean(log(d$t))), tolerance = 1e-10)
    expect_equal(est("count"), emm_est(emm_frame(emmeans::emmeans(rz, "b", mode = "count", lin.pred = TRUE,
                                                                  type = "response", data = d))), tolerance = 1e-6)
    # offset = 0 reproduces emmeans for pscl, which leaves the offset out.
    for (m in c("response", "count", "prob0"))
        expect_equal(est(m, offset = 0), emm_est(emm_frame(emmeans::emmeans(rz, "b", mode = m, data = d))),
                     tolerance = 1e-6)
    # A nonzero override enters the linear predictor, not the estimate.
    expect_equal(est("count", offset = log(2)), mu(log(2)), tolerance = 1e-10)
    expect_equal(est("response", offset = log(2)), (1 - pi) * mu(log(2)), tolerance = 1e-10)
    expect_equal(est("prob0", offset = log(2)), pi + (1 - pi) * exp(-mu(log(2))), tolerance = 1e-10)
    rg <- emmeans::ref_grid(fz, mode = "count", offset = log(2))
    expect_equal(emm_est(emm_frame(emmeans::emmeans(rg, "b"))), mu(log(2)), tolerance = 1e-10)
    se <- function(...) emm_frame(emmeans::emmeans(fz, "b", mode = "count", ...))$SE
    expect_equal(se(offset = log(2)), se(offset = 0) * 2, tolerance = 1e-10)
    # A formula offset is evaluated at the grid value of its variable.
    dp <- d[d$y > 0, ]
    ft <- fastglm_truncated(y ~ b + x + offset(log(t)), data = dp)
    Xt <- model.matrix(~ b + x, data.frame(b = factor(levels(d$b)), x = mean(dp$x)))
    lp <- function(o) unname(drop(Xt %*% coef(ft)) + o)
    e <- function(...) emm_est(emm_frame(emmeans::emmeans(ft, "b", mode = "count", ...)))
    expect_equal(e(), exp(lp(log(mean(dp$t)))), tolerance = 1e-10)
    expect_equal(e(at = list(t = 2)), exp(lp(log(2))), tolerance = 1e-10)
    expect_equal(e(offset = log(3)), exp(lp(log(3))), tolerance = 1e-10)
    mt <- function(o) { m <- exp(lp(o)); m / (1 - exp(-m)) }
    expect_equal(emm_est(emm_frame(emmeans::emmeans(ft, "b", mode = "response", offset = log(3)))), mt(log(3)),
                 tolerance = 1e-10)
})
