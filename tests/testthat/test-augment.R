## broom augment() methods: columns equal predict() and residuals(), agree with
## broom::augment.glm() for dense GLMs, and refuse streamed and big.matrix fits.

aug_data <- function(n = 120L, seed = 31L) {
    set.seed(seed)
    d <- data.frame(x = rnorm(n), g = factor(sample(c("a", "b", "c"), n, TRUE)))
    d$yb <- rbinom(n, 1, plogis(0.4 * d$x))
    d$yc <- rpois(n, exp(0.2 + 0.3 * d$x)) * rbinom(n, 1, 0.8)
    d$yp <- d$yc + 1L
    d$yk <- factor(sample(c("u", "v", "w"), n, TRUE))
    d
}

test_that("formula and matrix GLM fits agree with broom::augment.glm", {
    skip_if_not_installed("broom")
    d <- aug_data()
    for (fam in list(binomial(), poisson())) {
        yv <- if (identical(fam$family, "binomial")) "yb" else "yc"
        fo <- stats::reformulate(c("x", "g"), yv)
        f <- fastglm(fo, data = d, family = fam, tol = 1e-12)
        g <- glm(fo, data = d, family = fam, control = glm.control(epsilon = 1e-12, maxit = 100))
        for (tp in c("link", "response")) for (tr in c("deviance", "pearson")) {
            a <- suppressWarnings(broom::augment(f, type.predict = tp, type.residuals = tr, se_fit = TRUE))
            b <- broom::augment(g, type.predict = tp, type.residuals = tr, se_fit = TRUE)
            expect_equal(a$.fitted, b$.fitted, tolerance = 1e-8)
            expect_equal(a$.se.fit, b$.se.fit, tolerance = 1e-6)
            expect_equal(a$.resid, b$.resid, tolerance = 1e-8)
            expect_equal(a$.hat, b$.hat, tolerance = 1e-6)
            expect_equal(a$.sigma, b$.sigma, tolerance = 1e-6)
            expect_identical(names(a), names(b)[names(b) %in% names(a)])
            expect_s3_class(a, "tbl_df")
            expect_equal(a$.std.resid, b$.std.resid, tolerance = 1e-6)
            expect_equal(a$.cooksd, b$.cooksd, tolerance = 1e-6)
            expect_equal(a$x, d$x)
        }
        expect_identical(names(a)[1:3], c(yv, "x", "g"))
    }
    ## gaussian with an estimated dispersion
    d$yg <- d$x + rnorm(nrow(d))
    f <- fastglm(yg ~ x + g, data = d); g <- glm(yg ~ x + g, data = d)
    a <- broom::augment(f); b <- broom::augment(g)
    expect_equal(a$.std.resid, b$.std.resid, tolerance = 1e-6)
    expect_equal(a$.cooksd, b$.cooksd, tolerance = 1e-6)
    expect_equal(a$.sigma, b$.sigma, tolerance = 1e-6)
    ## matrix interface (class "fastglm") matches predict() and residuals()
    X <- model.matrix(~ x + g, d)
    fm <- fastglm(X, d$yb, family = binomial())
    a <- broom::augment(fm, se_fit = TRUE)
    expect_equal(a$.fitted, unname(predict(fm)))
    expect_equal(a$.se.fit, unname(predict(fm, se.fit = TRUE)$se.fit))
    expect_equal(a$.resid, unname(residuals(fm)))
    expect_true(all(c("y", "x", "gb", "gc") %in% names(a)))
    expect_false("(Intercept)" %in% names(a))
})

test_that("GLM augment supports newdata, data, and NA handling", {
    skip_if_not_installed("broom")
    d <- aug_data(); d$x[c(4, 9)] <- NA
    nd <- data.frame(x = c(-1, 0, 1), g = factor(c("a", "b", "c"), levels = levels(d$g)))
    f <- fastglm(yb ~ x + g, data = d, family = binomial())
    a <- broom::augment(f, newdata = nd, type.predict = "response", se_fit = TRUE)
    p <- predict(f, newdata = nd, type = "response", se.fit = TRUE)
    expect_equal(a$.fitted, unname(p$fit)); expect_equal(a$.se.fit, unname(p$se.fit))
    expect_false(any(c(".resid", ".hat") %in% names(a)))
    expect_error(broom::augment(f, data = d, newdata = nd), "either")
    ## na.omit: model frame rows, and full data with the omitted rows dropped
    a1 <- broom::augment(f); a2 <- broom::augment(f, data = d)
    expect_equal(nrow(a1), sum(complete.cases(d))); expect_equal(a1$.fitted, a2$.fitted)
    expect_equal(a1$.resid, unname(residuals(f)))
    ## na.exclude: rows restored with missing predictions
    fe <- fastglm(yb ~ x + g, data = d, family = binomial(), na.action = na.exclude)
    ae <- broom::augment(fe, data = d)
    expect_equal(nrow(ae), nrow(d)); expect_true(all(is.na(ae$.fitted[c(4, 9)])))
    expect_equal(ae$.fitted[-c(4, 9)], a1$.fitted)
    ae0 <- broom::augment(fe)
    expect_equal(nrow(ae0), nrow(d)); expect_true(".hat" %in% names(ae0)); expect_true(all(is.na(ae0$.hat[c(4, 9)])))
    expect_error(broom::augment(f, data = d[1:50, ]), "do not match")
    expect_error(broom::augment(f, data = transform(d, .fitted = 1)), "already contain")
})

test_that("GLM augment refuses streamed and big.matrix fits", {
    skip_if_not_installed("broom")
    d <- aug_data(60L)
    chunks <- function(k) d[((k - 1) * 20 + 1):(k * 20), ]
    src <- fastglm_source(chunks, 3, formula = yb ~ x + g, prototype = d)
    fs <- fastglm_streaming(src, family = binomial())
    expect_error(broom::augment(fs), "predict\\(\\)")
    an <- broom::augment(fs, newdata = d[1:5, ], type.predict = "response")
    expect_equal(an$.fitted, unname(predict(fs, newdata = d[1:5, ], type = "response")))
    expect_false(".resid" %in% names(an))
    skip_if_not_installed("bigmemory")
    X <- bigmemory::as.big.matrix(model.matrix(~ x + g, d))
    fb <- fastglm(X, d$yb, family = binomial(), method = 2L)
    expect_error(broom::augment(fb), "predict\\(\\)")
    an <- broom::augment(fb, newdata = model.matrix(~ x + g, d)[1:4, ])
    expect_equal(an$.fitted, unname(predict(fb, newdata = model.matrix(~ x + g, d)[1:4, ])))
})

test_that("negative binomial augment matches predict() and residuals()", {
    skip_if_not_installed("broom")
    d <- aug_data(); set.seed(5); d$yn <- MASS::rnegbin(nrow(d), exp(0.5 + 0.3 * d$x), 2)
    f <- fastglm_nb(yn ~ x + g, data = d)
    a <- broom::augment(f, type.predict = "response", se_fit = TRUE)
    expect_equal(a$.fitted, unname(predict(f, type = "response")))
    expect_equal(a$.se.fit, unname(predict(f, type = "response", se.fit = TRUE)$se.fit))
    expect_equal(a$.resid, unname(residuals(f)))
    fm <- MASS::glm.nb(yn ~ x + g, data = d)
    expect_equal(a$.fitted, unname(fitted(fm)), tolerance = 1e-4)
})

test_that("count-model augment matches predict() and residuals() and pscl", {
    skip_if_not_installed("broom"); skip_if_not_installed("pscl")
    d <- aug_data(); d$x[7] <- NA
    nd <- data.frame(x = c(-0.5, 0.7), g = factor(c("a", "c"), levels = levels(d$g)))
    models <- list(hurdle = fastglm_hurdle(yc ~ x + g, data = d),
                   zi = fastglm_zi(yc ~ x + g, data = d),
                   hurdle_nb = fastglm_hurdle(yc ~ x, data = d, dist = "negbin"))
    for (nm in names(models)) {
        f <- models[[nm]]
        a <- broom::augment(f, se_fit = TRUE)
        p <- predict(f, se.fit = TRUE)
        expect_equal(a$.fitted, unname(p$fit)); expect_equal(a$.se.fit, unname(p$se.fit))
        expect_equal(a$.resid, unname(residuals(f, type = "pearson")))
        expect_equal(nrow(a), nobs(f))
        ar <- broom::augment(f, type.residuals = "response")
        expect_equal(ar$.resid, unname(residuals(f, type = "response")))
        an <- broom::augment(f, newdata = nd, type.predict = "zero")
        expect_equal(an$.fitted, unname(predict(f, newdata = nd, type = "zero")))
        expect_false(".resid" %in% names(an))
        al <- broom::augment(f, type.predict = "link")
        expect_equal(dim(al$.fitted), c(nobs(f), 2L))
    }
    ## pscl reference for the Poisson hurdle and zero-inflated models
    ph <- pscl::hurdle(yc ~ x + g, data = d); pz <- pscl::zeroinfl(yc ~ x + g, data = d)
    ah <- broom::augment(models$hurdle); az <- broom::augment(models$zi)
    expect_equal(ah$.fitted, unname(fitted(ph)), tolerance = 1e-4)
    expect_equal(ah$.resid, unname(residuals(ph, "pearson")), tolerance = 1e-4)
    expect_equal(az$.fitted, unname(fitted(pz)), tolerance = 1e-4)
    expect_equal(az$.resid, unname(residuals(pz, "pearson")), tolerance = 1e-3)
    ap <- broom::augment(models$hurdle, type.predict = "prob", at = 0:2)
    expect_equal(dim(ap$.fitted), c(nobs(models$hurdle), 3L))
})

test_that("truncated augment matches predict() and residuals()", {
    skip_if_not_installed("broom")
    d <- aug_data()
    f <- fastglm_truncated(yp ~ x + g, data = d)
    a <- broom::augment(f, se_fit = TRUE)
    p <- predict(f, se.fit = TRUE)
    expect_equal(a$.fitted, unname(p$fit)); expect_equal(a$.se.fit, unname(p$se.fit))
    expect_equal(a$.resid, unname(residuals(f, type = "pearson")))
    an <- broom::augment(f, newdata = d[1:4, ], type.predict = "count")
    expect_equal(an$.fitted, unname(predict(f, newdata = d[1:4, ], type = "count")))
})

test_that("multinomial augment adds probability, class, and residual columns", {
    skip_if_not_installed("broom"); skip_if_not_installed("nnet")
    d <- aug_data(); d$x[3] <- NA
    f <- fastglm_multinom(yk ~ x + g, data = d)
    lv <- f$levels
    a <- broom::augment(f, se_fit = TRUE)
    p <- predict(f, se.fit = TRUE); r <- residuals(f)
    for (l in lv) {
        expect_equal(a[[paste0(".prob_", l)]], unname(p$fit[, l]))
        expect_equal(a[[paste0(".se.fit_", l)]], unname(p$se.fit[, l]))
        expect_equal(a[[paste0(".resid_", l)]], unname(r[, l]))
    }
    expect_identical(as.character(a$.class), as.character(unname(predict(f, type = "class"))))
    expect_equal(nrow(a), nobs(f))
    ref <- nnet::multinom(yk ~ x + g, data = d, trace = FALSE)
    expect_equal(as.matrix(a[paste0(".prob_", lv)]), unname(predict(ref, type = "probs")),
                 tolerance = 1e-3, ignore_attr = TRUE)
    nd <- d[1:5, ]; nd$x[is.na(nd$x)] <- 0
    an <- broom::augment(f, newdata = nd, type.predict = "link")
    expect_equal(an[[paste0(".link_", f$levels[2])]],
                 unname(predict(f, newdata = nd, type = "link")[, f$levels[2]]))
    expect_false(any(grepl("^\\.resid", names(an))))
    ## fits from a design matrix work without data
    X <- model.matrix(~ x + g, d[-3, ])
    fm <- fastglm_multinom(X, d$yk[-3])
    expect_equal(nrow(broom::augment(fm)), nrow(X))
})

test_that("augment returns tibbles with .rownames as broom does", {
    skip_if_not_installed("broom")
    f <- fastglm(mpg ~ wt, data = mtcars); g <- glm(mpg ~ wt, data = mtcars)
    a <- broom::augment(f); b <- broom::augment(g)
    expect_s3_class(a, "tbl_df")
    expect_identical(a$.rownames, rownames(mtcars))
    expect_identical(names(a), names(b))
    expect_equal(a$.sigma, b$.sigma, tolerance = 1e-6)
    expect_false(".rownames" %in% names(broom::augment(fastglm(yb ~ x, data = aug_data(), family = binomial()))))
    ## row names of omitted-row model frames follow the original rows
    d <- mtcars; d$wt[3] <- NA
    expect_identical(broom::augment(fastglm(mpg ~ wt, data = d))$.rownames, rownames(d)[-3])
    fe <- fastglm(mpg ~ wt, data = d, na.action = na.exclude)
    expect_identical(broom::augment(fe)$.rownames, rownames(d))
})

test_that("influence columns handle zero weights and unit leverage", {
    skip_if_not_installed("broom")
    d <- aug_data(60L); w <- rep(1, 60); w[c(2, 5)] <- 0
    f <- fastglm(yb ~ x, data = d, family = binomial(), weights = w, tol = 1e-12)
    a <- broom::augment(f)
    expect_true(all(is.na(a$.std.resid[c(2, 5)])) && all(is.na(a$.cooksd[c(2, 5)])))
    g <- glm(yb ~ x, data = d, family = binomial(), weights = w, control = glm.control(epsilon = 1e-12))
    live <- w > 0
    expect_equal(a$.std.resid[live], unname(rstandard(g)), tolerance = 1e-6)
    expect_equal(a$.cooksd[live], unname(cooks.distance(g)), tolerance = 1e-6)
    expect_equal(a$.sigma[live], unname(influence(g, do.coef = FALSE)$sigma), tolerance = 1e-6)
    ## a level with one observation has leverage one: NaN, not Inf
    d1 <- data.frame(y = c(1.1, 2.3, 0.7, 3.1, 2.2, 5), g = factor(c("a", "a", "a", "b", "b", "c")))
    a1 <- broom::augment(fastglm(y ~ g, data = d1))
    g1 <- glm(y ~ g, data = d1)
    expect_equal(is.nan(a1$.std.resid), is.nan(unname(rstandard(g1))))
    expect_false(any(is.infinite(a1$.cooksd)))
})

test_that("matrix-interface count fits and sparse designs augment without densifying", {
    skip_if_not_installed("broom"); skip_if_not_installed("Matrix")
    d <- aug_data()
    X <- model.matrix(~ x + g, d)
    for (fit in list(fastglm_hurdle(X, response = d$yc), fastglm_zi(X, response = d$yc),
                     fastglm_hurdle(Matrix::Matrix(X, sparse = TRUE), response = d$yc))) {
        a <- broom::augment(fit)
        expect_equal(nrow(a), nrow(X)); expect_true("y" %in% names(a))
        expect_equal(a$.fitted, unname(predict(fit)))
        expect_equal(a$.resid, unname(residuals(fit, type = "pearson")))
    }
    ## sparse GLM, nb, and multinomial fits: the frame holds only the response
    Xs <- Matrix::Matrix(X, sparse = TRUE)
    fs <- fastglm(Xs, d$yb, family = binomial(), method = 2L)
    a <- broom::augment(fs)
    expect_identical(names(a)[1], "y"); expect_false("x" %in% names(a))
    expect_equal(a$.fitted, unname(predict(fs)))
    ad <- broom::augment(fs, data = d); expect_true("x" %in% names(ad))
    fm <- fastglm_multinom(Xs, d$yk); expect_false("x" %in% names(broom::augment(fm)))
    ## without a stored response the frame is a row index
    f0 <- fs; f0$y <- NULL
    expect_identical(names(broom::augment(f0, newdata = NULL))[1], "row")
    ## a design whose dense form would be huge is never densified (1e6 x 1e4 sparse)
    big <- list(x = Matrix::sparseMatrix(i = 1:10, j = 1:10, x = 1, dims = c(1e6, 1e4)),
                y = rep(0, 1e6), na.action = NULL)
    fr <- .augment_frame(big, NULL, NULL, 1e6)
    expect_identical(names(fr), "y"); expect_equal(nrow(fr), 1e6)
    ## a dense design above the cell limit gets the same treatment
    dense <- list(x = matrix(0, 2e4, 600), y = rep(0, 2e4), na.action = NULL)
    expect_identical(names(.augment_frame(dense, NULL, NULL, 2e4)), "y")
})

test_that("weighted Pearson residuals of count models match pscl", {
    skip_if_not_installed("pscl")
    d <- aug_data(150L); d$w <- sample(1:3, nrow(d), TRUE); d$yp <- d$yc + 1L
    h <- fastglm_hurdle(yc ~ x, data = d, weights = w, tol = 1e-12)
    ph <- pscl::hurdle(yc ~ x, data = d, weights = w, control = pscl::hurdle.control(reltol = 1e-14))
    z <- fastglm_zi(yc ~ x, data = d, weights = w, tol = 1e-12)
    pz <- pscl::zeroinfl(yc ~ x, data = d, weights = w, control = pscl::zeroinfl.control(reltol = 1e-14))
    expect_equal(unname(residuals(h, "pearson")), unname(residuals(ph, "pearson")), tolerance = 1e-5)
    expect_equal(unname(residuals(z, "pearson")), unname(residuals(pz, "pearson")), tolerance = 1e-5)
    ## the weight factor itself is exact: weights 1 versus w on the same fit
    r1 <- residuals(h, "pearson"); h$weights[] <- 1
    expect_equal(unname(residuals(h, "pearson")) * sqrt(d$w), unname(r1), tolerance = 1e-12)
    ## truncated fits carry the factor too
    t <- fastglm_truncated(yp ~ x, data = d, weights = w)
    rt <- residuals(t, "pearson"); t$prior.weights[] <- 1
    expect_equal(unname(residuals(t, "pearson")) * sqrt(d$w), unname(rt), tolerance = 1e-12)
    expect_equal(unname(residuals(t, "response")), unname(d$yp - fitted(t)), tolerance = 1e-10)
})

test_that("ordinal augment keeps its matrix columns and adds .prob_<level> columns", {
    skip_if_not_installed("broom")
    d <- aug_data(); d$o <- ordered(d$yk)
    f <- fastglm_ordinal(o ~ x + g, d)
    a <- broom::augment(f, se_fit = TRUE)
    expect_true(is.matrix(a$.fitted)); expect_s3_class(a$.class, "ordered")
    for (l in f$levels) expect_equal(a[[paste0(".prob_", l)]], unname(predict(f)[, l]))
    expect_equal(as.matrix(a$.fitted), predict(f), ignore_attr = TRUE)
})

test_that("multinomial na.exclude with full data aligns the rows", {
    skip_if_not_installed("broom")
    d <- aug_data(); d$x[c(4, 9)] <- NA
    f <- fastglm_multinom(yk ~ x + g, data = d, na.action = na.exclude)
    a <- broom::augment(f, data = d)
    expect_equal(nrow(a), sum(complete.cases(d)))
})
