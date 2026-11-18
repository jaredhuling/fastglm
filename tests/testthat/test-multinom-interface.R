# Tests of fastglm_multinom() for formula and matrix input.

interface_data <- function() {
    d <- vglm_sim_data(3000L, 3L, 4L, seed = 5L)
    set.seed(105L)
    df <- data.frame(y = d$y, x1 = d$x[, "x1"], x2 = d$x[, "x2"],
                     x3 = d$x[, "x3"],
                     f = factor(sample(c("u", "v", "w"), 3000L, TRUE)))
    df
}

# (K - 1) by p matrix of coefficients, rows in the level order of the fit.
coef_matrix <- function(fit) matrix(fit$coefficients, length(fit$levels) - 1L)
se_matrix <- function(fit)
    matrix(sqrt(diag(fit$cov.unscaled)), length(fit$levels) - 1L)

nnet_fit <- function(f, dat) {
    # do.call embeds the formula and data in the stored call, so summary()
    # can re-evaluate it from any environment.
    do.call(nnet::multinom, list(formula = f, data = dat, maxit = 10000,
                                 reltol = 1e-14, abstol = 1e-14,
                                 MaxNWts = 1e6, trace = FALSE))
}

test_that("formula and matrix fits agree and the object has the documented fields", {
    df <- interface_data()
    ff <- fastglm_multinom(y ~ x1 + x2 + x3 + f, data = df, tol = 1e-12)
    X <- model.matrix(~ x1 + x2 + x3 + f, df)
    fm <- fastglm_multinom(X, df$y, tol = 1e-12)
    expect_s3_class(ff, c("fastglm_multinom", "fastglm_vglm"))
    expect_true(ff$converged)
    expect_max_diff(ff$coefficients, fm$coefficients, 1e-12)
    expect_max_diff(ff$cov.unscaled, fm$cov.unscaled, 1e-12)
    expect_max_diff(ff$loglik, fm$loglik, 1e-12)
    expect_identical(names(ff$coefficients)[1:3], c("B:(Intercept)", "C:(Intercept)", "D:(Intercept)"))
    expect_identical(names(ff$coefficients)[4:6], c("B:x1", "C:x1", "D:x1"))
    expect_identical(dimnames(ff$cov.unscaled), rep(list(names(ff$coefficients)), 2L))
    expect_identical(ff$levels, c("A", "B", "C", "D"))
    expect_identical(ff$reference, "A")
    expect_identical(ff$family, "multinomial")
    expect_identical(ff$nobs, 3000L)
    expect_equal(ff$df.residual, 3000 * 3 - length(ff$coefficients))
    expect_identical(dim(ff$fitted.values), c(3000L, 4L))
    expect_max_diff(rowSums(ff$fitted.values), rep(1, 3000L), 1e-12)
    expect_identical(ff$xlevels, list(f = c("u", "v", "w")))
    expect_false(ff$firth)
    expect_equal(ff$rank, length(ff$coefficients))
    expect_false(any(ff$aliased))
    expect_equal(ff$null.deviance,
                 -2 * sum(table(df$y) * log(table(df$y) / 3000)), tolerance = 1e-10)
    expect_identical(ff$call[[1L]], quote(fastglm::fastglm_multinom))
    for (nm in c("deviance", "iter", "passes", "convergence.reason", "constraints",
                 "terms", "contrasts", "y", "prior.weights", "offset"))
        expect_true(nm %in% names(ff), info = nm)
})

test_that("fit agrees with tightened nnet (n = 3000, K = 4, factor predictor)", {
    skip_if_not_installed("nnet")
    df <- interface_data()
    ff <- fastglm_multinom(y ~ x1 + x2 + x3 + f, data = df, tol = 1e-12)
    nn <- nnet_fit(y ~ x1 + x2 + x3 + f, df)
    expect_max_diff(ff$loglik, -nn$deviance / 2, 1e-8)
    expect_max_diff(coef_matrix(ff), coef(nn), 1e-6)
    expect_max_diff(se_matrix(ff), summary(nn)$standard.errors, 1e-6)
})

test_that("grouped and individual forms agree", {
    df <- interface_data()
    ind <- fastglm_multinom(y ~ x1 + x2 + x3, data = df, tol = 1e-13)
    # Group on a coarse covariate pattern.
    df$g <- cut(df$x1, c(-Inf, -1, 0, 1, Inf))
    ind2 <- fastglm_multinom(y ~ g, data = df, tol = 1e-13)
    tab <- table(df$g, df$y)
    gd <- data.frame(g = factor(rownames(tab), levels = levels(df$g)))
    gd$counts <- unclass(tab)[, , drop = FALSE]
    grp <- fastglm_multinom(counts ~ g, data = gd, tol = 1e-13)
    expect_max_diff(grp$coefficients, ind2$coefficients, 1e-10)
    expect_max_diff(grp$loglik, ind2$loglik, 1e-10)
    expect_identical(grp$levels, levels(df$y))
    # Matrix interface with a count matrix.
    X <- model.matrix(~ g, gd)
    grp2 <- fastglm_multinom(X, unclass(tab)[, , drop = FALSE], tol = 1e-13)
    expect_max_diff(grp2$coefficients, grp$coefficients, 1e-12)
    expect_true(ind$converged)
})

test_that("reference level by name and index matches nnet after releveling", {
    skip_if_not_installed("nnet")
    df <- interface_data()
    fc <- fastglm_multinom(y ~ x1 + x2 + x3 + f, data = df, reference = "C", tol = 1e-12)
    expect_identical(fc$reference, "C")
    expect_identical(names(fc$coefficients)[1:3],
                     c("A:(Intercept)", "B:(Intercept)", "D:(Intercept)"))
    df2 <- df
    df2$y <- relevel(df2$y, ref = "C")
    nn <- nnet_fit(y ~ x1 + x2 + x3 + f, df2)
    expect_max_diff(coef_matrix(fc), coef(nn), 1e-6)
    expect_max_diff(se_matrix(fc), summary(nn)$standard.errors, 1e-6)
    fi <- fastglm_multinom(y ~ x1 + x2 + x3 + f, data = df, reference = 3, tol = 1e-12)
    expect_max_diff(fi$coefficients, fc$coefficients, 1e-12)
})

test_that("offsets as a vector and as a matrix match VGAM", {
    skip_if_not_installed("VGAM")
    d <- vglm_sim_data(500L, 2L, 3L, seed = 9L)
    df <- data.frame(y = d$y, x1 = d$x[, "x1"], x2 = d$x[, "x2"])
    set.seed(3L)
    ov <- rnorm(500L, sd = 0.4)
    om <- matrix(rnorm(500L * 2L, sd = 0.4), 500L, 2L)
    ref <- function(off) {
        df$off <- off
        v <- suppressWarnings(VGAM::vglm(
            y ~ x1 + x2, VGAM::multinomial(refLevel = 1), data = df,
            offset = off, control = VGAM::vglm.control(epsilon = 1e-12, maxit = 200)))
        t(VGAM::coef(v, matrix = TRUE))
    }
    X <- d$x[, c("(Intercept)", "x1", "x2")]
    fv <- fastglm_multinom(X, d$y, offset = ov, tol = 1e-12)
    fm <- fastglm_multinom(X, d$y, offset = om, tol = 1e-12)
    expect_max_diff(coef_matrix(fv), ref(matrix(ov, 500L, 2L)), 1e-6)
    expect_max_diff(coef_matrix(fm), ref(om), 1e-6)
    # Formula offset (vector) gives the same fit as the matrix interface.
    ffo <- fastglm_multinom(y ~ x1 + x2, data = df, offset = ov, tol = 1e-12)
    expect_max_diff(ffo$coefficients, fv$coefficients, 1e-12)
    expect_identical(dim(fv$offset), c(500L, 2L))
})

test_that("weights match nnet with weights", {
    skip_if_not_installed("nnet")
    df <- interface_data()
    set.seed(8L)
    df$w <- sample(1:3, nrow(df), TRUE)
    ff <- fastglm_multinom(y ~ x1 + x2 + x3 + f, data = df, weights = w, tol = 1e-12)
    nn <- nnet::multinom(y ~ x1 + x2 + x3 + f, data = df, weights = w,
                         maxit = 10000, reltol = 1e-14, abstol = 1e-14,
                         MaxNWts = 1e6, trace = FALSE)
    expect_max_diff(coef_matrix(ff), coef(nn), 1e-6)
    expect_max_diff(se_matrix(ff), summary(nn)$standard.errors, 1e-6)
    expect_max_diff(ff$loglik, -nn$deviance / 2, 1e-8)
})

test_that("rows with missing values are removed and nobs counts the rest", {
    df <- interface_data()
    df$x1[c(5, 40)] <- NA
    df$y[c(7, 40, 99)] <- NA
    complete <- complete.cases(df)
    ff <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    expect_identical(ff$nobs, sum(complete))
    expect_identical(nrow(ff$fitted.values), sum(complete))
    ref <- fastglm_multinom(y ~ x1 + x2 + f, data = df[complete, ], tol = 1e-12)
    expect_max_diff(ff$coefficients, ref$coefficients, 1e-12)
    expect_false(is.null(ff$na.action))
    # The matrix interface rejects missing values rather than dropping them.
    X <- model.matrix(~ x2, df)
    yy <- df$y
    expect_error(fastglm_multinom(X, yy), "missing")
})

test_that("an empty category errors with its name", {
    df <- interface_data()
    df$y <- factor(df$y, levels = c(levels(df$y), "Z"))
    expect_error(fastglm_multinom(y ~ x1, data = df),
                 "category 'Z' has no observations")
    cm <- vglm_indicator(df$y)
    expect_error(fastglm_multinom(cbind("(Intercept)" = 1, x1 = df$x1), cm),
                 "category 'Z' has no observations")
    # A level emptied by subset is also named.
    df$y <- droplevels(df$y)
    expect_error(fastglm_multinom(y ~ x1, data = df, subset = y != "B"),
                 "category 'B' has no observations")
    # A level held only by zero-weight rows is empty.
    w <- as.numeric(df$y != "C")
    expect_error(fastglm_multinom(y ~ x1, data = df, weights = w),
                 "category 'C' has no observations")
})

test_that("separation warns without bias reduction and not with it", {
    s <- vglm_separated_data()
    expect_warning(fit <- fastglm_multinom(s$x, s$y),
                   "fastglm: fitted probabilities numerically 0 or 1 occurred; consider firth = TRUE",
                   fixed = TRUE)
    expect_true(fit$separation)
    expect_gt(max(abs(fit$coefficients)), 10)
    expect_no_warning(fb <- fastglm_multinom(s$x, s$y, firth = TRUE))
    expect_true(fb$firth)
    expect_true(fb$converged)
    expect_false(fb$separation)  # finite estimates: no fitted probability at 0 or 1
    expect_lt(max(abs(fb$coefficients)), 10)
})

test_that("small fitted probabilities without separation do not trigger the separation flag", {
    # A wide, non-separated fit in which an offset of -26 on category D for
    # five rows gives fitted probabilities near 1e-12: below the old 1e-10
    # threshold, above glm.fit's 10 * .Machine$double.eps.
    d <- vglm_sim_data(n = 800L, p = 20L, K = 4L, seed = 31L)
    off <- matrix(0, nrow(d$x), 3L)
    off[1:5, 3L] <- -26
    d$y[1:5] <- "A"
    expect_no_warning(fit <- fastglm_multinom(d$x, d$y, offset = off))
    pmin <- min(fit$fitted.values)
    expect_lt(pmin, 1e-10)
    expect_gt(pmin, 1e-14)
    expect_false(fit$separation)
    expect_true(fit$converged)

    s <- vglm_separated_data()
    expect_warning(sep <- fastglm_multinom(s$x, s$y), "numerically 0 or 1")
    expect_true(sep$separation)
})

test_that("bias-reduced fit agrees with brglm2", {
    skip_if_not_installed("brglm2")
    br_fit <- function(df)
        suppressWarnings(brglm2::brmultinom(
            y ~ x1 + x2, data = df, type = "AS_mean", ref = 1,
            epsilon = 1e-12, maxit = 1000))
    # Away from separation the agreement is to 1e-6.
    d <- vglm_sim_data(150L, 2L, 3L, seed = 12L)
    df <- data.frame(y = d$y, x1 = d$x[, "x1"], x2 = d$x[, "x2"])
    fb <- fastglm_multinom(y ~ x1 + x2, data = df, firth = TRUE, tol = 1e-12)
    expect_max_diff(coef_matrix(fb), coef(br_fit(df), matrix = TRUE), 1e-6)
    # Near separation the bias-reduced iteration converges linearly, so the
    # stopped coefficients are within about 3e-6 at tol = 1e-12.
    s <- vglm_separated_data()
    df <- data.frame(y = s$y, x1 = s$x[, "x1"], x2 = s$x[, "x2"])
    fb <- fastglm_multinom(y ~ x1 + x2, data = df, firth = TRUE, tol = 1e-12)
    expect_max_diff(coef_matrix(fb), coef(br_fit(df), matrix = TRUE), 1e-5)
})

test_that("input validation gives clear messages", {
    d <- vglm_sim_data(100L, 2L, 3L, seed = 2L)
    expect_error(fastglm_multinom(d$x, d$y, weights = c(-1, rep(1, 99))), "weights")
    expect_error(fastglm_multinom(d$x, d$y, weights = c(NA, rep(1, 99))), "weights")
    expect_error(fastglm_multinom(d$x, d$y, weights = rep(1, 99)), "weights")
    expect_error(fastglm_multinom(d$x, d$y, reference = "Q"), "reference")
    expect_error(fastglm_multinom(d$x, d$y, reference = 4), "reference")
    expect_error(fastglm_multinom(d$x, d$y, offset = matrix(0, 100, 3)), "offset")
    expect_error(fastglm_multinom(d$x, d$y, offset = rep(0, 5)), "offset")
    cm <- d$counts
    cm[3, 2] <- -1
    expect_error(fastglm_multinom(d$x, cm), "counts")
    cm[3, 2] <- Inf
    expect_error(fastglm_multinom(d$x, cm), "counts")
    cm[3, 2] <- NaN
    expect_error(fastglm_multinom(d$x, cm), "counts")
    expect_error(fastglm_multinom(d$x, d$y[-1]), "rows")
    xx <- d$x; xx[2, 2] <- Inf
    expect_error(fastglm_multinom(xx, d$y), "finite")
})

test_that("zero-weight rows are not counted in nobs and do not produce NaN", {
    d <- vglm_sim_data(300L, 2L, 3L, seed = 4L)
    w <- rep(1, 300L); w[1:20] <- 0
    ff <- fastglm_multinom(d$x, d$y, weights = w, tol = 1e-12)
    expect_identical(ff$nobs, 280L)
    expect_false(anyNA(ff$coefficients))
    ref <- fastglm_multinom(d$x[-(1:20), ], d$y[-(1:20)], tol = 1e-12)
    expect_max_diff(ff$coefficients, ref$coefficients, 1e-10)
    expect_max_diff(ff$null.deviance, ref$null.deviance, 1e-10)
})

test_that("engine errors under bias reduction are rethrown with a clear prefix", {
    d <- vglm_sim_data(50L, 1L, 3L, seed = 6L)
    local_mocked_bindings(
        vglm_fit_dense = function(...) stop("the information is singular at the starting values"),
        .package = "fastglm")
    expect_error(fastglm_multinom(d$x, d$y, firth = TRUE), "^fastglm_multinom: ")
})

test_that("the formula can be supplied by name and the stored call works with update()", {
    df <- interface_data()
    pos <- fastglm_multinom(y ~ x1 + x2, data = df, tol = 1e-12)
    nam <- fastglm_multinom(formula = y ~ x1 + x2, data = df, tol = 1e-12)
    expect_max_diff(nam$coefficients, pos$coefficients, 1e-12)
    expect_identical(nam$call[[1L]], quote(fastglm::fastglm_multinom))
    up <- update(nam, . ~ . + x3)
    expect_s3_class(up, "fastglm_multinom")
    expect_identical(names(up$coefficients)[10:12], c("B:x3", "C:x3", "D:x3"))
    expect_max_diff(update(nam, tol = 1e-12)$coefficients, pos$coefficients, 1e-12)
    expect_error(fastglm_multinom(data = df), "supply a design matrix or model formula")
})

test_that("a matrix offset works through the formula method", {
    d <- vglm_sim_data(300L, 2L, 3L, seed = 14L)
    df <- data.frame(y = d$y, x1 = d$x[, "x1"], x2 = d$x[, "x2"])
    set.seed(15L)
    om <- matrix(rnorm(300L * 2L, sd = 0.4), 300L, 2L)
    ff <- fastglm_multinom(y ~ x1 + x2, data = df, offset = om, tol = 1e-12)
    fm <- fastglm_multinom(d$x, d$y, offset = om, tol = 1e-12)
    expect_max_diff(ff$coefficients, fm$coefficients, 1e-12)
    expect_max_diff(ff$offset, om, 1e-14)
    # The offset matters: the fit differs from the one without it.
    f0 <- fastglm_multinom(y ~ x1 + x2, data = df, tol = 1e-12)
    expect_gt(max(abs(ff$coefficients - f0$coefficients)), 1e-3)
})
