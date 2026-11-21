# Tests of the standard S3 methods for multinomial fits.

methods_data <- function(n = 1500L) {
    d <- vglm_sim_data(n, 2L, 4L, seed = 8L)
    set.seed(108L)
    data.frame(y = d$y, x1 = d$x[, "x1"], x2 = d$x[, "x2"],
               f = factor(sample(c("u", "v", "w"), n, TRUE)))
}

methods_nnet <- function(f, dat)
    do.call(nnet::multinom, list(formula = f, data = dat, maxit = 10000,
                                 reltol = 1e-14, abstol = 1e-14,
                                 MaxNWts = 1e6, trace = FALSE))

test_that("predict agrees with nnet for training data, newdata, and subsets of levels", {
    skip_if_not_installed("nnet")
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    nn <- methods_nnet(y ~ x1 + x2 + f, df)
    expect_max_diff(predict(fit), predict(nn, type = "probs"), 1e-6)
    expect_max_diff(predict(fit, type = "probability"), fit$fitted.values, 1e-10)
    nd <- data.frame(x1 = c(-1, 0.3, 2), x2 = c(0.5, -0.2, 1),
                     f = c("u", "w", "v"))
    pf <- predict(fit, nd)
    expect_identical(dimnames(pf), list(c("1", "2", "3"), fit$levels))
    expect_max_diff(pf, predict(nn, nd, type = "probs"), 1e-6)
    expect_identical(as.character(predict(fit, nd, type = "class")),
                     as.character(predict(nn, nd, type = "class")))
    expect_identical(as.character(predict(fit, type = "class")),
                     as.character(predict(nn, type = "class")))
    expect_s3_class(predict(fit, nd, type = "class"), "factor")
    expect_identical(levels(predict(fit, nd, type = "class")), fit$levels)
    # Review Focus 3: a subset of the factor levels, no response column.
    sub <- data.frame(x1 = c(0.1, -0.4), x2 = c(0, 1), f = factor(c("w", "w")))
    expect_max_diff(predict(fit, sub),
                    predict(nn, data.frame(x1 = sub$x1, x2 = sub$x2, f = c("w", "w")),
                            type = "probs"), 1e-6)
    expect_max_diff(rowSums(predict(fit, sub)), c(1, 1), 1e-12)
})

test_that("link predictions are the log odds against the reference", {
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    nd <- df[1:20, ]
    eta <- predict(fit, nd, type = "link")
    p <- predict(fit, nd)
    expect_identical(colnames(eta), c("B", "C", "D"))
    expect_max_diff(eta, log(p[, -1] / p[, 1]), 1e-10)
})

test_that("delta-method standard errors match numDeriv times vcov", {
    skip_if_not_installed("numDeriv")
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    nd <- data.frame(x1 = c(-1, 0.3), x2 = c(0.5, -0.2), f = c("u", "w"))
    pr <- predict(fit, nd, se.fit = TRUE)
    X <- model.matrix(~ x1 + x2 + f, transform(nd, f = factor(f, levels = c("u", "v", "w"))))
    K <- length(fit$levels); M <- K - 1L
    probfun <- function(beta, xrow) {
        eta <- c(0, as.vector(matrix(beta, M) %*% xrow))
        e <- exp(eta - max(eta)); e / sum(e)
    }
    V <- vcov(fit)
    se <- t(sapply(1:2, function(i) {
        J <- numDeriv::jacobian(probfun, fit$coefficients, xrow = X[i, ],
                                method.args = list(eps = 1e-6, d = 1e-4, r = 6))
        sqrt(diag(J %*% V %*% t(J)))
    }))
    expect_identical(names(pr), c("fit", "se.fit"))
    expect_identical(dim(pr$se.fit), c(2L, K))
    expect_max_diff(pr$se.fit, se, 1e-6)
    expect_max_diff(pr$fit, predict(fit, nd), 1e-12)
    lk <- predict(fit, nd, type = "link", se.fit = TRUE)
    selink <- t(sapply(1:2, function(i) {
        J <- kronecker(t(X[i, ]), diag(M))
        sqrt(diag(J %*% V %*% t(J)))
    }))
    expect_max_diff(lk$se.fit, selink, 1e-10)
    expect_error(predict(fit, nd, type = "class", se.fit = TRUE), "se.fit")
})

test_that("information criteria, logLik, and coefficient layout match nnet", {
    skip_if_not_installed("nnet")
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    nn <- methods_nnet(y ~ x1 + x2 + f, df)
    expect_max_diff(AIC(fit), AIC(nn), 1e-8)
    expect_max_diff(BIC(fit), BIC(nn), 1e-8)
    ll <- logLik(fit)
    expect_identical(attr(ll, "df"), fit$rank)
    expect_identical(attr(ll, "nobs"), nrow(df))
    expect_identical(nobs(fit), nrow(df))
    expect_max_diff(deviance(fit), nn$deviance, 1e-8)
    cm <- coef(fit, matrix = TRUE)
    expect_identical(dim(cm), dim(coef(nn)))
    expect_identical(dimnames(cm), dimnames(coef(nn)))
    expect_max_diff(cm, coef(nn), 1e-6)
    expect_identical(coef(fit), fit$coefficients)
    expect_max_diff(sqrt(diag(vcov(fit))), as.vector(summary(nn)$standard.errors), 1e-6)
    expect_identical(dimnames(vcov(fit)), rep(list(names(coef(fit))), 2L))
})

test_that("fitted, residuals, model.matrix, and confint behave", {
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    expect_identical(fitted(fit), fit$fitted.values)
    r <- residuals(fit)
    expect_identical(dim(r), c(nrow(df), 4L))
    expect_max_diff(r, vglm_indicator(df$y) - fit$fitted.values, 1e-12)
    rp <- residuals(fit, type = "pearson")
    p <- fit$fitted.values
    expect_max_diff(rp, (vglm_indicator(df$y) - p) / sqrt(p * (1 - p)), 1e-10)
    expect_identical(model.matrix(fit), fit$x)
    ci <- confint(fit)
    se <- sqrt(diag(vcov(fit)))
    expect_identical(rownames(ci), names(coef(fit)))
    expect_max_diff(ci[, 1], coef(fit) - qnorm(0.975) * se, 1e-12)
    expect_max_diff(ci[, 2], coef(fit) + qnorm(0.975) * se, 1e-12)
    ci2 <- confint(fit, parm = "B:x1", level = 0.9)
    expect_identical(dim(ci2), c(1L, 2L))
    expect_identical(colnames(ci2), c("5 %", "95 %"))
})

test_that("tidy and glance give the documented columns", {
    skip_if_not_installed("broom")
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df, tol = 1e-12)
    td <- broom::tidy(fit)
    expect_identical(names(td), c("term", "y.level", "estimate", "std.error",
                                  "statistic", "p.value"))
    expect_identical(nrow(td), length(coef(fit)))
    expect_identical(td$y.level[1:3], c("B", "C", "D"))
    expect_identical(td$term[1:4], c("(Intercept)", "(Intercept)", "(Intercept)", "x1"))
    expect_max_diff(td$estimate, coef(fit), 1e-12)
    expect_max_diff(td$statistic, td$estimate / td$std.error, 1e-12)
    expect_max_diff(td$p.value, 2 * pnorm(-abs(td$statistic)), 1e-12)
    tc <- broom::tidy(fit, conf.int = TRUE, exponentiate = TRUE)
    expect_true(all(c("conf.low", "conf.high") %in% names(tc)))
    expect_max_diff(tc$estimate, exp(coef(fit)), 1e-12)
    gl <- broom::glance(fit)
    expect_identical(nrow(gl), 1L)
    expect_true(all(c("nobs", "df", "logLik", "AIC", "BIC", "deviance",
                      "null.deviance", "df.residual") %in% names(gl)))
    expect_max_diff(gl$AIC, AIC(fit), 1e-12)
})

test_that("print and summary run and report convergence", {
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2 + f, data = df)
    out <- capture.output(print(fit))
    expect_true(any(grepl("^Call:", out)))
    expect_false(any(grepl("fastglm::", out, fixed = TRUE)))
    s <- summary(fit)
    expect_s3_class(s, "summary.fastglm_multinom")
    expect_identical(dim(s$coefficients), c(length(coef(fit)), 4L))
    expect_identical(colnames(s$coefficients),
                     c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
    so <- capture.output(print(s))
    expect_true(any(grepl("Convergence: converged", so, fixed = TRUE)))
    expect_true(any(grepl("AIC", so)))
    expect_true(any(grepl("Residual deviance", so)))
    expect_true(any(grepl("Reference", so)))
})

test_that("matrix fits predict from design matrices and aliased columns are handled", {
    d <- vglm_sim_data(400L, 2L, 3L, seed = 3L)
    fit <- fastglm_multinom(d$x, d$y, tol = 1e-12)
    expect_max_diff(predict(fit, d$x[1:5, ]), fit$fitted.values[1:5, ], 1e-12)
    expect_error(predict(fit, d$x[1:5, 1:2]), "columns")
    X2 <- cbind(d$x, dup = d$x[, "x1"])
    f2 <- fastglm_multinom(X2, d$y, tol = 1e-12)
    expect_true(any(f2$aliased))
    expect_max_diff(predict(f2, X2[1:5, ]), f2$fitted.values[1:5, ], 1e-10)
    pr <- predict(f2, X2[1:5, ], se.fit = TRUE)
    expect_false(anyNA(pr$se.fit))
    expect_max_diff(pr$se.fit, predict(fit, d$x[1:5, ], se.fit = TRUE)$se.fit, 1e-6)
})

test_that("grouped fits give proportion residuals and offsets enter predictions", {
    df <- methods_data()
    df$g <- cut(df$x1, c(-Inf, -1, 0, 1, Inf))
    tab <- table(df$g, df$y)
    gd <- data.frame(g = factor(rownames(tab), levels = levels(df$g)))
    gd$counts <- unclass(tab)[, , drop = FALSE]
    fit <- fastglm_multinom(counts ~ g, data = gd, tol = 1e-12)
    r <- residuals(fit)
    expect_max_diff(r, unclass(tab) / rowSums(tab) - fit$fitted.values, 1e-12)
    # Formula offset() terms are applied to newdata.
    fo <- fastglm_multinom(y ~ x1 + offset(x2), data = df, tol = 1e-12)
    nd <- data.frame(x1 = c(0, 1), x2 = c(2, -1))
    nd0 <- nd; nd0$x2 <- 0
    eta <- predict(fo, nd, type = "link")
    eta0 <- predict(fo, nd0, type = "link")
    expect_max_diff(eta - eta0, matrix(nd$x2, 2L, 3L), 1e-12)
})

test_that("an offset given through the fit's offset argument is replayed in predict", {
    df <- methods_data()
    set.seed(9L)
    df$e <- runif(nrow(df), 0.5, 2)
    fit <- fastglm_multinom(y ~ x1 + x2, data = df, offset = log(e), tol = 1e-12)
    idx <- c(3L, 10L, 77L, 500L)
    expect_max_diff(predict(fit, df[idx, ]), fit$fitted.values[idx, ], 1e-12)
    expect_max_diff(predict(fit, df[idx, ], type = "link"),
                    predict(fit, type = "link")[idx, ], 1e-12)
    # Combined with a formula offset() term.
    f2 <- fastglm_multinom(y ~ x1 + offset(x2), data = df, offset = log(e), tol = 1e-12)
    expect_max_diff(predict(f2, df[idx, ]), f2$fitted.values[idx, ], 1e-12)
    # An offset matrix, one column per non-reference category.
    om <- cbind(0.1 * df$x1, -0.2 * df$x2, 0.3)
    f3 <- fastglm_multinom(y ~ x1, data = df, offset = cbind(0.1 * x1, -0.2 * x2, 0.3),
                           tol = 1e-12)
    expect_max_diff(predict(f3, df[idx, ]), f3$fitted.values[idx, ], 1e-12)
    # An offset written against the fitting data has the wrong row count.
    f4 <- fastglm_multinom(y ~ x1, data = df, offset = df$e, tol = 1e-12)
    expect_error(predict(f4, df[idx, ]), "rows in newdata")
    # An explicit offset overrides.
    expect_max_diff(predict(fit, df[idx, ], offset = 0, type = "link"),
                    predict(fit, df[idx, ], type = "link") -
                        matrix(log(df$e[idx]), 4L, 3L), 1e-12)
})

test_that("matrix fits with a nonzero offset require an explicit prediction offset", {
    d <- vglm_sim_data(300L, 2L, 3L, seed = 4L)
    off <- rep(0.3, 300L)
    fit <- fastglm_multinom(d$x, d$y, offset = off, tol = 1e-12)
    expect_error(predict(fit, d$x[1:5, ]), "offset")
    expect_max_diff(predict(fit, d$x[1:5, ], offset = off[1:5]),
                    fit$fitted.values[1:5, ], 1e-12)
    expect_max_diff(predict(fit), fit$fitted.values, 1e-12)
})

test_that("summary standard errors are not masked by non-convergence", {
    df <- methods_data()
    fit <- suppressWarnings(fastglm_multinom(y ~ x1 + x2, data = df, maxit = 1L))
    expect_false(fit$converged)
    s <- summary(fit)
    expect_true(all(is.finite(s$coefficients[, "Std. Error"])))
    expect_max_diff(unname(s$coefficients[, "Std. Error"]), sqrt(diag(vcov(fit))), 1e-12)
    so <- capture.output(print(s))
    expect_true(any(grepl("Convergence: not converged", so, fixed = TRUE)))
})

test_that("print.summary notes separation unless firth is used", {
    d <- vglm_separated_data()
    fit <- suppressWarnings(fastglm_multinom(d$x, d$y))
    expect_true(fit$separation)
    so <- capture.output(print(summary(fit)))
    expect_true(any(grepl("numerically 0 or 1", so)))
    expect_true(any(grepl("firth = TRUE", so, fixed = TRUE)))
    ff <- fastglm_multinom(d$x, d$y, firth = TRUE, tol = 1e-10)
    expect_false(any(grepl("numerically 0 or 1", capture.output(print(summary(ff))))))
})

test_that("predict handles zero-row newdata", {
    df <- methods_data()
    fit <- fastglm_multinom(y ~ x1 + x2, data = df)
    p <- predict(fit, df[0L, ], se.fit = TRUE)
    expect_identical(dim(p$fit), c(0L, 4L))
    expect_identical(dim(p$se.fit), c(0L, 4L))
    expect_identical(dim(predict(fit, df[0L, ], type = "link", se.fit = TRUE)$se.fit), c(0L, 3L))
    expect_identical(length(predict(fit, df[0L, ], type = "class")), 0L)
})

test_that("nobs() is the total frequency, as for ordinal and MASS::polr fits", {
    skip_if_not_installed("MASS")
    h <- as.data.frame(MASS::housing)
    ord <- fastglm_ordinal(Sat ~ Infl + Type + Cont, weights = Freq, data = h)
    mn <- fastglm_multinom(factor(Sat, ordered = FALSE) ~ Infl + Type + Cont, weights = Freq, data = h)
    polr <- MASS::polr(Sat ~ Infl + Type + Cont, weights = Freq, data = h)
    expect_identical(nobs(mn), 1681L)
    expect_equal(nobs(mn), nobs(ord))
    expect_equal(nobs(mn), nobs(polr))
    expect_identical(attr(logLik(mn), "nobs"), 1681L)
    expect_equal(BIC(mn), -2 * as.numeric(logLik(mn)) + log(1681) * attr(logLik(mn), "df"))
    # The rows with positive weight still give the residual degrees of freedom.
    expect_identical(mn$nobs, 72L)
    expect_identical(mn$df.residual, 72L * 2L - mn$rank)
    expect_silent(AIC(ord, mn))
    # The same data as individual rows: the same nobs().
    ind <- h[rep(seq_len(nrow(h)), h$Freq), ]
    expect_identical(nobs(fastglm_multinom(factor(Sat, ordered = FALSE) ~ Infl + Type + Cont, data = ind)), 1681L)
    # A count-matrix response counts its row totals.
    w <- reshape(h[, c("Infl", "Type", "Cont", "Sat", "Freq")], idvar = c("Infl", "Type", "Cont"),
                 timevar = "Sat", direction = "wide")
    counts <- as.matrix(w[, c("Freq.Low", "Freq.Medium", "Freq.High")])
    colnames(counts) <- c("Low", "Medium", "High")
    g <- fastglm_multinom(counts ~ Infl + Type + Cont, data = w)
    expect_identical(nobs(g), 1681L)
    expect_equal(as.numeric(logLik(g)), as.numeric(logLik(mn)), tolerance = 1e-8)
    # Without the response, individual data give the sum of the weights, and
    # grouped data the number of rows with positive weight.
    m2 <- mn; m2$y <- NULL; m2$fingerprint.response <- NULL
    expect_equal(nobs(m2), 1681)
    g2 <- g; g2$y <- NULL; g2$fingerprint.response <- NULL
    expect_identical(nobs(g2), g$nobs)
})

test_that("model = FALSE leaves out the model frame of a formula fit", {
    df <- data.frame(x = rnorm(300), g = factor(sample(c("a", "b", "c"), 300, TRUE)))
    df$y <- factor(sample(c("u", "v", "w"), 300, TRUE))
    full <- fastglm_multinom(y ~ x + g, data = df)
    lean <- fastglm_multinom(y ~ x + g, data = df, model = FALSE)
    expect_false(is.null(full$model))
    expect_null(lean$model)
    expect_lt(as.numeric(object.size(lean)), as.numeric(object.size(full)))
    expect_equal(coef(lean), coef(full))
    # drop1() refits from the stored design (no recoding is needed here).
    expect_equal(as.matrix(drop1(lean, test = "LRT")), as.matrix(drop1(full, test = "LRT")), tolerance = 1e-8)
    expect_error(fastglm_multinom(y ~ x, data = df, model = NA), "'model' must be TRUE or FALSE")
})
