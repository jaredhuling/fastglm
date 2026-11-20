# Comparison records and the refit helper (internal groundwork for anova()
# and drop1() across the fitted classes).

compare_core_data <- function(n = 400L, seed = 20261120L) {
    set.seed(seed)
    d <- data.frame(a = rnorm(n), b = factor(sample(c("p", "q", "r"), n, TRUE)), c = runif(n, 0.5, 2),
                    e = factor(sample(c("s", "t", "u"), n, TRUE)))
    eta <- 0.3 + 0.4 * d$a + c(0, 0.3, -0.2)[d$b]
    d$yp <- rpois(n, exp(eta))
    d$ynb <- rnbinom(n, mu = exp(eta), size = 2)
    d$yzi <- ifelse(runif(n) < plogis(-1 + 0.6 * d$a), 0, rpois(n, exp(eta)))
    d$ytp <- ifelse(d$ynb > 0, d$ynb * d$c, 0)
    d$yt <- d$yp + 1
    d$yo <- factor(cut(eta + rlogis(n), c(-Inf, -0.5, 0.5, 1.5, Inf), labels = c("l1", "l2", "l3", "l4")),
                   ordered = TRUE)
    d$ym <- factor(c("u", "v", "w")[apply(cbind(0, 0.5 * d$a, -0.3 * d$a + 0.2) +
                                          matrix(-log(-log(runif(3 * n))), n, 3), 1, which.max)])
    d$yb <- rbinom(n, 1, plogis(-0.5 + d$a))
    d
}

compare_core_source <- function(d, formula, n_chunks = 4L, ...) {
    idx <- split(seq_len(nrow(d)), rep(seq_len(n_chunks), length.out = nrow(d)))
    fastglm_source(function(k) d[idx[[k]], , drop = FALSE], n_chunks,
                   formula = formula, prototype = d, ...)
}

compare_core_fits <- function(d) {
    X <- model.matrix(~ a + b, d)
    Xb <- bigmemory::as.big.matrix(X)
    list(
        glm = fastglm(yp ~ a + b, data = d, family = poisson()),
        matrix = fastglm(X, d$yp, family = poisson()),
        nb = fastglm_nb(ynb ~ a + b, data = d),
        zi = fastglm_zi(yzi ~ a + b | a, data = d),
        hurdle = fastglm_hurdle(ynb ~ a + b | a, data = d),
        twopart = fastglm_twopart(ytp ~ a + b | a, data = d),
        truncated = fastglm_truncated(yt ~ a + b, data = d),
        ordinal = fastglm_ordinal(yo ~ a + b, data = d),
        partial = fastglm_ordinal(yo ~ a + b, data = d, nonparallel = ~ a),
        multinom = fastglm_multinom(ym ~ a + b, data = d),
        stream = fastglm_streaming(compare_core_source(d, yp ~ a + b), family = poisson()),
        bigmatrix = fastglm(Xb, d$yp, family = poisson(), method = 2L)
    )
}

same_coef <- function(a, b, info, tolerance = 1e-10)
    expect_equal(unname(coef(a)), unname(coef(b)), tolerance = tolerance, info = info)

test_that("comparison records describe every class", {
    skip_if_not_installed("bigmemory")
    d <- compare_core_data()
    fits <- compare_core_fits(d)
    expected <- c(glm = "glm", matrix = "glm", nb = "nb", zi = "zi", hurdle = "hurdle",
                  twopart = "twopart", truncated = "truncated", ordinal = "ordinal",
                  partial = "ordinal", multinom = "multinom", stream = "glm", bigmatrix = "glm")
    # Number of estimated parameters: coefficients plus theta (NB), the Gamma
    # dispersion (two-part), and thresholds (ordinal).
    df <- c(glm = 4L, matrix = 4L, nb = 5L, zi = 6L, hurdle = 6L, twopart = 7L, truncated = 4L,
            ordinal = 6L, partial = 8L, multinom = 8L, stream = 4L, bigmatrix = 4L)
    dispersion <- c(glm = "fixed", matrix = "fixed", nb = "ml", zi = "fixed", hurdle = "fixed",
                    twopart = "ml", truncated = "fixed", ordinal = "fixed", partial = "fixed",
                    multinom = "fixed", stream = "fixed", bigmatrix = "fixed")
    for (nm in names(fits)) {
        fit <- fits[[nm]]
        rec <- fastglm:::.fastglm_cmp_record(fit)
        expect_equal(rec$nobs, nobs(fit), info = nm)
        expect_identical(rec$df, df[[nm]], info = nm)
        expect_identical(rec$class_family, expected[[nm]], info = nm)
        expect_identical(rec$dispersion, dispersion[[nm]], info = nm)
        expect_named(rec$fingerprint, c("n", "wsum", "sy", "syy"))
        expect_false(anyNA(rec$fingerprint), info = nm)
        expect_equal(rec$wsum, rec$fingerprint[["wsum"]], info = nm)
        expect_true(all(names(rec$coef_names) %in% c("count", "zero", "location", "nominal", "main")), info = nm)
        expect_equal(as.numeric(rec$loglik), as.numeric(logLik(fit)), info = nm)
        expect_identical(rec$streamed, nm == "stream", info = nm)
        expect_true(rec$source_reusable, info = nm)
        expect_false(rec$penalized, info = nm)
        expect_true(rec$lr_available, info = nm)
        expect_identical(rec$lr_method, "lr", info = nm)
    }
    expect_identical(fastglm:::.fastglm_cmp_record(fits$zi)$link, c(count = "log", zero = "logit"))
    expect_identical(fastglm:::.fastglm_cmp_record(fits$zi)$coef_names,
                     list(count = c("(Intercept)", "a", "bq", "br"), zero = c("(Intercept)", "a")))
    expect_identical(fastglm:::.fastglm_cmp_record(fits$partial)$coef_names,
                     list(location = c("bq", "br"),
                          nominal = paste0("nonparallel:a:", c("l1|l2", "l2|l3", "l3|l4"))))
    expect_identical(fastglm:::.fastglm_cmp_record(fits$ordinal)$coef_names, list(location = c("a", "bq", "br")))
    expect_identical(fastglm:::.fastglm_cmp_record(fits$glm)$coef_names, list(main = c("(Intercept)", "a", "bq", "br")))
    expect_named(fits$bigmatrix$fingerprint.response, c("n", "wsum", "sy", "syy"))
    expect_identical(fastglm:::.fastglm_cmp_record(fastglm(yp ~ a, data = d, family = gaussian()))$dispersion,
                     "moment")
    expect_identical(fastglm:::.fastglm_cmp_record(fastglm(yt ~ a, data = d, family = Gamma("log")))$dispersion,
                     "moment")
})

test_that("Firth records carry the penalized log likelihood or refuse a likelihood ratio test", {
    skip_if_not_installed("logistf")
    d <- compare_core_data()
    f <- fastglm(yb ~ a + b, data = d, family = binomial(), firth = TRUE)
    rec <- fastglm:::.fastglm_cmp_record(f)
    expect_true(rec$penalized)
    expect_identical(rec$lr_method, "profile")
    expect_true(rec$lr_available)
    ref <- logistf::logistf(yb ~ a + b, data = d)
    expect_equal(as.numeric(rec$loglik), unname(ref$loglik[["full"]]), tolerance = 1e-6)
    expect_identical(rec$df, 4L)
    # The comparison Task 2 uses: the profile test of the coefficients absent
    # from the smaller model, under the larger model's penalty, which is
    # logistf's nested comparison (drop1() and anova(method = "nested")).
    pt <- fastglm_profile_test(f, parm = c("bq", "br"))
    nested <- anova(ref, logistf::logistf(yb ~ a, data = d), method = "nested")
    expect_equal(unname(pt$statistic), unname(nested$chisq), tolerance = 1e-4)
    expect_equal(unname(pt$statistic), drop1(ref)["b", "ChiSq"], tolerance = 1e-4)
    # The same for a streamed bias-reduced fit.
    s <- fastglm_streaming(model.matrix(~ a + b, d), y = d$yb, family = binomial(), firth = TRUE)
    rs <- fastglm:::.fastglm_cmp_record(s)
    expect_equal(as.numeric(rs$loglik), unname(ref$loglik[["full"]]), tolerance = 1e-6)
    expect_identical(rs$lr_method, "none")
    # Mean bias reduction that is not a penalized likelihood.
    p <- fastglm(yb ~ a + b, data = d, family = binomial("probit"), firth = TRUE)
    rp <- fastglm:::.fastglm_cmp_record(p)
    expect_true(is.na(rp$penalized))
    expect_false(rp$lr_available)
    expect_identical(rp$lr_method, "none")
    ro <- fastglm:::.fastglm_cmp_record(fastglm_ordinal(yo ~ a, data = d, firth = TRUE))
    expect_true(is.na(ro$penalized))
    expect_false(ro$lr_available)
    # Multinomial logit is canonical: the stored penalized log likelihood.
    m <- fastglm_multinom(ym ~ a, data = d, firth = TRUE)
    rm <- fastglm:::.fastglm_cmp_record(m)
    expect_true(rm$penalized)
    expect_equal(as.numeric(rm$loglik), m$penalized.loglik)
    expect_identical(rm$lr_method, "none")
})

test_that("streamed and in-memory fingerprints agree", {
    d <- compare_core_data()
    d$w <- runif(nrow(d), 0.5, 2)
    dense <- fastglm(yp ~ a + b, data = d, family = poisson(), weights = w)
    stream <- fastglm_streaming(compare_core_source(d, yp ~ a + b, weights = "w"), family = poisson())
    a <- fastglm:::.fastglm_cmp_record(dense)$fingerprint
    b <- fastglm:::.fastglm_cmp_record(stream)$fingerprint
    expect_equal(b, a, tolerance = 1e-12)
    expect_true(fastglm:::.fastglm_same_fingerprint(a, b))
    expect_equal(unname(a), c(nrow(d), sum(d$w), sum(d$w * d$yp), sum(d$w * d$yp^2)))
    nb <- fastglm_nb(fastglm_source(function(k) list(X = model.matrix(~ a + b, d), y = d$ynb), 1L))
    expect_equal(nb$fingerprint.response, fastglm:::.fastglm_cmp_record(fastglm_nb(ynb ~ a + b, data = d))$fingerprint,
                 tolerance = 1e-12)
    os <- fastglm_ordinal_streaming(compare_core_source(d, yo ~ a + b))
    expect_equal(os$fingerprint.response,
                 fastglm:::.fastglm_cmp_record(fastglm_ordinal(yo ~ a + b, data = d))$fingerprint, tolerance = 1e-12)
    ms <- fastglm_multinom(compare_core_source(d, ym ~ a + b))
    expect_equal(ms$fingerprint.response,
                 fastglm:::.fastglm_cmp_record(fastglm_multinom(ym ~ a + b, data = d))$fingerprint, tolerance = 1e-12)
    # Grouped and individual forms of the same categorical data agree, and n is
    # the total frequency.
    tab <- table(d$b, d$yo)
    counts <- matrix(as.numeric(tab), nrow(tab), dimnames = dimnames(tab))
    gm <- fastglm_multinom(model.matrix(~ b, data.frame(b = factor(rownames(tab), levels(d$b)))), counts)
    im <- fastglm_multinom(model.matrix(~ b, d), factor(d$yo, ordered = FALSE))
    expect_equal(fastglm:::.fastglm_cmp_fingerprint(gm), fastglm:::.fastglm_cmp_fingerprint(im), tolerance = 1e-12)
    expect_equal(fastglm:::.fastglm_cmp_fingerprint(im)[["n"]], nrow(d))
    other <- fastglm(yt ~ a + b, data = d, family = poisson(), weights = w)
    expect_false(fastglm:::.fastglm_same_fingerprint(a, fastglm:::.fastglm_cmp_record(other)$fingerprint))
})

test_that("refits reproduce a direct fit", {
    skip_if_not_installed("bigmemory")
    d <- compare_core_data()
    fits <- compare_core_fits(d)
    refit <- fastglm:::.fastglm_refit

    same_coef(refit(fits$glm, formula = . ~ . - a), fastglm(yp ~ b, data = d, family = poisson()), "glm")
    same_coef(refit(fits$nb, formula = . ~ . - a), fastglm_nb(ynb ~ b, data = d), "nb")
    same_coef(refit(fits$zi, formula = yzi ~ b | a), fastglm_zi(yzi ~ b | a, data = d), "zi")
    same_coef(refit(fits$zi, formula = ~ . - a, part = "zero"), fastglm_zi(yzi ~ a + b | 1, data = d), "zi zero")
    same_coef(refit(fits$hurdle, formula = . ~ . - a, part = "count"),
              fastglm_hurdle(ynb ~ b | a, data = d), "hurdle")
    same_coef(refit(fits$twopart, formula = ytp ~ b | a), fastglm_twopart(ytp ~ b | a, data = d), "twopart")
    same_coef(refit(fits$truncated, formula = . ~ . - a), fastglm_truncated(yt ~ b, data = d), "truncated")
    same_coef(refit(fits$ordinal, formula = . ~ . - a), fastglm_ordinal(yo ~ b, data = d), "ordinal")
    same_coef(refit(fits$partial, formula = . ~ . - b),
              fastglm_ordinal(yo ~ a, data = d, nonparallel = ~ a), "partial location")
    nominal <- refit(fits$partial, formula = ~ . - a, part = "nominal")
    expect_false(inherits(nominal, "fastglm_ordinal_partial"))
    same_coef(nominal, fits$ordinal, "partial nominal")
    same_coef(refit(fits$multinom, formula = . ~ . - a), fastglm_multinom(ym ~ b, data = d), "multinom")
    s <- refit(fits$stream, formula = . ~ . - a)
    expect_s3_class(s, "fastglm_stream")
    same_coef(s, fastglm_streaming(compare_core_source(d, yp ~ b), family = poisson()), "stream")
    expect_equal(predict(s, newdata = d[1:5, ]),
                 predict(fastglm(yp ~ b, data = d, family = poisson()), newdata = d[1:5, ]), tolerance = 1e-8)

    X <- model.matrix(~ a + b, d)
    m <- refit(fits$matrix, drop_columns = 2L)
    same_coef(m, fastglm(X[, -2L], d$yp, family = poisson()), "matrix")
    expect_identical(names(coef(m)), colnames(X)[-2L])
    same_coef(refit(fits$matrix, drop_columns = "a"), m, "matrix by name")
    ms <- fastglm_streaming(X, y = d$yp, family = poisson(), block.size = 100L)
    same_coef(refit(ms, drop_columns = 2L), fastglm_streaming(X[, -2L], y = d$yp, family = poisson(), block.size = 100L),
              "matrix stream")

    # Refits run with the fit's threads and the trace off.
    invisible(capture.output(fit <- fastglm(yp ~ a + b, data = d, family = poisson(), threads = 1L, trace = TRUE)))
    out <- capture.output(r <- refit(fit, formula = . ~ . - a))
    expect_length(out, 0L)
    expect_identical(r$threads, 1L)

    # The wrong interface, and terms a refit cannot add, are errors.
    expect_error(refit(fits$glm, drop_columns = 2L), "formula")
    expect_error(refit(fits$matrix, formula = . ~ . - a), "drop_columns")
    expect_error(refit(fits$glm, formula = . ~ . + c), "only remove terms")
})

test_that("refits use the stored data, not objects found in a frame", {
    d <- compare_core_data()
    refit <- fastglm:::.fastglm_refit
    target <- fastglm_nb(ynb ~ b, data = d)
    # Data changed after the fit are not used.
    d2 <- d
    f <- fastglm_nb(ynb ~ a + b, data = d2)
    d2$b <- factor(sample(levels(d$b), nrow(d), TRUE))
    same_coef(refit(f, formula = . ~ . - a), target, "changed data")
    # A fit made in a function, refitted where another object has its name.
    mk <- function(dd) fastglm_nb(ynb ~ a + b, data = dd)
    g <- mk(d)
    dd <- d; dd$b <- rev(dd$b)
    same_coef(refit(g, formula = . ~ . - a), target, "fit in a function")
    # Matrix fits made in a function, with their response, weights, offset,
    # and controls local to it.
    mkm <- function() {
        X <- model.matrix(~ a + b, d); yy <- d$yp; ww <- d$c; oo <- d$a / 10
        ctl <- 1e-10
        fastglm(X, yy, family = poisson(), weights = ww, offset = oo, tol = ctl)
    }
    m <- mkm()
    X <- model.matrix(~ a + b, d)
    same_coef(refit(m, drop_columns = 2L),
              fastglm(X[, -2L], d$yp, family = poisson(), weights = d$c, offset = d$a / 10, tol = 1e-10),
              "matrix in a function")
    mkn <- function() { X <- model.matrix(~ a + b, d); yy <- d$ynb; fastglm_nb(X, yy) }
    same_coef(refit(mkn(), drop_columns = 2L), fastglm_nb(X[, -2L], d$ynb), "nb matrix in a function")
    mkh <- function() { X <- model.matrix(~ a + b, d); yy <- d$ynb; fastglm_hurdle(X, response = yy) }
    same_coef(refit(mkh(), drop_columns = 2L, part = "zero"),
              fastglm_hurdle(X, response = d$ynb, z = X[, -2L]), "hurdle matrix in a function")
    mkmn <- function() { X <- model.matrix(~ a + b, d); yy <- d$ym; fastglm_multinom(X, yy) }
    same_coef(refit(mkmn(), drop_columns = 2L), fastglm_multinom(X[, -2L], d$ym), "multinom matrix in a function")
    # Rows dropped for missing values at fit time stay dropped.
    dna <- d; dna$a[1:3] <- NA
    gna <- fastglm(yp ~ a + b, data = dna, family = poisson())
    r <- refit(gna, formula = . ~ . - a)
    expect_equal(nobs(r), nrow(d) - 3L)
    same_coef(r, fastglm(yp ~ b, data = d[-(1:3), ], family = poisson()), "missing values")
    # A refit whose data differ from the fit is an error.
    expect_error(fastglm:::.fastglm_refit_check(gna, fastglm(yt ~ b, data = dna[-(1:3), ], family = poisson())),
                 "same response")
    expect_error(fastglm:::.fastglm_refit_check(gna, fastglm(yp ~ b, data = d, family = poisson())),
                 "observations")
})

test_that("refits rebuild codings as a fit of the reduced formula does", {
    d <- compare_core_data()
    refit <- fastglm:::.fastglm_refit
    # Removing the first factor of a no-intercept model recodes the second.
    f <- fastglm(yp ~ 0 + b + e, data = d, family = poisson())
    r <- refit(f, formula = . ~ . - b)
    direct <- fastglm(yp ~ 0 + e, data = d, family = poisson())
    expect_identical(names(coef(r)), names(coef(direct)))
    same_coef(r, direct, "in memory")
    s <- fastglm_streaming(compare_core_source(d, yp ~ 0 + b + e), family = poisson())
    rs <- refit(s, formula = . ~ . - b)
    ds <- fastglm_streaming(compare_core_source(d, yp ~ 0 + e), family = poisson())
    expect_identical(names(coef(rs)), names(coef(ds)))
    same_coef(rs, ds, "streamed")
    expect_equal(as.numeric(logLik(rs)), as.numeric(logLik(direct)), tolerance = 1e-8)
    # Removing a main effect whose interaction remains, and the intercept.
    gs <- fastglm_streaming(compare_core_source(d, yp ~ a * b), family = poisson())
    same_coef(refit(gs, formula = . ~ . - a), fastglm_streaming(compare_core_source(d, yp ~ b + a:b),
                                                                 family = poisson()), "marginality")
    same_coef(refit(gs, formula = . ~ . - 1), fastglm_streaming(compare_core_source(d, yp ~ a * b - 1),
                                                                 family = poisson()), "intercept")
    g <- fastglm(yp ~ a * b, data = d, family = poisson())
    same_coef(refit(g, formula = . ~ . - a), fastglm(yp ~ b + a:b, data = d, family = poisson()), "in-memory marginality")
    # Data-dependent bases are kept from the fit.
    p <- fastglm(yp ~ poly(a, 2) + b, data = d, family = poisson())
    same_coef(refit(p, formula = . ~ . - b), fastglm(yp ~ poly(a, 2), data = d, family = poisson()), "poly")
})

test_that("streamed and matrix variants of each class refit", {
    skip_on_cran()
    skip_if_not_installed("bigmemory")
    d <- compare_core_data()
    refit <- fastglm:::.fastglm_refit
    src <- function(f) compare_core_source(d, f)
    X <- model.matrix(~ a + b, d)

    same_coef(refit(fastglm_nb(src(ynb ~ a + b)), formula = . ~ . - a), fastglm_nb(src(ynb ~ b)), "nb stream")
    same_coef(refit(fastglm_nb(X, d$ynb), drop_columns = 2L), fastglm_nb(X[, -2L], d$ynb), "nb matrix")
    same_coef(refit(fastglm_truncated(src(yt ~ a + b)), formula = . ~ . - a), fastglm_truncated(src(yt ~ b)),
              "truncated stream")
    hs <- fastglm_hurdle(src(ynb ~ a + b | a))
    same_coef(refit(hs, formula = . ~ . - a, part = "count"), fastglm_hurdle(src(ynb ~ b | a)), "hurdle stream count")
    same_coef(refit(hs, formula = ~ . - a, part = "zero"), fastglm_hurdle(src(ynb ~ a + b | 1)), "hurdle stream zero")
    same_coef(refit(fastglm_twopart(src(ytp ~ a + b | a)), formula = ytp ~ b | a), fastglm_twopart(src(ytp ~ b | a)),
              "twopart stream")
    same_coef(refit(fastglm_zi(src(yzi ~ a + b | a)), formula = . ~ . - a, part = "count"),
              fastglm_zi(src(yzi ~ b | a)), "zi stream")
    same_coef(refit(fastglm_ordinal_streaming(src(yo ~ a + b)), formula = . ~ . - a),
              fastglm_ordinal_streaming(src(yo ~ b)), "ordinal stream")
    same_coef(refit(fastglm_ordinal_streaming(src(yo ~ a + b), nonparallel = ~ a), formula = ~ . - a, part = "nominal"),
              fastglm_ordinal_streaming(src(yo ~ a + b)), "partial stream")
    same_coef(refit(fastglm_ordinal(X[, -1L], d$yo, nonparallel = "a"), drop_columns = "bq"),
              fastglm_ordinal(X[, -c(1L, 3L)], d$yo, nonparallel = "a"), "partial matrix")
    same_coef(refit(fastglm_ordinal(bigmemory::as.big.matrix(X[, -1L]), d$yo, chunk.size = 100L), drop_columns = 1L),
              fastglm_ordinal(bigmemory::as.big.matrix(X[, -(1:2)]), d$yo, chunk.size = 100L), "ordinal big.matrix")
    same_coef(refit(fastglm_multinom(src(ym ~ a + b)), formula = . ~ . - a), fastglm_multinom(src(ym ~ b)),
              "multinom stream")
    same_coef(refit(fastglm_multinom(X, d$ym), drop_columns = 2L), fastglm_multinom(X[, -2L], d$ym), "multinom matrix")
})

test_that("big.matrix refits read the kept columns in blocks without copying the design", {
    skip_if_not_installed("bigmemory")
    d <- compare_core_data()
    X <- model.matrix(~ a + b, d)
    w <- d$c; o <- d$a / 10
    fit <- fastglm(bigmemory::as.big.matrix(X), d$yp, family = poisson(), method = 2L, weights = w, offset = o)
    before <- list.files(tempdir(), all.files = TRUE, recursive = TRUE)
    r <- fastglm:::.fastglm_refit(fit, drop_columns = 2L)
    expect_identical(list.files(tempdir(), all.files = TRUE, recursive = TRUE), before)
    same_coef(r, fastglm(bigmemory::as.big.matrix(X[, -2L]), d$yp, family = poisson(), method = 2L,
                         weights = w, offset = o), "big.matrix")
    expect_equal(as.numeric(logLik(r)),
                 as.numeric(logLik(fastglm(X[, -2L], d$yp, family = poisson(), weights = w, offset = o))),
                 tolerance = 1e-10)
})

test_that("update works for nb, truncated, and streamed fits as stats::update does", {
    d <- compare_core_data()
    nb <- fastglm_nb(ynb ~ a + b, data = d)
    same_coef(update(nb, . ~ . - a), fastglm_nb(ynb ~ b, data = d), "nb")
    tr <- fastglm_truncated(yt ~ a + b, data = d)
    same_coef(update(tr, . ~ . - a), fastglm_truncated(yt ~ b, data = d), "truncated")
    st <- fastglm_streaming(compare_core_source(d, yp ~ a + b), family = poisson())
    same_coef(update(st, . ~ . - a), fastglm_streaming(compare_core_source(d, yp ~ b), family = poisson()), "stream")
    src <- compare_core_source(d, yp ~ a + b)
    st2 <- fastglm_streaming(src, family = poisson())
    cl <- update(st2, . ~ . - a, evaluate = FALSE)
    expect_identical(deparse1(cl$chunk_callback), "stats::update(src, yp ~ b)")
    same_coef(eval(cl), fastglm_streaming(compare_core_source(d, yp ~ b), family = poisson()), "stream call")
    # Other arguments change as in stats::update(), unevaluated in the call.
    same_coef(update(nb, . ~ . - a, link = "sqrt"), fastglm_nb(ynb ~ b, data = d, link = "sqrt"), "nb link")
    cl <- update(nb, . ~ . - a, evaluate = FALSE)
    expect_identical(deparse1(cl$x), "ynb ~ b")
    expect_identical(deparse1(cl$data), "d")
    sub <- update(nb, data = d[1:300, ])
    expect_equal(nobs(sub), 300)
    expect_identical(deparse1(sub$call$data), "d[1:300, ]")
    # A source's update() changes the arguments a source takes, and refuses others.
    d$w <- d$c
    ws <- update(compare_core_source(d, yp ~ a + b), . ~ . - a, weights = "w")
    same_coef(fastglm_streaming(ws, family = poisson()),
              fastglm_streaming(compare_core_source(d, yp ~ b, weights = "w"), family = poisson()), "source weights")
    expect_false(isTRUE(update(src, reusable = FALSE)$reusable))
    expect_error(update(src, . ~ . - a, family = poisson()), "cannot change: family")
    # A formula fit made without data = (variables found in the formula's environment).
    yy <- d$ynb; aa <- d$a
    expect_equal(nobs(fastglm_nb(yy ~ aa)), nrow(d))
})

test_that("one-shot sources are reported as not reusable", {
    d <- compare_core_data()
    idx <- split(seq_len(nrow(d)), rep(1:4, length.out = nrow(d)))
    reads <- integer(4)
    one_shot <- function(k) {
        reads[k] <<- reads[k] + 1L
        if (reads[k] > 1L) stop("block ", k, " was already read")
        d[idx[[k]], ]
    }
    src <- fastglm_source(one_shot, 4L, formula = yp ~ a + b, prototype = d, cache = TRUE, reusable = FALSE)
    fit <- fastglm_streaming(src, family = poisson())
    rec <- fastglm:::.fastglm_cmp_record(fit)
    expect_true(rec$streamed)
    expect_false(rec$source_reusable)
    expect_error(fastglm:::.fastglm_refit(fit, formula = . ~ . - a), "one-shot")
    expect_true(isTRUE(compare_core_source(d, yp ~ a)$reusable))
    expect_error(fastglm_source(one_shot, 4L, reusable = NA), "reusable")
})

test_that("a fit without its response cannot be refitted or checked", {
    d <- compare_core_data()
    f <- fastglm_zi(yzi ~ a + b | a, data = d, y = FALSE, model = FALSE, x = TRUE)
    expect_true(anyNA(fastglm:::.fastglm_cmp_fingerprint(f)))
    expect_error(fastglm:::.fastglm_refit(f, formula = . ~ . - a, part = "count"), "response")
    expect_warning(fastglm:::.fastglm_refit_check(f, f), "could not be checked")
})

test_that("refits of cached and big.matrix designs stay bounded in disk and memory", {
    skip_on_cran()
    skip_if_not_installed("bigmemory")
    d <- compare_core_data()
    # A cached formula source: refits read the blocks without writing a cache.
    src <- compare_core_source(d, yp ~ a + b, cache = TRUE)
    fit <- fastglm_streaming(src, family = poisson())
    # Collect garbage first: the finalizer of an unreachable cached source of
    # an earlier test would otherwise remove its cache during the refit.
    invisible(gc())
    before <- list.files(tempdir(), all.files = TRUE, recursive = TRUE, include.dirs = TRUE)
    r <- fastglm:::.fastglm_refit(fit, formula = . ~ . - a)
    expect_identical(list.files(tempdir(), all.files = TRUE, recursive = TRUE, include.dirs = TRUE), before)
    same_coef(r, fastglm_streaming(compare_core_source(d, yp ~ b), family = poisson()), "cached source")
    # A wide big.matrix: blocks of max(256, 2^20 / p) rows (about 8 MB), so
    # the R memory peak of a refit does not grow with the number of rows. With
    # 65536-row blocks this refit peaked near 350 MB.
    set.seed(7); n <- 70000L; p <- 300L
    Xb <- bigmemory::big.matrix(n, p, type = "double", init = 1)
    for (j in 2:p) Xb[, j] <- rnorm(n)
    y <- rpois(n, exp(0.2 + 0.05 * Xb[, 2]))
    big <- fastglm(Xb, y, family = poisson(), method = 2L)
    invisible(gc())
    base <- gc(reset = TRUE)[2L, 2L]
    r <- fastglm:::.fastglm_refit(big, drop_columns = 3L)
    g <- gc()
    expect_lt(g[2L, ncol(g)] - base, 200)
    expect_identical(r$source$n_chunks, as.integer(ceiling(n / max(256, floor(2^20 / (p - 1L))))))
    expect_identical(length(coef(r)), p - 1L)
})
