# drop1() and single-model sequential anova() across the fitted classes.

drop1_data <- function(n = 500L, seed = 20261122L) {
    set.seed(seed)
    d <- data.frame(a = rnorm(n), c = rnorm(n), b = factor(sample(c("p", "q", "r"), n, TRUE)),
                    e = factor(sample(c("x", "y"), n, TRUE)))
    eta <- 0.3 + 0.4 * d$a + c(0, 0.3, -0.2)[d$b]
    d$yp <- rpois(n, exp(eta))
    d$ynb <- rnbinom(n, mu = exp(eta), size = 1.5)
    d$yzi <- ifelse(runif(n) < plogis(-1 + 0.6 * d$a), 0, rpois(n, exp(eta)))
    d$yt <- d$ynb + 1
    d$yg <- ifelse(runif(n) < 0.3, 0, rgamma(n, shape = 2, rate = 2 / exp(0.2 + 0.3 * d$a)))
    d$yo <- factor(cut(eta + rlogis(n), c(-Inf, -0.5, 0.5, 1.5, Inf), labels = c("l1", "l2", "l3", "l4")),
                   ordered = TRUE)
    d$ym <- factor(c("u", "v", "w")[apply(cbind(0, 0.5 * d$a, -0.3 * d$a + 0.2) +
                                          matrix(-log(-log(runif(3 * n))), n, 3), 1, which.max)])
    d$yb <- rbinom(n, 1, plogis(-0.5 + d$a))
    d
}

# A formula source over d in n_chunks blocks; reads counts the blocks read.
drop1_source <- function(d, formula, n_chunks = 4L, counter = NULL, ...) {
    idx <- split(seq_len(nrow(d)), rep(seq_len(n_chunks), length.out = nrow(d)))
    fastglm_source(function(k) {
        if (!is.null(counter)) counter$reads <- counter$reads + 1L
        d[idx[[k]], , drop = FALSE]
    }, n_chunks, formula = formula, prototype = d, ...)
}

# The messages a call emits, and its value.
with_messages <- function(expr) {
    msgs <- character()
    value <- withCallingHandlers(expr, message = function(m) {
        msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage")
    })
    list(value = value, messages = msgs)
}

ll <- function(f) as.numeric(logLik(f))

test_that("negative binomial drop1() and anova() match MASS (theta held fixed)", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    g <- MASS::glm.nb(ynb ~ a + b + c, data = d, control = glm.control(epsilon = 1e-12, maxit = 100))
    nb <- fastglm_nb(ynb ~ a + b + c, data = d, tol = 1e-12, theta.tol = 1e-12, outer.tol = 1e-12)
    ref <- drop1(g, test = "Chisq")
    tab <- drop1(nb, test = "LRT")
    expect_s3_class(tab, "anova")
    expect_identical(dimnames(tab), dimnames(ref))
    expect_equal(as.matrix(tab), as.matrix(ref), tolerance = 1e-8)
    expect_equal(as.matrix(drop1(nb, test = "F")), as.matrix(drop1(g, test = "F")), tolerance = 1e-8)
    expect_equal(as.matrix(drop1(nb, test = "none")), as.matrix(drop1(g)), tolerance = 1e-8)
    # Theta is held at the full model's estimate: the refits are those of
    # glm(family = negative.binomial(theta)) with the same theta, to rounding.
    gf <- glm(ynb ~ a + b + c, data = d, family = MASS::negative.binomial(nb$theta),
              control = glm.control(epsilon = 1e-12, maxit = 100))
    expect_equal(tab$Deviance, drop1(gf)$Deviance, tolerance = 1e-10)
    # ... and not refits that re-estimate theta.
    nb.c <- fastglm_nb(ynb ~ a + b, data = d, tol = 1e-12, theta.tol = 1e-12, outer.tol = 1e-12)
    expect_gt(abs(tab["c", "LRT"] - 2 * (ll(nb) - ll(nb.c))), 1e-6)
    # Sequential table, as anova() of one glm.nb fit.
    expect_warning(aref <- anova(g), "without re-estimating")
    atab <- anova(nb)
    expect_identical(dimnames(atab), dimnames(aref))
    expect_equal(as.matrix(atab), as.matrix(aref), tolerance = 1e-8)
    # Rao is not offered (fastglm_score_test() does not support NB fits).
    expect_error(drop1(nb, test = "Rao"), "available tests: \"none\", \"LRT\", \"Chisq\", \"F\"")
    # A streamed NB fit gives the same tables.
    s <- fastglm_nb(drop1_source(d, ynb ~ a + b + c), tol = 1e-12, theta.tol = 1e-12, outer.tol = 1e-12)
    r <- with_messages(drop1(s, test = "LRT"))
    expect_identical(r$messages, "drop1: 3 refits, each a streamed fit making several passes over the source\n")
    expect_equal(as.matrix(r$value), as.matrix(ref), tolerance = 1e-8)
    r <- with_messages(anova(s))
    expect_identical(r$messages, "anova: 2 refits, each a streamed fit making several passes over the source\n")
    expect_equal(as.matrix(r$value), as.matrix(aref), tolerance = 1e-8)
})

test_that("GLM drop1() follows stats::drop1.glm (matrix, streamed, Rao)", {
    d <- drop1_data()
    f <- fastglm(yp ~ a + b + c, data = d, family = poisson())
    ref <- drop1(f, test = "LRT")
    expect_equal(as.matrix(ref), as.matrix(drop1(glm(yp ~ a + b + c, poisson, d), test = "LRT")),
                 tolerance = 1e-8)
    # A streamed Poisson fit against the dense formula fit.
    s <- fastglm_streaming(drop1_source(d, yp ~ a + b + c), family = poisson())
    r <- with_messages(drop1(s, test = "LRT"))
    expect_length(r$messages, 1L)
    expect_identical(dimnames(r$value), dimnames(ref))
    expect_equal(as.matrix(r$value), as.matrix(ref), tolerance = 1e-8)
    # The default test is "none", as stats::drop1.glm (and the formula fit).
    expect_equal(as.matrix(suppressMessages(drop1(s))), as.matrix(drop1(f)), tolerance = 1e-8)
    expect_named(drop1(f), c("Df", "Deviance", "AIC"))
    # Sequential table of a streamed fit, as anova.glm of the dense fit.
    r <- with_messages(anova(s))
    expect_identical(r$messages, "anova: 2 refits, each a streamed fit making several passes over the source\n")
    aref <- anova(glm(yp ~ a + b + c, poisson, d), test = "Chisq")
    expect_identical(dimnames(r$value), dimnames(aref))
    expect_equal(as.matrix(r$value), as.matrix(aref), tolerance = 1e-8)
    # A matrix fit: each non-intercept column on its own, or column groups.
    X <- model.matrix(~ a + b + c, d)
    m <- fastglm(X, d$yp, family = poisson())
    tab <- drop1(m, test = "LRT")
    expect_identical(rownames(tab), c("<none>", "a", "bq", "br", "c"))
    g <- glm(yp ~ 0 + X, poisson, d)
    expect_equal(tab["bq", "LRT"], deviance(glm(d$yp ~ 0 + X[, -3], poisson)) - deviance(g), tolerance = 1e-8)
    tg <- drop1(m, scope = list(b = c("bq", "br")), test = "LRT")
    expect_equal(unlist(tg["b", ]), unlist(ref["b", ]), tolerance = 1e-8)
    # Rao score: matches stats::drop1.glm(test = "Rao") (with glm's refits
    # converged tightly; its default tolerance moves the score by about 1e-4).
    gt <- glm(yp ~ a + b + c, poisson, d, control = glm.control(epsilon = 1e-14, maxit = 100))
    expect_equal(unname(as.matrix(drop1(m, test = "Rao", scope = list(a = 2, b = 3:4, c = 5)))),
                 unname(as.matrix(drop1(gt, test = "Rao"))), tolerance = 1e-8)
    # Gaussian and Gamma fits: scaled deviance and F, as drop1.glm.
    d$yn <- 1 + 0.5 * d$a + rnorm(nrow(d))
    gn <- glm(yn ~ a + b + c, data = d)
    mn <- fastglm(model.matrix(gn), d$yn)
    for (t in c("Chisq", "F"))
        expect_equal(unname(as.matrix(drop1(mn, test = t, scope = list(a = 2, b = 3:4, c = 5)))),
                     unname(as.matrix(drop1(gn, test = t))), tolerance = 1e-8)
    # F for a Poisson model warns, as drop1.glm.
    expect_warning(drop1(m, test = "F"), "quasipoisson")
    # A sequential table needs a formula.
    expect_error(anova(m), "sequential table")
})

test_that("streamed and negative binomial refits keep weights and offsets", {
    skip_if_not_installed("MASS")
    d <- drop1_data(n = 400L)
    d$off <- runif(nrow(d)); d$w <- rep(1:3, length.out = nrow(d))
    s <- fastglm_streaming(drop1_source(d, yp ~ a + b + c + offset(off), weights = "w"), family = poisson())
    g <- glm(yp ~ a + b + c + offset(off), poisson, d, weights = w)
    expect_equal(as.matrix(suppressMessages(drop1(s, test = "LRT"))), as.matrix(drop1(g, test = "LRT")), tolerance = 1e-8)
    expect_equal(as.matrix(suppressMessages(anova(s))), as.matrix(anova(g, test = "Chisq")), tolerance = 1e-8)
    gn <- MASS::glm.nb(ynb ~ a + b + c + offset(off), data = d, weights = w,
                       control = glm.control(epsilon = 1e-12, maxit = 100))
    nb <- fastglm_nb(ynb ~ a + b + c + offset(off), data = d, weights = d$w,
                     tol = 1e-12, theta.tol = 1e-12, outer.tol = 1e-12)
    expect_equal(as.matrix(drop1(nb, test = "Chisq")), as.matrix(drop1(gn, test = "Chisq")), tolerance = 1e-8)
    expect_equal(as.matrix(anova(nb)), as.matrix(suppressWarnings(anova(gn))), tolerance = 1e-8)
})

test_that("drop1() respects marginality (y ~ b * e offers only b:e)", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    g <- glm(yp ~ b * e, poisson, d)
    ref <- drop1(g, test = "LRT")
    expect_identical(rownames(ref), c("<none>", "b:e"))
    s <- fastglm_streaming(drop1_source(d, yp ~ b * e), family = poisson())
    r <- with_messages(drop1(s, test = "LRT"))
    expect_identical(r$messages, "drop1: 1 refit, a streamed fit making several passes over the source\n")
    expect_equal(as.matrix(r$value), as.matrix(ref), tolerance = 1e-8)
    nb <- fastglm_nb(ynb ~ b * e, data = d)
    expect_identical(rownames(drop1(nb)), c("<none>", "b:e"))
    expect_identical(rownames(drop1(fastglm_ordinal(yo ~ b * e, data = d))), c("<none>", "b:e"))
    expect_identical(rownames(drop1(fastglm_zi(yzi ~ b * e | a, data = d))), c("<none>", "count: b:e", "zero: a"))
    # An explicit scope is taken as given, as in stats::drop1.
    expect_identical(rownames(drop1(nb, scope = ~ e)), c("<none>", "e"))
    expect_error(drop1(nb, scope = "z"), "subset of term labels")
})

test_that("ordinal drop1() and anova() match MASS::polr", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    p <- MASS::polr(yo ~ a + b + c, data = d, Hess = TRUE, control = list(reltol = 1e-14, maxit = 1000))
    o <- fastglm_ordinal(yo ~ a + b + c, data = d)
    ref <- drop1(p, test = "Chisq")
    tab <- drop1(o, test = "LRT")
    expect_identical(dimnames(tab), dimnames(ref))
    expect_equal(as.matrix(tab), as.matrix(ref), tolerance = 1e-8)
    # Rao: the score test of each term's coefficients.
    rt <- drop1(o, test = "Rao")
    expect_equal(rt["b", "Rao score"], unname(fastglm_score_test(o, terms = "b")$statistic), tolerance = 1e-10)
    # The sequential table: each model against the one before it, as
    # anova.polr of the sequence.
    seq.fits <- list(MASS::polr(yo ~ 1, data = d), MASS::polr(yo ~ a, data = d),
                     MASS::polr(yo ~ a + b, data = d), p)
    seq.ll <- vapply(seq.fits, function(f) as.numeric(logLik(f)), 0)
    at <- anova(o)
    expect_identical(rownames(at), c("NULL", "a", "b", "c"))
    expect_equal(at$LogLik, seq.ll, tolerance = 1e-8)
    expect_equal(at$Chisq[-1], 2 * diff(seq.ll), tolerance = 1e-6)
    expect_equal(at$Df[-1], c(1, 2, 1))
    # Streamed: the same table.
    s <- fastglm_ordinal_streaming(drop1_source(d, yo ~ a + b + c))
    r <- with_messages(drop1(s, test = "LRT"))
    expect_identical(r$messages, "drop1: 3 refits, each a streamed fit making several passes over the source\n")
    expect_equal(as.matrix(r$value), as.matrix(ref), tolerance = 1e-6)
    expect_error(drop1(s, test = "Rao"), "available tests: \"none\", \"LRT\", \"Chisq\"(\\.|$)")
})

test_that("multinomial drop1() matches nnet refits", {
    skip_if_not_installed("nnet")
    d <- drop1_data()
    fm <- fastglm_multinom(ym ~ a + b + c, data = d)
    tab <- drop1(fm, test = "LRT")
    expect_identical(rownames(tab), c("<none>", "a", "b", "c"))
    nn <- function(f) nnet::multinom(f, data = d, trace = FALSE, reltol = 1e-14, maxit = 1000)
    full <- nn(ym ~ a + b + c)
    reds <- list(nn(ym ~ b + c), nn(ym ~ a + c), nn(ym ~ a + b))
    expect_equal(tab$AIC, c(full$AIC, vapply(reds, `[[`, 0, "AIC")), tolerance = 1e-8)
    expect_equal(tab$LRT[-1], vapply(reds, function(r) deviance(r) - deviance(full), 0), tolerance = 1e-6)
    expect_equal(tab$Df[-1], c(2, 4, 2))
    # nnet's own drop1() gives the same AIC column. (It evaluates `trace` in
    # the caller's frame, so a variable of that name is needed there.)
    trace <- FALSE
    nd <- suppressMessages(utils::capture.output(ref <- drop1(full)))
    expect_equal(tab$AIC, ref$AIC, tolerance = 1e-8)
    at <- anova(fm)
    expect_equal(at$LogLik, c(as.numeric(logLik(nn(ym ~ 1))), as.numeric(logLik(nn(ym ~ a))),
                              as.numeric(logLik(nn(ym ~ a + b))), as.numeric(logLik(full))), tolerance = 1e-8)
    expect_error(drop1(fm, test = "Rao"), "available tests")
})

test_that("models with two linear predictors get one row per term per part", {
    d <- drop1_data()
    zi <- fastglm_zi(yzi ~ a + b | a, data = d)
    tab <- drop1(zi, test = "LRT")
    expect_identical(rownames(tab), c("<none>", "count: a", "count: b", "zero: a"))
    manual <- list(fastglm_zi(yzi ~ b | a, data = d), fastglm_zi(yzi ~ a | a, data = d),
                   fastglm_zi(yzi ~ a + b | 1, data = d))
    expect_equal(tab$LRT[-1], vapply(manual, function(m) 2 * (ll(zi) - ll(m)), 0), tolerance = 1e-8)
    expect_equal(tab$AIC, c(AIC(zi), vapply(manual, AIC, 0)), tolerance = 1e-8)
    expect_equal(tab$Df[-1], c(1, 2, 1))
    # Scope by part, or a term in every part that has it.
    expect_identical(rownames(drop1(zi, scope = list(zero = ~ a))), c("<none>", "zero: a"))
    expect_identical(rownames(drop1(zi, scope = "a")), c("<none>", "count: a", "zero: a"))
    expect_error(drop1(zi, scope = list(a = 1)), "named 'count' or 'zero'")
    expect_error(drop1(zi, scope = "z"), "not terms of any part")
    # No sequential table across parts.
    expect_error(anova(zi), "use drop1\\(\\)")
    expect_error(drop1(zi, test = "Rao"), "available tests: \"none\", \"LRT\", \"Chisq\"(\\.|$)")
    expect_error(drop1(zi, test = "F"), "available tests")
    # Against pscl refits.
    skip_if_not_installed("pscl")
    ctl <- pscl::zeroinfl.control(reltol = 1e-14, maxit = 10000)
    zp <- pscl::zeroinfl(yzi ~ a + b | a, data = d, control = ctl)
    zr <- pscl::zeroinfl(yzi ~ b | a, data = d, control = ctl)
    expect_equal(tab["count: a", "LRT"], 2 * (ll(zp) - ll(zr)), tolerance = 1e-6)
    # Hurdle and two-part fits.
    h <- fastglm_hurdle(ynb ~ a + b | a + c, data = d, dist = "negbin")
    th <- drop1(h, test = "LRT")
    expect_identical(rownames(th), c("<none>", "count: a", "count: b", "zero: a", "zero: c"))
    expect_equal(th["zero: c", "LRT"], 2 * (ll(h) - ll(fastglm_hurdle(ynb ~ a + b | a, data = d, dist = "negbin"))),
                 tolerance = 1e-8)
    expect_error(anova(h), "more than one linear predictor")
    tp <- fastglm_twopart(yg ~ a + b | a, data = d)
    tt <- drop1(tp, test = "LRT")
    expect_identical(rownames(tt), c("<none>", "positive: a", "positive: b", "zero: a"))
    # The statistic uses the profile likelihood over the Gamma shape, as anova().
    tp0 <- fastglm_twopart(yg ~ a | a, data = d)
    expect_equal(tt["positive: b", "LRT"], anova(tp0, tp)$Chisq[2], tolerance = 1e-10)
    # A streamed zero-inflated fit gives the same rows.
    zs <- fastglm_zi(drop1_source(d, yzi ~ a + b | a))
    r <- with_messages(drop1(zs, test = "LRT"))
    expect_identical(r$messages, "drop1: 3 refits, each a streamed fit making several passes over the source\n")
    expect_equal(as.matrix(r$value), as.matrix(tab), tolerance = 1e-6)
})

test_that("a partial proportional odds fit lists parallel and nominal effects separately", {
    d <- drop1_data()
    op <- fastglm_ordinal(yo ~ a + b + c, data = d, nonparallel = ~ a)
    tab <- drop1(op, test = "LRT")
    expect_identical(rownames(tab), c("<none>", "a (all effects)", "b", "c", "a (nonparallel)"))
    # The location row removes a entirely; the nominal row makes its effect
    # common to the cutpoints (the proportional odds model).
    expect_equal(tab["a (all effects)", "LRT"], 2 * (ll(op) - ll(fastglm_ordinal(yo ~ b + c, data = d))),
                 tolerance = 1e-8)
    expect_equal(tab["a (nonparallel)", "LRT"], 2 * (ll(op) - ll(fastglm_ordinal(yo ~ a + b + c, data = d))),
                 tolerance = 1e-8)
    expect_equal(tab$Df[-1], c(3, 2, 1, 2))
    # Rao: the nominal row tests equal cutpoint-specific slopes.
    rt <- drop1(op, test = "Rao")
    expect_equal(rt$Df, tab$Df)
    expect_true(all(abs(rt[-1, "Rao score"] - tab[-1, "LRT"]) < 0.1 * tab[-1, "LRT"] + 0.1))
    # The nonparallel row's score statistic is the score test of equal
    # cutpoint-specific slopes of a.
    b <- names(coef(op)); np <- b[startsWith(b, "nonparallel:a:")]
    L <- rbind(setNames(c(1, -1, 0), np), setNames(c(0, 1, -1), np))
    expect_equal(rt["a (nonparallel)", "Rao score"],
                 unname(fastglm_score_test(op, contrast = L)$statistic), tolerance = 1e-10)
    expect_identical(rownames(drop1(op, scope = list(nominal = "a"))), c("<none>", "a (nonparallel)"))
    expect_match(paste(attr(tab, "heading"), collapse = ""), "proportional odds restriction")
    expect_error(anova(op), "more than one linear predictor")
})

test_that("truncated drop1() and anova() match manual refits", {
    d <- drop1_data()
    tr <- fastglm_truncated(yt ~ a + b + c, data = d, dist = "negbin")
    tab <- drop1(tr, test = "LRT")
    red <- fastglm_truncated(yt ~ a + c, data = d, dist = "negbin")
    expect_equal(tab["b", "LRT"], 2 * (ll(tr) - ll(red)), tolerance = 1e-8)
    expect_equal(tab["b", "AIC"], AIC(red), tolerance = 1e-8)
    at <- anova(tr)
    expect_equal(at$LogLik, c(ll(fastglm_truncated(yt ~ 1, data = d, dist = "negbin")),
                              ll(fastglm_truncated(yt ~ a, data = d, dist = "negbin")),
                              ll(fastglm_truncated(yt ~ a + b, data = d, dist = "negbin")), ll(tr)),
                 tolerance = 1e-8)
    expect_error(anova(tr, test = "F"), "not available")
})

test_that("big.matrix drop1() takes column groups", {
    skip_if_not_installed("bigmemory")
    d <- drop1_data()
    X <- model.matrix(~ a + b + c, d)
    bm <- fastglm(bigmemory::as.big.matrix(X), d$yp, family = poisson(), method = 2L)
    m <- fastglm(X, d$yp, family = poisson())
    r <- with_messages(drop1(bm, test = "LRT"))
    expect_identical(r$messages,
                     "drop1: 4 refits, each a streamed fit making several passes over the big.matrix design in blocks of rows\n")
    expect_identical(rownames(r$value), c("<none>", "a", "bq", "br", "c"))
    expect_equal(as.matrix(r$value), as.matrix(drop1(m, test = "LRT")), tolerance = 1e-8)
    r <- with_messages(drop1(bm, scope = list(g = c(2, 3)), test = "LRT"))
    expect_length(r$messages, 1L)
    expect_identical(rownames(r$value), c("<none>", "g"))
    expect_equal(r$value["g", "Deviance"], fastglm(X[, -(2:3)], d$yp, family = poisson())$deviance,
                 tolerance = 1e-8)
    expect_equal(r$value["g", "Df"], 2)
    expect_identical(rownames(suppressMessages(drop1(bm, scope = list(c("a", "c")))))[2], "a + c")
    expect_error(drop1(bm, scope = list(g = 9)), "index design columns")
    expect_error(drop1(bm, test = "Rao"), "not available")
    expect_error(anova(bm), "sequential table")
})

test_that("streamed refits: one announcement, one-shot sources refused first, trace off, interrupts", {
    d <- drop1_data(n = 400L)
    counter <- new.env(); counter$reads <- 0L
    s <- fastglm_streaming(drop1_source(d, yp ~ a + b + c, counter = counter), family = poisson())
    counter$reads <- 0L
    r <- with_messages(drop1(s, scope = c("a", "b")))
    expect_identical(r$messages, "drop1: 2 refits, each a streamed fit making several passes over the source\n")
    expect_gt(counter$reads, 0L)
    # Rao names the available tests, before any refit.
    counter$reads <- 0L
    expect_error(drop1(s, test = "Rao"), "available tests: \"none\", \"LRT\", \"Chisq\", \"F\"")
    expect_identical(counter$reads, 0L)
    # A one-shot source: an error before any refit (no read, no announcement).
    one <- new.env(); one$reads <- 0L
    s1 <- fastglm_streaming(drop1_source(d, yp ~ a + b, counter = one, reusable = FALSE), family = poisson())
    one$reads <- 0L
    r <- tryCatch(with_messages(drop1(s1)), error = function(e) conditionMessage(e))
    expect_match(r, "one-shot")
    expect_identical(one$reads, 0L)
    expect_error(anova(s1), "one-shot")
    expect_identical(one$reads, 0L)
    # Refits are not traced, even under options(fastglm.trace = TRUE).
    old <- options(fastglm.trace = TRUE)
    on.exit(options(old), add = TRUE)
    out <- capture.output(tab <- suppressMessages(drop1(s, scope = "c")))
    expect_length(out, 0L)
    options(old)
})

test_that("refits use at most the session's threads and an interrupt leaves the session usable", {
    skip_on_cran()
    skip_on_os("windows")
    d <- drop1_data(n = 400L)
    session <- fastglm_threads()$threads
    fastglm_threads(1L)
    on.exit(fastglm_threads(session), add = TRUE)
    seen <- integer()
    idx <- split(seq_len(nrow(d)), rep(1:4, length.out = nrow(d)))
    log.src <- fastglm_source(function(k) {
        seen <<- c(seen, getOption("fastglm.threads"))
        d[idx[[k]], , drop = FALSE]
    }, 4L, formula = yp ~ a + b, prototype = d)
    s <- fastglm_streaming(log.src, family = poisson(), threads = 2L)
    seen <- integer()
    suppressMessages(drop1(s))
    expect_true(length(seen) > 0L)
    # The fit used 2 threads (at most); its refits are capped by the session's 1.
    expect_true(all(seen == 1L))
    expect_identical(fastglm_threads()$threads, 1L)
    # Interrupt during the second refit.
    reads <- 0L
    int.src <- fastglm_source(function(k) {
        reads <<- reads + 1L
        if (reads == 30L) { tools::pskill(Sys.getpid(), tools::SIGINT); Sys.sleep(0.5) }
        d[idx[[k]], , drop = FALSE]
    }, 4L, formula = yp ~ a + b + c, prototype = d)
    s2 <- fastglm_streaming(int.src, family = poisson(), threads = 2L)
    old <- options(fastglm.trace = NULL)
    on.exit(options(old), add = TRUE)
    reads <- 0L
    got <- tryCatch({ suppressMessages(drop1(s2)); "finished" }, interrupt = function(e) "interrupted")
    expect_identical(got, "interrupted")
    expect_identical(fastglm_threads()$threads, 1L)
    expect_null(getOption("fastglm.trace"))
    expect_false(fastglm:::fastglm_set_trace_cpp(FALSE))
    # The session is usable: drop1() runs again and matches the dense fit.
    reads <- 1000L
    tab <- suppressMessages(drop1(s2, test = "LRT"))
    expect_equal(as.matrix(tab), as.matrix(drop1(fastglm(yp ~ a + b + c, data = d, family = poisson()), test = "LRT")),
                 tolerance = 1e-8)
})

test_that("bias-reduced fits use penalized likelihood ratio tests", {
    d <- drop1_data()
    ff <- fastglm(yb ~ a + b + c, data = d, family = binomial(), firth = TRUE)
    tab <- drop1(ff, test = "LRT")
    expect_identical(rownames(tab), c("<none>", "a", "b", "c"))
    expect_named(tab, c("Df", "Penalized LRT", "Pr(>Chi)"))
    expect_equal(tab["b", "Penalized LRT"],
                 unname(fastglm_profile_test(ff, parm = c("bq", "br"))$statistic), tolerance = 1e-12)
    # Sequential: each model of the sequence against the one before it.
    at <- anova(ff)
    expect_identical(rownames(at), c("NULL", "a", "b", "c"))
    f2 <- fastglm(yb ~ a + b, data = d, family = binomial(), firth = TRUE)
    expect_equal(at["b", "Penalized LRT"], unname(fastglm_profile_test(f2, parm = c("bq", "br"))$statistic),
                 tolerance = 1e-6)
    expect_equal(at["c", "Penalized LRT"], tab["c", "Penalized LRT"], tolerance = 1e-10)
    expect_error(anova(ff, test = "F"), "not available")
    expect_error(drop1(ff, test = "Rao"), "available tests: \"none\", \"LRT\", \"Chisq\"(\\.|$)")
    # A matrix Firth fit: column groups.
    fm <- fastglm(model.matrix(~ a + b + c, d), d$yb, family = binomial(), firth = TRUE)
    expect_equal(drop1(fm, scope = list(b = 3:4), test = "LRT")["b", "Penalized LRT"], tab["b", "Penalized LRT"],
                 tolerance = 1e-8)
    # Bias-reduced fits without a penalized likelihood ratio test.
    fp <- fastglm(yp ~ a + b, data = d, family = poisson(), firth = TRUE)
    expect_error(drop1(fp), "no likelihood ratio test")
    expect_error(anova(fp), "no likelihood ratio test")
    expect_error(drop1(fastglm_ordinal(yo ~ a + b, data = d, firth = TRUE)), "no likelihood ratio test")
    # Against logistf (the nested comparison).
    skip_if_not_installed("logistf")
    lf <- logistf::logistf(yb ~ a + b + c, data = d)
    ref <- drop1(lf)
    expect_equal(tab[c("a", "b", "c"), "Penalized LRT"], unname(ref[, "ChiSq"]), tolerance = 1e-5)
    l1 <- logistf::logistf(yb ~ a, data = d); l2 <- logistf::logistf(yb ~ a + b, data = d)
    expect_equal(at["b", "Penalized LRT"], unname(anova(l2, l1, method = "nested")$chisq), tolerance = 1e-5)
})

test_that("drop1() warns about unused arguments and checks its test", {
    d <- drop1_data()
    nb <- fastglm_nb(ynb ~ a + b, data = d)
    expect_warning(drop1(nb, foo = 3), "not used")
    expect_error(drop1(nb, test = "Wald"), "should be one of")
})

test_that("drop1() follows stats::drop1 arguments: test matching, k, scale, positional order", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    g <- MASS::glm.nb(ynb ~ a + b + c, data = d, control = glm.control(epsilon = 1e-12, maxit = 100))
    nb <- fastglm_nb(ynb ~ a + b + c, data = d, tol = 1e-12, theta.tol = 1e-12, outer.tol = 1e-12)
    # Positional scope, scale, test; k = log(n) in the AIC column.
    k <- log(nrow(d))
    expect_equal(as.matrix(drop1(nb, ~ . - c, 0, "LRT", k = k)),
                 as.matrix(drop1(g, ~ . - c, 0, "LRT", k = k)), tolerance = 1e-8)
    expect_identical(rownames(drop1(nb, ~ . - c)), c("<none>", "a", "b"))
    # test is matched as stats does: case sensitive, partial matching.
    expect_error(drop1(nb, test = "chisq"), "should be one of")
    expect_equal(as.matrix(drop1(nb, test = "Chi")), as.matrix(drop1(nb, test = "Chisq")))
    # Ordinal with k, against polr.
    p <- MASS::polr(yo ~ a + b + c, data = d, Hess = TRUE, control = list(reltol = 1e-14, maxit = 1000))
    o <- fastglm_ordinal(yo ~ a + b + c, data = d)
    expect_equal(as.matrix(drop1(o, test = "Chisq", k = k)), as.matrix(drop1(p, test = "Chisq", k = k)),
                 tolerance = 1e-8)
    expect_equal(as.matrix(drop1(o)), as.matrix(drop1(p)), tolerance = 1e-8)
    # scale: replaces the dispersion of a GLM, as drop1.glm; refused elsewhere.
    d$yn <- 1 + 0.5 * d$a + rnorm(nrow(d))
    gn <- glm(yn ~ a + b, data = d)
    mn <- fastglm(model.matrix(gn), d$yn)
    for (t in c("none", "LRT", "F"))
        expect_equal(unname(as.matrix(drop1(mn, scope = list(a = 2, b = 3:4), scale = 2, test = t))),
                     unname(as.matrix(drop1(gn, scale = 2, test = t))), tolerance = 1e-8)
    expect_error(drop1(o, scale = 1), "'scale' applies")
    expect_silent(drop1(o, scale = 0))
    # A part-named scope that is not in that part names the part.
    zi <- fastglm_zi(yzi ~ a + b | a, data = d)
    expect_error(drop1(zi, scope = list(count = "z")), "scope terms of part 'count' are not terms of that part: z")
    expect_identical(rownames(drop1(zi, scope = ~ . - b)), c("<none>", "count: a", "zero: a"))
})

test_that("the sequential anova() matches its test before any refit", {
    skip_if_not_installed("MASS")
    d <- drop1_data(n = 400L)
    nb <- fastglm_nb(ynb ~ a + b, data = d)
    g <- MASS::glm.nb(ynb ~ a + b, data = d)
    expect_error(anova(nb, test = "none"), "should be one of")
    expect_error(suppressWarnings(anova(g, test = "none")), "should be one of")
    expect_error(anova(nb, test = "chisq"), "should be one of")
    expect_error(anova(nb, test = "Cp"), "not available")
    expect_error(anova(nb, test = "Rao"), "not available")
    expect_named(anova(nb, test = FALSE), c("Df", "Deviance", "Resid. Df", "Resid. Dev"))
    o <- fastglm_ordinal(yo ~ a + b, data = d)
    expect_error(anova(o, test = "none"), "should be one of")
    expect_error(anova(o, test = "F"), "not available")
    expect_named(anova(o, test = FALSE), c("#Df", "LogLik", "Df"))
    # On a streamed fit an invalid test reads nothing.
    counter <- new.env(); counter$reads <- 0L
    s <- fastglm_streaming(drop1_source(d, yp ~ a + b, counter = counter), family = poisson())
    counter$reads <- 0L
    expect_error(anova(s, test = "chisq"), "should be one of")
    expect_error(drop1(s, test = "chisq"), "should be one of")
    expect_identical(counter$reads, 0L)
})

test_that("trace = TRUE traces each refit after a header line; trace = 1 stays silent", {
    d <- drop1_data(n = 400L)
    s <- fastglm_streaming(drop1_source(d, yp ~ a + b + c), family = poisson())
    out <- capture.output(tab <- suppressMessages(drop1(s, trace = TRUE)))
    expect_identical(grep("^drop1 refit", out, value = TRUE),
                     c("drop1 refit 1/3: - a", "drop1 refit 2/3: - b", "drop1 refit 3/3: - c"))
    expect_gt(length(out), 3L)
    expect_length(capture.output(tab <- suppressMessages(drop1(s, trace = 1))), 0L)
    out <- capture.output(tab <- suppressMessages(anova(s, trace = TRUE)))
    expect_identical(grep("^anova refit", out, value = TRUE),
                     c("anova refit 1/2: terms up to a", "anova refit 2/2: terms up to b"))
    expect_gt(length(out), 2L)
})

test_that("step() reaches the same model as on the MASS reference fits", {
    skip_if_not_installed("MASS")
    set.seed(34); n <- 500
    d <- data.frame(a = rnorm(n), c = rnorm(n), z = rnorm(n), b = factor(sample(c("p", "q", "r"), n, TRUE)))
    eta <- 0.3 + 0.4 * d$a + c(0, 0.3, -0.2)[d$b]
    d$ynb <- rnbinom(n, mu = exp(eta), size = 1.5)
    d$yo <- factor(cut(eta + rlogis(n), c(-Inf, -0.5, 0.5, 1.5, Inf), labels = c("l1", "l2", "l3", "l4")),
                   ordered = TRUE)
    lab <- function(f) sort(attr(terms(f), "term.labels"))
    g <- MASS::glm.nb(ynb ~ a + b + c + z, data = d)
    nb <- fastglm_nb(ynb ~ a + b + c + z, data = d)
    expect_equal(extractAIC(nb), extractAIC(g), tolerance = 1e-6)
    sg <- step(g, trace = 0); sf <- step(nb, trace = 0)
    expect_identical(lab(sf), lab(sg))
    expect_lt(length(lab(sf)), 4L)
    expect_identical(lab(MASS::stepAIC(nb, trace = 0)), lab(sg))
    p <- MASS::polr(yo ~ a + b + c + z, data = d)
    o <- fastglm_ordinal(yo ~ a + b + c + z, data = d)
    expect_equal(extractAIC(o), extractAIC(p), tolerance = 1e-6)
    sp <- step(p, trace = 0); so <- step(o, trace = 0)
    expect_identical(lab(so), lab(sp))
    expect_lt(length(lab(so)), 4L)
    expect_equal(extractAIC(so), extractAIC(sp), tolerance = 1e-6)
})

test_that("the Firth formula path matches arguments positionally and defaults to no test", {
    d <- drop1_data()
    ff <- fastglm(yb ~ a + b + c, data = d, family = binomial(), firth = TRUE)
    expect_named(drop1(ff), "Df")
    expect_named(drop1(ff, ~ a, 0, "LRT"), c("Df", "Penalized LRT", "Pr(>Chi)"))
    expect_identical(rownames(drop1(ff, ~ a, 0, "LRT")), c("<none>", "a"))
    expect_error(drop1(ff, test = "F"), "available tests")
})

test_that("drop1() and anova() keep a few numbers of each refit, not the refit", {
    d <- drop1_data()
    nb <- fastglm_nb(ynb ~ a + b + c, data = d)
    # The runner reduces each refit before the next one starts.
    sizes <- numeric()
    out <- fastglm:::.fastglm_drop1_run(3L, function(i)
        fastglm:::.fastglm_refit(nb, drop_columns = i + 1L, columns = TRUE, nb_theta_fixed = TRUE),
        c("a", "bq", "br"), "drop1", FALSE, function(fit, i) {
            sizes[i] <<- as.numeric(object.size(fit))
            fastglm:::.fastglm_drop1_keep_glm(fit, i)
        })
    expect_length(out, 3L)
    for (o in out) {
        expect_lt(as.numeric(object.size(o)), 1e4)
        expect_named(o, c("deviance", "rank", "df.residual"))
    }
    expect_true(all(sizes > 1e5))
    o <- fastglm_ordinal(yo ~ a + b + c, data = d)
    keep <- fastglm:::.fastglm_drop1_keep_loglik(2)(fastglm_ordinal(yo ~ a + c, data = d), 1L)
    expect_lt(as.numeric(object.size(keep)), 1e4)
    expect_named(keep, c("npar", "loglik", "aic"))
    # The tables are unchanged: they equal refits made by hand.
    tab <- drop1(o, test = "LRT")
    expect_equal(tab["b", "LRT"], 2 * (ll(o) - ll(fastglm_ordinal(yo ~ a + c, data = d))), tolerance = 1e-8)
    expect_equal(tab["b", "AIC"], AIC(fastglm_ordinal(yo ~ a + c, data = d)), tolerance = 1e-8)
})

test_that("drop1() and anova() of a cached formula source read the callback once per call", {
    d <- drop1_data(n = 400L)
    counter <- new.env(); counter$reads <- 0L
    zs <- fastglm_zi(drop1_source(d, yzi ~ a + b | a, counter = counter, cache = TRUE))
    ref <- drop1(fastglm_zi(yzi ~ a + b | a, data = d), test = "LRT")
    tmp <- function() sort(list.files(tempdir(), all.files = TRUE, no.. = TRUE))
    before <- tmp()
    counter$reads <- 0L
    r <- with_messages(drop1(zs, test = "LRT"))
    # One pass over the 4 blocks builds the cache of the call; the 3 refits
    # read the cache, which is removed when drop1() returns.
    expect_identical(counter$reads, 4L)
    expect_identical(tmp(), before)
    expect_identical(r$messages, paste0("drop1: one pass over the source to cache its blocks for this call ",
                                        "(removed when it ends), then 3 refits, each a streamed fit making ",
                                        "several passes over the cached blocks\n"))
    expect_equal(as.matrix(r$value), as.matrix(ref), tolerance = 1e-6)
    # The sequential table of a cached ordinal source.
    os <- fastglm_ordinal_streaming(drop1_source(d, yo ~ a + b + c, counter = counter, cache = TRUE))
    before <- tmp()
    counter$reads <- 0L
    r <- with_messages(anova(os))
    expect_identical(counter$reads, 4L)
    expect_identical(tmp(), before)
    expect_equal(r$value$LogLik, anova(fastglm_ordinal(yo ~ a + b + c, data = d))$LogLik, tolerance = 1e-6)
    # Without a cache, each refit reads the source on every pass, and
    # nothing is written.
    zu <- fastglm_zi(drop1_source(d, yzi ~ a + b | a, counter = counter))
    before <- tmp()
    counter$reads <- 0L
    r <- with_messages(drop1(zu, test = "LRT"))
    expect_gt(counter$reads, 3L * 4L)
    expect_identical(tmp(), before)
    expect_identical(r$messages, "drop1: 3 refits, each a streamed fit making several passes over the source\n")
    # Column refits of a cached GLM source read the source's own cache.
    gs <- fastglm_streaming(drop1_source(d, yp ~ a + b + c, counter = counter, cache = TRUE), family = poisson())
    counter$reads <- 0L
    r <- with_messages(drop1(gs, test = "LRT"))
    expect_identical(counter$reads, 0L)
    expect_identical(r$messages,
                     "drop1: 3 refits, each a streamed fit making several passes over the block cache of the source\n")
    # An error while the cache is built (here, the third block) leaves no
    # files behind.
    failing <- fastglm_source(function(k) {
        counter$reads <- counter$reads + 1L
        if (isTRUE(counter$broken) && k == 3L) stop("block 3 is unavailable")
        d[seq(k, nrow(d), by = 4L), , drop = FALSE]
    }, 4L, formula = yzi ~ a + b | a, prototype = d, cache = TRUE)
    zf <- fastglm_zi(failing)
    counter$broken <- TRUE
    before <- tmp()
    expect_error(suppressMessages(drop1(zf, test = "LRT")), "block 3 is unavailable")
    expect_identical(tmp(), before)
})

test_that("negative binomial fits at the Poisson limit: drop1(), anova(), and step() use Poisson refits", {
    d <- drop1_data()
    set.seed(3)
    d$yu <- rbinom(nrow(d), 4, plogis(0.3 * d$a))
    nb <- suppressWarnings(fastglm_nb(yu ~ a + b + c, data = d))
    expect_identical(nb$theta, Inf)
    pg <- glm(yu ~ a + b + c, poisson, d)
    tab <- drop1(nb, test = "LRT")
    ref <- drop1(pg, test = "LRT")
    expect_equal(tab$Deviance, ref$Deviance, tolerance = 1e-8)
    expect_equal(tab$LRT, ref$LRT, tolerance = 1e-6)
    expect_match(paste(attr(tab, "heading"), collapse = ""), "Poisson limit")
    at <- anova(nb)
    expect_equal(at$Deviance, anova(pg, test = "Chisq")$Deviance, tolerance = 1e-8)
    s <- suppressWarnings(step(nb, trace = 0))
    expect_s3_class(s, "fastglm_nb")
    expect_setequal(attr(terms(s), "term.labels"), attr(terms(step(pg, trace = 0)), "term.labels"))
})

test_that("the fit returned by step() can be evaluated again from its call and updated", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    d$o <- d$yo
    # A noise term (z) that step() removes, and fits where it removes nothing.
    set.seed(9); d$z <- rnorm(nrow(d))
    same <- function(a, b) expect_equal(unlist(coef(a)), unlist(coef(b)), tolerance = 1e-8)
    fits <- list(
        nb = fastglm_nb(ynb ~ a + b + z, data = d),
        nb0 = fastglm_nb(ynb ~ a, data = d),
        trunc = fastglm_truncated(yt ~ a + b + z, data = d),
        trunc0 = fastglm_truncated(yt ~ a, data = d),
        ordinal = fastglm_ordinal(yo ~ a + b + z, data = d),
        ordinal0 = fastglm_ordinal(yo ~ a, data = d),
        multinom = fastglm_multinom(ym ~ a + b + z, data = d),
        multinom0 = fastglm_multinom(ym ~ a, data = d),
        zi = fastglm_zi(yzi ~ a + b + z | a, data = d),
        hurdle = fastglm_hurdle(yzi ~ a + b | a, data = d),
        twopart = fastglm_twopart(yg ~ a + b | a, data = d))
    for (nm in names(fits)) {
        f <- fits[[nm]]
        s <- suppressWarnings(step(f, trace = 0))
        expect_true(is.call(s$call), info = nm)
        same(eval(s$call), s)
        same(update(s, . ~ .), s)
        expect_false(inherits(formula(s), "terms"), info = nm)
        expect_null(attr(formula(s), "factors"))
        # step() stopped where no deletion lowers the AIC.
        if (!nm %in% c("zi", "hurdle", "twopart")) {
            tab <- drop1(s)
            expect_identical(rownames(tab)[which.min(tab$AIC)], "<none>", info = nm)
        }
    }
    expect_setequal(attr(terms(step(fits$nb, trace = 0)), "term.labels"),
                    attr(terms(MASS::stepAIC(MASS::glm.nb(ynb ~ a + b + z, data = d), trace = 0)), "term.labels"))
    # Models with two linear predictors: step() returns the fit unchanged.
    expect_warning(sz <- step(fits$zi, trace = 0), "does not select terms")
    same(sz, fits$zi)
    expect_identical(format(formula(sz)), format(formula(fits$zi)))
})

test_that("formula() of every class is a formula without terms attributes", {
    d <- drop1_data()
    fits <- list(fastglm_nb(ynb ~ a + b, data = d), fastglm_truncated(yt ~ a + b, data = d),
                 fastglm_zi(yzi ~ a + b | a, data = d), fastglm_hurdle(yzi ~ a + b | a, data = d),
                 fastglm_twopart(yg ~ a + b | a, data = d), fastglm_ordinal(yo ~ a + b, data = d),
                 fastglm_multinom(ym ~ a + b, data = d),
                 fastglm_streaming(drop1_source(d, yp ~ a + b), family = poisson()))
    for (f in fits) {
        fo <- formula(f)
        expect_true(inherits(fo, "formula"), info = class(f)[1L])
        expect_false(inherits(fo, "terms"), info = class(f)[1L])
        expect_null(attr(fo, "factors"))
    }
    expect_identical(deparse(formula(fits[[1L]])), "ynb ~ a + b")
    expect_identical(deparse(formula(fits[[3L]])), "yzi ~ a + b | a")
    expect_error(formula(fastglm_nb(model.matrix(~ a, d), d$ynb)), "no formula")
})

test_that("two-part drop1() names the positive part", {
    d <- drop1_data()
    tp <- fastglm_twopart(yg ~ a + b | a, data = d)
    tt <- drop1(tp, test = "LRT")
    expect_identical(rownames(tt), c("<none>", "positive: a", "positive: b", "zero: a"))
    expect_equal(tt["positive: b", "LRT"],
                 2 * (ll(tp) - ll(fastglm_twopart(yg ~ a | a, data = d))), tolerance = 1e-8)
    # `count` is accepted for `positive` in a list scope.
    expect_identical(rownames(drop1(tp, scope = list(count = "b"))), c("<none>", "positive: b"))
    expect_identical(rownames(drop1(tp, scope = list(positive = "b", zero = "a"))),
                     c("<none>", "positive: b", "zero: a"))
})

test_that("extractAIC() counts the rank; with zero weights step() selects what MASS selects", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    set.seed(12); d$z <- rnorm(nrow(d))
    d$w <- rep(c(1, 1, 1, 0), length.out = nrow(d))
    k <- log(sum(d$w > 0))
    nb <- fastglm_nb(ynb ~ a + b + c + z, data = d, weights = w)
    ref <- MASS::glm.nb(ynb ~ a + b + c + z, data = d, weights = w)
    s <- step(nb, trace = 0, k = k)
    r <- MASS::stepAIC(ref, trace = 0, k = k)
    expect_setequal(attr(terms(s), "term.labels"), attr(terms(r), "term.labels"))
    # MASS counts n - df.residual (the rank plus the zero-weight rows): the
    # values differ by the same constant for every model.
    nzero <- sum(d$w == 0)
    expect_equal(extractAIC(nb, k = k)[1L], nb$rank)
    expect_equal(extractAIC(ref, k = k)[2L] - extractAIC(nb, k = k)[2L], k * nzero, tolerance = 1e-6)
    expect_equal(extractAIC(r, k = k)[2L] - extractAIC(s, k = k)[2L], k * nzero, tolerance = 1e-6)
})

test_that("step() of streamed fits returns a fit whose call can be evaluated and updated", {
    d <- drop1_data(n = 400L)
    set.seed(9); d$z <- rnorm(nrow(d))
    same <- function(a, b, info) expect_equal(unlist(coef(a)), unlist(coef(b)), tolerance = 1e-8, info = info)
    fits <- list(
        glm = fastglm_streaming(drop1_source(d, yp ~ a + b + z), family = poisson()),
        glm0 = fastglm_streaming(drop1_source(d, yp ~ a), family = poisson()),
        nb = fastglm_nb(drop1_source(d, ynb ~ a + b + z)),
        nb0 = fastglm_nb(drop1_source(d, ynb ~ a)),
        trunc = fastglm_truncated(drop1_source(d, yt ~ a + b + z)),
        ordinal = fastglm_ordinal_streaming(drop1_source(d, yo ~ a + b + z)),
        ordinal0 = fastglm_ordinal_streaming(drop1_source(d, yo ~ a)),
        multinom = fastglm_multinom(drop1_source(d, ym ~ a + z)))
    for (nm in names(fits)) {
        s <- suppressMessages(step(fits[[nm]], trace = 0))
        same(eval(s$call), s, nm)
        same(suppressMessages(update(s, . ~ .)), s, nm)
        tab <- suppressMessages(drop1(s))
        expect_identical(rownames(tab)[which.min(tab$AIC)], "<none>", info = nm)
    }
    # The selected models are those of the same fits held in memory.
    expect_identical(attr(terms(formula(suppressMessages(step(fits$nb, trace = 0)))), "term.labels"),
                     attr(terms(step(fastglm_nb(ynb ~ a + b + z, data = d), trace = 0)), "term.labels"))
    expect_identical(attr(terms(formula(suppressMessages(step(fits$ordinal, trace = 0)))), "term.labels"),
                     attr(terms(formula(step(fastglm_ordinal(yo ~ a + b + z, data = d), trace = 0))), "term.labels"))
    # A formula with a design matrix is an error.
    expect_error(fastglm_nb(model.matrix(~ a, d), d$ynb, formula = ynb ~ a), "not with a design matrix")
})

test_that("step() and stepAIC() warn once for models with two linear predictors; direct use does not warn", {
    skip_if_not_installed("MASS")
    d <- drop1_data()
    set.seed(9); d$noise <- rnorm(nrow(d))
    fits <- list(zi = fastglm_zi(yzi ~ a + b + noise | a + noise, data = d),
                 hurdle = fastglm_hurdle(yzi ~ a + b | a, data = d),
                 twopart = fastglm_twopart(yg ~ a + b | a, data = d))
    count <- function(expr) {
        n <- 0L
        value <- withCallingHandlers(expr, warning = function(w) {
            if (grepl("does not select terms", conditionMessage(w))) n <<- n + 1L
            invokeRestart("muffleWarning")
        })
        list(n = n, value = value)
    }
    for (nm in names(fits)) {
        f <- fits[[nm]]
        r <- count(step(f, trace = 0))
        expect_identical(r$n, 1L, info = nm)
        expect_equal(unlist(coef(r$value)), unlist(coef(f)), info = nm)
        r <- count(MASS::stepAIC(f, trace = 0))
        expect_identical(r$n, 1L, info = nm)
        expect_identical(count(MASS::stepAIC(f, trace = 0, k = log(nobs(f))))$n, 1L, info = nm)
        # Direct use never warns.
        expect_no_warning(extractAIC(f))
        expect_no_warning(extractAIC(f, k = 3))
        expect_no_warning(AIC(f))
        expect_no_warning(BIC(f))
        expect_no_warning(drop1(f, test = "LRT"))
        expect_no_warning(stats::extractAIC(f))
    }
    # Other classes do not warn under step().
    expect_no_warning(step(fastglm_nb(ynb ~ a + b, data = d), trace = 0))
    expect_no_warning(MASS::stepAIC(fastglm_ordinal(yo ~ a + b, data = d), trace = 0))
    # A wrapper that calls step() still warns (the caller of extractAIC() is step()).
    wrap <- function(m) step(m, trace = 0)
    expect_identical(count(wrap(fits$zi))$n, 1L)
})
