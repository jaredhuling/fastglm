trace_pattern <- "^iter +[0-9]+  passes +[0-9]+  (deviance|loglik) [-0-9.e+]+  halvings [0-9]+  [0-9.]+s$"

trace_lines <- function(out) grep("^iter ", out, value = TRUE)

test_that("trace prints one line per iteration and is off by default", {
    set.seed(1); x <- cbind(1, rnorm(500)); y <- rbinom(500, 1, 0.4)
    out <- capture.output(f <- fastglm(x, y, family = binomial(), trace = TRUE))
    expect_length(grep("^iter +[0-9]+ +passes +[0-9]+ +deviance ", out), f$iter)
    expect_true(all(grepl(trace_pattern, out)))
    # The last line reports the fit's deviance and the iterations are numbered 1, 2, ...
    expect_equal(as.numeric(sub(".*deviance ([^ ]+) .*", "\\1", out[length(out)])), f$deviance,
                 tolerance = 1e-9)
    expect_identical(as.integer(sub("^iter +([0-9]+) .*", "\\1", out)), seq_len(f$iter))
    expect_silent(fastglm(x, y, family = binomial()))
    old <- options(fastglm.trace = TRUE)
    on.exit(options(old), add = TRUE)
    expect_output(fastglm(x, y, family = binomial()), "iter +1 ")
    # An explicit FALSE overrides the option.
    expect_silent(fastglm(x, y, family = binomial(), trace = FALSE))
})

test_that("trace leaves the fit unchanged and restores the session setting", {
    set.seed(2); x <- cbind(1, matrix(rnorm(2000), 1000)); y <- rpois(1000, exp(0.3 + 0.2 * x[, 2]))
    capture.output(a <- fastglm(x, y, family = poisson(), method = 2L, trace = TRUE))
    b <- fastglm(x, y, family = poisson(), method = 2L)
    expect_identical(coef(a), coef(b))
    expect_identical(a$iter, b$iter)
    expect_null(getOption("fastglm.trace"))
    expect_false(fastglm:::fastglm_set_trace_cpp(FALSE))
    expect_error(fastglm(x, y, family = poisson(), trace = NA), "trace")
    expect_error(fastglm(x, y, family = poisson(), trace = "yes"), "trace")
})

test_that("sparse, big.matrix, Firth, and glm(method = fastglm_fit) fits trace", {
    set.seed(3); x <- cbind(1, rnorm(400)); y <- rbinom(400, 1, 0.3)
    out <- capture.output(f <- fastglm(Matrix::Matrix(x, sparse = TRUE), y, family = binomial(),
                                       method = 2L, trace = TRUE))
    expect_length(trace_lines(out), f$iter)
    out <- capture.output(f <- fastglm(x, y, family = binomial(), firth = TRUE, trace = TRUE))
    expect_length(trace_lines(out), f$iter)
    d <- data.frame(y = y, x = x[, 2])
    out <- capture.output(g <- glm(y ~ x, family = binomial(), data = d, method = fastglm_fit,
                                   control = list(trace = TRUE)))
    expect_true(length(trace_lines(out)) >= 1L)
    out <- capture.output(g <- fastglm(y ~ x, data = d, family = binomial(), trace = TRUE))
    expect_true(length(trace_lines(out)) >= 1L)
    skip_if_not_installed("bigmemory")
    bx <- bigmemory::as.big.matrix(x)
    out <- capture.output(f <- fastglm(bx, y, family = binomial(), method = 2L, trace = TRUE))
    expect_length(trace_lines(out), f$iter)
})

test_that("a streamed fit is silent by default when not interactive and traces on request", {
    expect_false(interactive())
    set.seed(4); x <- cbind(1, rnorm(3000)); y <- rbinom(3000, 1, plogis(0.2 + 0.5 * x[, 2]))
    expect_silent(f <- fastglm_streaming(x, y = y, family = binomial(), block.size = 1000L))
    out <- capture.output(g <- fastglm_streaming(x, y = y, family = binomial(), block.size = 1000L,
                                                 trace = TRUE))
    lines <- trace_lines(out)
    expect_length(lines, g$iter)
    expect_true(all(grepl(trace_pattern, lines)))
    expect_identical(coef(f), coef(g))
    # Passes count the passes over the source, the validation pass included.
    passes <- as.integer(sub(".*passes +([0-9]+) .*", "\\1", lines))
    expect_true(all(diff(passes) >= 1L))
    expect_true(max(passes) <= g$passes)
    # A QR streamed fit traces as well.
    out <- capture.output(h <- fastglm_streaming(x, y = y, family = binomial(), method = 0L,
                                                 block.size = 1000L, trace = TRUE))
    expect_length(trace_lines(out), h$iter)
})

test_that("multinomial fits trace with control$trace and keep it out of the refit controls", {
    set.seed(5); n <- 600; x <- cbind(x1 = rnorm(n), x2 = rnorm(n))
    y <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
    out <- capture.output(f <- fastglm_multinom(x, y, control = list(trace = TRUE)))
    lines <- trace_lines(out)
    expect_length(lines, f$iter)
    expect_true(all(grepl("deviance", lines)))
    expect_true(all(grepl(trace_pattern, lines)))
    expect_silent(g <- fastglm_multinom(x, y))
    expect_identical(coef(f), coef(g))
    expect_error(fastglm_multinom(x, y, control = list(trace = 1)), "control\\$trace")
    # A source fit traces on request.
    src <- fastglm_source(function(k) {
        i <- (k - 1L) * 200L + seq_len(200L)
        list(X = cbind("(Intercept)" = 1, x[i, ]), y = y[i])
    }, 3L)
    out <- capture.output(s <- fastglm_multinom(src, control = list(trace = TRUE)))
    expect_true(length(trace_lines(out)) >= 1L)
})

test_that("ordinal fits trace with loglik in the common format", {
    set.seed(6); n <- 800; x <- cbind(a = rnorm(n), b = rnorm(n))
    y <- cut(x[, 1] + rlogis(n), c(-Inf, -1, 0, 1, Inf), ordered_result = TRUE)
    out <- capture.output(f <- fastglm_ordinal(x, y, control = list(trace = TRUE)))
    lines <- trace_lines(out)
    expect_length(lines, f$iterations)
    expect_true(all(grepl("loglik", lines)))
    expect_true(all(grepl(trace_pattern, lines)))
    expect_silent(fastglm_ordinal(x, y))
    old <- options(fastglm.trace = TRUE)
    on.exit(options(old), add = TRUE)
    expect_output(fastglm_ordinal(x, y), "loglik")
    options(old)
    # Partial proportional odds and streamed fits.
    out <- capture.output(fastglm_ordinal(x, y, nonparallel = "b", control = list(trace = TRUE)))
    expect_true(length(trace_lines(out)) >= 1L && all(grepl("loglik", trace_lines(out))))
    src <- function(k) { i <- (k - 1L) * 400L + seq_len(400L); list(X = x[i, ], y = y[i]) }
    expect_silent(s <- fastglm_ordinal_streaming(src, 2L))
    out <- capture.output(s <- fastglm_ordinal_streaming(src, 2L, control = list(trace = TRUE)))
    expect_true(length(trace_lines(out)) >= 1L && all(grepl("loglik", trace_lines(out))))
    expect_error(fastglm_ordinal(x, y, control = list(trace = NA)), "trace")
})

test_that("count models trace their iterations", {
    set.seed(7); n <- 1500; x <- cbind(1, rnorm(n)); mu <- exp(0.5 + 0.4 * x[, 2])
    y <- rnbinom(n, mu = mu, size = 2)
    out <- capture.output(f <- fastglm_nb(x, y, trace = TRUE))
    expect_true(length(trace_lines(out)) >= 1L)
    expect_true(all(grepl(trace_pattern, trace_lines(out))))
    expect_silent(fastglm_nb(x, y))
    yp <- y + 1
    out <- capture.output(fastglm_truncated(x, yp, dist = "negbin", trace = TRUE))
    expect_true(length(trace_lines(out)) >= 1L && all(grepl("loglik", trace_lines(out))))
    out <- capture.output(fastglm_truncated(x, yp, trace = TRUE))
    expect_true(length(trace_lines(out)) >= 1L)
    d <- data.frame(y = y, x = x[, 2])
    out <- capture.output(fastglm_zi(y ~ x, data = d, trace = TRUE))
    expect_true(length(trace_lines(out)) >= 1L && all(grepl("loglik", trace_lines(out))))
    out <- capture.output(fastglm_hurdle(y ~ x, data = d, trace = TRUE))
    expect_true(length(trace_lines(out)) >= 2L)
    expect_silent(fastglm_hurdle(y ~ x, data = d))
    # A source-backed negative binomial fit traces its outer iterations, and
    # not those of the fits it runs internally.
    src <- fastglm_source(function(k) { i <- (k - 1L) * 500L + seq_len(500L); list(X = x[i, ], y = y[i]) }, 3L)
    out <- capture.output(g <- fastglm_nb(src, trace = TRUE, poisson.limit = FALSE))
    expect_true(length(trace_lines(out)) >= 1L)
    expect_identical(length(trace_lines(out)), as.integer(g$iter.theta))
})

test_that("bootstrap and batch refits are not traced", {
    set.seed(8); x <- cbind(1, rnorm(300)); y <- rbinom(300, 1, 0.4)
    f <- fastglm(x, y, family = binomial())
    old <- options(fastglm.trace = TRUE)
    on.exit(options(old), add = TRUE)
    out <- capture.output(b <- fastglm_bootstrap(f, R = 5L, seed = 1L))
    expect_length(trace_lines(out), 0L)
    expect_true(isTRUE(getOption("fastglm.trace")))
    design <- fastglm_prepare(x)
    out <- capture.output(fits <- fastglm_batch(design, list(y, 1 - y), family = binomial()))
    expect_length(trace_lines(out), 0L)
})

test_that("an interrupt during a streamed fit unwinds and restores the session settings", {
    skip_on_cran()
    skip_on_os("windows")
    set.seed(9); x <- cbind(1, rnorm(4000)); y <- rbinom(4000, 1, 0.4)
    session <- fastglm_threads()$threads
    fastglm_threads(1L)
    on.exit(fastglm_threads(session), add = TRUE)
    old <- options(fastglm.trace = NULL)
    on.exit(options(old), add = TRUE)
    reads <- 0L
    src <- function(k) {
        reads <<- reads + 1L
        # Interrupt during the second pass (after the validation pass).
        if (reads == 6L) { tools::pskill(Sys.getpid(), tools::SIGINT); Sys.sleep(0.5) }
        i <- (k - 1L) * 1000L + seq_len(1000L)
        list(X = x[i, ], y = y[i])
    }
    got <- tryCatch({
        fastglm_streaming(src, 4L, family = binomial(), threads = 2L, trace = TRUE)
        "finished"
    }, interrupt = function(e) "interrupted")
    expect_identical(got, "interrupted")
    expect_true(reads < 12L)
    # The thread setting, the trace flag, and the trace option are restored.
    expect_identical(fastglm_threads()$threads, 1L)
    expect_identical(getOption("fastglm.threads"), 1L)
    expect_null(getOption("fastglm.trace"))
    expect_false(fastglm:::fastglm_set_trace_cpp(FALSE))
    # The session is usable: a following fit works and matches a fresh fit.
    f <- fastglm(x, y, family = binomial(), method = 2L)
    expect_true(f$converged)
    g <- fastglm_streaming(x, y = y, family = binomial(), block.size = 1000L)
    expect_equal(unname(coef(g)), unname(coef(f)), tolerance = 1e-6)
})

test_that("the trace default follows the option, interactivity, and knitr", {
    resolve <- fastglm:::.fastglm_resolve_trace
    old <- options(fastglm.trace = NULL, knitr.in.progress = NULL)
    on.exit(options(old), add = TRUE)
    expect_false(resolve(NULL, streamed = FALSE))
    # Not interactive here, so streamed fits are off by default too.
    expect_identical(resolve(NULL, streamed = TRUE), interactive())
    options(knitr.in.progress = TRUE)
    expect_false(resolve(NULL, streamed = TRUE))
    options(fastglm.trace = TRUE)
    expect_true(resolve(NULL, streamed = FALSE))
    expect_false(resolve(FALSE, streamed = TRUE))
    expect_error(resolve(c(TRUE, FALSE)), "trace")
})

test_that("an ordinal trace measures elapsed time from the start of its own fit", {
    set.seed(10); n <- 600; x <- cbind(a = rnorm(n), b = rnorm(n))
    y <- cut(x[, 1] + rlogis(n), c(-Inf, -1, 0, 1, Inf), ordered_result = TRUE)
    elapsed <- function(out) as.numeric(sub(".*  ([0-9.]+)s$", "\\1", trace_lines(out)))
    capture.output(fastglm(cbind(1, x), as.numeric(as.numeric(y) > 2), family = binomial(), trace = TRUE))
    Sys.sleep(1.1)
    out <- capture.output(fastglm_ordinal(x, y, control = list(trace = TRUE)))
    expect_lt(elapsed(out)[1], 1)
    Sys.sleep(1.1)
    out <- capture.output(fastglm_ordinal(x, y, nonparallel = "b", control = list(trace = TRUE)))
    expect_lt(elapsed(out)[1], 1)
    Sys.sleep(1.1)
    src <- function(k) { i <- (k - 1L) * 300L + seq_len(300L); list(X = x[i, ], y = y[i]) }
    out <- capture.output(fastglm_ordinal_streaming(src, 2L, control = list(trace = TRUE)))
    expect_lt(elapsed(out)[1], 1)
    # The fit keeps trace = FALSE in its controls, so refits from them are silent.
    capture.output(fit <- fastglm_ordinal(x, y, control = list(trace = TRUE)))
    expect_false(fit$control$trace)
})

test_that("inference refits are not traced under options(fastglm.trace = TRUE)", {
    set.seed(11); n <- 400
    d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), o = runif(n))
    d$y <- rbinom(n, 1, plogis(0.3 + d$x1 + d$o))
    capture.output(f <- fastglm(y ~ x1 + x2, data = d, family = binomial()))
    d$z <- cut(d$x1 + rlogis(n), c(-Inf, -1, 0, 1, Inf), ordered_result = TRUE)
    capture.output(o <- fastglm_ordinal(z ~ x1 + x2, data = d, control = list(trace = TRUE)))
    expect_false(isTRUE(o$control$trace))
    old <- options(fastglm.trace = TRUE)
    on.exit(options(old), add = TRUE)
    expect_length(trace_lines(capture.output(a <- anova(f, test = "Chisq"))), 0L)
    expect_length(trace_lines(capture.output(b <- drop1(f, test = "LRT"))), 0L)
    if (getRversion() >= "4.4.0")
        expect_length(trace_lines(capture.output(p <- profile(f))), 0L)
    expect_length(trace_lines(capture.output(ci <- confint(o, method = "profile"))), 0L)
    expect_length(trace_lines(capture.output(st <- fastglm_score_test(o, terms = "x2"))), 0L)
    expect_length(trace_lines(capture.output(po <- fastglm_po_test(o))), 0L)
    expect_true(isTRUE(getOption("fastglm.trace")))
    # glm(method = fastglm_fit) with an offset traces the fit only, not the
    # intercept-only refit that glm() makes for the null deviance.
    options(fastglm.trace = NULL)
    out <- capture.output(g <- glm(y ~ x1 + offset(o), data = d, family = binomial(),
                                   method = fastglm_fit, control = list(trace = TRUE)))
    expect_length(trace_lines(out), g$iter)
})

test_that("drop1, add1, profile, anova, and update work on formula fits as on glm", {
    set.seed(12); n <- 500
    d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
    d$y <- rbinom(n, 1, plogis(-0.2 + 0.8 * d$x1 - 0.4 * d$x2))
    f <- fastglm(y ~ x1 + x2, data = d, family = binomial())
    g <- glm(y ~ x1 + x2, data = d, family = binomial())
    df <- drop1(f, test = "LRT"); dg <- drop1(g, test = "LRT")
    expect_equal(as.matrix(df), as.matrix(dg), tolerance = 1e-8)
    af <- add1(f, ~ . + x3, test = "LRT"); ag <- add1(g, ~ . + x3, test = "LRT")
    expect_equal(as.matrix(af), as.matrix(ag), tolerance = 1e-8)
    if (getRversion() >= "4.4.0") {   # profile.glm() is in stats from R 4.4.0
        # profile.glm() steps in units of the standard errors; glm() takes
        # them from its last iteration, which covariance = "previous" matches.
        pf <- profile(update(f, covariance = "previous")); pg <- profile(g)
        expect_identical(names(pf), names(pg))
        for (nm in names(pg)) {
            expect_equal(pf[[nm]]$z, pg[[nm]]$z, tolerance = 1e-8)
            expect_equal(pf[[nm]]$par.vals, pg[[nm]]$par.vals, tolerance = 1e-8)
        }
    }
    an <- anova(f, test = "Chisq"); ang <- anova(g, test = "Chisq")
    expect_equal(as.matrix(an), as.matrix(ang), tolerance = 1e-8)
    u <- update(f, . ~ . - x2)
    expect_equal(unname(coef(u)), unname(coef(glm(y ~ x1, data = d, family = binomial()))),
                 tolerance = 1e-8)
    # A Gaussian fit (dispersion estimated) as well.
    h <- fastglm(x3 ~ x1 + x2, data = d)
    expect_equal(as.matrix(drop1(h, test = "F")),
                 as.matrix(drop1(glm(x3 ~ x1 + x2, data = d), test = "F")), tolerance = 1e-8)
    # Bias-reduced fits are refused: glm.fit() would refit by maximum likelihood.
    fb <- fastglm(y ~ x1 + x2, data = d, family = binomial(), firth = TRUE)
    expect_error(drop1(fb), "bias-reduced")
})
