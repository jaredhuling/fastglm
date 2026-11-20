# parsnip engines: fit and predict agree with direct fastglm fits.

skip_if_not_installed("parsnip")
skip_if_not_installed("tibble")

suppressPackageStartupMessages(library(parsnip))

parsnip_data <- function(n = 160L, seed = 61L) {
    set.seed(seed)
    d <- data.frame(x1 = rnorm(n), x2 = rnorm(n),
                    g = factor(sample(c("u", "v", "w"), n, replace = TRUE)))
    eta <- -0.3 + 0.8 * d$x1 - 0.5 * d$x2 + 0.4 * (d$g == "v")
    # The first level ("yes") is the event for yardstick; glm and fastglm
    # model the probability of the second level ("no").
    d$yb <- factor(ifelse(runif(n) < plogis(eta), "no", "yes"), levels = c("yes", "no"))
    d$yc <- rpois(n, exp(0.2 + 0.3 * d$x1 - 0.2 * d$x2))
    d$yg <- 1 + d$x1 - 2 * d$x2 + rnorm(n)
    lin <- cbind(0, 0.6 * d$x1, -0.7 * d$x2 + 0.3)
    p <- exp(lin) / rowSums(exp(lin))
    # Levels in an order that is not alphabetical.
    d$ym <- factor(c("c", "a", "b")[apply(p, 1L, function(q) sample.int(3L, 1L, prob = q))],
                   levels = c("c", "a", "b"))
    z <- 0.9 * d$x1 - 0.6 * d$x2 + rlogis(n)
    d$yo <- factor(cut(z, c(-Inf, -0.8, 0.4, Inf), labels = c("low", "mid", "high")),
                   levels = c("low", "mid", "high"), ordered = TRUE)
    d$w <- sample(1:3, n, replace = TRUE)
    d
}

pd <- parsnip_data()
newd <- parsnip_data(n = 25L, seed = 62L)

test_that("logistic_reg: probabilities in level order match fastglm and glm (Review Focus 5)", {
    spec <- logistic_reg() |> set_engine("fastglm")
    fit <- fit(spec, yb ~ x1 + x2 + g, data = pd)
    expect_s3_class(extract_fit_engine(fit), "fastglmFit")
    direct <- fastglm(yb ~ x1 + x2 + g, data = pd, family = binomial())
    p2 <- unname(predict(direct, newd, type = "response"))
    pr <- predict(fit, newd, type = "prob")
    expect_named(pr, c(".pred_yes", ".pred_no"))
    expect_equal(pr$.pred_no, p2, tolerance = 1e-10)
    expect_equal(pr$.pred_yes, 1 - p2, tolerance = 1e-10)
    ref <- unname(predict(glm(yb ~ x1 + x2 + g, data = pd, family = binomial()), newd,
                          type = "response"))
    expect_equal(pr$.pred_no, ref, tolerance = 1e-7)
    cls <- predict(fit, newd, type = "class")$.pred_class
    expect_identical(levels(cls), c("yes", "no"))
    expect_identical(as.character(cls), ifelse(p2 >= 0.5, "no", "yes"))
    # The fit is the fastglm fit, with one thread unless threads is given.
    expect_identical(extract_fit_engine(fit)$threads, 1L)
    expect_equal(coef(extract_fit_engine(fit)), coef(direct), tolerance = 1e-10)
})

test_that("logistic_reg engine arguments: link, firth, method, threads", {
    spec <- logistic_reg() |> set_engine("fastglm", link = "probit", firth = TRUE, method = 2L)
    fit <- fit(spec, yb ~ x1 + x2, data = pd)
    direct <- fastglm(yb ~ x1 + x2, data = pd, family = binomial(link = "probit"),
                      firth = TRUE, method = 2L)
    expect_identical(extract_fit_engine(fit)$family$link, "probit")
    expect_equal(predict(fit, newd, type = "prob")$.pred_no,
                 unname(predict(direct, newd, type = "response")), tolerance = 1e-10)
    # Engine fits use one thread unless threads is given (dense QR fits are
    # always serial, so the Cholesky decomposition shows the setting).
    fit1 <- fit(logistic_reg() |> set_engine("fastglm", method = 2L), yb ~ x1, data = pd)
    expect_identical(extract_fit_engine(fit1)$threads, 1L)
    fit2 <- fit(logistic_reg() |> set_engine("fastglm", method = 2L, threads = 2L),
                yb ~ x1, data = pd)
    expect_identical(extract_fit_engine(fit2)$threads,
                     fastglm:::.fastglm_resolve_threads(2L))
    # Results are identical for every number of threads.
    expect_identical(coef(extract_fit_engine(fit2)), coef(extract_fit_engine(fit1)))
})

test_that("logistic_reg with case weights matches weighted fastglm", {
    fit <- fit(logistic_reg() |> set_engine("fastglm"), yb ~ x1 + x2, data = pd,
               case_weights = parsnip::frequency_weights(pd$w))
    direct <- fastglm(yb ~ x1 + x2, data = pd, family = binomial(), weights = w)
    expect_equal(predict(fit, newd, type = "prob")$.pred_no,
                 unname(predict(direct, newd, type = "response")), tolerance = 1e-10)
})

test_that("linear_reg: numeric predictions match fastglm", {
    fit <- fit(linear_reg() |> set_engine("fastglm"), yg ~ x1 + x2 + g, data = pd)
    direct <- fastglm(yg ~ x1 + x2 + g, data = pd)
    pr <- predict(fit, newd)
    expect_named(pr, ".pred")
    expect_equal(pr$.pred, unname(predict(direct, newd, type = "response")), tolerance = 1e-10)
    expect_equal(pr$.pred, unname(predict(lm(yg ~ x1 + x2 + g, data = pd), newd)), tolerance = 1e-8)
})

test_that("poisson_reg: numeric predictions match fastglm", {
    skip_if_not("poisson_reg" %in% ls(parsnip::get_model_env()))
    fit <- fit(poisson_reg() |> set_engine("fastglm"), yc ~ x1 + x2, data = pd)
    direct <- fastglm(yc ~ x1 + x2, data = pd, family = poisson())
    expect_equal(predict(fit, newd)$.pred, unname(predict(direct, newd, type = "response")),
                 tolerance = 1e-10)
    fit2 <- fit(poisson_reg() |> set_engine("fastglm", link = "sqrt", firth = TRUE),
                yc ~ x1 + x2, data = pd)
    direct2 <- fastglm(yc ~ x1 + x2, data = pd, family = poisson(link = "sqrt"), firth = TRUE)
    expect_equal(predict(fit2, newd)$.pred, unname(predict(direct2, newd, type = "response")),
                 tolerance = 1e-10)
})

test_that("multinom_reg: probability columns follow the outcome levels", {
    fit <- fit(multinom_reg() |> set_engine("fastglm"), ym ~ x1 + x2 + g, data = pd)
    expect_s3_class(extract_fit_engine(fit), "fastglm_multinom")
    expect_identical(extract_fit_engine(fit)$threads, 1L)
    direct <- fastglm_multinom(ym ~ x1 + x2 + g, data = pd)
    pdir <- predict(direct, newd, type = "probability")
    pr <- predict(fit, newd, type = "prob")
    expect_named(pr, c(".pred_c", ".pred_a", ".pred_b"))
    expect_equal(unname(as.matrix(pr)), unname(pdir[, c("c", "a", "b")]), tolerance = 1e-10)
    cls <- predict(fit, newd, type = "class")$.pred_class
    expect_identical(levels(cls), c("c", "a", "b"))
    expect_identical(as.character(cls), as.character(predict(direct, newd, type = "class")))
    skip_if_not_installed("nnet")
    ref <- nnet::multinom(ym ~ x1 + x2 + g, data = pd, trace = FALSE, reltol = 1e-14, maxit = 1000)
    expect_equal(unname(as.matrix(pr)), unname(predict(ref, newd, type = "probs")), tolerance = 1e-6)
    fitf <- fit(multinom_reg() |> set_engine("fastglm", firth = TRUE), ym ~ x1 + x2, data = pd)
    dirf <- fastglm_multinom(ym ~ x1 + x2, data = pd, firth = TRUE)
    expect_equal(unname(as.matrix(predict(fitf, newd, type = "prob"))),
                 unname(predict(dirf, newd, type = "probability")), tolerance = 1e-10)
})

test_that("ordinal_reg: probabilities and classes match fastglm_ordinal", {
    skip_if_not("ordinal_reg" %in% ls(parsnip::get_model_env()))
    fit <- fit(ordinal_reg() |> set_engine("fastglm"), yo ~ x1 + x2 + g, data = pd)
    expect_s3_class(extract_fit_engine(fit), "fastglm_ordinal")
    direct <- fastglm_ordinal(yo ~ x1 + x2 + g, data = pd)
    pr <- predict(fit, newd, type = "prob")
    expect_named(pr, c(".pred_low", ".pred_mid", ".pred_high"))
    expect_equal(unname(as.matrix(pr)), unname(predict(direct, newd, type = "probability")),
                 tolerance = 1e-10)
    cls <- predict(fit, newd, type = "class")$.pred_class
    expect_identical(levels(cls), c("low", "mid", "high"))
    expect_identical(as.character(cls), as.character(predict(direct, newd, type = "class")))
    skip_if_not_installed("MASS")
    ref <- MASS::polr(yo ~ x1 + x2 + g, data = pd)
    expect_equal(unname(as.matrix(pr)), unname(predict(ref, newd, type = "probs")), tolerance = 1e-5)
    # The ordinal_link main argument maps to the link of fastglm_ordinal().
    fitp <- fit(ordinal_reg(ordinal_link = "probit") |> set_engine("fastglm", firth = TRUE),
                yo ~ x1 + x2, data = pd)
    dirp <- fastglm_ordinal(yo ~ x1 + x2, data = pd, link = "probit", firth = TRUE)
    expect_equal(unname(as.matrix(predict(fitp, newd, type = "prob"))),
                 unname(predict(dirp, newd, type = "probability")), tolerance = 1e-10)
    expect_error(fit(ordinal_reg(ordinal_link = "cauchit") |> set_engine("fastglm"),
                     yo ~ x1, data = pd), "cauchit")
    # Only cumulative links with flexible thresholds are fitted; other odds
    # links and threshold structures are errors, never silently ignored.
    fitc <- fit(ordinal_reg(odds_link = "cumulative_link", threshold_structure = "flexible") |>
                    set_engine("fastglm"), yo ~ x1 + x2, data = pd)
    expect_equal(unname(as.matrix(predict(fitc, newd, type = "prob"))),
                 unname(predict(fastglm_ordinal(yo ~ x1 + x2, data = pd), newd,
                                type = "probability")), tolerance = 1e-10)
    for (odds in c("adjacent_categories", "continuation_ratio", "stopping_ratio"))
        expect_error(fit(ordinal_reg(odds_link = odds) |> set_engine("fastglm"), yo ~ x1, data = pd),
                     "cumulative_link")
    expect_error(fit(ordinal_reg(threshold_structure = "equidistant") |> set_engine("fastglm"),
                     yo ~ x1, data = pd), "flexible")
})

test_that("conf_int predictions match the glm engine", {
    # Tight convergence on both sides, and the covariance of glm.fit()
    # (weights of the last iteration, covariance = "previous"), so the
    # comparison is not limited by the convergence tolerance. With the
    # default covariance = "estimate", the logistic standard errors differ
    # from glm's by about 1e-8 here: glm's are those of the next-to-last
    # iterate.
    glm.ctl <- stats::glm.control(epsilon = 1e-14, maxit = 100)
    for (lvl in c(0.95, 0.8)) {
        fb <- fit(logistic_reg() |> set_engine("fastglm", tol = 1e-14, covariance = "previous"),
                  yb ~ x1 + x2 + g, data = pd)
        gb <- fit(logistic_reg() |> set_engine("glm", control = glm.ctl), yb ~ x1 + x2 + g, data = pd)
        cf <- predict(fb, newd, type = "conf_int", level = lvl, std_error = TRUE)
        cg <- predict(gb, newd, type = "conf_int", level = lvl, std_error = TRUE)
        expect_named(cf, names(cg))
        expect_equal(as.data.frame(cf), as.data.frame(cg), tolerance = 1e-8)
        fl <- fit(linear_reg() |> set_engine("fastglm", tol = 1e-14), yg ~ x1 + x2 + g, data = pd)
        gl <- fit(linear_reg() |> set_engine("glm", control = glm.ctl), yg ~ x1 + x2 + g, data = pd)
        cf <- predict(fl, newd, type = "conf_int", level = lvl, std_error = TRUE)
        cg <- predict(gl, newd, type = "conf_int", level = lvl, std_error = TRUE)
        expect_named(cf, c(".pred_lower", ".pred_upper", ".std_error"))
        expect_equal(as.data.frame(cf), as.data.frame(cg), tolerance = 1e-8)
    }
})

test_that("registration is idempotent and declares fastglm as a dependency", {
    fastglm:::.fastglm_register_parsnip()
    fastglm:::.fastglm_register_poissonreg()
    fastglm:::.fastglm_register_ordered()
    env <- parsnip::get_model_env()
    for (m in c("logistic_reg", "linear_reg", "multinom_reg", "poisson_reg", "ordinal_reg")) {
        if (is.null(env[[m]])) next
        expect_identical(sum(env[[m]]$engine == "fastglm"), 1L)
        pkgs <- env[[paste0(m, "_pkgs")]]
        expect_true("fastglm" %in% unlist(pkgs$pkg[pkgs$engine == "fastglm"]))
    }
    expect_true("fastglm" %in% parsnip::required_pkgs(logistic_reg() |> set_engine("fastglm")))
})

test_that("a workflow with a recipe matches a direct fit on the processed data", {
    skip_if_not_installed("workflows")
    skip_if_not_installed("recipes")
    rec <- recipes::recipe(yb ~ x1 + x2 + g, data = pd) |>
        recipes::step_normalize(recipes::all_numeric_predictors()) |>
        recipes::step_dummy(recipes::all_nominal_predictors())
    wf <- workflows::workflow(rec, logistic_reg() |> set_engine("fastglm"))
    wfit <- fit(wf, data = pd)
    baked <- recipes::bake(recipes::prep(rec, training = pd), new_data = NULL)
    bnew <- recipes::bake(recipes::prep(rec, training = pd), new_data = newd)
    direct <- fastglm(yb ~ ., data = baked, family = binomial())
    pr <- predict(wfit, newd, type = "prob")
    expect_named(pr, c(".pred_yes", ".pred_no"))
    expect_equal(pr$.pred_no, unname(predict(direct, bnew, type = "response")), tolerance = 1e-10)
})

# Run R code in a fresh R process (base R only) with this session's library
# path, returning its last line of output.
run_fresh_r <- function(code) {
    script <- tempfile(fileext = ".R")
    on.exit(unlink(script), add = TRUE)
    writeLines(code, script)
    libs <- paste0("R_LIBS=", shQuote(paste(.libPaths(), collapse = .Platform$path.sep)))
    out <- suppressWarnings(system2(file.path(R.home("bin"), "Rscript"), c("--vanilla", shQuote(script)),
                                    stdout = TRUE, stderr = FALSE, env = libs))
    out[length(out)]
}

test_that("loading parsnip after fastglm registers the engines", {
    skip_on_cran()
    models <- 'c("logistic_reg", "linear_reg", "multinom_reg", "poisson_reg", "ordinal_reg")'
    count <- paste0('env <- parsnip::get_model_env(); ',
                    'cat(vapply(', models, ', function(m) if (is.null(env[[m]])) "NA" else ',
                    'as.character(sum(env[[m]]$engine == "fastglm")), ""), "\\n")')
    expected <- function(out) {
        v <- strsplit(trimws(out), " +")[[1L]]
        expect_length(v, 5L)
        expect_true(all(v[1:4] == "1"))
        expect_true(v[5] %in% c("1", "NA"))
    }
    # fastglm first: registered when parsnip loads.
    out <- run_fresh_r(c("suppressPackageStartupMessages(library(fastglm))",
                         "stopifnot(!'parsnip' %in% loadedNamespaces())",
                         "suppressPackageStartupMessages(library(parsnip))", count))
    expected(out)
    # parsnip first: registered when fastglm loads.
    out <- run_fresh_r(c("suppressPackageStartupMessages(library(parsnip))",
                         "suppressPackageStartupMessages(library(fastglm))", count))
    expected(out)
    # parsnip loaded first, then unloaded and loaded again: registered again.
    out <- run_fresh_r(c("suppressPackageStartupMessages(library(parsnip))",
                         "suppressPackageStartupMessages(library(fastglm))",
                         "detach('package:parsnip', unload = TRUE)",
                         "suppressPackageStartupMessages(loadNamespace('parsnip'))", count))
    expected(out)
    # poissonreg and ordered loaded after fastglm leave one fastglm engine.
    skip_if_not_installed("poissonreg")
    skip_if_not_installed("ordered")
    out <- run_fresh_r(c("suppressPackageStartupMessages(library(fastglm))",
                         "suppressPackageStartupMessages(library(poissonreg))",
                         "suppressPackageStartupMessages(library(ordered))", count))
    expected(out)
})
