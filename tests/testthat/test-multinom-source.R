# Streamed data sources for multinomial fits: a fit over a fastglm_source
# (callback, DBI, or arrow) must agree with the dense fit of the same rows to
# 1e-10, and inference must reread the source block by block.

source_sim_frame <- function(n = 700L, seed = 301L) {
    set.seed(seed)
    d <- data.frame(id = seq_len(n), x1 = rnorm(n), x2 = rnorm(n),
                    f = factor(sample(c("u", "v", "w"), n, TRUE)),
                    g = sample.int(40L, n, TRUE), w = runif(n, 0.5, 2),
                    o1 = rnorm(n, sd = 0.2), o2 = rnorm(n, sd = 0.2))
    eta <- cbind(0, 0.3 + 0.8 * d$x1 + d$o1, -0.2 - 0.5 * d$x2 + 0.4 * (d$f == "v") + d$o2)
    pr <- exp(eta) / rowSums(exp(eta))
    d$y <- factor(c("A", "B", "C")[apply(pr, 1L, function(q) sample.int(3L, 1L, prob = q))],
                  levels = c("A", "B", "C"))
    d$blk <- ceiling(seq_len(n) / ceiling(n / 7))
    d
}

source_blocks <- function(d) split(seq_len(nrow(d)), d$blk)

expect_source_agrees <- function(sf, de, tol = 1e-10) {
    expect_s3_class(sf, "fastglm_multinom")
    expect_true(sf$converged)
    expect_identical(names(sf$coefficients), names(de$coefficients))
    expect_max_diff(sf$coefficients, de$coefficients, tol)
    expect_max_diff(sf$cov.unscaled, de$cov.unscaled, tol)
    expect_max_diff(sf$loglik, de$loglik, tol)
    expect_max_diff(sf$deviance, de$deviance, tol)
    expect_max_diff(sf$null.deviance, de$null.deviance, tol)
    expect_identical(sf$nobs, de$nobs)
    expect_identical(sf$rank, de$rank)
    expect_identical(sf$df.residual, de$df.residual)
}

test_that("a callback source over 7 blocks agrees with the dense fit", {
    d <- source_sim_frame()
    idx <- source_blocks(d)
    expect_length(idx, 7L)
    src <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = y ~ x1 + x2 + f,
                          prototype = d[1:20, ], weights = "w", offset = c("o1", "o2"))
    de <- fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                           offset = cbind(o1, o2), tol = 1e-12)
    sf <- fastglm_multinom(src, tol = 1e-12)
    expect_source_agrees(sf, de)
    expect_identical(sf$passes, sf$iter + 1L)
    # Every pass over the source: the preliminary pass, the engine's passes,
    # and those of the intercept-only fit for the null deviance.
    null.passes <- fastglm_multinom(matrix(1, nrow(d), 1L), d$y, weights = d$w,
                                    offset = cbind(d$o1, d$o2), tol = 1e-12)$passes
    expect_identical(sf$source.passes, 1L + sf$passes + null.passes)
    expect_identical(sf$source, src)
    expect_null(sf$fitted.values)
    expect_identical(sf$levels, levels(d$y))

    # Bias reduction: one extra (adjustment) pass per iteration.
    def <- fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                            offset = cbind(o1, o2), firth = TRUE, tol = 1e-12)
    sff <- fastglm_multinom(src, firth = TRUE, tol = 1e-12)
    expect_source_agrees(sff, def)
    expect_max_diff(sff$penalized.loglik, def$penalized.loglik, 1e-10)
    expect_identical(sff$passes, 2L * sff$iter + 2L)
    expect_identical(sff$source.passes, 1L + sff$passes + null.passes)
})

test_that("prediction, sandwich, and cluster covariance agree with the dense fit", {
    d <- source_sim_frame()
    idx <- source_blocks(d)
    src <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = y ~ x1 + x2 + f,
                          prototype = d, weights = "w", offset = c("o1", "o2"))
    de <- fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                           offset = cbind(o1, o2), tol = 1e-12)
    sf <- fastglm_multinom(src, tol = 1e-12)

    # newdata: the source offset columns are read from newdata, as for
    # fastglm_streaming fits.
    nd <- d[c(5, 99, 400, 650), ]
    ref <- predict(de, newdata = nd, offset = cbind(nd$o1, nd$o2))
    expect_max_diff(predict(sf, newdata = nd), ref, 1e-10)
    expect_max_diff(predict(sf, newdata = nd, type = "link"),
                    predict(de, newdata = nd, type = "link", offset = cbind(nd$o1, nd$o2)), 1e-10)
    expect_max_diff(predict(sf, newdata = nd, se.fit = TRUE)$se.fit,
                    predict(de, newdata = nd, se.fit = TRUE, offset = cbind(nd$o1, nd$o2))$se.fit,
                    1e-10)
    expect_error(predict(sf, newdata = nd[, setdiff(names(nd), "o2")]), "o2")

    # Without newdata the source is reread block by block.
    expect_max_diff(predict(sf), predict(de), 1e-10)
    expect_identical(predict(sf, type = "class"), unname(predict(de, type = "class")))
    expect_max_diff(predict(sf, se.fit = TRUE)$se.fit, predict(de, se.fit = TRUE)$se.fit, 1e-10)

    # A sink receives every block; the concatenation is the in-memory result.
    got <- list()
    res <- predict(sf, sink = function(value, k) got[[k]] <<- value)
    expect_identical(res, 7L)
    expect_identical(vapply(got, nrow, 1L), unname(lengths(idx)))
    expect_max_diff(do.call(rbind, got), predict(de), 1e-12)
    expect_error(predict(de, sink = function(value, k) NULL), "sink")

    expect_max_diff(fastglm_vcov(sf), fastglm_vcov(de), 1e-10)
    expect_max_diff(fastglm_vcov(sf, "HC1"), fastglm_vcov(de, "HC1"), 1e-10)
    # Clusters are a vector over the rows in source order (formula clusters
    # are not accepted for vector GLM fits).
    expect_max_diff(fastglm_vcov(sf, cluster = d$g), fastglm_vcov(de, cluster = d$g), 1e-10)
    expect_max_diff(fastglm_vcov(sf, "HC1", cluster = d$g),
                    fastglm_vcov(de, "HC1", cluster = d$g), 1e-10)
    expect_error(fastglm_vcov(sf, cluster = d$g[-1]), "cluster")

    expect_max_diff(fastglm_wald_test(sf, "f")$statistic,
                    fastglm_wald_test(de, "f")$statistic, 1e-8)
    dg <- fastglm_diagnostics(sf)
    expect_max_diff(dg$score, fastglm_diagnostics(de)$score, 1e-8)
    expect_output(print(summary(sf)), "Multinomial")
    expect_error(residuals(sf), "source")
})

test_that("callback sources of model matrices, grouped counts, and vector offsets agree", {
    dd <- vglm_sim_data(n = 900L, p = 3L, K = 4L, seed = 17L)
    set.seed(3L)
    w <- runif(900L, 0.5, 2)
    off <- rnorm(900L, sd = 0.1)
    idx <- split(seq_len(900L), ceiling(seq_len(900L) / 129))
    # Individual data with a vector offset applied to every linear predictor.
    src <- fastglm_source(function(k) list(X = dd$x[idx[[k]], ], y = dd$y[idx[[k]]],
                                           weights = w[idx[[k]]], offset = off[idx[[k]]]), 7L)
    de <- fastglm_multinom(dd$x, dd$y, weights = w, offset = off, tol = 1e-12)
    sf <- fastglm_multinom(src, tol = 1e-12)
    expect_source_agrees(sf, de)
    expect_identical(sf$passes, sf$iter + 1L)
    expect_max_diff(fastglm_vcov(sf), fastglm_vcov(de), 1e-10)
    nd <- dd$x[1:5, ]
    expect_max_diff(predict(sf, newdata = nd, offset = off[1:5]),
                    predict(de, newdata = nd, offset = off[1:5]), 1e-10)
    # A nonzero source offset cannot be replayed for a design matrix.
    expect_error(predict(sf, newdata = nd), "offset")

    # Grouped counts with a reference category; levels from the count columns.
    srcg <- fastglm_source(function(k) list(X = dd$x[idx[[k]], ], y = dd$counts[idx[[k]], ],
                                            weights = w[idx[[k]]]), 7L)
    deg <- fastglm_multinom(dd$x, dd$counts, weights = w, reference = "C", tol = 1e-12)
    sfg <- fastglm_multinom(srcg, reference = "C", tol = 1e-12)
    expect_source_agrees(sfg, deg)
    expect_identical(sfg$reference, "C")
    expect_max_diff(fastglm_vcov(sfg), fastglm_vcov(deg), 1e-10)
})

test_that("response levels come from the prototype or 'levels'; unknown levels stop", {
    d <- source_sim_frame(n = 140L)
    idx <- source_blocks(d)
    bad <- d
    bad$y <- as.character(bad$y)
    bad$y[130] <- "Z"
    src <- fastglm_source(function(k) bad[idx[[k]], ], 7L, formula = y ~ x1,
                          prototype = d)
    expect_error(fastglm_multinom(src), "'Z'")

    # A character response is matched against the levels supplied.
    chr <- d
    chr$y <- as.character(chr$y)
    src <- fastglm_source(function(k) chr[idx[[k]], ], 7L, formula = y ~ x1,
                          prototype = chr)
    de <- fastglm_multinom(y ~ x1, data = d, reference = "B", tol = 1e-12)
    sf <- fastglm_multinom(src, levels = c("A", "B", "C"), reference = "B", tol = 1e-12)
    expect_max_diff(sf$coefficients, de$coefficients, 1e-10)
    expect_identical(names(sf$coefficients), names(de$coefficients))
    # A level declared but never observed is an empty category.
    expect_error(fastglm_multinom(src, levels = c("A", "B", "C", "D")), "'D'")
})

test_that("per-predictor source offsets generalize offset without changing existing models", {
    d <- source_sim_frame(n = 140L)
    idx <- source_blocks(d)
    d$o3 <- 0
    src <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = y ~ x1,
                          prototype = d, offset = c("o1", "o2", "o3"))
    expect_s3_class(src, "fastglm_source")
    # Three offset columns for a model with two linear predictors.
    expect_error(fastglm_multinom(src), "offset")
    # Existing models keep the message for more than one offset column with
    # a one-part formula.
    d$count <- rpois(nrow(d), 2)
    src2 <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = count ~ x1,
                           prototype = d, offset = c("o1", "o2"))
    expect_error(fastglm_streaming(src2, family = poisson()),
                 "two offset columns require a two-component formula")
    expect_error(fastglm_source(function(k) d, 1L, formula = count ~ x1 | x2,
                                prototype = d, offset = c("o1", "o2", "o3")),
                 "offset must name one or two data columns")
    expect_error(fastglm_source(function(k) d, 1L, offset = c("o1", "o2", "o3")),
                 "offset must name one or two data columns")
    # Three offset names with a one-part formula: accepted for multinomial
    # fits; existing models keep the message for more than two names.
    src3 <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = count ~ x1,
                           prototype = d, offset = c("o1", "o2", "o3"))
    expect_error(fastglm_streaming(src3, family = poisson()),
                 "offset must name one or two data columns")
    # One offset column keeps its vector form and applies to every linear
    # predictor, also in prediction from newdata.
    src1 <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = y ~ x1,
                           prototype = d, offset = "o1")
    expect_null(dim(src1$callback(1L)$offset))
    sf1 <- fastglm_multinom(src1, tol = 1e-12)
    de1 <- fastglm_multinom(y ~ x1, data = d, offset = o1, tol = 1e-12)
    expect_max_diff(sf1$coefficients, de1$coefficients, 1e-10)
    expect_max_diff(predict(sf1, newdata = d[1:4, ]), predict(de1, newdata = d[1:4, ]), 1e-10)
    expect_max_diff(predict(sf1, newdata = d[1:4, ], type = "link"),
                    predict(de1, newdata = d[1:4, ], type = "link"), 1e-10)
    expect_max_diff(fastglm_vcov(sf1), fastglm_vcov(de1), 1e-10)
    # M offset columns in prediction from newdata.
    src2 <- fastglm_source(function(k) d[idx[[k]], ], 7L, formula = y ~ x1,
                           prototype = d, offset = c("o1", "o2"))
    sf2 <- fastglm_multinom(src2, tol = 1e-12)
    de2 <- fastglm_multinom(y ~ x1, data = d, offset = cbind(o1, o2), tol = 1e-12)
    expect_max_diff(predict(sf2, newdata = d[1:4, ], type = "link"),
                    predict(de2, newdata = d[1:4, ], type = "link"), 1e-10)
})

test_that("an SQLite database source agrees with the dense fit", {
    skip_if_not_installed("DBI")
    skip_if_not_installed("RSQLite")
    d <- source_sim_frame()
    con <- DBI::dbConnect(RSQLite::SQLite(), ":memory:")
    on.exit(DBI::dbDisconnect(con))
    DBI::dbWriteTable(con, "d", transform(d, y = as.character(y), f = as.character(f)))
    queries <- sprintf("SELECT * FROM d WHERE blk = %d ORDER BY id", 1:7)
    src <- fastglm_source_dbi(con, queries, formula = y ~ x1 + x2 + f, prototype = d,
                              weights = "w", offset = c("o1", "o2"))
    de <- fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                           offset = cbind(o1, o2), tol = 1e-12)
    sf <- fastglm_multinom(src, tol = 1e-12)
    expect_source_agrees(sf, de)
    expect_identical(sf$passes, sf$iter + 1L)
    sff <- fastglm_multinom(src, firth = TRUE, tol = 1e-12)
    expect_source_agrees(sff, fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                                               offset = cbind(o1, o2), firth = TRUE, tol = 1e-12))
    expect_max_diff(fastglm_vcov(sf, cluster = d$g), fastglm_vcov(de, cluster = d$g), 1e-10)
    expect_max_diff(predict(sf, newdata = d[1:9, ]),
                    predict(de, newdata = d[1:9, ], offset = cbind(d$o1, d$o2)[1:9, ]), 1e-10)
})

test_that("an arrow (Parquet) source agrees with the dense fit", {
    skip_if_not_installed("arrow")
    skip_if_not(arrow::arrow_with_parquet(), "arrow was built without Parquet support")
    d <- source_sim_frame()
    directory <- tempfile("multinom-parquet-")
    dir.create(directory)
    on.exit(unlink(directory, recursive = TRUE))
    paths <- file.path(directory, paste0(1:7, ".parquet"))
    for (k in 1:7) arrow::write_parquet(d[d$blk == k, ], paths[k])
    src <- fastglm_source_arrow(paths, formula = y ~ x1 + x2 + f, prototype = d,
                                weights = "w", offset = c("o1", "o2"))
    de <- fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                           offset = cbind(o1, o2), tol = 1e-12)
    sf <- fastglm_multinom(src, tol = 1e-12)
    expect_source_agrees(sf, de)
    expect_identical(sf$passes, sf$iter + 1L)
    expect_max_diff(predict(sf, newdata = d[1:9, ]),
                    predict(de, newdata = d[1:9, ], offset = cbind(d$o1, d$o2)[1:9, ]), 1e-10)
    sff <- fastglm_multinom(src, firth = TRUE, tol = 1e-12)
    expect_source_agrees(sff, fastglm_multinom(y ~ x1 + x2 + f, data = d, weights = w,
                                               offset = cbind(o1, o2), firth = TRUE, tol = 1e-12))
    expect_max_diff(fastglm_vcov(sf), fastglm_vcov(de), 1e-10)
    expect_max_diff(fastglm_vcov(sf, cluster = d$g), fastglm_vcov(de, cluster = d$g), 1e-10)
})
